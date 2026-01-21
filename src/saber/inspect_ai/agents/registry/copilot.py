"""Copilot agent implementation for SABER.

This module provides the Copilot agent that uses GitHub Copilot CLI
as the reasoning engine for SABER benchmark tasks.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from inspect_ai.event._model import ModelEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.log._transcript import transcript
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.model._chat_message import ToolCall, ToolCallError
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import active_model
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage
from inspect_ai.solver import Solver, TaskState
from inspect_ai.util import sandbox

from ....logging_config import LogCategory, get_saber_logger
from ...integration.copilot_tools import (
    Tool,
    ToolCallTracker,
    convert_mcp_tools_to_copilot,
    create_submit_tool,
    get_saber_mcp_tools,
)
from .models import (
    AnthropicProviderConfig,
    AzureProviderConfig,
    CopilotSessionConfig,
    DefaultUrl,
    DefaultValue,
    EnvVar,
    ModelPrefix,
    OpenAIProviderConfig,
    ProviderConfig,
    ProviderType,
)

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Try to import Copilot SDK - will fail gracefully if not installed
try:
    from copilot import CopilotClient

    COPILOT_SDK_AVAILABLE = True
except ImportError:
    COPILOT_SDK_AVAILABLE = False
    CopilotClient = None


# ==============================================================================
# COPILOT SDK WORKAROUND: Raw Message Retrieval
# ==============================================================================
#
# CONTEXT:
# The Copilot SDK v0.0.388 has a bug in session_event_from_dict() that causes
# session.get_messages() to fail with an AssertionError when parsing the "context"
# field. The SDK expects "context" to be a string or None, but the server sends
# it as a dict containing git/workspace info.
#
# ERROR:
#   File "copilot/generated/session_events.py", line 388, in from_dict
#     context = from_union([from_str, from_none], obj.get("context"))
#   AssertionError
#
# WORKAROUND:
# We bypass the broken parsing by making a raw JSON-RPC request directly to the
# Copilot CLI server. This returns the raw event dicts before the broken
# session_event_from_dict() is applied, allowing us to extract the conversation
# history ourselves.
#
# WHY THIS MATTERS:
# When the model calls tools, the SDK's event handler receives an assistant.message
# event with NO text content - only tool_requests. The model's "thinking" is not
# exposed. By retrieving the full conversation history after each turn, we can:
# 1. Get all assistant messages (text and tool calls) in order
# 2. Reconstruct the full transcript for Inspect AI evaluation
# 3. Not rely on broken streaming events that may miss content
#
# This workaround should be removed once the SDK is fixed.
# Tracking: https://github.com/github/copilot-sdk/issues/XXX (if filed)
# ==============================================================================


async def _get_raw_session_messages(
    session: Any,
    timeout: float = 5.0,
    retries: int = 1,
) -> list[dict[str, Any]]:
    """Get raw message events from Copilot session, bypassing broken SDK parsing.

    This is a WORKAROUND for Copilot SDK v0.0.388 where session.get_messages()
    fails due to a parsing bug with the "context" field.

    Args:
        session: CopilotSession instance
        timeout: Timeout in seconds for each attempt (default 5s, reduced for responsiveness)
        retries: Number of retry attempts on failure (default 1)

    Returns:
        List of raw event dicts from the session history

    Note:
        This accesses private SDK internals (_client.request) and may break
        in future SDK versions. Remove this workaround when SDK is fixed.
    """
    for attempt in range(retries + 1):
        try:
            # Access the internal client to make a raw JSON-RPC request
            # The session._client is the CopilotClient which can make requests
            raw_response = await asyncio.wait_for(
                session._client.request("session.getMessages", {"sessionId": session.session_id}),
                timeout=timeout,
            )
            events: list[dict[str, Any]] = raw_response.get("events", [])
            if attempt > 0:
                logger.debug(
                    f"Raw message retrieval succeeded on attempt {attempt + 1}",
                    extra={"attempt": attempt + 1, "event_count": len(events)},
                )
            return events
        except asyncio.TimeoutError:
            if attempt < retries:
                logger.debug(
                    f"Timeout ({timeout}s) getting raw messages, retrying... (attempt {attempt + 1}/{retries + 1})",
                    extra={"attempt": attempt + 1, "timeout": timeout},
                )
                await asyncio.sleep(0.2)
        except Exception as e:
            # Don't log at warning level - this is expected when session is closing
            logger.debug(
                f"Failed to get raw session messages: {type(e).__name__}: {e}",
                extra={"error": str(e), "error_type": type(e).__name__, "attempt": attempt + 1},
            )
            break

    return []


async def _poll_raw_messages_background(
    session: Any,
    all_raw_assistant_messages: list[dict[str, Any]],
    stop_event: asyncio.Event,
    poll_interval: float = 2.0,
) -> None:
    """Background task to periodically poll for raw messages during session execution.

    This runs in parallel with send_and_wait() to capture messages while the session
    is still active. The session becomes unresponsive after send_and_wait() returns,
    so we need to get messages DURING execution.

    Args:
        session: CopilotSession instance
        all_raw_assistant_messages: List to append captured messages to (shared state)
        stop_event: Event to signal when to stop polling
        poll_interval: Seconds between polls (default 2.0)
    """
    seen_message_ids: set[str] = set()
    poll_count = 0

    while not stop_event.is_set():
        try:
            # Short timeout since we're running in background
            raw_events = await _get_raw_session_messages(session, timeout=3.0, retries=0)

            if raw_events:
                poll_count += 1
                # Extract new assistant messages
                for evt in raw_events:
                    if evt.get("type") == "assistant.message":
                        data = evt.get("data", {})
                        msg_id = data.get("messageId", "")

                        # Skip if we've already seen this message
                        if msg_id and msg_id in seen_message_ids:
                            continue

                        if msg_id:
                            seen_message_ids.add(msg_id)

                        content = data.get("content", "")
                        tool_requests = data.get("toolRequests", [])

                        if content or tool_requests:
                            msg_data = {
                                "message_id": msg_id,
                                "content": content,
                                "tool_requests": tool_requests,
                            }
                            all_raw_assistant_messages.append(msg_data)
                            logger.debug(
                                "Background poll captured assistant message",
                                extra={
                                    "poll_count": poll_count,
                                    "has_content": bool(content),
                                    "tool_count": len(tool_requests),
                                    "total_messages": len(all_raw_assistant_messages),
                                },
                            )

            # Wait for next poll or stop signal
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Continue polling

        except Exception as e:
            # Don't crash on errors - just log and continue
            logger.debug(f"Background poll error: {e}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                break
            except asyncio.TimeoutError:
                pass

    logger.debug(
        "Background message polling stopped",
        extra={
            "poll_count": poll_count,
            "total_messages_captured": len(all_raw_assistant_messages),
        },
    )


def _extract_assistant_messages_from_raw(
    raw_events: list[dict[str, Any]],
    since_index: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Extract assistant messages from raw SDK events.

    Parses raw event dicts to extract assistant.message events, handling both
    text responses and tool-calling responses.

    Args:
        raw_events: List of raw event dicts from _get_raw_session_messages()
        since_index: Only process events at or after this index (for incremental)

    Returns:
        Tuple of (list of assistant message dicts, new index for next call)

    Each message dict contains:
        - message_id: str - unique message identifier
        - content: str - text content (empty string for tool-only messages)
        - tool_requests: list[dict] - tool calls made (empty list for text-only)

    Note:
        When the model calls tools, content is empty string "". This is standard
        OpenAI API behavior - the tool call IS the response, there's no separate
        "thinking" text exposed.
    """
    messages = []

    for i, evt in enumerate(raw_events):
        if i < since_index:
            continue

        if evt.get("type") == "assistant.message":
            data = evt.get("data", {})
            message_id = data.get("messageId", "")
            content = data.get("content", "")  # Empty string for tool-calling turns
            tool_requests = data.get("toolRequests", [])

            # Include messages that have content OR tool_requests (or both)
            if content or tool_requests:
                messages.append(
                    {
                        "message_id": message_id,
                        "content": content,
                        "tool_requests": tool_requests,
                    }
                )

    return messages, len(raw_events)


def _extract_user_messages_from_raw(
    raw_events: list[dict[str, Any]],
    since_index: int = 0,
) -> list[str]:
    """Extract user message content from raw SDK events.

    Args:
        raw_events: List of raw event dicts
        since_index: Only process events at or after this index

    Returns:
        List of user message content strings
    """
    messages = []

    for i, evt in enumerate(raw_events):
        if i < since_index:
            continue

        if evt.get("type") == "user.message":
            content = evt.get("data", {}).get("content", "")
            if content:
                messages.append(content)

    return messages


# ==============================================================================
# END COPILOT SDK WORKAROUND
# ==============================================================================


def _get_provider_config_from_inspect() -> ProviderConfig | None:
    """Extract provider configuration from the active Inspect AI model.

    This derives the BYOK (Bring Your Own Key) provider configuration
    automatically from the model configured for Inspect AI, including
    Azure OpenAI, OpenAI, and Anthropic.

    Returns:
        ProviderConfig subclass instance, or None if using Copilot auth
    """
    model = active_model()
    if model is None:
        logger.debug("No active Inspect AI model, using Copilot default auth")
        return None

    api = model.api
    model_name = api.model_name

    # Detect provider type from model name prefix or API class
    provider_type: ProviderType | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_version: str | None = None

    # Check for Azure OpenAI (model name starts with "openai/azure/" or "azure/")
    if model_name.startswith(ModelPrefix.AZURE_OPENAI) or model_name.startswith(ModelPrefix.AZURE):
        provider_type = ProviderType.AZURE
        # Get base URL from API or environment
        base_url = getattr(api, "base_url", None) or getattr(api, "endpoint_url", None)
        if not base_url:
            base_url = os.environ.get(EnvVar.AZUREAI_OPENAI_BASE_URL) or os.environ.get(EnvVar.AZURE_OPENAI_BASE_URL)
        # Get API key from API or environment
        api_key = getattr(api, "api_key", None)
        if not api_key:
            api_key = os.environ.get(EnvVar.AZUREAI_OPENAI_API_KEY) or os.environ.get(EnvVar.AZURE_OPENAI_API_KEY)
        # Get API version
        api_version = os.environ.get(EnvVar.AZUREAI_OPENAI_API_VERSION) or os.environ.get(
            EnvVar.OPENAI_API_VERSION, DefaultValue.AZURE_API_VERSION
        )
        # Extract deployment name from model name (e.g., "openai/azure/gpt-4o" -> "gpt-4o")
        # and construct full Azure deployment URL
        deployment_name = model_name.split("/")[-1]
        if base_url and not base_url.endswith("/openai/deployments/"):
            base_url = base_url.rstrip("/") + f"/openai/deployments/{deployment_name}"

    # Check for standard OpenAI
    elif model_name.startswith(ModelPrefix.OPENAI) or hasattr(api, "__class__") and "OpenAI" in api.__class__.__name__:
        provider_type = ProviderType.OPENAI
        base_url = getattr(api, "base_url", None) or DefaultUrl.OPENAI
        api_key = getattr(api, "api_key", None) or os.environ.get(EnvVar.OPENAI_API_KEY)

    # Check for Anthropic
    elif model_name.startswith(ModelPrefix.ANTHROPIC) or "claude" in model_name.lower():
        provider_type = ProviderType.ANTHROPIC
        base_url = getattr(api, "base_url", None) or DefaultUrl.ANTHROPIC
        api_key = getattr(api, "api_key", None) or os.environ.get(EnvVar.ANTHROPIC_API_KEY)

    # Build provider config if we have the required fields
    if provider_type and base_url and api_key:
        provider_config: ProviderConfig

        if provider_type == ProviderType.AZURE:
            provider_config = AzureProviderConfig(
                base_url=base_url,
                api_key=api_key,
                api_version=api_version or DefaultValue.AZURE_API_VERSION,
            )
        elif provider_type == ProviderType.OPENAI:
            provider_config = OpenAIProviderConfig(
                base_url=base_url,
                api_key=api_key,
            )
        elif provider_type == ProviderType.ANTHROPIC:
            provider_config = AnthropicProviderConfig(
                base_url=base_url,
                api_key=api_key,
            )

        logger.info(
            "Derived provider config from Inspect AI model",
            extra={
                "provider_type": provider_type.value,
                "model_name": model_name,
                "base_url": base_url[:50] + "..." if len(base_url) > 50 else base_url,
            },
        )
        return provider_config

    logger.debug(
        "Could not derive provider config from model",
        extra={"model_name": model_name, "has_base_url": bool(base_url), "has_api_key": bool(api_key)},
    )
    return None


class CopilotClientWrapper:
    """Wrapper around CopilotClient that handles SDK availability.

    This wrapper provides a consistent interface whether or not the
    Copilot SDK is installed, allowing for better testing and graceful
    degradation.
    """

    def __init__(self, options: dict[str, Any] | None = None):
        if not COPILOT_SDK_AVAILABLE:
            raise RuntimeError(
                "GitHub Copilot SDK is not installed. " "Install it with: pip install github-copilot-sdk"
            )

        # Set up environment for fnm-installed Node.js and Copilot CLI
        opts = options or {}
        env = opts.get("env", dict(os.environ))

        # Add fnm to PATH if available
        fnm_path = os.path.expanduser("~/.local/share/fnm")
        if os.path.exists(fnm_path):
            # fnm stores Node versions, find copilot in the active version
            import subprocess

            try:
                # Get fnm environment to find the right Node/npm bin
                fnm_cmd = (
                    f'export PATH="{fnm_path}:$PATH" && eval "$(fnm env)" && '
                    'dirname $(which copilot 2>/dev/null || echo "")'
                )
                result = subprocess.run(
                    ["bash", "-c", fnm_cmd],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                copilot_bin_dir = result.stdout.strip()
                if copilot_bin_dir and os.path.exists(copilot_bin_dir):
                    current_path = env.get("PATH", "")
                    env["PATH"] = f"{copilot_bin_dir}:{fnm_path}:{current_path}"
                    logger.debug(f"Added fnm copilot bin to PATH: {copilot_bin_dir}")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                logger.debug(f"Could not set up fnm environment: {e}")

        opts["env"] = env
        self._client = CopilotClient(opts)
        self._started = False

    async def start(self) -> None:
        """Start the Copilot CLI server."""
        await self._client.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the Copilot CLI server."""
        if self._started:
            await self._client.stop()
            self._started = False

    async def create_session(self, config: dict[str, Any]) -> Any:
        """Create a new Copilot session."""
        return await self._client.create_session(config)


def copilot_solver(
    instruction_prompt: str,
    assistant_prompt: str,
    submit_prompt: str,
    continue_prompt: str,
    model: str = DefaultValue.MODEL,
    max_turns: int = 50,
    submit: bool | None = None,
    transcript_config: dict[str, Any] | None = None,
    streaming: bool = False,
    timeout: float = 60.0,
    # Provider configuration for BYOK (Bring Your Own Key)
    provider_type: str | None = None,  # ProviderType.AZURE, OPENAI, or ANTHROPIC
    provider_base_url: str | None = None,  # API endpoint URL
    provider_api_key: str | None = None,  # API key
    provider_api_version: str | None = None,  # Azure API version (e.g., '2024-02-15-preview')
) -> Callable[[TaskState], Awaitable[TaskState]]:
    """SABER solver using GitHub Copilot CLI as the reasoning engine.

    This solver creates a Copilot session with SABER's MCP tools registered,
    allowing the Copilot agent to interact with the sandbox environment.

    Provider Configuration:
        The solver automatically derives API credentials from Inspect AI's
        active model configuration. If you run with `--model openai/azure/gpt-4o`,
        the Azure OpenAI endpoint and API key will be extracted from environment
        variables (AZUREAI_OPENAI_BASE_URL, AZUREAI_OPENAI_API_KEY) and passed
        to the Copilot SDK.

        Supported providers:
        - Azure OpenAI: model names starting with "openai/azure/" or "azure/"
        - OpenAI: model names starting with "openai/"
        - Anthropic: model names starting with "anthropic/" or containing "claude"

        You can also explicitly override with provider_* parameters.

    Args:
        instruction_prompt: The main task instructions for the agent
        assistant_prompt: System prompt defining assistant behavior
        submit_prompt: Instructions about task submission
        continue_prompt: Message shown after each step to guide the agent
        model: Copilot model to use (default: gpt-4o)
        max_turns: Maximum conversation turns before stopping (default: 50)
        submit: Whether to enable the submit tool (default: True)
        transcript_config: Optional transcript synchronization config
        streaming: Whether to enable streaming responses (default: False)
        timeout: Timeout in seconds for each turn (default: 60.0)
        provider_type: Override provider type - 'openai', 'azure', or 'anthropic'
        provider_base_url: Override API endpoint URL
        provider_api_key: Override API key
        provider_api_version: Override Azure API version

    Returns:
        Solver function for Inspect AI

    Example - Automatic from Inspect AI:
        # Just run with Inspect's --model flag, credentials auto-derived:
        # uv run inspect eval ... --model openai/azure/gpt-4o

    Example - Explicit BYOK override:
        # Azure OpenAI
        copilot_solver(
            ...,
            provider_type="azure",
            provider_base_url="https://your-resource.openai.azure.com",
            provider_api_key="your-api-key",
            provider_api_version="2024-02-15-preview",
            model="gpt-4o",
        )

        # OpenAI direct
        copilot_solver(
            ...,
            provider_type="openai",
            provider_base_url="https://api.openai.com/v1",
            provider_api_key="sk-...",
            model="gpt-4o",
        )

        # Anthropic
        copilot_solver(
            ...,
            provider_type="anthropic",
            provider_base_url="https://api.anthropic.com",
            provider_api_key="sk-ant-...",
            model="claude-sonnet-4",
        )
    """
    submit_enabled = submit if submit is not None else True

    logger.info(
        "Creating Copilot solver",
        extra={
            "model": model,
            "max_turns": max_turns,
            "submit_enabled": submit_enabled,
            "instruction_length": len(instruction_prompt),
        },
    )

    async def solve(state: TaskState) -> TaskState:
        """Execute the Copilot agent loop.

        Args:
            state: Current task state with metadata and messages

        Returns:
            Updated TaskState with conversation history

        Note:
            This follows the Agent interface (state only) rather than Solver
            interface (state, generate) because SABER's solver_factory calls
            agent(state) directly without the generate parameter.
        """
        # Initialize Copilot client
        client = CopilotClientWrapper({"auto_start": True})

        # Create tool call tracker for transcript visibility
        tool_tracker = ToolCallTracker()

        try:
            await client.start()
            logger.debug("Copilot client started")

            # Get sandbox from Inspect AI context
            sb = sandbox("saber")

            # Get MCP tools from sandbox with tracker
            mcp_tools = await get_saber_mcp_tools(sb)
            copilot_tools = convert_mcp_tools_to_copilot(mcp_tools, _get_mcp_client(sb), tool_tracker)

            # Create submission handler that marks task as submitted
            async def handle_submission(answer: str) -> None:
                """Handle answer submission."""
                state.store.set("submitted", True)
                state.store.set("submission_answer", answer)

                # If sandbox has submit method, call it
                actual_sandbox = sb
                if hasattr(sb, "_sandbox"):
                    actual_sandbox = sb._sandbox

                if hasattr(actual_sandbox, "submit_answer"):
                    await actual_sandbox.submit_answer(answer)

                logger.info(
                    "Answer submitted via Copilot agent",
                    extra={"answer_length": len(answer)},
                )

            # Add submit tool if enabled (with tracker)
            all_tools: list[Tool] = list(copilot_tools)
            if submit_enabled:
                submit_tool = create_submit_tool(handle_submission, tool_tracker)
                all_tools.append(submit_tool)
                logger.debug("Added submit_answer tool")

            # Build system message - include assistant prompt, instructions, and submit prompt
            # This mirrors how the react agent builds its prompt via AgentPrompt
            system_parts = []
            if assistant_prompt:
                system_parts.append(assistant_prompt)
            if instruction_prompt:
                system_parts.append(f"\n\n## Task Instructions\n\n{instruction_prompt}")
            if submit_prompt and submit_enabled:
                system_parts.append(f"\n\n## Submission Guidelines\n\n{submit_prompt}")
            system_content = "".join(system_parts)

            logger.debug(
                "Built system content for Copilot session",
                extra={
                    "assistant_prompt_length": len(assistant_prompt) if assistant_prompt else 0,
                    "instruction_prompt_length": len(instruction_prompt) if instruction_prompt else 0,
                    "submit_prompt_length": len(submit_prompt) if submit_prompt else 0,
                    "total_system_length": len(system_content),
                },
            )

            # Get the list of tool names we're providing
            # This is used to restrict Copilot to ONLY these tools (no filesystem access)
            tool_names = [t.name for t in all_tools]

            # Build provider configuration - either explicit BYOK or derived from Inspect AI
            provider_config: ProviderConfig | None = None

            if provider_type and provider_base_url and provider_api_key:
                # Use explicitly provided BYOK config
                if provider_type == ProviderType.AZURE or provider_type == ProviderType.AZURE.value:
                    provider_config = AzureProviderConfig(
                        base_url=provider_base_url,
                        api_key=provider_api_key,
                        api_version=provider_api_version or DefaultValue.AZURE_API_VERSION,
                    )
                elif provider_type == ProviderType.OPENAI or provider_type == ProviderType.OPENAI.value:
                    provider_config = OpenAIProviderConfig(
                        base_url=provider_base_url,
                        api_key=provider_api_key,
                    )
                elif provider_type == ProviderType.ANTHROPIC or provider_type == ProviderType.ANTHROPIC.value:
                    provider_config = AnthropicProviderConfig(
                        base_url=provider_base_url,
                        api_key=provider_api_key,
                    )

                if provider_config:
                    logger.info(
                        "Using explicit BYOK provider configuration",
                        extra={
                            "provider_type": provider_type,
                            "base_url": provider_base_url[:50] + "..."
                            if len(provider_base_url) > 50
                            else provider_base_url,
                        },
                    )
            else:
                # Try to derive from Inspect AI's active model
                provider_config = _get_provider_config_from_inspect()

            # Create strongly-typed session config
            # IMPORTANT: We use available_tools to restrict Copilot to ONLY our tools
            # This prevents the agent from accessing the local filesystem, reading code, etc.
            session_config = CopilotSessionConfig.create(
                model=model,
                tools=all_tools,
                system_content=system_content,
                streaming=streaming,
                system_mode="append",
                provider=provider_config,
            )

            logger.info(
                "Creating Copilot session with restricted tools",
                extra={
                    "model": model,
                    "tool_count": len(all_tools),
                    "tool_names": tool_names,
                    "using_byok": provider_config is not None,
                    "tools_restricted": True,  # available_tools limits to only our tools
                },
            )

            session = await client.create_session(session_config.to_dict())

            # Track assistant messages and reasoning received during the session
            # The Copilot SDK emits events for assistant reasoning/thinking
            assistant_messages_this_turn: list[str] = []
            assistant_reasoning_this_turn: list[str] = []

            # Track delta content for streaming events
            current_message_deltas: list[str] = []
            current_reasoning_deltas: list[str] = []

            def on_session_event(event: Any) -> None:
                """Capture assistant messages and reasoning from Copilot events."""
                nonlocal current_message_deltas, current_reasoning_deltas

                if not event:
                    return

                # Get event type safely
                event_type = "unknown"
                if hasattr(event, "type"):
                    event_type = str(event.type.value) if hasattr(event.type, "value") else str(event.type)

                # Log ALL events at INFO level during debugging
                logger.info(
                    f"Copilot event received: {event_type}",
                    extra={
                        "event_type": event_type,
                        "has_data": hasattr(event, "data") and event.data is not None,
                    },
                )

                # Only process events with data
                if not hasattr(event, "data") or event.data is None:
                    return

                data = event.data

                # Log all data fields for debugging
                data_fields = {
                    attr: getattr(data, attr, None)
                    for attr in [
                        "content",
                        "delta_content",
                        "summary",
                        "message_id",
                        "turn_id",
                        "reasoning_id",
                        "tool_name",
                        "arguments",
                        "tool_call_id",
                        "result",
                        "text",
                        "message",
                    ]
                    if hasattr(data, attr) and getattr(data, attr, None)
                }
                if data_fields:
                    # For debugging - log ALL event types with their data
                    logger.info(
                        f"Event {event_type} data fields",
                        extra={"fields": {k: str(v)[:200] if v else None for k, v in data_fields.items()}},
                    )

                if event_type == "assistant.message":
                    content = getattr(data, "content", None)
                    if content:
                        # Always append - we'll deduplicate later if needed
                        # Don't deduplicate here as we might miss legitimate separate messages
                        logger.info(
                            "EVENT: assistant.message received with content",
                            extra={
                                "content_length": len(content),
                                "content_preview": content[:200],
                                "current_list_size": len(assistant_messages_this_turn),
                            },
                        )
                        assistant_messages_this_turn.append(content)
                    else:
                        # No content - this is likely a tool-calling message
                        # Check for tool_requests which contain the model's decision
                        tool_requests = getattr(data, "tool_requests", None)
                        if tool_requests:
                            # The model's "message" is the tool call decision
                            # We can construct a description of what tools are being called
                            tool_desc = ", ".join([f"{tr.name}({tr.arguments})" for tr in tool_requests])
                            logger.info(
                                "EVENT: assistant.message with tool_requests (no text content)",
                                extra={"tool_requests": tool_desc[:200]},
                            )
                            # Don't add this as a text message - the tool call IS the message
                        else:
                            logger.debug("assistant.message event has no content and no tool_requests")

                elif event_type == "assistant.intent":
                    # Capture the assistant's intent/reasoning before tool calls
                    intent = getattr(data, "intent", None)
                    if intent:
                        logger.info(
                            "EVENT: assistant.intent received",
                            extra={"intent_length": len(intent), "intent_preview": intent[:200]},
                        )
                        # Add intent as a reasoning message
                        assistant_reasoning_this_turn.append(intent)

                elif event_type == "assistant.message_delta":
                    # Streaming delta - accumulate
                    delta = getattr(data, "delta_content", None) or getattr(data, "content", None)
                    if delta:
                        current_message_deltas.append(delta)
                        logger.debug(f"Accumulated message delta: {len(delta)} chars")

                elif event_type == "assistant.reasoning":
                    content = getattr(data, "content", None)
                    if content:
                        assistant_reasoning_this_turn.append(content)
                        logger.info(
                            "Captured assistant reasoning",
                            extra={"content_length": len(content), "content_preview": content[:200]},
                        )
                    else:
                        logger.debug("assistant.reasoning event has no content (deltas may follow)")

                elif event_type == "assistant.reasoning_delta":
                    # Streaming delta - accumulate
                    delta = getattr(data, "delta_content", None) or getattr(data, "content", None)
                    if delta:
                        current_reasoning_deltas.append(delta)
                        logger.debug(f"Accumulated reasoning delta: {len(delta)} chars")

                elif event_type == "assistant.turn_end":
                    # Turn ended - finalize accumulated deltas if any
                    # NOTE: We might get the same content from both deltas AND assistant.message event
                    # So we deduplicate here by checking if it's already in the list
                    if current_message_deltas:
                        full_message = "".join(current_message_deltas)
                        if full_message:
                            # Only add if not already captured by assistant.message event
                            if full_message not in assistant_messages_this_turn:
                                assistant_messages_this_turn.append(full_message)
                                logger.info(
                                    "Finalized streamed message from deltas (new)",
                                    extra={"content_length": len(full_message)},
                                )
                            else:
                                logger.debug(
                                    "Finalized streamed message from deltas (already captured by event)",
                                    extra={"content_length": len(full_message)},
                                )
                        current_message_deltas.clear()

                    if current_reasoning_deltas:
                        full_reasoning = "".join(current_reasoning_deltas)
                        if full_reasoning and full_reasoning not in assistant_reasoning_this_turn:
                            assistant_reasoning_this_turn.append(full_reasoning)
                            logger.info(
                                "Finalized streamed reasoning from deltas",
                                extra={"content_length": len(full_reasoning)},
                            )
                        current_reasoning_deltas.clear()

                    # Also check for content/summary in turn_end itself
                    content = getattr(data, "content", None) or getattr(data, "summary", None)
                    if content and content not in assistant_messages_this_turn:
                        assistant_messages_this_turn.append(content)
                        logger.info(
                            "Captured content from turn_end",
                            extra={"content_length": len(content)},
                        )

                elif event_type == "assistant.turn_start":
                    # New turn starting - clear delta accumulators
                    current_message_deltas.clear()
                    current_reasoning_deltas.clear()
                    logger.debug("Turn started - cleared delta accumulators")

            # Register single handler for all events
            unsubscribe = session.on(on_session_event)

            # Build the proper message order for the transcript
            # Inspect AI samples come with messages already in state.messages (e.g., user task, assistant prompts)
            # We need to prepend our system message at the start, keeping the existing messages
            #
            # Expected order: system -> existing sample messages (inspect_assistant, user_1) -> instruction
            existing_messages = list(state.messages)  # Make a copy
            state.messages.clear()

            # Add system message FIRST
            state.messages.append(ChatMessageSystem(content=system_content))
            logger.debug(
                "Added system message to transcript",
                extra={"content_length": len(system_content)},
            )

            # Then add all existing messages from the sample (task description, assistant prompt, etc.)
            for msg in existing_messages:
                state.messages.append(msg)
            logger.debug(
                "Added existing sample messages to transcript",
                extra={"message_count": len(existing_messages)},
            )

            # Run agent loop
            submitted = False
            turn = 0
            consecutive_timeouts = 0
            max_consecutive_timeouts = 3  # Stop after 3 consecutive timeouts

            # Cache of all raw assistant messages captured by background poller
            # This is part of the SDK workaround - see _poll_raw_messages_background()
            all_raw_assistant_messages: list[dict[str, Any]] = []

            while turn < max_turns and not submitted and consecutive_timeouts < max_consecutive_timeouts:
                # Determine prompt for this turn
                prompt = instruction_prompt if turn == 0 else continue_prompt

                # Add user message for this turn to transcript
                state.messages.append(ChatMessageUser(content=prompt))

                # Capture the input messages for ModelEvent
                # For turn 0: Include all context (system, sample messages, instruction)
                # For subsequent turns: Only include the new prompt to avoid duplication in viewer
                # The SDK maintains full conversation context internally
                if turn == 0:
                    # First turn - include full context
                    input_messages_for_event = list(state.messages)
                else:
                    # Subsequent turns - only include the new user prompt
                    # The SDK has full history, but we don't want to duplicate in the viewer
                    input_messages_for_event = [ChatMessageUser(content=prompt)]

                logger.debug(
                    f"Sending turn {turn + 1}",
                    extra={
                        "turn": turn + 1,
                        "max_turns": max_turns,
                        "prompt_type": "instruction" if turn == 0 else "continue",
                        "input_message_count": len(input_messages_for_event),
                    },
                )

                # ==============================================================
                # SDK WORKAROUND: Background polling for raw messages
                # ==============================================================
                # The session becomes unresponsive AFTER send_and_wait() returns,
                # so we must capture raw messages DURING execution.
                # Start a background task that polls for raw messages while
                # send_and_wait() is running.
                # ==============================================================
                stop_polling = asyncio.Event()
                poll_task = asyncio.create_task(
                    _poll_raw_messages_background(
                        session=session,
                        all_raw_assistant_messages=all_raw_assistant_messages,
                        stop_event=stop_polling,
                        poll_interval=2.0,
                    )
                )

                # Send message and wait for response
                response = None
                timed_out = False
                try:
                    # Clear messages from previous turn
                    assistant_messages_this_turn.clear()
                    assistant_reasoning_this_turn.clear()

                    response = await asyncio.wait_for(
                        session.send_and_wait({"prompt": prompt}, timeout=timeout),
                        timeout=timeout + 5,  # Extra buffer for cleanup
                    )

                except asyncio.TimeoutError:
                    timed_out = True
                    consecutive_timeouts += 1
                    logger.warning(
                        f"Turn {turn + 1} timed out after {timeout}s "
                        f"({consecutive_timeouts}/{max_consecutive_timeouts} consecutive)",
                        extra={"turn": turn + 1, "consecutive_timeouts": consecutive_timeouts},
                    )
                finally:
                    # Stop the background polling task
                    stop_polling.set()
                    # Give it a moment to finish gracefully
                    try:
                        await asyncio.wait_for(poll_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        poll_task.cancel()
                        try:
                            await poll_task
                        except asyncio.CancelledError:
                            pass
                # ==============================================================
                # END BACKGROUND POLLING
                # ==============================================================

                # Log the response object for debugging
                if response:
                    resp_type = (
                        str(response.type.value)
                        if hasattr(response, "type") and hasattr(response.type, "value")
                        else str(getattr(response, "type", "unknown"))
                    )
                    resp_data = response.data if hasattr(response, "data") else None
                    resp_content = getattr(resp_data, "content", None) if resp_data else None
                    resp_summary = getattr(resp_data, "summary", None) if resp_data else None
                    resp_transformed = getattr(resp_data, "transformed_content", None) if resp_data else None

                    logger.info(
                        f"Response from send_and_wait: type={resp_type}",
                        extra={
                            "has_data": resp_data is not None,
                            "content": resp_content[:300] if resp_content else None,
                            "summary": resp_summary[:300] if resp_summary else None,
                            "transformed_content": resp_transformed[:300] if resp_transformed else None,
                        },
                    )
                else:
                    logger.info("Response from send_and_wait is None")

                # ==============================================================
                # SDK WORKAROUND: Process raw messages captured by background poller
                # ==============================================================
                # The background polling task captured raw messages DURING execution.
                # Now we process them to merge with event-captured content.
                # This is more reliable than post-turn retrieval which times out.
                # ==============================================================
                if all_raw_assistant_messages:
                    logger.info(
                        f"Turn {turn + 1} - background poller captured "
                        f"{len(all_raw_assistant_messages)} raw assistant message(s)",
                        extra={
                            "total_cached_messages": len(all_raw_assistant_messages),
                            "messages": [
                                {
                                    "has_content": bool(m["content"]),
                                    "content_preview": m["content"][:100] if m["content"] else "",
                                    "tool_count": len(m["tool_requests"]),
                                }
                                for m in all_raw_assistant_messages
                            ],
                        },
                    )

                    # Merge raw messages with event-captured messages
                    # Raw messages are authoritative - use them to fill gaps
                    for raw_msg in all_raw_assistant_messages:
                        content = raw_msg["content"]
                        if content and content not in assistant_messages_this_turn:
                            assistant_messages_this_turn.append(content)
                            logger.debug(
                                "Added assistant content from raw history (not captured by events)",
                                extra={"content_preview": content[:100]},
                            )
                else:
                    # No raw messages captured - rely on event-based capture
                    logger.warning(
                        f"Turn {turn + 1} - no raw messages from background poller, using event-based capture only",
                        extra={
                            "event_captured_count": len(assistant_messages_this_turn),
                        },
                    )
                # ==============================================================
                # END SDK WORKAROUND
                # ==============================================================
                # ==============================================================
                # END SDK WORKAROUND
                # ==============================================================

                # ALWAYS record tool calls and messages, even on timeout
                # Tool calls happen during send_and_wait and are tracked separately

                # Log what we captured during this turn for debugging
                logger.info(
                    f"Turn {turn + 1} - captured content before processing",
                    extra={
                        "reasoning_count": len(assistant_reasoning_this_turn),
                        "reasoning_contents": [r[:100] for r in assistant_reasoning_this_turn],
                        "message_count": len(assistant_messages_this_turn),
                        "message_contents": [m[:100] for m in assistant_messages_this_turn],
                    },
                )

                # First, add any assistant reasoning captured during this turn
                # This shows the agent's thinking process
                for reasoning_content in assistant_reasoning_this_turn:
                    state.messages.append(ChatMessageAssistant(content=reasoning_content))
                    logger.info(
                        "Added captured assistant reasoning to transcript",
                        extra={"content_length": len(reasoning_content), "content_preview": reasoning_content[:100]},
                    )

                # Add assistant messages captured during this turn
                for msg_content in assistant_messages_this_turn:
                    # Skip if same as reasoning (avoid duplicates)
                    if msg_content not in assistant_reasoning_this_turn:
                        state.messages.append(ChatMessageAssistant(content=msg_content))
                        logger.info(
                            "Added captured assistant message to transcript",
                            extra={"content_length": len(msg_content), "content_preview": msg_content[:100]},
                        )

                # Collect tool calls from tracker and add to transcript
                # These are the tools that were called during this turn
                tool_calls_this_turn = tool_tracker.get_and_clear()
                if tool_calls_this_turn:
                    # Build ToolCall objects for the assistant message
                    tool_call_objects = []
                    for tc in tool_calls_this_turn:
                        tool_call_objects.append(
                            ToolCall(
                                id=tc.tool_call_id,
                                function=tc.tool_name,
                                arguments=tc.arguments,
                                type="function",
                            )
                        )

                    # Add assistant message with tool calls to state.messages
                    state.messages.append(
                        ChatMessageAssistant(
                            content="",  # Tool calls don't have separate content
                            tool_calls=tool_call_objects,
                        )
                    )

                    # Add tool response messages to state.messages
                    # Note: ToolEvent recording is done later, AFTER ModelEvent for correct transcript order
                    for tc in tool_calls_this_turn:
                        # Build error object if this was an error
                        error_obj = None
                        if tc.is_error:
                            error_obj = ToolCallError(type="unknown", message=tc.result)

                        state.messages.append(
                            ChatMessageTool(
                                content=tc.result,
                                tool_call_id=tc.tool_call_id,
                                error=error_obj,
                            )
                        )

                    logger.info(
                        f"Added {len(tool_calls_this_turn)} tool calls to state.messages",
                        extra={
                            "turn": turn + 1,
                            "tool_names": [tc.tool_name for tc in tool_calls_this_turn],
                            "timed_out": timed_out,
                        },
                    )

                # Also capture the final response content if different from what we captured
                if response and hasattr(response, "data") and hasattr(response.data, "content"):
                    content = response.data.content
                    # Only add if we didn't already capture this message
                    all_captured = set(assistant_messages_this_turn) | set(assistant_reasoning_this_turn)
                    if content and content not in all_captured:
                        assistant_messages_this_turn.append(content)
                        state.messages.append(ChatMessageAssistant(content=content))
                        logger.info(
                            "Added final response content to transcript",
                            extra={"content_length": len(content), "content_preview": content[:200]},
                        )
                    # Reset timeout counter on successful response
                    consecutive_timeouts = 0

                # NOTE: session.get_messages() is broken in copilot SDK 0.0.388 due to a parsing
                # bug with the "context" field. We rely on events captured via on_session_event()
                # and the response object from send_and_wait() instead.
                #
                # Assistant content sources (in order of reliability):
                # 1. response.data.content - the final message from send_and_wait
                # 2. assistant.message events - captured via on_session_event
                # 3. assistant.message_delta events - streaming chunks (accumulated at turn_end)
                # 4. assistant.reasoning events - reasoning/thinking content

                # Log what we captured this turn
                logger.info(
                    f"Turn {turn + 1} captured assistant content",
                    extra={
                        "reasoning_count": len(assistant_reasoning_this_turn),
                        "message_count": len(assistant_messages_this_turn),
                        "total_reasoning_chars": sum(len(r) for r in assistant_reasoning_this_turn),
                        "total_message_chars": sum(len(m) for m in assistant_messages_this_turn),
                    },
                )

                # Record ModelEvent for this turn FIRST
                # This captures the model's decision (including tool calls) before tool execution
                # Correct transcript order: ModelEvent (model response) -> ToolEvent (tool execution)
                turn_end_time = datetime.now(timezone.utc)

                # Build the assistant message with all content from this turn
                all_assistant_content = "\n\n".join(assistant_reasoning_this_turn + assistant_messages_this_turn)

                # Build tool calls for ModelEvent (if any)
                assistant_tool_calls = []
                if tool_calls_this_turn:
                    for tc in tool_calls_this_turn:
                        assistant_tool_calls.append(
                            ToolCall(
                                id=tc.tool_call_id,
                                function=tc.tool_name,
                                arguments=tc.arguments,
                                type="function",
                            )
                        )

                # Create the assistant message for the ModelOutput
                assistant_message = ChatMessageAssistant(
                    content=all_assistant_content or "",
                    tool_calls=assistant_tool_calls if assistant_tool_calls else None,
                )

                # Build model output
                model_output = ModelOutput(
                    model=model if model != DefaultValue.MODEL else "copilot",
                    choices=[
                        ChatCompletionChoice(
                            message=assistant_message,
                            stop_reason="tool_calls" if assistant_tool_calls else "stop",
                        )
                    ],
                    usage=ModelUsage(),  # We don't have usage info from Copilot SDK
                )

                # Record ModelEvent FIRST - this shows the model's response and decision to call tools
                # Use input_messages_for_event which captures the full conversation history
                # up to and including the user prompt for this turn (before the response)
                model_event = ModelEvent(
                    model=model if model != DefaultValue.MODEL else "copilot",
                    input=input_messages_for_event,
                    tools=[],  # Could populate with ToolInfo if needed
                    tool_choice="auto",
                    config=GenerateConfig(),
                    output=model_output,
                    completed=turn_end_time,
                )
                transcript()._event(model_event)

                logger.debug(
                    "Recorded ModelEvent for turn",
                    extra={
                        "turn": turn + 1,
                        "input_message_count": len(input_messages_for_event),
                        "output_length": len(all_assistant_content) if all_assistant_content else 0,
                        "tool_calls": len(assistant_tool_calls),
                    },
                )

                # Record ToolEvents AFTER ModelEvent - this shows the actual tool execution results
                for tc in tool_calls_this_turn:
                    # Build error object if this was an error
                    error_obj = None
                    if tc.is_error:
                        error_obj = ToolCallError(type="unknown", message=tc.result)

                    tool_event = ToolEvent(
                        id=tc.tool_call_id,
                        function=tc.tool_name,
                        arguments=tc.arguments,
                        result=tc.result,
                        error=error_obj,
                        completed=datetime.now(timezone.utc),
                    )
                    transcript()._event(tool_event)

                if tool_calls_this_turn:
                    logger.debug(
                        f"Recorded {len(tool_calls_this_turn)} ToolEvents after ModelEvent",
                        extra={
                            "turn": turn + 1,
                            "tool_names": [tc.tool_name for tc in tool_calls_this_turn],
                        },
                    )

                # Check if submitted
                submitted = state.store.get("submitted") or False

                turn += 1

            # ==============================================================
            # SDK WORKAROUND: Final transcript reconciliation
            # ==============================================================
            # Log all raw assistant messages we captured during the session.
            # This provides a complete audit trail even if individual turn
            # retrievals had issues.
            # ==============================================================
            if all_raw_assistant_messages:
                logger.info(
                    f"SDK workaround: captured {len(all_raw_assistant_messages)} total raw "
                    "assistant messages across all turns",
                    extra={
                        "total_messages": len(all_raw_assistant_messages),
                        "messages_with_content": sum(1 for m in all_raw_assistant_messages if m["content"]),
                        "messages_with_tools": sum(1 for m in all_raw_assistant_messages if m["tool_requests"]),
                        "sample_contents": [
                            m["content"][:100] if m["content"] else f"[{len(m['tool_requests'])} tool calls]"
                            for m in all_raw_assistant_messages[:5]  # First 5 for logging
                        ],
                    },
                )
            else:
                logger.warning(
                    "SDK workaround: no raw assistant messages were captured during session "
                    "(background poller may have failed)",
                )
            # ==============================================================
            # END SDK WORKAROUND
            # ==============================================================

            # Clean up session
            try:
                # Unsubscribe from events
                unsubscribe()
                await session.destroy()
            except Exception as e:
                logger.warning(f"Error destroying session: {e}")

            logger.info(
                "Copilot solver completed",
                extra={
                    "turns": turn,
                    "submitted": submitted,
                    "message_count": len(state.messages),
                },
            )

        except Exception as e:
            logger.error(
                f"Error in Copilot solver: {e}",
                exc_info=True,
            )
            raise

        finally:
            # Always stop client
            await client.stop()
            logger.debug("Copilot client stopped")

        return state

    return solve


def _get_mcp_client(sandbox: Any) -> Any:
    """Extract MCP client from sandbox for tool execution.

    Args:
        sandbox: SABERSandboxEnvironment (possibly wrapped in proxy)

    Returns:
        MCP client instance
    """
    actual_sandbox = sandbox
    if hasattr(sandbox, "_sandbox"):
        actual_sandbox = sandbox._sandbox

    return actual_sandbox._mcp_client


def create_agent(**kwargs: Any) -> Callable[..., Any]:
    """Create a Copilot agent with SABER integration.

    This agent uses the GitHub Copilot CLI as the reasoning engine,
    with SABER's MCP tools registered for sandbox interaction.

    Args:
        **kwargs: Additional parameters passed to copilot_solver()
            - model: Copilot model to use (default: gpt-4o)
            - max_turns: Maximum conversation turns (default: 50)
            - streaming: Enable streaming responses (default: False)
            - timeout: Per-turn timeout in seconds (default: 60.0)
            - provider_type: 'openai', 'azure', or 'anthropic' for BYOK
            - provider_base_url: API endpoint URL for BYOK
            - provider_api_key: API key for BYOK
            - provider_api_version: Azure API version (e.g., '2024-02-15-preview')

    Returns:
        Factory function that receives prompts from task execution

    Usage:
        # In domain configuration:
        roles:
          red:
            agent: copilot
            model: gpt-4o

        # BYOK with Azure OpenAI:
        roles:
          red:
            agent: copilot
            model: gpt-4o
            provider_type: azure
            provider_base_url: https://your-resource.openai.azure.com
            provider_api_key: ${AZURE_OPENAI_API_KEY}
            provider_api_version: 2024-02-15-preview

        # The factory is called by solver_factory with runtime prompts
    """

    def create_with_prompts(
        instruction_prompt: str,
        assistant_prompt: str,
        submit_prompt: str,
        continue_prompt: str,
        transcript_config: dict[str, Any] | None = None,
        submit: bool | None = None,
    ) -> Solver:
        """Inner factory that receives prompts from task execution.

        Args:
            instruction_prompt: The main instructions for the agent
            assistant_prompt: Assistant behavior prompt
            submit_prompt: Instructions about submission
            continue_prompt: Message shown after each step
            transcript_config: Optional transcript configuration dict
            submit: Whether to enable the submit tool (default: True)
        """
        logger.debug(
            "Creating Copilot agent with prompts",
            extra={
                "instruction_length": len(instruction_prompt),
                "assistant_length": len(assistant_prompt),
                "submit_length": len(submit_prompt),
                "continue_length": len(continue_prompt),
                "submit_enabled": submit if submit is not None else True,
            },
        )

        return copilot_solver(
            instruction_prompt=instruction_prompt,
            assistant_prompt=assistant_prompt,
            submit_prompt=submit_prompt,
            continue_prompt=continue_prompt,
            transcript_config=transcript_config,
            submit=submit,
            **kwargs,
        )

    return create_with_prompts


__all__ = ["create_agent", "copilot_solver"]
