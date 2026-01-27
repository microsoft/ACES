"""Copilot agent implementation for SABER.

This module provides the Copilot agent that uses GitHub Copilot SDK
as the reasoning engine for SABER benchmark tasks.

Key Design:
- Uses SDK events (not raw message hacks) for capturing transcripts
- Clean OOP design with separate classes for event capture and session management
- Modular helper functions for provider config and tool setup

Transcript Capture Notes:
- The SDK defines assistant.reasoning and assistant.intent event types, but the
  Copilot server does not currently send them for gpt-4o or gpt-5 models.
- GPT-5 uses a built-in `report_intent` tool to report intent before tool calls.
  We capture these as reasoning/intent messages for the transcript.
- All assistant.message content (including explanatory text) is captured.
- Tool calls and results are fully captured.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from inspect_ai.event._model import ModelEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.log._transcript import transcript
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._chat_message import ToolCall, ToolCallError
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import active_model
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage
from inspect_ai.solver import Solver, TaskState
from inspect_ai.util import sandbox

from ....logging_config import LogCategory, get_saber_logger
from ...integration.agent_transcript_sync import AgentTranscriptSync
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
    CopilotBuiltInTools,
    CopilotSessionConfig,
    DefaultValue,
    EnvVar,
    ModelPrefix,
    OpenAIProviderConfig,
    ProviderConfig,
    ProviderType,
)

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Check SDK availability
try:
    from copilot import CopilotClient

    COPILOT_SDK_AVAILABLE = True
except ImportError:
    COPILOT_SDK_AVAILABLE = False
    CopilotClient = None


# =============================================================================
# Event Capture Classes
# =============================================================================


@dataclass
class CapturedMessage:
    """A captured message from the Copilot session."""

    message_id: str | None
    content: str
    message_type: str  # "text", "tool_call", "reasoning", "intent"
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None


@dataclass
class CapturedToolResult:
    """A captured tool execution result."""

    tool_call_id: str | None
    result: str


@dataclass
class AssistantMessageGroup:
    """A group of content from a single assistant.message event.

    In the Copilot SDK, each assistant.message event can contain:
    - Intent (from report_intent tool)
    - Text content (the assistant's explanation)
    - Multiple tool_calls (actions to take)

    These should be rendered together, not split apart.
    """

    message_id: str | None
    intent: str | None = None
    text: str | None = None
    tool_calls: list[CapturedMessage] = field(default_factory=list)
    # tool_call_id -> result mapping
    tool_results: dict[str, str] = field(default_factory=dict)

    def all_results_received(self) -> bool:
        """Check if we have results for all tool calls."""
        if not self.tool_calls:
            return True
        return all(tc.tool_call_id in self.tool_results for tc in self.tool_calls)


@dataclass
class TurnCapture:
    """Captured data from a single conversation turn.

    Tracks message groups to maintain proper association between
    assistant text/intent and their tool calls/results.
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    # Ordered list of message groups (each from one assistant.message event)
    message_groups: list[AssistantMessageGroup] = field(default_factory=list)
    # Legacy: tool_results for backward compatibility
    tool_results: list[CapturedToolResult] = field(default_factory=list)
    # Map tool_call_id -> message_group for quick lookup when results arrive
    _tool_call_to_group: dict[str, AssistantMessageGroup] = field(default_factory=dict)

    def has_content(self) -> bool:
        """Check if any content was captured this turn."""
        return bool(self.message_groups)

    def get_all_text_content(self) -> list[str]:
        """Get all text content in order."""
        content = []
        for group in self.message_groups:
            if group.intent:
                content.append(group.intent)
            if group.text:
                content.append(group.text)
        return content

    # Convenience properties for backward compatibility
    @property
    def assistant_messages(self) -> list[CapturedMessage]:
        return [CapturedMessage(g.message_id, g.text or "", "text") for g in self.message_groups if g.text]

    @property
    def reasoning_messages(self) -> list[CapturedMessage]:
        return [CapturedMessage(g.message_id, g.intent or "", "intent") for g in self.message_groups if g.intent]

    @property
    def tool_calls(self) -> list[CapturedMessage]:
        result = []
        for g in self.message_groups:
            result.extend(g.tool_calls)
        return result

    def format_transcript(self, system_message: str = "", user_prompt: str = "") -> str:
        """Format a clean transcript showing the conversation flow.

        Format:
        <system>...</system>
        <user>...</user>
        <assistant>
          [Intent] ...
          reasoning text...
        </assistant>
        <tool name="bash" call_id="...">
          command: ls -la
        </tool>
        <tool_result call_id="...">
          output...
        </tool_result>
        ...

        Args:
            system_message: Optional system message to include
            user_prompt: Optional user prompt to include

        Returns:
            Formatted transcript string
        """
        lines = []

        # System message
        if system_message:
            lines.append("<system>")
            # Truncate long system messages
            if len(system_message) > 500:
                lines.append(f"  {system_message[:500]}...")
            else:
                for line in system_message.split("\n"):
                    lines.append(f"  {line}")
            lines.append("</system>")

        # User prompt
        if user_prompt:
            lines.append("\n<user>")
            for line in user_prompt.split("\n"):
                lines.append(f"  {line}")
            lines.append("</user>")

        # Process message groups - each group keeps intent/text/tool_calls together
        for group in self.message_groups:
            # Collect assistant content for this group
            assistant_parts = []

            if group.intent:
                assistant_parts.append(group.intent)

            if group.text:
                # Wrap long content
                text_lines = []
                for line in group.text.split("\n"):
                    if len(line) > 100:
                        text_lines.append(f"{line[:100]}...")
                    else:
                        text_lines.append(line)
                assistant_parts.append("\n".join(text_lines))

            # Output assistant block if there's content
            if assistant_parts:
                lines.append("\n<assistant>")
                for part in assistant_parts:
                    for line in part.split("\n"):
                        lines.append(f"  {line}")
                lines.append("</assistant>")

            # Output tool calls with their results interleaved
            for tc in group.tool_calls:
                call_id = tc.tool_call_id or ""
                lines.append(f'\n<tool name="{tc.tool_name}" call_id="{call_id}">')
                if tc.tool_arguments:
                    for k, v in tc.tool_arguments.items():
                        v_str = str(v)
                        if len(v_str) > 100:
                            v_str = v_str[:100] + "..."
                        lines.append(f"  {k}: {v_str}")
                lines.append("</tool>")

                # Immediately add the result for this tool call
                if call_id in group.tool_results:
                    result = group.tool_results[call_id]
                    lines.append(f'\n<tool_result call_id="{call_id}">')
                    # Truncate long results
                    if len(result) > 300:
                        result = result[:300] + "...(truncated)"
                    for line in result.split("\n")[:10]:  # Max 10 lines
                        lines.append(f"  {line}")
                    if result.count("\n") > 10:
                        lines.append("  ...(more lines)")
                    lines.append("</tool_result>")

        return "\n".join(lines)


class EventCapture:
    """Captures events from a Copilot session via the SDK event system.

    This class registers as an event handler on the session and captures
    all relevant events (assistant messages, reasoning, tool calls, tool results).
    This approach uses the official SDK event API rather than raw message hacks.
    """

    def __init__(self, debug_all_events: bool = False) -> None:
        self._current_turn = TurnCapture()
        self._idle_event = asyncio.Event()
        self._error: Exception | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._debug_all_events = debug_all_events
        # Track seen message IDs to avoid duplicates
        self._seen_message_ids: set[str] = set()

    def attach(self, session: Any) -> None:
        """Attach to a session and start capturing events."""
        self._unsubscribe = session.on(self._on_event)

    def detach(self) -> None:
        """Detach from the session."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def reset_turn(self) -> TurnCapture:
        """Reset for a new turn, returning the previous turn's data."""
        previous = self._current_turn
        self._current_turn = TurnCapture()
        self._idle_event.clear()
        self._error = None
        # Don't clear _seen_message_ids - we want to track across turns
        return previous

    async def wait_for_idle(self, timeout: float) -> None:
        """Wait for the session to become idle."""
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for session idle after {timeout}s")

        if self._error:
            raise self._error

    @property
    def current_turn(self) -> TurnCapture:
        """Get the current turn's captured data."""
        return self._current_turn

    def _on_event(self, event: Any) -> None:
        """Handle events from the Copilot session."""
        if not event:
            return

        # Get event type string
        event_type = str(event.type.value) if hasattr(event.type, "value") else str(event.type)

        # Record all events with their data for debugging
        event_record = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
        }

        # Get event data
        data = getattr(event, "data", None)

        # Log ALL events at debug level so we can see what's happening
        if self._debug_all_events and data:
            # Extract common fields for debugging
            data_fields = {}
            for attr in ["content", "message_id", "tool_requests", "intent", "reasoning", "summary"]:
                val = getattr(data, attr, None)
                if val:
                    data_fields[attr] = str(val)[:200] if isinstance(val, str) else str(val)[:200]
            if data_fields:
                logger.info(f"Event {event_type}: {data_fields}")

        self._current_turn.events.append(event_record)

        if not data:
            # Handle control events without data
            if event_type == "session.idle":
                logger.debug("Session idle event received")
                self._idle_event.set()
            return

        # Route to specific handler based on event type
        if event_type == "assistant.message":
            self._handle_assistant_message(data)
        elif event_type == "assistant.reasoning":
            self._handle_reasoning(data)
        elif event_type == "assistant.intent":
            self._handle_intent(data)
        elif event_type == "assistant.turn_end":
            self._handle_turn_end(data)
        elif event_type == "tool.execution_complete":
            self._handle_tool_result(data)
        elif event_type == "session.idle":
            logger.debug("Session idle event received")
            self._idle_event.set()
        elif event_type == "session.error":
            error_msg = getattr(data, "message", str(data))
            self._error = Exception(f"Session error: {error_msg}")
            self._idle_event.set()

    def _handle_assistant_message(self, data: Any) -> None:
        """Handle assistant.message event.

        Each assistant.message event is a unit containing:
        - Optional intent (from report_intent tool)
        - Optional text content
        - Optional tool calls

        We create an AssistantMessageGroup for each event to keep
        these elements together for proper transcript rendering.
        """
        message_id = getattr(data, "message_id", None)
        content = getattr(data, "content", None)
        tool_requests = getattr(data, "tool_requests", None)

        # Check for duplicates based on message_id
        if message_id and message_id in self._seen_message_ids:
            logger.debug(f"Skipping duplicate message {message_id}")
            return
        if message_id:
            self._seen_message_ids.add(message_id)

        # Create a new message group for this assistant.message event
        group = AssistantMessageGroup(message_id=message_id)

        # Extract intent from report_intent tool call
        if tool_requests:
            for tr in tool_requests:
                tool_name = getattr(tr, "name", None)
                tool_arguments = getattr(tr, "arguments", None)

                if tool_name == CopilotBuiltInTools.REPORT_INTENT:
                    intent_text = tool_arguments.get("intent", "") if tool_arguments else ""
                    if intent_text:
                        group.intent = f"[Intent] {intent_text}"
                        logger.info(
                            "Captured intent from report_intent tool",
                            extra={"intent": intent_text},
                        )

        # Capture text content
        if content:
            group.text = content
            logger.info(
                "Captured assistant text message",
                extra={"content_preview": content[:100], "message_id": message_id},
            )

        # Capture tool calls (excluding report_intent)
        if tool_requests:
            for tr in tool_requests:
                tool_name = getattr(tr, "name", None)
                tool_arguments = getattr(tr, "arguments", None)
                tool_call_id = getattr(tr, "tool_call_id", None)

                # Skip report_intent - already captured as intent above
                if tool_name == CopilotBuiltInTools.REPORT_INTENT:
                    continue

                msg = CapturedMessage(
                    message_id=message_id,
                    content="",
                    message_type="tool_call",
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                )
                group.tool_calls.append(msg)
                # Map tool_call_id to this group for result lookup
                if tool_call_id:
                    self._current_turn._tool_call_to_group[tool_call_id] = group
                logger.debug(
                    "Captured tool call",
                    extra={"tool_name": tool_name, "arguments": tool_arguments},
                )

        # Only add group if it has content
        if group.intent or group.text or group.tool_calls:
            self._current_turn.message_groups.append(group)

    def _handle_reasoning(self, data: Any) -> None:
        """Handle assistant.reasoning event - captures model's thinking.

        Creates a standalone message group for reasoning.
        """
        content = getattr(data, "content", None)
        message_id = getattr(data, "reasoning_id", None) or getattr(data, "message_id", None)

        if content:
            group = AssistantMessageGroup(
                message_id=message_id,
                text=f"[Reasoning] {content}",
            )
            self._current_turn.message_groups.append(group)
            logger.info(
                "Captured assistant reasoning",
                extra={"content_preview": content[:100], "message_id": message_id},
            )

    def _handle_intent(self, data: Any) -> None:
        """Handle assistant.intent event - captures model's intent before tool calls.

        Creates a standalone message group for intent (if not from report_intent tool).
        """
        intent = getattr(data, "intent", None) or getattr(data, "content", None)
        message_id = getattr(data, "message_id", None)

        if intent:
            group = AssistantMessageGroup(
                message_id=message_id,
                intent=f"[Intent] {intent}",
            )
            self._current_turn.message_groups.append(group)
            logger.info(
                "Captured assistant intent",
                extra={"content_preview": intent[:100], "message_id": message_id},
            )

    def _handle_turn_end(self, data: Any) -> None:
        """Handle assistant.turn_end event - may contain summary or content."""
        content = getattr(data, "content", None) or getattr(data, "summary", None)
        message_id = getattr(data, "message_id", None)

        if content:
            # Only add if not already captured
            existing_contents = {g.text for g in self._current_turn.message_groups if g.text}
            if content not in existing_contents:
                group = AssistantMessageGroup(
                    message_id=message_id,
                    text=content,
                )
                self._current_turn.message_groups.append(group)
                logger.info(
                    "Captured content from turn_end",
                    extra={"content_preview": content[:100]},
                )

    def _handle_tool_result(self, data: Any) -> None:
        """Handle tool.execution_complete event.

        Associates the result with its corresponding tool call in the message group.
        """
        tool_call_id = getattr(data, "tool_call_id", None)
        result_str = str(getattr(data, "result", ""))

        # Add to legacy list for backward compatibility
        result = CapturedToolResult(
            tool_call_id=tool_call_id,
            result=result_str,
        )
        self._current_turn.tool_results.append(result)

        # Associate result with its message group
        if tool_call_id and tool_call_id in self._current_turn._tool_call_to_group:
            group = self._current_turn._tool_call_to_group[tool_call_id]
            group.tool_results[tool_call_id] = result_str
            logger.debug(f"Associated tool result with group {group.message_id}")

        logger.debug(f"Captured tool result for {tool_call_id}")


# =============================================================================
# Client Wrapper
# =============================================================================


class CopilotClientWrapper:
    """Wrapper around CopilotClient that handles SDK availability and setup."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        if not COPILOT_SDK_AVAILABLE:
            raise RuntimeError(
                "GitHub Copilot SDK is not installed. " "Install it with: pip install github-copilot-sdk"
            )

        opts = options or {}
        env = opts.get("env", dict(os.environ))

        # Add fnm to PATH if available
        env = self._setup_fnm_path(env)
        opts["env"] = env

        self._client = CopilotClient(opts)
        self._started = False

    def _setup_fnm_path(self, env: dict[str, str]) -> dict[str, str]:
        """Set up fnm path for Node.js access."""
        fnm_path = os.path.expanduser("~/.local/share/fnm")
        if os.path.exists(fnm_path):
            import subprocess

            try:
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

        return env

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


# =============================================================================
# Provider Configuration
# =============================================================================


def get_provider_config_from_inspect() -> ProviderConfig | None:
    """Extract provider configuration from the active Inspect AI model.

    Supports both API key and Entra ID (bearer token) authentication.
    For Azure, if no API key is found but a token_provider exists (from
    DefaultAzureCredential via az login), it will use bearer token auth.

    Returns:
        ProviderConfig subclass instance, or None if using Copilot auth
    """
    model = active_model()
    if model is None:
        logger.debug("No active Inspect AI model, using Copilot default auth")
        return None

    api = model.api
    model_name = api.model_name

    # Check for token_provider (Entra ID auth from inspect_ai)
    token_provider = getattr(api, "token_provider", None)

    logger.info(
        "Checking active model for provider config",
        extra={
            "model_name": model_name,
            "api_class": api.__class__.__name__,
            "has_base_url": hasattr(api, "base_url"),
            "has_api_key": hasattr(api, "api_key"),
            "has_token_provider": token_provider is not None,
        },
    )

    provider_type: ProviderType | None = None
    base_url: str | None = None
    api_key: str | None = None
    bearer_token: str | None = None
    api_version: str | None = None

    # Azure OpenAI
    if model_name.startswith(ModelPrefix.AZURE_OPENAI) or model_name.startswith(ModelPrefix.AZURE):
        provider_type = ProviderType.AZURE
        base_url = getattr(api, "base_url", None) or getattr(api, "endpoint_url", None)
        if not base_url:
            base_url = os.environ.get(EnvVar.AZUREAI_OPENAI_BASE_URL) or os.environ.get(EnvVar.AZURE_OPENAI_BASE_URL)

        # Try API key first
        api_key = (
            getattr(api, "api_key", None)
            or os.environ.get(EnvVar.AZUREAI_OPENAI_API_KEY)
            or os.environ.get(EnvVar.AZURE_OPENAI_API_KEY)
        )

        # If no API key, try Entra ID token provider (from inspect_ai's DefaultAzureCredential)
        if not api_key and token_provider is not None:
            try:
                bearer_token = token_provider()
                logger.info(
                    "Using Entra ID bearer token from inspect_ai token_provider",
                    extra={"model_name": model_name},
                )
            except Exception as e:
                logger.warning(f"Failed to get bearer token from token_provider: {e}")

        api_version = os.environ.get(EnvVar.AZUREAI_OPENAI_API_VERSION) or os.environ.get(
            EnvVar.OPENAI_API_VERSION, DefaultValue.AZURE_API_VERSION
        )
        deployment_name = model_name.split("/")[-1]
        if base_url and "/openai/deployments/" not in base_url:
            base_url = base_url.rstrip("/") + f"/openai/deployments/{deployment_name}"

    # Standard OpenAI
    elif model_name.startswith(ModelPrefix.OPENAI) or (
        hasattr(api, "__class__") and "OpenAI" in api.__class__.__name__
    ):
        provider_type = ProviderType.OPENAI
        base_url = getattr(api, "base_url", None) or "https://api.openai.com/v1"
        api_key = getattr(api, "api_key", None) or os.environ.get(EnvVar.OPENAI_API_KEY)

    # Anthropic
    elif model_name.startswith(ModelPrefix.ANTHROPIC) or "claude" in model_name.lower():
        provider_type = ProviderType.ANTHROPIC
        base_url = getattr(api, "base_url", None) or "https://api.anthropic.com"
        api_key = getattr(api, "api_key", None) or os.environ.get(EnvVar.ANTHROPIC_API_KEY)

    # Build provider config (supports both api_key and bearer_token)
    if provider_type and base_url and (api_key or bearer_token):
        config: AzureProviderConfig | OpenAIProviderConfig | AnthropicProviderConfig
        if provider_type == ProviderType.AZURE:
            config = AzureProviderConfig(
                base_url=base_url,
                api_key=api_key,
                bearer_token=bearer_token,
                api_version=api_version or DefaultValue.AZURE_API_VERSION,
            )
        elif provider_type == ProviderType.OPENAI:
            config = OpenAIProviderConfig(base_url=base_url, api_key=api_key)
        else:  # provider_type == ProviderType.ANTHROPIC
            config = AnthropicProviderConfig(base_url=base_url, api_key=api_key)

        logger.info(
            "Derived provider config from Inspect AI model",
            extra={
                "provider_type": provider_type.value,
                "model_name": model_name,
                "auth_method": "bearer_token" if bearer_token else "api_key",
            },
        )
        return config

    return None


def build_provider_config(
    provider_type: str | None,
    provider_base_url: str | None,
    provider_api_key: str | None,
    provider_api_version: str | None,
) -> ProviderConfig | None:
    """Build provider config from explicit parameters or derive from Inspect AI."""
    if provider_type and provider_base_url and provider_api_key:
        if provider_type == ProviderType.AZURE or provider_type == "azure":
            return AzureProviderConfig(
                base_url=provider_base_url,
                api_key=provider_api_key,
                api_version=provider_api_version or DefaultValue.AZURE_API_VERSION,
            )
        elif provider_type == ProviderType.OPENAI or provider_type == "openai":
            return OpenAIProviderConfig(
                base_url=provider_base_url,
                api_key=provider_api_key,
            )
        elif provider_type == ProviderType.ANTHROPIC or provider_type == "anthropic":
            return AnthropicProviderConfig(
                base_url=provider_base_url,
                api_key=provider_api_key,
            )

    # Fall back to deriving from Inspect AI
    return get_provider_config_from_inspect()


# =============================================================================
# Tool Setup
# =============================================================================


def get_mcp_client(sb: Any) -> Any:
    """Extract MCP client from sandbox."""
    actual_sandbox = sb
    if hasattr(sb, "_sandbox"):
        actual_sandbox = sb._sandbox
    return actual_sandbox._mcp_client


async def setup_tools(
    sb: Any,
    tool_tracker: ToolCallTracker,
    submit_enabled: bool,
) -> list[Tool]:
    """Set up all tools for the Copilot session."""
    mcp_tools = await get_saber_mcp_tools(sb)
    copilot_tools = convert_mcp_tools_to_copilot(mcp_tools, get_mcp_client(sb), tool_tracker)

    all_tools: list[Tool] = list(copilot_tools)
    if submit_enabled:
        submit_tool = create_submit_tool(tool_tracker)
        all_tools.append(submit_tool)
        logger.debug("Added submit tool")

    return all_tools


# =============================================================================
# Transcript Recording
# =============================================================================


def build_system_message(
    assistant_prompt: str,
    submit_prompt: str,
    submit_enabled: bool,
) -> str:
    """Build the system message content to append to Copilot's default preamble.

    With system_mode='append', Copilot SDK provides its CLI foundation prompt
    (environment context, tool instructions, security guardrails) and our content
    is appended after. We include:
    - assistant_prompt: SABER's assistant behavior instructions
    - submit_prompt: Instructions for task submission

    The instruction_prompt (task details) goes in the user message instead.
    """
    parts = []
    if assistant_prompt:
        parts.append(assistant_prompt)
    if submit_prompt and submit_enabled:
        parts.append(submit_prompt)
    return "\n\n".join(parts)


def record_model_event(
    state: TaskState,
    model: str,
    input_messages: list[Any],
    assistant_content: str,
    tool_calls: list[ToolCall],
) -> None:
    """Record a ModelEvent to the transcript."""
    assistant_message = ChatMessageAssistant(
        content=assistant_content,
        tool_calls=tool_calls if tool_calls else None,
    )

    model_output = ModelOutput(
        model=model,
        choices=[
            ChatCompletionChoice(
                message=assistant_message,
                stop_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=ModelUsage(),
    )

    model_event = ModelEvent(
        model=model,
        input=input_messages,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=model_output,
        completed=datetime.now(timezone.utc),
    )
    transcript()._event(model_event)


def record_tool_events(tool_records: list[Any]) -> None:
    """Record ToolEvents to the transcript."""
    for tc in tool_records:
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


# =============================================================================
# Main Solver
# =============================================================================


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
    provider_type: str | None = None,
    provider_base_url: str | None = None,
    provider_api_key: str | None = None,
    provider_api_version: str | None = None,
    skill_directories: list[str] | None = None,
    agent_persona: str | None = None,
) -> Callable[[TaskState], Awaitable[TaskState]]:
    """SABER solver using GitHub Copilot SDK as the reasoning engine.

    This solver creates a Copilot session with SABER's MCP tools registered,
    allowing the Copilot agent to interact with the sandbox environment.

    Args:
        instruction_prompt: The main task instructions for the agent
        assistant_prompt: System prompt defining assistant behavior
        submit_prompt: Instructions about task submission
        continue_prompt: Message shown after each step to guide the agent
        model: Copilot model to use (default: gpt-5, which provides intent via report_intent tool)
        max_turns: Maximum conversation turns before stopping (default: 50)
        submit: Whether to enable the submit tool (default: True)
        transcript_config: Optional transcript synchronization config
        streaming: Whether to enable streaming responses (default: False)
        timeout: Timeout in seconds for each turn (default: 60.0)
        provider_type: Override provider type - 'openai', 'azure', or 'anthropic'
        provider_base_url: Override API endpoint URL
        provider_api_key: Override API key
        provider_api_version: Override Azure API version
        skill_directories: Optional list of directories containing skill files.
            Skills are markdown files with YAML frontmatter that provide
            contextual knowledge/instructions to guide agent behavior.
        agent_persona: Optional path to an agent.md file defining a custom agent persona.
            The file contains YAML frontmatter with agent metadata and markdown content
            with the agent's system prompt.

    Returns:
        Solver function for Inspect AI
    """
    submit_enabled = submit if submit is not None else True

    logger.info(
        "Creating Copilot solver",
        extra={
            "model": model,
            "max_turns": max_turns,
            "submit_enabled": submit_enabled,
            "skill_directories": skill_directories,
        },
    )

    async def solve(state: TaskState) -> TaskState:
        """Execute the Copilot agent loop."""
        client = CopilotClientWrapper({"auto_start": True})
        tool_tracker = ToolCallTracker()
        event_capture = EventCapture()

        try:
            await client.start()
            logger.debug("Copilot client started")

            # Get sandbox and set up tools
            sb = sandbox("saber")

            all_tools = await setup_tools(sb, tool_tracker, submit_enabled)
            tool_names = [t.name for t in all_tools]

            # Build session config
            # Note: instruction_prompt is NOT included in system message - it goes in the user message.
            # With append mode, Copilot SDK provides its default preamble and we append our behavioral prompts.
            system_content = build_system_message(assistant_prompt, submit_prompt, submit_enabled)
            system_mode: Literal["append", "replace"] = "append"

            provider_config = build_provider_config(
                provider_type, provider_base_url, provider_api_key, provider_api_version
            )

            # Load custom agent persona if provided
            # When a persona is supplied, we use 'replace' mode to fully override Copilot's defaults
            # with the persona's prompt, giving full control to the persona author
            persona_skill_directories: list[str] | None = None
            if agent_persona:
                from pathlib import Path

                from .custom_agent import map_tools_to_saber, parse_agent_file

                metadata, prompt_content = parse_agent_file(agent_persona)

                # Map tools if specified in agent file
                mapped_tools: list[str] | None = None
                if metadata.tools and metadata.tools.allowed:
                    mapped_tools = map_tools_to_saber(metadata.tools.allowed)

                # Resolve skill_directories relative to the agent.md file location
                if metadata.skill_directories:
                    agent_dir = Path(agent_persona).parent
                    persona_skill_directories = [str((agent_dir / sd).resolve()) for sd in metadata.skill_directories]

                # Replace system prompt with persona content for full control
                system_content = prompt_content
                system_mode = "replace"

                logger.info(
                    "Loaded custom agent persona (replacing system prompt)",
                    extra={
                        "agent_name": metadata.name,
                        "tool_count": len(mapped_tools) if mapped_tools else "all",
                        "system_mode": system_mode,
                        "skill_directories": persona_skill_directories,
                    },
                )

            # Use skill_directories from persona if available, otherwise use CLI parameter
            effective_skill_directories = persona_skill_directories or skill_directories

            session_config = CopilotSessionConfig.create(
                model=model,
                tools=all_tools,
                system_content=system_content,
                streaming=streaming,
                system_mode=system_mode,
                provider=provider_config,
                skill_directories=effective_skill_directories,
            )

            logger.info(
                "Creating Copilot session",
                extra={
                    "model": model,
                    "tool_count": len(all_tools),
                    "tool_names": tool_names,
                    "available_tools": session_config.available_tools,
                    "skill_directories": effective_skill_directories,
                    "using_byok": provider_config is not None,
                    "system_mode": system_mode,
                },
            )

            session = await client.create_session(session_config.to_dict())
            event_capture.attach(session)

            # Use instruction_prompt as the initial user message
            # This contains the task details (what to do) while system message
            # contains behavioral guidance (how to behave)
            initial_user_prompt = instruction_prompt

            # Set up initial state messages
            state.messages.clear()
            state.messages.append(ChatMessageSystem(content=system_content))

            # Initialize transcript sync for SABER server
            # This uses the WebSocketTranscriptSyncingModelWrapper created by solver_factory
            transcript_sync = AgentTranscriptSync(state)
            await transcript_sync.initialize()

            # Run agent loop
            submitted = False
            turn = 0
            consecutive_timeouts = 0
            max_consecutive_timeouts = 3

            while turn < max_turns and not submitted and consecutive_timeouts < max_consecutive_timeouts:
                prompt = initial_user_prompt if turn == 0 else continue_prompt
                state.messages.append(ChatMessageUser(content=prompt))

                # Build input messages for ModelEvent
                if turn == 0:
                    input_messages = list(state.messages)
                else:
                    input_messages = [ChatMessageUser(content=prompt)]

                logger.debug(
                    f"Sending turn {turn + 1}",
                    extra={"turn": turn + 1, "max_turns": max_turns},
                )

                # Reset capture for this turn
                event_capture.reset_turn()

                # Send message using send() and wait via events
                timed_out = False
                try:
                    await session.send({"prompt": prompt})
                    await event_capture.wait_for_idle(timeout=timeout)
                except asyncio.TimeoutError:
                    timed_out = True
                    consecutive_timeouts += 1
                    logger.warning(
                        f"Turn {turn + 1} timed out",
                        extra={"consecutive_timeouts": consecutive_timeouts},
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for fatal auth/authorization errors - fail fast instead of retrying
                    if any(
                        pattern in error_str
                        for pattern in [
                            "authorization error",
                            "authentication",
                            "401",
                            "403",
                            "/login",
                            "key based authentication is disabled",
                            "access denied",
                            "unauthorized",
                            "invalid api key",
                            "invalid_api_key",
                        ]
                    ):
                        logger.error(
                            f"Fatal authentication/authorization error during turn {turn + 1}: {e}",
                            extra={"error_type": "auth_error"},
                        )
                        raise RuntimeError(
                            f"Copilot SDK authentication failed: {e}\n\n"
                            "Possible causes:\n"
                            "  1. API key authentication is disabled on your Azure resource (use Entra ID instead)\n"
                            "  2. Invalid or expired API key\n"
                            "  3. Missing or invalid bearer token\n"
                            "  4. Run 'az login' to refresh your Azure credentials"
                        ) from e

                    logger.error(f"Error during turn {turn + 1}: {e}")
                    timed_out = True
                    consecutive_timeouts += 1

                # Get captured data
                turn_data = event_capture.current_turn

                logger.info(
                    f"Turn {turn + 1} captured",
                    extra={
                        "event_count": len(turn_data.events),
                        "message_groups": len(turn_data.message_groups),
                        "text_messages": len(turn_data.assistant_messages),
                        "reasoning_messages": len(turn_data.reasoning_messages),
                        "tool_calls": len(turn_data.tool_calls),
                        "tool_results": len(turn_data.tool_results),
                    },
                )

                # Log formatted transcript for this turn
                turn_transcript = turn_data.format_transcript(
                    system_message="" if turn > 0 else system_content,
                    user_prompt=prompt,
                )
                logger.debug(f"Turn {turn + 1} transcript:\n{turn_transcript}")

                # Process message groups in order
                # Each group represents one assistant.message event with its
                # intent, text, tool_calls, and tool_results kept together
                all_content_parts = []

                for group in turn_data.message_groups:
                    group_content_parts = []

                    # Add intent if present
                    if group.intent:
                        group_content_parts.append(group.intent)
                        logger.info(f"Added intent to transcript: {group.intent[:100]}...")

                    # Add text if present
                    if group.text:
                        group_content_parts.append(group.text)

                    # Build combined content for this group
                    group_content = "\n\n".join(group_content_parts)

                    if group.tool_calls:
                        # This group has tool calls - output as assistant message with tool_calls
                        tool_call_objects = [
                            ToolCall(
                                id=tc.tool_call_id or "",
                                function=tc.tool_name or "",
                                arguments=tc.tool_arguments or {},
                                type="function",
                            )
                            for tc in group.tool_calls
                        ]
                        state.messages.append(ChatMessageAssistant(content=group_content, tool_calls=tool_call_objects))

                        # Add tool results
                        for tc in group.tool_calls:
                            result = group.tool_results.get(tc.tool_call_id or "", "")
                            state.messages.append(
                                ChatMessageTool(
                                    content=result,
                                    tool_call_id=tc.tool_call_id or "",
                                )
                            )
                    elif group_content:
                        # No tool calls - just text/intent
                        state.messages.append(ChatMessageAssistant(content=group_content))

                    if group_content_parts:
                        all_content_parts.extend(group_content_parts)

                # Build tool_call_objects for ModelEvent (all tool calls)
                tool_call_objects = [
                    ToolCall(
                        id=tc.tool_call_id or "",
                        function=tc.tool_name or "",
                        arguments=tc.tool_arguments or {},
                        type="function",
                    )
                    for tc in turn_data.tool_calls
                ]

                # Get tool call records from tracker (for recording ToolEvents)
                tool_records = tool_tracker.get_and_clear()

                # Record ModelEvent
                assistant_content = "\n\n".join(all_content_parts)
                record_model_event(
                    state,
                    model if model != DefaultValue.MODEL else "copilot",
                    input_messages,
                    assistant_content,
                    tool_call_objects,
                )

                # Record ToolEvents
                if tool_records:
                    record_tool_events(tool_records)
                    logger.debug(
                        f"Recorded {len(tool_records)} tool events",
                        extra={"tool_names": [tc.tool_name for tc in tool_records]},
                    )

                # Reset timeout counter on success
                if not timed_out and turn_data.has_content():
                    consecutive_timeouts = 0

                # Check for submission - look for "submit" tool call
                # This matches inspect_ai's react pattern: detect submit tool, extract answer
                for tc in turn_data.tool_calls:
                    if tc.tool_name == "submit":
                        answer = (tc.tool_arguments or {}).get("answer", "")
                        if answer:
                            # Set state.output.completion like inspect_ai's react agent does
                            state.output.completion = answer
                            submitted = True
                            logger.info(
                                "Submission detected via submit tool",
                                extra={"answer_length": len(answer)},
                            )
                            break

                # Sync transcript to SABER server after each turn
                # This pushes all new messages (user prompt, assistant response, tool results)
                await transcript_sync.sync_state_messages(state)

                turn += 1

            # Cleanup
            event_capture.detach()
            try:
                await session.destroy()
            except Exception as e:
                logger.warning(f"Error destroying session: {e}")

            logger.info(
                "Copilot solver completed",
                extra={
                    "turns": turn,
                    "submitted": submitted,
                    "message_count": len(state.messages),
                    "transcript_sync_enabled": transcript_sync.is_enabled,
                },
            )

        except Exception as e:
            logger.error(f"Error in Copilot solver: {e}", exc_info=True)
            raise
        finally:
            await client.stop()
            logger.debug("Copilot client stopped")

        return state

    return solve


# =============================================================================
# Agent Factory
# =============================================================================


def create_agent(**kwargs: Any) -> Callable[..., Any]:
    """Create a Copilot agent with SABER integration.

    Args:
        **kwargs: Parameters passed to copilot_solver()

    Returns:
        Factory function that receives prompts from task execution
    """

    def create_with_prompts(
        instruction_prompt: str,
        assistant_prompt: str,
        submit_prompt: str,
        continue_prompt: str,
        transcript_config: dict[str, Any] | None = None,
        submit: bool | None = None,
        skill_directories: list[str] | None = None,
        agent_persona: str | None = None,
    ) -> Solver:
        """Inner factory that receives prompts from task execution."""
        logger.debug(
            "Creating Copilot agent with prompts",
            extra={
                "instruction_length": len(instruction_prompt),
                "assistant_length": len(assistant_prompt),
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
            skill_directories=skill_directories,
            agent_persona=agent_persona,
            **kwargs,
        )

    return create_with_prompts


__all__ = ["create_agent", "copilot_solver"]
