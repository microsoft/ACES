"""Copilot agent implementation for SABER.

This module provides the Copilot agent that uses GitHub Copilot CLI
as the reasoning engine for SABER benchmark tasks.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model._model import active_model
from inspect_ai.solver import Solver, TaskState
from inspect_ai.util import sandbox

from ....logging_config import LogCategory, get_saber_logger
from ...integration.copilot_tools import (
    Tool,
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

        try:
            await client.start()
            logger.debug("Copilot client started")

            # Get sandbox from Inspect AI context
            sb = sandbox("saber")

            # Get MCP tools from sandbox
            mcp_tools = await get_saber_mcp_tools(sb)
            copilot_tools = convert_mcp_tools_to_copilot(mcp_tools, _get_mcp_client(sb))

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

            # Add submit tool if enabled
            all_tools: list[Tool] = list(copilot_tools)
            if submit_enabled:
                submit_tool = create_submit_tool(handle_submission)
                all_tools.append(submit_tool)
                logger.debug("Added submit_answer tool")

            # Build system message
            system_content = assistant_prompt
            if submit_prompt and submit_enabled:
                system_content += f"\n\n{submit_prompt}"

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

            # Run agent loop
            submitted = False
            turn = 0

            while turn < max_turns and not submitted:
                # Determine prompt for this turn
                prompt = instruction_prompt if turn == 0 else continue_prompt

                logger.debug(
                    f"Sending turn {turn + 1}",
                    extra={
                        "turn": turn + 1,
                        "max_turns": max_turns,
                        "prompt_type": "instruction" if turn == 0 else "continue",
                    },
                )

                # Send message and wait for response
                try:
                    response = await asyncio.wait_for(
                        session.send_and_wait({"prompt": prompt}, timeout=timeout),
                        timeout=timeout + 5,  # Extra buffer for cleanup
                    )

                    # Collect assistant message if available
                    if response and hasattr(response, "data") and hasattr(response.data, "content"):
                        content = response.data.content
                        if content:
                            state.messages.append(ChatMessageAssistant(content=content))
                            logger.debug(
                                "Collected assistant message",
                                extra={"content_length": len(content)},
                            )

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Turn {turn + 1} timed out after {timeout}s",
                        extra={"turn": turn + 1},
                    )

                # Check if submitted
                submitted = state.store.get("submitted") or False

                turn += 1

            # Clean up session
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
