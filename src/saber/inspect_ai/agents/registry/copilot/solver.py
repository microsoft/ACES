"""Copilot agent implementation for SABER.

This module provides the Copilot agent that uses GitHub Copilot SDK
as the reasoning engine for SABER benchmark tasks.

Architecture:
- SessionTracker: Minimal class for tracking session state (idle + submit)
- events.jsonl parsing: Post-session transcript extraction via events.py
- Single transcript sync at session end (no per-turn sync)

The Copilot SDK saves all events to ~/.copilot/session-state/<session-id>/events.jsonl.
We parse this file after session completion for reliable, ordered transcript data.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from inspect_ai.model._model import active_model
from inspect_ai.solver import Solver, TaskState
from inspect_ai.util import sandbox

if TYPE_CHECKING:
    from inspect_ai.util import SandboxEnvironment

from .....logging_config import LogCategory, get_saber_logger
from ....integration.agent_transcript_sync import AgentTranscriptSync
from ..provider import build_provider_config
from ..tools import (
    build_system_message,
    get_mcp_client_from_sandbox,
    record_tool_event,
)
from .events import SessionEventLog, events_to_chat_messages, events_to_model_events
from .models import (
    CopilotSessionConfig,
    DefaultValue,
)
from .tools import (
    Tool,
    ToolCallTracker,
    convert_mcp_tools_to_copilot,
    create_submit_tool,
    get_saber_mcp_tools,
)

logger = get_saber_logger(LogCategory.AGENT, __name__)


class CopilotSDKEvent(Protocol):
    """Protocol for Copilot SDK events.

    The SDK event has a `.type` attribute with a `.value` property,
    and optionally a `.data` attribute with event-specific data.
    """

    @property
    def type(self) -> object:
        """Event type with .value property."""
        ...

    @property
    def data(self) -> object | None:
        """Event data (optional, varies by event type)."""
        ...


# Check SDK availability
try:
    from copilot import CopilotClient

    COPILOT_SDK_AVAILABLE = True
except ImportError:
    COPILOT_SDK_AVAILABLE = False
    CopilotClient = None


# =============================================================================
# Session Tracker
# =============================================================================


@dataclass
class SessionTracker:
    """Minimal tracker for session state during execution.

    Tracks only what's needed for the session loop:
    - idle_event: Set when session becomes idle (turn complete)
    - submitted: Set by submit tool handler when answer is submitted
    - submit_answer: The submitted answer (for state.output.completion)
    - session_id: Captured from session.start for events.jsonl path
    """

    idle_event: asyncio.Event = field(default_factory=asyncio.Event)
    submitted: bool = False
    submit_answer: str | None = None
    session_id: str | None = None
    _error: Exception | None = field(default=None, repr=False)

    def on_event(self, event: CopilotSDKEvent | None) -> None:
        """Handle SDK events - only care about idle and session start.

        Args:
            event: Copilot SDK event object
        """
        if not event:
            return

        event_type = str(event.type.value) if hasattr(event.type, "value") else str(event.type)
        logger.debug(f"SessionTracker received event: {event_type}")

        if event_type == "session.idle":
            self.idle_event.set()
        elif event_type == "session.start":
            data = getattr(event, "data", None)
            logger.debug(f"session.start data: {data}, type: {type(data)}")
            if data:
                # Try multiple attribute names for session ID
                self.session_id = (
                    getattr(data, "session_id", None)
                    or getattr(data, "sessionId", None)
                    or (data.get("session_id") if isinstance(data, dict) else None)
                    or (data.get("sessionId") if isinstance(data, dict) else None)
                )
                logger.info(f"Session started with ID: {self.session_id}")
                if not self.session_id:
                    logger.warning(f"Could not extract session_id from data: {data}")
        elif event_type == "session.error":
            data = getattr(event, "data", None)
            error_msg = getattr(data, "message", str(data)) if data else "Unknown error"
            self._error = Exception(f"Session error: {error_msg}")
            self.idle_event.set()

    async def wait_for_idle(self, timeout: float) -> None:
        """Wait for session to become idle.

        Args:
            timeout: Maximum seconds to wait

        Raises:
            asyncio.TimeoutError: If timeout expires before idle
            Exception: If session error occurred
        """
        try:
            await asyncio.wait_for(self.idle_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for session idle after {timeout}s")
            raise

        if self._error:
            raise self._error

        # Reset for next turn
        self.idle_event.clear()

    def set_submission(self, answer: str) -> None:
        """Mark submission and store the answer.

        Called by submit tool handler when agent submits.

        Args:
            answer: The submitted answer
        """
        self.submitted = True
        self.submit_answer = answer
        logger.info(f"Submission recorded: {answer[:100]}...")


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
# Tool Setup
# =============================================================================


async def setup_tools(
    sb: SandboxEnvironment,
    tool_tracker: ToolCallTracker,
    session_tracker: SessionTracker,
    submit_enabled: bool,
) -> list[Tool]:
    """Set up all tools for the Copilot session.

    Args:
        sb: Sandbox instance
        tool_tracker: Tracker for recording tool calls
        session_tracker: Session state tracker for submit detection
        submit_enabled: Whether to include the submit tool

    Returns:
        List of Copilot SDK Tool objects
    """
    mcp_tools = await get_saber_mcp_tools(sb)
    mcp_client = get_mcp_client_from_sandbox("saber")
    copilot_tools = convert_mcp_tools_to_copilot(mcp_tools, mcp_client, tool_tracker)

    all_tools: list[Tool] = list(copilot_tools)
    if submit_enabled:
        submit_tool = create_submit_tool(tool_tracker, session_tracker)
        all_tools.append(submit_tool)
        logger.debug("Added submit tool")

    return all_tools


def _record_tool_events(tool_records: list[Any]) -> None:
    """Record ToolEvents to the transcript using the shared utility."""
    for tc in tool_records:
        record_tool_event(
            tool_call_id=tc.tool_call_id,
            function_name=tc.tool_name,
            arguments=tc.arguments,
            result=tc.result,
            is_error=tc.is_error,
        )


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

    Architecture:
    - Uses SessionTracker for minimal real-time state (idle + submit flag)
    - Parses events.jsonl after session for complete transcript
    - Single transcript sync at session end

    Args:
        instruction_prompt: The main task instructions for the agent
        assistant_prompt: System prompt defining assistant behavior
        submit_prompt: Instructions about task submission
        continue_prompt: Message shown after each step to guide the agent
        model: Copilot model to use (default: gpt-5)
        max_turns: Maximum conversation turns before stopping (default: 50)
        submit: Whether to enable the submit tool (default: True)
        transcript_config: Optional transcript synchronization config
        streaming: Whether to enable streaming responses (default: False)
        timeout: Timeout in seconds for each turn (default: 60.0)
        provider_type: Override provider type - 'openai', 'azure', or 'anthropic'
        provider_base_url: Override API endpoint URL
        provider_api_key: Override API key
        provider_api_version: Override Azure API version
        skill_directories: Optional list of directories containing skill files
        agent_persona: Optional path to an agent.md file defining a custom agent persona

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
        session_tracker = SessionTracker()

        try:
            await client.start()
            logger.info("Copilot client started")

            # Get sandbox and set up tools
            sb = sandbox("saber")
            all_tools = await setup_tools(sb, tool_tracker, session_tracker, submit_enabled)
            tool_names = [t.name for t in all_tools]

            # Build session config
            system_content = build_system_message(assistant_prompt, submit_prompt, submit_enabled)
            system_mode: Literal["append", "replace"] = "append"

            provider_config = build_provider_config(
                provider_type, provider_base_url, provider_api_key, provider_api_version
            )

            # Extract the actual model name from the active Inspect AI model.
            # Model names like "openai/azure/gpt-5.2" need the service prefix stripped
            # to get just "gpt-5.2" for the Copilot SDK session config.
            actual_model = model  # default from function parameter
            active = active_model()
            if active and active.api:
                # Use service_model_name() if available (handles all prefix stripping)
                # This is the standard method all Inspect AI providers implement
                if hasattr(active.api, "service_model_name"):
                    actual_model = active.api.service_model_name()
                else:
                    # Fallback: just use the model_name as-is
                    actual_model = active.api.model_name
                logger.debug(
                    "Extracted model name from active Inspect AI model",
                    extra={
                        "original": active.api.model_name,
                        "extracted": actual_model,
                    },
                )

            # Load custom agent persona if provided
            persona_skill_directories: list[str] | None = None
            if agent_persona:
                from pathlib import Path

                from ..custom_agent import map_tools_to_saber, parse_agent_file

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
                model=actual_model,
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
                    "model": actual_model,
                    "model_param": model,
                    "tool_count": len(all_tools),
                    "tool_names": tool_names,
                    "available_tools": session_config.available_tools,
                    "skill_directories": effective_skill_directories,
                    "using_byok": provider_config is not None,
                    "system_mode": system_mode,
                },
            )

            session = await client.create_session(session_config.to_dict())
            session.on(session_tracker.on_event)
            logger.info("Copilot session created")

            # Try to get session_id directly from session object (in case session.start event was missed)
            if not session_tracker.session_id:
                # Try various attribute names the SDK might use
                session_id_from_obj = (
                    getattr(session, "session_id", None)
                    or getattr(session, "sessionId", None)
                    or getattr(session, "id", None)
                    or getattr(session, "_session_id", None)
                )
                if session_id_from_obj:
                    session_tracker.session_id = session_id_from_obj
                    logger.info(f"Got session_id from session object: {session_id_from_obj}")
                else:
                    logger.debug(
                        f"Could not get session_id from session object. "
                        f"Session attrs: {[a for a in dir(session) if not a.startswith('__')]}"
                    )

            # Use instruction_prompt as the initial user message
            initial_user_prompt = instruction_prompt

            # Initialize transcript sync for SABER server
            transcript_sync = AgentTranscriptSync(state)
            await transcript_sync.initialize()

            # Run agent loop
            turn = 0
            consecutive_timeouts = 0
            max_consecutive_timeouts = 3

            while (
                turn < max_turns and not session_tracker.submitted and consecutive_timeouts < max_consecutive_timeouts
            ):
                prompt = initial_user_prompt if turn == 0 else continue_prompt

                logger.debug(
                    f"Sending turn {turn + 1}",
                    extra={"turn": turn + 1, "max_turns": max_turns},
                )

                # Send message and wait for idle
                timed_out = False
                try:
                    await session.send({"prompt": prompt})
                    await session_tracker.wait_for_idle(timeout=timeout)
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

                # Get tool call records from tracker (for recording ToolEvents)
                tool_records = tool_tracker.get_and_clear()

                # Record ToolEvents from tracker
                if tool_records:
                    _record_tool_events(tool_records)
                    logger.debug(
                        f"Recorded {len(tool_records)} tool events",
                        extra={"tool_names": [tc.tool_name for tc in tool_records]},
                    )

                # Reset timeout counter on success
                if not timed_out:
                    consecutive_timeouts = 0

                turn += 1

            # Cleanup session
            try:
                await session.destroy()
            except Exception as e:
                logger.warning(f"Error destroying session: {e}")

            # Parse events.jsonl for complete transcript
            if session_tracker.session_id:
                log = SessionEventLog(session_tracker.session_id)
                events = log.read_events()
                logger.debug(
                    f"Read {len(events)} events from events.jsonl",
                    extra={"session_id": session_tracker.session_id},
                )

                # Log the event types for debugging
                event_types: dict[str, int] = {}
                for event in events:
                    etype = type(event).__name__
                    event_types[etype] = event_types.get(etype, 0) + 1
                logger.debug(f"Event type breakdown: {event_types}")

                # IMPORTANT: Only use append/extend - never clear() or assign.
                # Inspect AI's transcript system only tracks append/extend operations.
                # The state already has [system, user] messages from task setup.
                # We need to append only the NEW messages (assistant, tool responses, etc.)
                chat_messages = events_to_chat_messages(events, system_content)

                # Skip messages that already exist (system + user from task setup)
                # Our parsed messages start with system, then user, then the conversation
                initial_count = len(state.messages)  # Usually 2: system + user
                new_messages = chat_messages[initial_count:] if len(chat_messages) > initial_count else []

                logger.debug(
                    f"Appending {len(new_messages)} new messages "
                    f"(skipping first {initial_count} that match existing state)"
                )
                state.messages.extend(new_messages)

                # Create synthetic ModelEvents for the Transcript tab in Inspect viewer
                # This allows the conversation to appear in the transcript view
                events_to_model_events(
                    events=events,
                    model_name=actual_model,
                    system_content=system_content,
                )

                # Log the message types for debugging
                msg_types: dict[str, int] = {}
                for msg in state.messages:
                    mtype = type(msg).__name__
                    msg_types[mtype] = msg_types.get(mtype, 0) + 1
                logger.info(
                    "Parsed session transcript from events.jsonl",
                    extra={
                        "session_id": session_tracker.session_id,
                        "event_count": len(events),
                        "message_count": len(state.messages),
                        "message_types": msg_types,
                    },
                )
            else:
                # Fallback: no events to parse, keep existing messages
                logger.warning("No session_id available for events.jsonl parsing")

            # Set submission answer if available
            if session_tracker.submit_answer:
                state.output.completion = session_tracker.submit_answer

            # Single transcript sync at end
            logger.debug(
                f"Before transcript sync: {len(state.messages)} messages, "
                f"sync enabled: {transcript_sync.is_enabled}"
            )
            sync_result = await transcript_sync.sync_state_messages(state)
            logger.debug(f"Transcript sync result: {sync_result}")

            # Debug: Log first few messages to verify they're populated
            for i, msg in enumerate(state.messages[:5]):
                content_preview = ""
                if hasattr(msg, "content") and msg.content:
                    content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                tool_calls_count = len(msg.tool_calls) if hasattr(msg, "tool_calls") and msg.tool_calls else 0
                content_len = len(msg.content) if hasattr(msg, "content") and msg.content else 0
                logger.debug(
                    f"Message[{i}]: role={msg.role}, content_len={content_len}, "
                    f"tool_calls={tool_calls_count}, preview={content_preview[:50]}"
                )

            logger.info(
                "Copilot solver completed",
                extra={
                    "turns": turn,
                    "submitted": session_tracker.submitted,
                    "message_count": len(state.messages),
                    "has_completion": state.output.completion is not None,
                },
            )

        except Exception as e:
            logger.error(f"Error in Copilot solver: {e}", exc_info=True)
            raise
        finally:
            await client.stop()
            logger.info("Copilot client stopped")

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
