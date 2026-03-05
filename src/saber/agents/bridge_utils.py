"""Shared utilities for sandbox_agent_bridge-based agents.

Common helpers used by both ``claude_code`` and ``copilot`` solvers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage
    from inspect_ai.tool import Tool
    from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
    from inspect_ai.util import SandboxEnvironment

from saber.logging import get_logger

logger = get_logger(__name__)


# Tools that Claude Code provides natively — don't bridge these via MCP
# Claude Code has: Bash, Read, Write, Edit, Glob, Grep, LS, MultiEdit, etc.
CLAUDE_CODE_NATIVE_TOOLS: frozenset[str] = frozenset(
    {
        "bash",
        "python",  # Claude Code can run python via bash
    }
)

# Tools that Copilot provides natively — don't bridge these via MCP
# Copilot has: run_in_terminal, python (via terminal), file operations, etc.
COPILOT_NATIVE_TOOLS: frozenset[str] = frozenset(
    {
        "bash",
        "python",
    }
)


def build_system_prompt(
    instruction_prompt: str,
    assistant_prompt: str,
) -> str:
    """Combine prompt parts into a single system prompt.

    Args:
        instruction_prompt: Main task instructions.
        assistant_prompt: Agent persona / role description.

    Returns:
        Combined prompt string, joined by double newlines.
    """
    parts = [p for p in (instruction_prompt, assistant_prompt) if p]
    return "\n\n".join(parts)


def build_bridged_tools(
    tools: Sequence[Tool] | None,
) -> list[object] | None:
    """Convert a tool sequence to a list of BridgedToolsSpec.

    Args:
        tools: Sequence of inspect_ai Tool objects, or None.

    Returns:
        A single-element list with a ``BridgedToolsSpec`` named
        ``"saber_tools"``, or ``None`` if no tools provided.
    """
    if not tools:
        return None

    from inspect_ai.agent import BridgedToolsSpec

    return [BridgedToolsSpec(name="saber_tools", tools=list(tools))]


def _get_tool_name(tool: Tool) -> str:
    """Extract the short name from an inspect_ai tool.

    inspect_ai tools store their name in ``__registry_info__.name`` as
    ``package/short_name`` (e.g., ``inspect_ai/bash``) or just ``short_name``.

    Args:
        tool: An inspect_ai Tool callable.

    Returns:
        The short tool name (e.g., ``"bash"``).
    """
    registry_info = getattr(tool, "__registry_info__", None)
    if registry_info is not None:
        full_name = getattr(registry_info, "name", "")
        return full_name.split("/")[-1] if "/" in full_name else full_name
    # Fallback to __name__ if no registry info
    return getattr(tool, "__name__", "")


def filter_native_tools(
    tools: Sequence[Tool] | None,
    native_tools: frozenset[str],
) -> list[Tool]:
    """Filter out tools that the agent provides natively.

    Args:
        tools: Sequence of inspect_ai Tool objects, or None.
        native_tools: Set of tool names to exclude (agent has these built-in).

    Returns:
        List of tools excluding those in ``native_tools``.
    """
    if not tools:
        return []
    return [t for t in tools if _get_tool_name(t) not in native_tools]


def build_bridged_tools_for_claude_code(
    tools: Sequence[Tool] | None,
) -> list[object] | None:
    """Build bridged tools for Claude Code, excluding native tools.

    Claude Code has built-in bash, file operations, etc. We only bridge
    custom SABER tools that Claude Code doesn't have natively.

    Args:
        tools: Sequence of inspect_ai Tool objects, or None.

    Returns:
        A BridgedToolsSpec list for non-native tools, or None if none remain.
    """
    filtered = filter_native_tools(tools, CLAUDE_CODE_NATIVE_TOOLS)
    return build_bridged_tools(filtered) if filtered else None


def build_bridged_tools_for_copilot(
    tools: Sequence[Tool] | None,
) -> list[object] | None:
    """Build bridged tools for Copilot, excluding native tools.

    Copilot has built-in terminal execution, file operations, etc. We only
    bridge custom SABER tools that Copilot doesn't have natively.

    Args:
        tools: Sequence of inspect_ai Tool objects, or None.

    Returns:
        A BridgedToolsSpec list for non-native tools, or None if none remain.
    """
    filtered = filter_native_tools(tools, COPILOT_NATIVE_TOOLS)
    return build_bridged_tools(filtered) if filtered else None


def build_user_prompt(messages: Sequence[ChatMessage]) -> tuple[str, bool]:
    """Extract user prompt text from message history.

    Finds the last assistant message, then collects all user message text
    after it. If no assistant message exists, collects all user messages.

    Adapted from the inspect_swe ``build_user_prompt`` pattern.

    Args:
        messages: Sequence of ChatMessage objects from ``state.messages``.

    Returns:
        A tuple of ``(prompt_text, has_assistant_response)`` where
        ``prompt_text`` is all user messages after the last assistant
        joined by ``"\\n\\n"``, and ``has_assistant_response`` indicates
        whether any assistant message was found.
    """
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    # Reject trailing assistant messages (matches reference implementation)
    if messages and isinstance(messages[-1], ChatMessageAssistant):
        raise ValueError("Messages input ends with an assistant message.")

    # Find index of last assistant message
    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, ChatMessageAssistant):
            last_assistant_idx = i

    has_assistant_response = last_assistant_idx >= 0

    # Collect user messages after the last assistant message
    start_idx = last_assistant_idx + 1 if has_assistant_response else 0
    user_texts: list[str] = []
    for msg in messages[start_idx:]:
        if isinstance(msg, ChatMessageUser) and msg.text:
            user_texts.append(msg.text)

    prompt_text = "\n\n".join(user_texts)
    return prompt_text, has_assistant_response


def resolve_mcp_servers(
    mcp_server_configs: Sequence[MCPServerConfigHTTP],
) -> tuple[str, list[str]]:
    """Build MCP config JSON and allowed-tools list from bridge configs.

    Uses ``model_dump()`` for proper serialization instead of ``getattr``
    hacks. Suitable for Claude Code's ``--mcp-config`` CLI flag.

    Args:
        mcp_server_configs: MCP server configs from the bridge.

    Returns:
        A tuple of ``(mcp_config_json, allowed_tools)`` where
        ``mcp_config_json`` is a JSON string for ``--mcp-config`` and
        ``allowed_tools`` is a list of tool patterns for ``--allowed-tools``.
    """
    mcp_servers_json: dict[str, object] = {}
    allowed_tools: list[str] = []

    for mcp_server in mcp_server_configs:
        mcp_servers_json[mcp_server.name] = mcp_server.model_dump(exclude={"name", "tools"}, exclude_none=True)
        if mcp_server.tools == "all":
            allowed_tools.append(f"mcp__{mcp_server.name}__*")
        elif isinstance(mcp_server.tools, list):
            allowed_tools.extend(f"mcp__{mcp_server.name}__{tool}" for tool in mcp_server.tools)

    mcp_config_json = json.dumps({"mcpServers": mcp_servers_json})
    return mcp_config_json, allowed_tools


# ---------------------------------------------------------------------------
# Pre-flight model validation
# ---------------------------------------------------------------------------

# Error patterns that indicate a fatal, non-retryable model configuration
# problem (e.g. wrong model name, invalid API key format).
_FATAL_ERROR_PATTERNS: tuple[str, ...] = (
    "not_found",
    "was not found",
    "model_not_found",
    "invalid_api_key",
    "permission_denied",
    "authentication_error",
)


class ModelValidationError(RuntimeError):
    """Raised when pre-flight model validation detects a fatal error.

    This indicates a configuration problem (e.g. wrong model name)
    that will never succeed on retry.
    """


async def validate_model_availability() -> None:
    """Validate the configured model is accessible before sandbox setup.

    Calls ``model.api.generate()`` directly (bypassing transcript recording)
    to verify the model exists and responds.  Raises
    :class:`ModelValidationError` immediately on fatal errors such as
    ``404 not_found`` (wrong model name) or ``401 authentication_error``
    (bad API key), preventing a long wait for the sandbox timeout.

    Raises:
        ModelValidationError: If the model is unreachable due to a
            configuration error (wrong name, bad key, etc.).
    """
    from inspect_ai.model import ChatMessageUser, GenerateConfig
    from inspect_ai.model._model import get_model

    model = get_model()
    logger.info("Pre-flight model validation: testing %s", model.name)

    try:
        # Use model.api.generate() directly to avoid recording a
        # "ping" message in the eval transcript.  model.generate()
        # writes to the transcript via _record_model_interaction();
        # the low-level API call does not.
        await model.api.generate(
            input=[ChatMessageUser(content="ping")],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(max_tokens=1),
        )
    except Exception as exc:
        error_str = str(exc).lower()
        if any(pattern in error_str for pattern in _FATAL_ERROR_PATTERNS):
            raise ModelValidationError(
                f"Model validation failed for '{model.name}'. Check your --model flag. API error: {exc}"
            ) from exc
        # Non-fatal errors (rate limits, transient network issues) are
        # swallowed — the bridge retry logic will handle them later.
        logger.warning(
            "Pre-flight model check got a non-fatal error (proceeding): %s",
            exc,
        )

    logger.info("Pre-flight model validation passed for %s", model.name)


# ---------------------------------------------------------------------------
# Tool-call-limit filter for sandbox_agent_bridge
# ---------------------------------------------------------------------------


def create_tool_call_limit_filter() -> tuple[object, Callable[[], None]]:
    """Create a GenerateFilter that records tool call usage for bridge agents.

    The ``sandbox_agent_bridge`` does not natively record tool calls against
    inspect_ai's ``tool_call_limit``.  This filter observes the conversation
    before each model generation, counts new tool calls from assistant
    messages, and records them via ``record_tool_call_usage()``.

    When the limit is exceeded the filter returns a text-only ``ModelOutput``
    telling the agent CLI to stop.  This avoids raising
    ``LimitExceededError`` inside the bridge's background service task,
    where the resulting cancel-scope propagation can stall waiting for
    the sandbox subprocess to terminate.  Instead, the companion
    ``check_after_exec`` callback should be called by the solver after
    ``sbox.exec()`` returns; it raises ``LimitExceededError`` in the
    main execution flow where ``apply_limits`` handles it cleanly.

    Returns:
        A ``(filter, check_after_exec)`` tuple.  ``filter`` is a
        ``GenerateFilter`` callable suitable for
        ``sandbox_agent_bridge(filter=...)``.  ``check_after_exec`` is a
        no-arg callable that raises ``LimitExceededError`` if the limit
        was hit during the bridge session.
    """
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    _recorded_count: int = 0
    _limit_exceeded: bool = False
    # After the limit is hit we allow one more model generation so
    # the agent can produce a final answer.  We inject a user message
    # telling the agent the limit is reached, clear all tools, and let
    # the real model generate text.
    _GRACE_GENERATIONS: int = 1
    _grace_remaining: int = 0

    _LIMIT_MESSAGE = (
        "IMPORTANT: You have reached the tool call limit. You cannot use "
        "any more tools. Please provide your final answer immediately as "
        "plain text in your next response."
    )

    async def _filter(
        model: object,
        messages: list[ChatMessage],
        tools: list[object],
        tool_choice: object | None,
        config: object,
    ) -> object | None:
        nonlocal _recorded_count, _limit_exceeded, _grace_remaining
        from inspect_ai.model._model import GenerateInput
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.util._limit import (
            LimitExceededError,
            record_tool_call_usage,
        )

        # Count total tool calls across all assistant messages
        total = sum(len(m.tool_calls) for m in messages if isinstance(m, ChatMessageAssistant) and m.tool_calls)

        # If limit was already hit, handle grace/hard-stop BEFORE
        # the delta block so grace does not reset infinitely (Bug 3).
        if _limit_exceeded:
            if _grace_remaining > 0:
                _grace_remaining -= 1
                logger.debug(
                    "Grace generation %d/%d — letting model produce final answer.",
                    _GRACE_GENERATIONS - _grace_remaining,
                    _GRACE_GENERATIONS,
                )
                # Patch any orphaned tool calls so the API doesn't
                # reject the request with a 400.
                from saber.agents.message_utils import patch_orphaned_tool_calls

                patch_orphaned_tool_calls(messages)
                # Return GenerateInput with empty tools so the real
                # model generates text only (Bug 1 & Bug 2 fix).
                return GenerateInput(  # type: ignore[no-any-return]
                    input=messages,
                    tools=[],
                    tool_choice="none",
                    config=config,
                )

            # Grace period exhausted — hard stop.
            return ModelOutput.from_content(  # type: ignore[no-any-return]
                model="inspect",
                content=("Tool call limit reached. No more generations allowed. Returning final state."),
                stop_reason="stop",
            )

        # Record only the delta (new tool calls since last check)
        delta = total - _recorded_count
        if delta > 0:
            record_tool_call_usage(delta)
            _recorded_count = total

            # Check whether we've hit the limit.
            try:
                from inspect_ai.util._limit import check_tool_call_limit

                check_tool_call_limit()
            except LimitExceededError:
                _limit_exceeded = True
                _grace_remaining = _GRACE_GENERATIONS
                logger.info(
                    "Tool call limit reached (%d calls). "
                    "Injecting limit message and allowing %d grace "
                    "generations for final answer.",
                    total,
                    _GRACE_GENERATIONS,
                )
                # Inject dummy tool results for any orphaned tool
                # calls so the API doesn't reject the request, then
                # add a user message telling the agent to stop.
                from saber.agents.message_utils import patch_orphaned_tool_calls

                patch_orphaned_tool_calls(messages)
                messages.append(ChatMessageUser(content=_LIMIT_MESSAGE))
                return GenerateInput(  # type: ignore[no-any-return]
                    input=messages,
                    tools=[],
                    tool_choice="none",
                    config=config,
                )

        # Return None to proceed with normal generation
        return None

    def check_after_exec() -> None:
        """Raise ``LimitExceededError`` if the tool-call limit was hit.

        Call this in the solver after ``sbox.exec()`` returns so the
        error is raised in the main execution flow where
        ``apply_limits(catch_errors=True)`` handles it properly.
        """
        if _limit_exceeded:
            from inspect_ai.util._limit import check_tool_call_limit

            # This will raise LimitExceededError with the correct
            # source so apply_limits recognises it as one of its own.
            check_tool_call_limit()

    return _filter, check_after_exec


async def upload_skills_to_sandbox(
    sbox: SandboxEnvironment,
    host_skills_dir: str | Path,
    sandbox_base: str,
) -> str:
    """Copy a host-side directory into the sandbox recursively.

    Walks all files in host_skills_dir and writes each to
    sandbox_base/<relative_path> using sbox.write_file().

    Each agent passes its native skills directory as sandbox_base:
    - Claude Code: ".claude/skills"
    - Copilot: ".github/skills"

    Args:
        sbox: The inspect_ai SandboxEnvironment.
        host_skills_dir: Path to skills directory on host.
        sandbox_base: Target directory path inside sandbox.

    Returns:
        The sandbox_base path (for passing to agent config).

    Raises:
        FileNotFoundError: If host_skills_dir does not exist.
        ValueError: If host_skills_dir is not a directory.
    """
    skills_path = Path(host_skills_dir)

    if not skills_path.exists():
        msg = f"Skills directory does not exist: {skills_path}"
        raise FileNotFoundError(msg)

    if not skills_path.is_dir():
        msg = f"Skills path is not a directory: {skills_path}"
        raise ValueError(msg)

    uploaded = 0
    for path in sorted(skills_path.rglob("*")):
        if path.is_symlink():
            logger.debug("Skipping symlink: %s", path)
            continue
        if path.is_dir():
            continue
        relative = path.relative_to(skills_path)
        sandbox_path = f"{sandbox_base}/{relative}"
        content = path.read_bytes()
        await sbox.write_file(sandbox_path, content)
        uploaded += 1

    logger.debug("Uploaded %d skill files to sandbox at %s", uploaded, sandbox_base)
    return sandbox_base
