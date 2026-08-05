# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Shared utilities for sandbox_agent_bridge-based agents.

Common helpers used by both ``claude_code`` and ``copilot`` solvers.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage
    from inspect_ai.model._model import GenerateFilter
    from inspect_ai.tool import Tool
    from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
    from inspect_ai.util import SandboxEnvironment

    from saber.agents.bridge_tracking_models import BridgeSessionSummary
    from saber.agents.persona import AgentBundle
    from saber.agents.registry.copilot.solver import IdleDecision

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


# ---------------------------------------------------------------------------
# Bridge stderr error parsing
# ---------------------------------------------------------------------------

# Compiled patterns for bridge proxy error detection.
_BRIDGE_ERROR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "BadRequestError",
        re.compile(r"BadRequestError\('Error code: (\d+) - (.+?)'\)", re.DOTALL),
    ),
    (
        "CAPIError",
        re.compile(r"CAPIError:\s*(.+?)(?:\n|$)"),
    ),
    (
        "ErrorCallingMethod",
        re.compile(r"Error calling method (\w+):", re.DOTALL),
    ),
    (
        "UnexpectedProxyError",
        re.compile(r"Unexpected error during model proxy call:\s*(.+?)(?:\n|$)"),
    ),
    (
        "RuntimeError",
        re.compile(r"RuntimeError:\s*(.+?)(?:\nTraceback|\n\n|\n|$)", re.DOTALL),
    ),
    (
        "ConnectionError",
        re.compile(r"Connection error:\s*(.+?)(?:\n|$)"),
    ),
]

# Keywords that indicate an error even without a structured pattern.
_BRIDGE_ERROR_KEYWORDS: tuple[str, ...] = (
    "Unknown parameter",
    "invalid_request_error",
    "Connection error",
)


def parse_bridge_stderr(stderr: str) -> str:
    """Parse bridge proxy stderr for structured error diagnostics.

    The bridge proxy's error messages often contain the full model request
    body before the actual error, causing the real error to be truncated
    in logs. This function extracts known error patterns to surface the
    root cause.

    Args:
        stderr: Raw stderr from the agent CLI subprocess.

    Returns:
        A diagnostic summary string. Empty string if no known patterns found.
    """
    if not stderr:
        return ""

    findings: list[str] = []
    # Track raw matched text for dedup against keyword scan
    matched_text: list[str] = []

    for label, pattern in _BRIDGE_ERROR_PATTERNS:
        match = pattern.search(stderr)
        if match:
            matched_text.append(match.group(0))
            groups = match.groups()
            # Strip newlines from detail to prevent multi-line findings
            detail = " | ".join(g.strip().replace("\n", " ")[:300] for g in groups if g)
            findings.append(f"[{label}] {detail}")

    # Keyword scan for indicators not caught by structured patterns.
    for keyword in _BRIDGE_ERROR_KEYWORDS:
        if keyword in stderr:
            # Avoid duplicate reporting if a structured pattern already
            # captured this keyword in its match span.
            if not any(keyword in span for span in matched_text):
                # Extract the line containing the keyword for context.
                for line in stderr.splitlines():
                    if keyword in line:
                        findings.append(f"[Keyword] {line.strip()[:300]}")
                        break

    return "\n".join(findings)


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
    result = build_bridged_tools(filtered) if filtered else None
    return result


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


def override_state_completion(state: object, completion: str) -> None:
    """Set a task state's final completion and assistant message.

    Args:
        state: Inspect TaskState-like object with ``messages`` and ``output``.
        completion: Final assistant text to expose to scorers.
    """
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    message = ChatMessageAssistant(content=completion, source="generate")
    messages = getattr(state, "messages", None)
    if isinstance(messages, list):
        if messages and isinstance(messages[-1], ChatMessageAssistant):
            messages[-1] = message
        else:
            messages.append(message)

    current_output = getattr(state, "output", None)
    if current_output is not None:
        state.output = current_output.model_copy(
            update={
                "choices": [ChatCompletionChoice(message=message, stop_reason="stop")],
                "completion": completion,
            }
        )
    else:
        state.output = ModelOutput(
            choices=[ChatCompletionChoice(message=message, stop_reason="stop")],
            completion=completion,
        )


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


def read_mcp_config(path: str | Path | None) -> dict[str, object]:
    """Read an MCP config JSON file.

    Args:
        path: Path to a JSON file containing ``{"mcpServers": ...}``.

    Returns:
        Parsed config. Empty dict when *path* is ``None``.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not a mapping or has invalid
            ``mcpServers`` shape.
    """
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"MCP config file not found: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MCP config JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"MCP config in {config_path} must be a JSON object.")
    servers = data.get("mcpServers")
    if servers is not None and not isinstance(servers, dict):
        raise ValueError(f"MCP config in {config_path} must contain object-valued mcpServers.")
    return data


def merge_mcp_configs(*configs: dict[str, object] | str | None) -> dict[str, object]:
    """Merge MCP config objects or JSON strings.

    Later configs override earlier configs when server names collide.
    Non-``mcpServers`` top-level fields from later configs also override
    earlier values.
    """
    merged: dict[str, object] = {}
    merged_servers: dict[str, object] = {}
    for config in configs:
        if not config:
            continue
        if isinstance(config, str):
            parsed = json.loads(config)
            if not isinstance(parsed, dict):
                raise ValueError("MCP config JSON must decode to an object.")
            config_obj = parsed
        else:
            config_obj = config
        for key, value in config_obj.items():
            if key == "mcpServers":
                if not isinstance(value, dict):
                    raise ValueError("mcpServers must be an object.")
                merged_servers.update(value)
            else:
                merged[key] = value
    if merged_servers:
        merged["mcpServers"] = merged_servers
    return merged


def mcp_allowed_tools(config: dict[str, object]) -> list[str]:
    """Return Claude Code ``--allowed-tools`` patterns for an MCP config."""
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        return []
    allowed: list[str] = []
    for server_name, raw_server in servers.items():
        tools: object = None
        if isinstance(raw_server, dict):
            tools = raw_server.get("tools")
        if tools in (None, "all", ["*"]):
            allowed.append(f"mcp__{server_name}__*")
        elif isinstance(tools, list):
            allowed.extend(f"mcp__{server_name}__{tool}" for tool in tools if isinstance(tool, str))
    return allowed


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
    from inspect_ai.model import ChatMessageUser

    from saber.agents.message_utils import (
        TOOL_CALL_LIMIT_MESSAGE,
        count_tool_calls,
    )

    _recorded_count: int = 0
    _limit_exceeded: bool = False
    # After the limit is hit we allow one more model generation so
    # the agent can produce a final answer.  We inject a user message
    # telling the agent the limit is reached, clear all tools, and let
    # the real model generate text.
    _GRACE_GENERATIONS: int = 1
    _grace_remaining: int = 0

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
        total = count_tool_calls(messages)

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
                messages.append(ChatMessageUser(content=TOOL_CALL_LIMIT_MESSAGE))
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


def create_responses_store_filter() -> GenerateFilter:
    """Create a GenerateFilter that keeps the Responses API ``store`` flag on.

    The ``sandbox_agent_bridge`` reconstructs each generation's request from
    the agent CLI runner (e.g. the Copilot/codex SDK), which defaults to
    ``store: false`` and captures that value into ``config.extra_body``.
    inspect_ai's Responses provider then re-injects ``extra_body["store"]``
    into the outgoing request (``completion_params_responses``), overriding
    the model's ``responses_store=True`` setting.

    For reasoning models the server only persists reasoning items
    (``rs_...``) when ``store`` is true; with ``store: false`` the next turn
    fails with ``Item with id 'rs_...' not found ... Items are not persisted
    when 'store' is set to false``.

    This filter aligns ``config.extra_body["store"]`` with the model's
    ``responses_store`` intent when that intent is ``True``, so the runner's
    ``store: false`` cannot override it. It mutates ``config`` in place and
    returns ``None`` (observation-only), so it never short-circuits other
    filters in the chain.

    Returns:
        A ``GenerateFilter`` callable suitable for
        ``sandbox_agent_bridge(filter=...)``.
    """

    async def _filter(
        model: object,
        messages: list[ChatMessage],
        tools: list[object],
        tool_choice: object | None,
        config: object,
    ) -> None:
        api = getattr(model, "api", None)
        if getattr(api, "responses_store", None) is not True:
            return None
        extra_body = getattr(config, "extra_body", None)
        if isinstance(extra_body, dict) and extra_body.get("store") is not True:
            extra_body["store"] = True
            logger.debug(
                "Responses store filter: forced extra_body['store']=True to honor model responses_store (model=%s)",
                getattr(model, "name", str(model)),
            )
        return None

    return _filter


def create_reasoning_effort_filter(effort: str | None = None) -> GenerateFilter | None:
    """Create a GenerateFilter that forces a Responses-API reasoning effort.

    Some Azure reasoning deployments (e.g. ``...flash-code``) are not matched
    by inspect_ai's name-based reasoning heuristic
    (``ResponsesModelInfo.has_reasoning_options()`` only recognises o-series /
    gpt-5 / codex names). As a result inspect drops any ``reasoning_effort``
    with a ``reasoning options ignored for non-reasoning model`` warning, and
    the server silently applies its own default effort.

    When ``effort`` is provided this filter, per generation:
      * marks the bridge model as reasoning-capable by overriding
        ``model.api.has_reasoning_options`` to return ``True`` (instance-level,
        so inspect emits the ``reasoning`` param), and
      * sets ``config.reasoning_effort`` to the requested level.

    It mutates ``config`` in place and returns ``None`` (observation-only).

    Args:
        effort: One of ``minimal`` / ``low`` / ``medium`` / ``high``. When
            falsy the factory returns ``None`` (no filter installed).

    Returns:
        A ``GenerateFilter`` callable, or ``None`` when ``effort`` is falsy.
    """
    if not effort:
        return None

    async def _filter(
        model: object,
        messages: list[ChatMessage],
        tools: list[object],
        tool_choice: object | None,
        config: object,
    ) -> None:
        api = getattr(model, "api", None)
        if api is not None and getattr(api, "has_reasoning_options", None) is not None:
            # Instance-level override so completion_params_responses emits the
            # reasoning param for this otherwise-unrecognised deployment.
            try:
                api.has_reasoning_options = lambda: True
            except Exception:
                logger.debug("Could not override has_reasoning_options on bridge model")
        if getattr(config, "reasoning_effort", None) != effort:
            try:
                config.reasoning_effort = effort  # type: ignore[attr-defined]
                logger.debug(
                    "Reasoning effort filter: set reasoning_effort=%s (model=%s)",
                    effort,
                    getattr(model, "name", str(model)),
                )
            except Exception:
                logger.debug("Could not set reasoning_effort on bridge config")
        return None

    return _filter


# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------


def resolve_model_aliases(
    raw_aliases: dict[str, str] | None,
) -> dict[str, str] | None:
    """Validate and normalize model alias mappings.

    Used with ``sandbox_agent_bridge(model_aliases=...)`` to route
    subagent model requests to specific providers.

    Args:
        raw_aliases: Optional mapping from requested name → model spec string.

    Returns:
        Validated aliases dict, or None if empty/not provided.

    Raises:
        ValueError: If any key or value is empty.
    """
    if not raw_aliases:
        return None
    for key, value in raw_aliases.items():
        if not key or not value:
            msg = f"Model alias mapping has empty key or value: {key!r} \u2192 {value!r}"
            raise ValueError(msg)
    return dict(raw_aliases)


# ---------------------------------------------------------------------------
# Filter composition
# ---------------------------------------------------------------------------


def compose_filters(
    *filters: GenerateFilter | None,
) -> GenerateFilter | None:
    """Chain multiple GenerateFilters into one.

    Filters run in order. The first filter that returns a non-None result
    (ModelOutput or GenerateInput) short-circuits — later filters are skipped.

    Observation-only filters (returning None) should be listed first.

    Args:
        *filters: GenerateFilter callables (None entries are skipped).

    Returns:
        A single composed GenerateFilter, or None if no non-None filters.
    """
    active = [f for f in filters if f is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    async def _composed(
        model: object,
        messages: list[ChatMessage],
        tools: list[object],
        tool_choice: object,
        config: object,
    ) -> object | None:
        for filt in active:
            result: object | None = await filt(model, messages, tools, tool_choice, config)
            if result is not None:
                return result
        return None

    return _composed


# ---------------------------------------------------------------------------
# Bridge session summary recording
# ---------------------------------------------------------------------------


def record_bridge_summary(
    get_tracking_summary: Callable[[], BridgeSessionSummary],
    log: logging.Logger,
) -> BridgeSessionSummary:
    """Record a bridge session summary as an InfoEvent and log it.

    Call this after the ``sandbox_agent_bridge`` context manager exits.

    Args:
        get_tracking_summary: Callable returned by ``create_tracking_filter()``.
        log: Logger instance for the calling module.

    Returns:
        The ``BridgeSessionSummary`` snapshot.
    """
    summary = get_tracking_summary()
    try:
        from inspect_ai.log._transcript import transcript

        transcript().info(
            summary.model_dump(mode="json"),
            source="saber.bridge_session_summary",
        )
    except Exception:
        log.debug("Could not record bridge session summary InfoEvent")
    log.info(
        "Bridge session complete: %d generations (%d main, %d subagent), %d tool calls",
        summary.total_generations,
        summary.main_generations,
        summary.subagent_generations,
        summary.total_tool_calls,
    )
    return summary


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


async def upload_agent_bundle_to_sandbox(
    sbox: SandboxEnvironment,
    bundle: AgentBundle,
    sandbox_agents_dir: str,
) -> str | None:
    """Upload parsed agent definition files into a sandbox agent directory.

    Args:
        sbox: Inspect sandbox environment.
        bundle: Runtime agent bundle.
        sandbox_agents_dir: Agent directory inside the sandbox, e.g.
            ``".github/agents"`` or ``".claude/agents"``.

    Returns:
        ``sandbox_agents_dir`` if files were uploaded, otherwise ``None``.
    """
    if not bundle.agents:
        return None
    uploaded = 0
    for agent in bundle.agents:
        relative = agent.relative_path
        if relative.is_absolute() or any(part == ".." for part in relative.parts):
            msg = f"Unsafe agent relative path: {relative}"
            raise ValueError(msg)
        sandbox_path = f"{sandbox_agents_dir}/{relative.as_posix()}"
        await sbox.write_file(sandbox_path, agent.raw_content.encode("utf-8"))
        uploaded += 1
    logger.debug("Uploaded %d agent file(s) to sandbox at %s", uploaded, sandbox_agents_dir)
    return sandbox_agents_dir


# ---------------------------------------------------------------------------
# Runner metrics parsing
# ---------------------------------------------------------------------------


def parse_runner_metrics(stderr: str) -> dict[str, object] | None:
    """Parse COPILOT_METRICS JSON from runner stderr.

    Searches for a line starting with ``COPILOT_METRICS:`` and parses the
    JSON payload after the prefix.

    Args:
        stderr: Raw stderr from the runner subprocess.

    Returns:
        Parsed metrics dict, or None if no metrics line found.
    """
    if not stderr:
        return None

    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("COPILOT_METRICS:"):
            json_str = stripped[len("COPILOT_METRICS:") :].strip()
            try:
                return json.loads(json_str)  # type: ignore[no-any-return]
            except (json.JSONDecodeError, ValueError):
                logger.debug("Failed to parse COPILOT_METRICS JSON: %s", json_str[:200])
                return None

    return None


def parse_workflow_status(stderr: str) -> dict[str, object] | None:
    """Parse COPILOT_WORKFLOW_STATUS JSON from runner stderr.

    Args:
        stderr: Raw stderr from the runner subprocess.

    Returns:
        Parsed workflow status dict, or None if no status line found.
    """
    if not stderr:
        return None

    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("COPILOT_WORKFLOW_STATUS:"):
            json_str = stripped[len("COPILOT_WORKFLOW_STATUS:") :].strip()
            try:
                return json.loads(json_str)  # type: ignore[no-any-return]
            except (json.JSONDecodeError, ValueError):
                logger.debug("Failed to parse COPILOT_WORKFLOW_STATUS JSON: %s", json_str[:200])
                return None

    return None


def parse_idle_decision(stderr: str) -> IdleDecision | None:
    """Parse COPILOT_IDLE_DECISION JSON from runner stderr.

    Searches for a line starting with ``COPILOT_IDLE_DECISION:`` and
    parses the JSON payload into an :class:`IdleDecision` model.

    Args:
        stderr: Raw stderr from the runner subprocess.

    Returns:
        Parsed :class:`IdleDecision`, or ``None`` if not found or invalid.
    """
    if not stderr:
        return None

    from saber.agents.registry.copilot.solver import IdleDecision as _IdleDecision

    last_decision: _IdleDecision | None = None
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("COPILOT_IDLE_DECISION:"):
            json_str = stripped[len("COPILOT_IDLE_DECISION:") :].strip()
            try:
                data = json.loads(json_str)
                last_decision = _IdleDecision.model_validate(data)
            except (json.JSONDecodeError, ValueError):
                logger.debug("Failed to parse COPILOT_IDLE_DECISION: %s", json_str[:200])
    return last_decision
