"""Utilities for capturing bridged agent tool trajectories."""

from __future__ import annotations

import json
from typing import Any

SABER_BRIDGE_TOOL_STEPS_KEY = "_saber_bridge_tool_steps"
BRIDGE_TOOL_EVENT_SCHEMA_VERSION = 1
BRIDGE_TOOL_EVENT_FIELD_MAX_CHARS = 20_000
_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "credential",
        "credentials",
        "device_code",
        "password",
        "private_key",
        "refresh_token",
        "sas_token",
        "secret",
        "token",
    }
)


def _field(data: dict[str, Any], *names: str) -> Any:
    """Return the first present field from camelCase/snake_case aliases."""
    for name in names:
        if name in data:
            return data[name]
    return None


def _has_field(data: dict[str, Any], *names: str) -> bool:
    """Return whether any field alias is present."""
    return any(name in data for name in names)


def _valid_tool_event(event: dict[str, Any]) -> bool:
    """Validate one versioned runner event against the scoring schema."""
    if event.get("schema_version") != BRIDGE_TOOL_EVENT_SCHEMA_VERSION:
        return False
    event_type = event.get("type")
    data = event.get("data")
    if not isinstance(data, dict):
        return False
    tool_call_id = _field(data, "toolCallId", "tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return False
    if event_type == "tool.execution_start":
        tool_name = _field(
            data,
            "mcpToolName",
            "mcp_tool_name",
            "toolName",
            "tool_name",
        )
        return (
            isinstance(tool_name, str)
            and bool(tool_name)
            and _has_field(data, "arguments")
        )
    if event_type == "tool.execution_complete":
        return (
            isinstance(_field(data, "success"), bool)
            and _has_field(data, "result")
        )
    return False


def _sanitize_value(value: Any, field_name: str = "") -> Any:
    """Redact known secret fields and bound persisted scoring values."""
    normalized_name = field_name.lower().replace("-", "_")
    if normalized_name in _SECRET_FIELD_NAMES:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(child, str(key))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(child) for child in value]
    if isinstance(value, tuple):
        return [_sanitize_value(child) for child in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                value = json.dumps(
                    _sanitize_value(parsed),
                    separators=(",", ":"),
                )
        if len(value) > BRIDGE_TOOL_EVENT_FIELD_MAX_CHARS:
            return (
                value[:BRIDGE_TOOL_EVENT_FIELD_MAX_CHARS]
                + "...[truncated]"
            )
    return value


def _parse_tool_input(raw_args: Any) -> dict[str, object]:
    """Normalize tool arguments from bridge event payloads."""
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return _sanitize_value(raw_args)
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"_raw": _sanitize_value(raw_args)}
        sanitized = _sanitize_value(parsed)
        return sanitized if isinstance(sanitized, dict) else {"_raw": sanitized}
    return {"_raw": _sanitize_value(raw_args)}


def _result_content(result: Any) -> str:
    """Extract user-visible tool result content from a bridge event result."""
    if isinstance(result, dict):
        content = result.get("content")
        if content is not None:
            return str(_sanitize_value(content))
        detailed = result.get("detailedContent")
        if detailed is not None:
            return str(_sanitize_value(detailed))
        return json.dumps(
            _sanitize_value(result),
            separators=(",", ":"),
        )
    if result is None:
        return ""
    return str(_sanitize_value(result))


def bridge_event_tool_steps_from_lines(lines: list[str]) -> list[dict[str, object]]:
    """Extract tool-call steps from Copilot/bridge JSONL event lines.

    Joins ``tool.execution_complete`` events back to their corresponding
    ``tool.execution_start`` event by ``toolCallId``. Nested/subagent calls are
    preserved because Copilot emits them to the same event stream with
    ``parentToolCallId``.
    """
    starts: dict[str, dict[str, object]] = {}
    steps: list[dict[str, object]] = []
    events: list[dict[str, Any]] = []

    for raw_line in lines:
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") not in {
            "tool.execution_start",
            "tool.execution_complete",
        }:
            continue
        if not _valid_tool_event(event):
            return []
        events.append(event)

    for event in events:
        event_type = event.get("type")
        data = event["data"]
        tool_call_id = str(_field(data, "toolCallId", "tool_call_id"))
        tool_name = str(
            _field(
                data,
                "mcpToolName",
                "mcp_tool_name",
                "toolName",
                "tool_name",
            )
            or ""
        )

        if event_type == "tool.execution_start":
            step = {
                "step_number": len(steps) + 1,
                "tool_name": tool_name,
                "tool_input": _parse_tool_input(data.get("arguments")),
                "output": "",
                "is_error": False,
                "error_type": None,
                "tool_call_id": tool_call_id,
                "parent_tool_call_id": _field(
                    data,
                    "parentToolCallId",
                    "parent_tool_call_id",
                ),
            }
            starts[tool_call_id] = step
            steps.append(step)
            continue

        if event_type != "tool.execution_complete":
            continue

        step = starts.get(tool_call_id)
        if step is None:
            step = {
                "step_number": len(steps) + 1,
                "tool_name": tool_name,
                "tool_input": {},
                "output": "",
                "is_error": False,
                "error_type": None,
                "tool_call_id": tool_call_id,
                "parent_tool_call_id": _field(
                    data,
                    "parentToolCallId",
                    "parent_tool_call_id",
                ),
            }
            starts[tool_call_id] = step
            steps.append(step)

        step["output"] = _result_content(_field(data, "result"))
        step["is_error"] = not bool(_field(data, "success"))
        if step["is_error"]:
            telemetry = _field(data, "toolTelemetry", "tool_telemetry")
            if isinstance(telemetry, dict):
                step["error_type"] = _field(
                    telemetry,
                    "errorType",
                    "error_type",
                )

    return steps
