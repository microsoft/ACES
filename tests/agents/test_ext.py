"""Tests for the public external adapter surface."""

from saber.ext import (
    AgentCapabilities,
    build_bridged_tools,
    build_system_prompt,
    build_user_prompt,
    compose_filters,
    create_tool_call_limit_filter,
    create_tracking_filter,
    parse_bridge_stderr,
    record_bridge_summary,
    resolve_model_aliases,
    tool_call_limit,
    validate_model_availability,
)


def test_public_ext_exports_bridge_helpers() -> None:
    assert callable(build_bridged_tools)
    assert callable(build_system_prompt)
    assert callable(build_user_prompt)
    assert callable(compose_filters)
    assert callable(create_tool_call_limit_filter)
    assert callable(create_tracking_filter)
    assert callable(parse_bridge_stderr)
    assert callable(record_bridge_summary)
    assert callable(resolve_model_aliases)
    assert callable(tool_call_limit)
    assert callable(validate_model_availability)


def test_public_ext_exports_agent_capabilities() -> None:
    caps = AgentCapabilities(supports_tools=False)
    assert caps.supports_tools is False
