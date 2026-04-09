"""Public extensibility surface for external SABER adapters.

External packages should import bridge-grade adapter helpers from this module
instead of reaching into ``saber.agents.bridge_utils`` or
``saber.agents.bridge_tracking`` directly.

This surface intentionally exposes:

- the standard adapter capability/type primitives
- generic bridge helper functions that are useful across external adapters

It intentionally does **not** expose solver-specific internals such as the
Copilot or Claude runner implementations.
"""

from saber.agents.bridge_tracking import create_tracking_filter
from saber.agents.bridge_utils import (
    build_bridged_tools,
    build_system_prompt,
    build_user_prompt,
    compose_filters,
    create_tool_call_limit_filter,
    parse_bridge_stderr,
    record_bridge_summary,
    resolve_model_aliases,
    tool_call_limit,
    validate_model_availability,
)
from saber.agents.models import AgentCapabilities, AgentFactory, AgentSolverFactory

__all__ = [
    "AgentCapabilities",
    "AgentFactory",
    "AgentSolverFactory",
    "build_bridged_tools",
    "build_system_prompt",
    "build_user_prompt",
    "compose_filters",
    "create_tool_call_limit_filter",
    "create_tracking_filter",
    "parse_bridge_stderr",
    "record_bridge_summary",
    "resolve_model_aliases",
    "tool_call_limit",
    "validate_model_availability",
]
