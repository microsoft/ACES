"""Agent-related type definitions."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict


class AgentPromptKwargs(TypedDict, total=False):
    """Typed kwargs for agent prompt injection."""

    instruction_prompt: str
    assistant_prompt: str
    max_steps: int  # Tool-call limit for graceful stop


class AgentCapabilities(BaseModel):
    """Capabilities supported by an agent implementation.

    Used by solver_factory to gate which kwargs are forwarded to each agent.

    All three built-in agents support tools:
    - ``react``: tools passed directly to ``react(tools=...)``
    - ``claude_code``: tools converted to ``BridgedToolsSpec`` for MCP inside sandbox
    - ``copilot``: tools converted to ``BridgedToolsSpec`` for MCP inside sandbox

    Plugin agents declare capabilities via a module-level ``AGENT_CAPABILITIES``
    attribute in their adapter module.
    """

    model_config = ConfigDict(frozen=True)

    #: Contract version — ACES checks this at solver creation time and fails
    #: fast with a clear error if the plugin was built against an incompatible
    #: version.  Increment when the factory contract changes in a breaking way.
    contract_version: int = 1

    #: How ACES should enforce limits around this agent.
    #:
    #: - ``inspect_managed``: The agent participates in Inspect/bridge limit
    #:   handling. ACES should not wrap it in ``asyncio.wait_for`` because that
    #:   can mask ``LimitExceededError``.
    #: - ``self_managed``: The agent brings its own execution loop/LLM client.
    #:   ACES should apply the ``asyncio.wait_for(max_steps * 120)`` backstop.
    execution_mode: Literal["inspect_managed", "self_managed"] = "inspect_managed"

    #: Whether the agent accepts ACES-resolved sandbox tools.  When False,
    #: the ``tools`` kwarg is omitted entirely (not passed as None).
    supports_tools: bool = True

    #: Whether the agent supports a limit-approaching callback.  When True,
    #: ACES passes ``limit_callback: Callable[[int, int], None]`` to the
    #: inner factory, where args are ``(current_step, max_steps)``.
    supports_limit_callback: bool = False

    #: External services the agent requires.  ACES validates these during
    #: preflight (``run_preflight=true``) by checking network reachability.
    #: Each entry is a URL string (e.g., ``"gremlin://localhost:8182"``).
    required_services: tuple[str, ...] = ()

    #: Optional module-level callable name for custom preflight validation.
    #: If set, ACES imports the agent module and calls this function during
    #: preflight.  The callable must accept no arguments and return True
    #: (healthy) or raise an exception with a diagnostic message.
    preflight_check_name: str | None = None


#: Current contract version — increment on breaking factory changes.
CURRENT_CONTRACT_VERSION = 1
