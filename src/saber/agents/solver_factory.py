"""Solver factory — bridges agent factory to inspect_ai Solver."""

from __future__ import annotations

import functools
import importlib
import types

from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util._limit import LimitExceededError

from saber.agents.message_utils import TOOL_CALL_LIMIT_MESSAGE
from saber.agents.models import AgentCapabilities, AgentFactory, AgentPromptKwargs
from saber.config.models import ToolConfig
from saber.logging import get_logger
from saber.tools.registry import ResolvedTools, ToolRegistry

logger = get_logger(__name__)

# Re-export so existing imports (tests, etc.) continue to work.
__all__ = ["TOOL_CALL_LIMIT_MESSAGE", "create_saber_solver"]

# Built-in agent capabilities — gates which kwargs are forwarded to each agent.
# Plugin agents may declare a module-level AGENT_CAPABILITIES object to opt out
# of ACES-resolved sandbox tools.
_BUILTIN_CAPABILITIES: dict[str, AgentCapabilities] = {
    "react": AgentCapabilities(supports_tools=True),
    "copilot": AgentCapabilities(supports_tools=True),
    "claude_code": AgentCapabilities(supports_tools=True),
}

# Keep AGENT_CAPABILITIES as a read-only public alias for backward compatibility.
AGENT_CAPABILITIES: types.MappingProxyType[str, AgentCapabilities] = types.MappingProxyType(_BUILTIN_CAPABILITIES)


@functools.lru_cache(maxsize=64)
def _resolve_capabilities(agent_name: str) -> AgentCapabilities:
    """Resolve capabilities for an agent, checking built-in then plugin modules."""
    caps = _BUILTIN_CAPABILITIES.get(agent_name)
    if caps is not None:
        return caps

    from saber.agents import AgentRegistry

    factory = AgentRegistry.get(agent_name)
    if factory is not None:
        module = getattr(factory, "__module__", None)
        if module:
            try:
                mod = importlib.import_module(module)
            except ImportError:
                return AgentCapabilities()

            mod_caps = getattr(mod, "AGENT_CAPABILITIES", None)
            if isinstance(mod_caps, AgentCapabilities):
                return mod_caps
            if mod_caps is not None:
                logger.warning(
                    "Agent '%s' exports AGENT_CAPABILITIES but it is %s, not AgentCapabilities. Using defaults.",
                    agent_name,
                    type(mod_caps).__name__,
                )

    return AgentCapabilities()


def create_saber_solver(
    agent_name: str,
    agent_factory: AgentFactory,
    tool_registry: ToolRegistry | None = None,
    **kwargs: object,
) -> Solver:
    """Create solver that extracts prompts from sample metadata and invokes agent.

    The key insight: prompts are stored in Sample.metadata by ConfigLoader,
    and extracted at solve-time to support per-sample prompt customization.
    Tool configs are also stored per-sample in metadata["tools"], enabling
    per-task tool resolution.

    Args:
        agent_name: Name of the agent (for capabilities lookup).
        agent_factory: Two-level factory — factory() -> create_with_prompts(**kwargs) -> Solver.
        tool_registry: Optional ToolRegistry for per-sample tool resolution.
            Tools are resolved from ``state.metadata["tools"]`` at solve-time.

    Returns:
        A Solver that invokes the agent with per-sample prompts and tools.
    """

    @solver
    def saber_agent_solver() -> Solver:
        _tools_cache: dict[frozenset[tuple[str, str]], ResolvedTools] = {}
        create_with_prompts = agent_factory(**kwargs)
        caps = _resolve_capabilities(agent_name)

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            metadata = state.metadata or {}
            instruction = metadata.get("instruction_prompt") or ""
            assistant = metadata.get("assistant_prompt") or ""

            agent_kwargs: AgentPromptKwargs = {
                "instruction_prompt": instruction,
                "assistant_prompt": assistant,
            }

            per_sample_max_steps = metadata.get("max_steps")
            if not isinstance(per_sample_max_steps, int) or per_sample_max_steps <= 0:
                raise ValueError(
                    f"Task metadata must define a positive integer 'max_steps', "
                    f"got {per_sample_max_steps!r}. "
                    f"Set max_steps in your task YAML (e.g. global_defaults.max_steps: 200)."
                )
            agent_kwargs["max_steps"] = per_sample_max_steps

            resolved: ResolvedTools | None = None
            if tool_registry is not None:
                raw_tools = metadata.get("tools") or {}
                if raw_tools:
                    cache_key = frozenset(
                        (name, str(sorted(cfg.items()) if isinstance(cfg, dict) else cfg))
                        for name, cfg in raw_tools.items()
                    )
                    if cache_key not in _tools_cache:
                        tool_configs = {
                            name: ToolConfig(**cfg) if isinstance(cfg, dict) else ToolConfig()
                            for name, cfg in raw_tools.items()
                        }
                        _tools_cache[cache_key] = tool_registry.resolve(tool_configs)
                    resolved = _tools_cache[cache_key]

            if resolved is not None and caps.supports_tools:
                agent_solver = create_with_prompts(
                    tools=list(resolved.tools),
                    **agent_kwargs,
                )
            else:
                agent_solver = create_with_prompts(**agent_kwargs)

            if not callable(agent_solver):
                raise TypeError(f"Agent '{agent_name}' factory returned non-callable: {type(agent_solver).__name__}")

            try:
                return await agent_solver(state, generate)
            except LimitExceededError:
                logger.info(
                    "Tool call limit exceeded for agent '%s'. Injecting final-answer prompt.",
                    agent_name,
                )
                from saber.agents.message_utils import patch_orphaned_tool_calls

                patch_orphaned_tool_calls(state.messages)
                state.messages.append(ChatMessageUser(content=TOOL_CALL_LIMIT_MESSAGE))
                state.output = await get_model().generate(
                    input=state.messages,
                    tools=[],
                )
                state.messages.append(state.output.message)
                return state

        return solve

    return saber_agent_solver()
