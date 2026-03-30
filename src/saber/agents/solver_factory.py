"""Solver factory — bridges agent factory to inspect_ai Solver."""

from __future__ import annotations

import asyncio
import functools
import importlib
import types
from collections.abc import Callable

from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util._limit import LimitExceededError

from saber.agents.message_utils import TOOL_CALL_LIMIT_MESSAGE
from saber.agents.models import CURRENT_CONTRACT_VERSION, AgentCapabilities, AgentPromptKwargs
from saber.config.models import ToolConfig
from saber.logging import get_logger
from saber.tools.registry import ResolvedTools, ToolRegistry

logger = get_logger(__name__)

# Re-export so existing imports (tests, etc.) continue to work.
__all__ = ["TOOL_CALL_LIMIT_MESSAGE", "create_saber_solver"]

# Built-in agent capabilities — gates which kwargs are forwarded to each agent.
# Plugin agents declare their own capabilities via AGENT_CAPABILITIES module attribute.
# Unknown agents fall back to the default AgentCapabilities() (supports_tools=True).
_BUILTIN_CAPABILITIES: dict[str, AgentCapabilities] = {
    "react": AgentCapabilities(supports_tools=True),
    "copilot": AgentCapabilities(supports_tools=True),
    "claude_code": AgentCapabilities(supports_tools=True),
}

# Keep AGENT_CAPABILITIES as a read-only public alias for backward compatibility.
AGENT_CAPABILITIES: types.MappingProxyType[str, AgentCapabilities] = types.MappingProxyType(_BUILTIN_CAPABILITIES)


@functools.lru_cache(maxsize=64)
def _resolve_capabilities(agent_name: str) -> AgentCapabilities:
    """Resolve capabilities for an agent, checking built-in then plugin modules.

    Results are cached — capabilities are immutable and should not change
    between samples within a single eval run.

    Lookup order:
    1. Built-in capabilities dict
    2. Plugin agent module's AGENT_CAPABILITIES attribute
    3. Default (supports_tools=True)
    """
    # Check built-in first
    caps = _BUILTIN_CAPABILITIES.get(agent_name)
    if caps is not None:
        return caps

    # Try to find capabilities from the agent's module
    from saber.agents import AgentRegistry

    factory = AgentRegistry.get(agent_name)
    if factory is not None:
        module = getattr(factory, "__module__", None)
        if module:
            try:
                mod = importlib.import_module(module)
                mod_caps = getattr(mod, "AGENT_CAPABILITIES", None)
                if isinstance(mod_caps, AgentCapabilities):
                    return mod_caps
                if mod_caps is not None:
                    logger.warning(
                        "Agent '%s' exports AGENT_CAPABILITIES but it is %s, "
                        "not AgentCapabilities. Using defaults.",
                        agent_name,
                        type(mod_caps).__name__,
                    )
            except ImportError:
                pass

    return AgentCapabilities()


def create_saber_solver(
    agent_name: str,
    agent_factory: Callable[..., Callable[..., Solver]],
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

    @solver  # type: ignore[misc]
    def saber_agent_solver() -> Solver:
        _tools_cache: dict[frozenset[tuple[str, str]], ResolvedTools] = {}
        # Level-1 factory call — receives -T flags which don't change between
        # samples.  Cache the result so we don't re-run per sample.
        _create_with_prompts = agent_factory(**kwargs)

        # Resolve capabilities once at solver creation (immutable per eval run).
        caps = _resolve_capabilities(agent_name)

        # Contract version check — fail fast if plugin uses a newer contract
        if caps.contract_version > CURRENT_CONTRACT_VERSION:
            raise ValueError(
                f"Agent '{agent_name}' declares contract_version={caps.contract_version}, "
                f"but ACES supports up to version {CURRENT_CONTRACT_VERSION}. "
                f"Upgrade ACES or downgrade the agent's AgentCapabilities."
            )

        # Pre-compute whether this is a self-managed agent (no ACES tools).
        is_self_managed = not caps.supports_tools

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Extract prompts from sample metadata (set by tasks_to_samples)
            metadata = state.metadata or {}
            instruction = metadata.get("instruction_prompt") or ""
            assistant = metadata.get("assistant_prompt") or ""

            # Build agent kwargs
            agent_kwargs: AgentPromptKwargs = {
                "instruction_prompt": instruction,
                "assistant_prompt": assistant,
            }

            # Per-sample max_steps from metadata — REQUIRED
            per_sample_max_steps = metadata.get("max_steps")
            if not isinstance(per_sample_max_steps, int) or per_sample_max_steps <= 0:
                raise ValueError(
                    f"Task metadata must define a positive integer 'max_steps', "
                    f"got {per_sample_max_steps!r}. "
                    f"Set max_steps in your task YAML (e.g. global_defaults.max_steps: 200)."
                )
            state.tool_call_limit = per_sample_max_steps  # safety net
            agent_kwargs["max_steps"] = per_sample_max_steps  # forwarded to agent

            # Limit callback for agents that support it — fires at 80% budget
            if caps.supports_limit_callback:
                threshold = int(per_sample_max_steps * 0.8)

                def _limit_callback(current_step: int, max_steps: int) -> None:
                    if current_step >= max_steps:
                        logger.warning(
                            "Agent '%s' exceeded max_steps (%d/%d)",
                            agent_name,
                            current_step,
                            max_steps,
                        )
                    elif current_step >= threshold:
                        logger.info(
                            "Agent '%s' approaching limit (%d/%d — 80%% threshold)",
                            agent_name,
                            current_step,
                            max_steps,
                        )

                agent_kwargs["limit_callback"] = _limit_callback  # type: ignore[typeddict-unknown-key]

            # Per-sample tool resolution from metadata
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
                agent_solver = _create_with_prompts(
                    tools=list(resolved.tools),
                    **agent_kwargs,
                )
            else:
                agent_solver = _create_with_prompts(**agent_kwargs)

            # If agent_solver is a Solver, call it
            if not callable(agent_solver):
                raise TypeError(f"Agent '{agent_name}' factory returned non-callable: {type(agent_solver).__name__}")

            # Per-agent timeout — tighter backstop for self-managed agents.
            # Built-in/Inspect-integrated agents already have LimitExceededError;
            # wrapping them would mask that signal.
            agent_timeout = per_sample_max_steps * 120 if is_self_managed else None

            try:
                if agent_timeout is not None:
                    return await asyncio.wait_for(
                        agent_solver(state, generate),
                        timeout=agent_timeout,
                    )
                else:
                    return await agent_solver(state, generate)
            except TimeoutError:
                logger.warning(
                    "Agent '%s' timed out after %ds (max_steps=%d). Writing partial output.",
                    agent_name,
                    agent_timeout,
                    per_sample_max_steps,
                )
                if not state.output:
                    state.output = ModelOutput.from_content(
                        model=agent_name,
                        content=f"Agent timed out after {agent_timeout}s with no output.",
                    )
                return state
            except LimitExceededError:
                # The agent's tool call limit was exceeded.  Give it one
                # tool-free generation to produce a final answer instead
                # of letting the sample die with no output.
                logger.info(
                    "Tool call limit exceeded for agent '%s'. Injecting final-answer prompt.",
                    agent_name,
                )
                # Patch any orphaned tool calls (assistant messages with
                # tool_calls that lack matching tool-result messages)
                # before sending to the API.  Without this, the API
                # rejects the request with a 400 error.
                from saber.agents.message_utils import patch_orphaned_tool_calls

                patch_orphaned_tool_calls(state.messages)
                state.messages.append(ChatMessageUser(content=TOOL_CALL_LIMIT_MESSAGE))
                state.output = await get_model().generate(
                    input=state.messages,
                    tools=[],
                )
                state.messages.append(state.output.message)
                return state
            except asyncio.CancelledError:
                # Let Inspect handle task cancellation — do not swallow.
                raise
            except Exception:
                # Catch uncaught plugin agent exceptions — fail this sample
                # with partial output rather than crashing the entire eval run.
                logger.exception(
                    "Agent '%s' raised an uncaught exception.",
                    agent_name,
                )
                if not state.output:
                    state.output = ModelOutput.from_content(
                        model=agent_name,
                        content=f"Agent '{agent_name}' failed with an internal error.",
                    )
                return state

        return solve

    return saber_agent_solver()
