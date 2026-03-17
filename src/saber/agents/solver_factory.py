"""Solver factory — bridges agent factory to inspect_ai Solver."""

from __future__ import annotations

from collections.abc import Callable

from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util._limit import LimitExceededError

from saber.agents.message_utils import TOOL_CALL_LIMIT_MESSAGE
from saber.agents.models import AgentCapabilities, AgentPromptKwargs
from saber.config.models import ToolConfig
from saber.logging import get_logger
from saber.tools.registry import ResolvedTools, ToolRegistry

logger = get_logger(__name__)

# Re-export so existing imports (tests, etc.) continue to work.
__all__ = ["TOOL_CALL_LIMIT_MESSAGE", "create_saber_solver"]

# Agent capabilities — gates which kwargs are forwarded to each agent.
# Unknown agents fall back to the default AgentCapabilities() (supports_tools=True).
AGENT_CAPABILITIES: dict[str, AgentCapabilities] = {
    "react": AgentCapabilities(supports_tools=True),
    "copilot": AgentCapabilities(supports_tools=True),
    "claude_code": AgentCapabilities(supports_tools=True),
}


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

            # Two-level factory call — forward kwargs (persona_file, skills_dir, etc.)
            create_with_prompts = agent_factory(**kwargs)

            # Capabilities-gated kwarg forwarding
            caps = AGENT_CAPABILITIES.get(agent_name, AgentCapabilities())
            if resolved is not None and caps.supports_tools:
                agent_solver = create_with_prompts(
                    tools=list(resolved.tools),
                    **agent_kwargs,
                )
            else:
                agent_solver = create_with_prompts(**agent_kwargs)

            # If agent_solver is a Solver, call it
            if not callable(agent_solver):
                raise TypeError(f"Agent '{agent_name}' factory returned non-callable: {type(agent_solver).__name__}")

            try:
                return await agent_solver(state, generate)
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

        return solve

    return saber_agent_solver()
