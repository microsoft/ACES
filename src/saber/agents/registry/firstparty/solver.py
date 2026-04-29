"""First-party agent solver — sandbox_agent_bridge integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from saber.agents.bridge_utils import (
    create_tool_call_limit_filter,
    parse_bridge_stderr,
)
from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry
from saber.agents.registry.firstparty.runtime_spec import interpolate_command
from saber.logging import get_logger

if TYPE_CHECKING:
    from inspect_ai.solver import Solver
    from inspect_ai.tool import Tool

logger = get_logger(__name__)

_STORE_PORT_KEY = "firstparty_model_port"
_MIN_TIMEOUT = 30
_METADATA_KEYS_TO_BRIDGE = frozenset({"repo_tarball", "repo_path", "repo_tarball_url", "task_id"})


def create_agent(
    runtime: str = "",
    **kwargs: object,
) -> Callable[..., Solver]:
    """Outer factory — accepts -T parameters.

    Args:
        runtime: Name of the registered runtime spec.
        **kwargs: Additional parameters from -T flags.

    Returns:
        Inner factory callable.

    Raises:
        ValueError: If runtime is not registered.
    """
    spec = RuntimeRegistry.get(runtime)
    if spec is None:
        raise ValueError(f"Unknown runtime '{runtime}'. Available: {RuntimeRegistry.list_runtimes()}")
    outer_kwargs = kwargs

    def create_with_prompts(
        instruction_prompt: str = "",
        assistant_prompt: str = "",
        tools: Sequence[Tool] | None = None,
        *,
        max_steps: int,
        **extra_kwargs: object,
    ) -> Solver:
        """Inner factory returning a Solver."""
        from inspect_ai.agent import Agent, AgentState, agent, as_solver, sandbox_agent_bridge
        from inspect_ai.util import sandbox as sandbox_env
        from inspect_ai.util import store, tool_call_limit

        # @agent's get_type_hints() resolves annotations against module globals.
        # With `from __future__ import annotations`, all type hints are strings,
        # so Agent/AgentState must be in module globals for resolution.
        globals().setdefault("Agent", Agent)
        globals().setdefault("AgentState", AgentState)

        if instruction_prompt or assistant_prompt:
            logger.debug(
                "1P runtime '%s' manages its own prompts; "
                "instruction_prompt and assistant_prompt are passed via env/hooks, not injected.",
                spec.name,
            )

        model_aliases = spec.build_model_aliases()

        typed_agent = cast("Callable[[Callable[[], Agent]], Callable[[], Agent]]", agent)

        @typed_agent
        def _firstparty_agent() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                port = store().get(_STORE_PORT_KEY, spec.port_base) + 1
                store().set(_STORE_PORT_KEY, port)

                timeout = _derive_timeout(spec.timeout)

                bridge_filter, check_tool_limit = create_tool_call_limit_filter()

                async with sandbox_agent_bridge(
                    state,
                    model="inspect",
                    port=port,
                    filter=bridge_filter,
                    model_aliases=model_aliases,
                ) as bridge:
                    sbox = sandbox_env()
                    bridge_url = f"http://localhost:{bridge.port}/v1"
                    bridge_api_key = "saber-bridge"

                    env = spec.build_env(
                        bridge_url=bridge_url,
                        bridge_api_key=bridge_api_key,
                    )

                    # Bridge specific metadata keys into env for pre_invoke_hook.
                    # Only repo-related keys are bridged — not large prompt strings.
                    # AgentState has no .metadata, so we read from store()
                    # (stashed by solver_factory from TaskState.metadata).
                    metadata: dict[str, str] = store().get("firstparty_sample_metadata", {})
                    for key in _METADATA_KEYS_TO_BRIDGE:
                        if key in metadata and key not in env:
                            env[key] = str(metadata[key])

                    # Pre-invoke hook
                    if spec.pre_invoke_hook is not None:
                        env = await spec.pre_invoke_hook(sbox, env)

                    # Interpolate command with env defaults + metadata + outer kwargs
                    # (env provides baseline values; metadata and outer_kwargs override)
                    cmd = interpolate_command(
                        spec.invoke_command,
                        **{k: str(v) for k, v in {**env, **metadata, **outer_kwargs}.items()},
                    )

                    # Execute
                    result = await sbox.exec(cmd, env=env, timeout=timeout)

                    if result.returncode != 0:
                        stderr = result.stderr or ""
                        stdout = result.stdout or ""
                        diagnostics = parse_bridge_stderr(stderr)
                        if diagnostics:
                            logger.error("Bridge error:\n%s", diagnostics)
                        logger.warning(
                            "1P agent '%s' exited %d: %s",
                            spec.name,
                            result.returncode,
                            stderr[:2000],
                        )
                        if stdout:
                            logger.warning(
                                "1P agent '%s' stdout:\n%s",
                                spec.name,
                                stdout[:2000],
                            )

                    # Post-invoke hook
                    if spec.post_invoke_hook is not None:
                        await spec.post_invoke_hook(sbox, env)

                    # Check tool call limits
                    check_tool_limit()

                    return bridge.state

            return execute

        return as_solver(_firstparty_agent(), limits=[tool_call_limit(max_steps)])

    return create_with_prompts


def _derive_timeout(default: int) -> int:
    """Derive solver timeout from inspect_ai sample limits or fallback.

    Args:
        default: Default timeout in seconds.

    Returns:
        Timeout in seconds.
    """
    try:
        from inspect_ai.util import sample_limits

        remaining = sample_limits().time.remaining
        return max(int(remaining), _MIN_TIMEOUT) if remaining is not None else default
    except RuntimeError:
        return default
