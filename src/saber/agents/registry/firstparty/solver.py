"""First-party agent solver — sandbox_agent_bridge integration."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from saber.agents.bridge_utils import (
    create_tool_call_limit_filter,
    parse_bridge_stderr,
)
from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry
from saber.agents.registry.firstparty.runtime_spec import AnalysisMode, interpolate_command
from saber.agents.transcript.copilot_v3 import parse_copilot_v3
from saber.agents.transcript.emit import emit_transcript_events
from saber.agents.transcript.models import CopilotSession
from saber.logging import get_logger

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage
    from inspect_ai.solver import Solver
    from inspect_ai.tool import Tool
    from inspect_ai.util import SandboxEnvironment

logger = get_logger(__name__)

_STORE_PORT_KEY = "firstparty_model_port"
_MIN_TIMEOUT = 30
# Headroom (seconds) reserved for transcript reading after sbox.exec times out
# but before inspect_ai's time_limit cancels the solver.
_TIMEOUT_HEADROOM = 30
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
        raise ValueError(
            f"Unknown runtime '{runtime}'. "
            f"Available: {RuntimeRegistry.list_runtimes()}"
        )
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

        @agent
        def _firstparty_agent() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                if spec.analysis_mode == AnalysisMode.DETACHED:
                    # Detached mode constructs its own AgentState from
                    # the transcript — incoming state is intentionally unused.
                    return await _execute_detached()

                return await _execute_attached(state)

            async def _execute_detached() -> AgentState:
                import anyio

                sbox = sandbox_env()

                env = spec.build_env()

                metadata: dict[str, str] = store().get("firstparty_sample_metadata", {})
                for key in _METADATA_KEYS_TO_BRIDGE:
                    if key in metadata and key not in env:
                        env[key] = str(metadata[key])

                kwargs_str = {k: str(v) for k, v in outer_kwargs.items()}

                if spec.pre_invoke_hook is not None:
                    env = await spec.pre_invoke_hook(sbox, env, kwargs_str)

                cmd = interpolate_command(
                    spec.invoke_command,
                    **{**env, **metadata, **kwargs_str},
                )

                # Derive timeout AFTER pre_invoke_hook so remaining time
                # accounts for any work the hook performed (e.g. tarball
                # download).
                timeout = _derive_timeout(spec.timeout)

                timed_out = False
                cancelled = False
                result = None
                try:
                    result = await sbox.exec(
                        cmd, env=env, timeout=timeout, timeout_retry=False
                    )
                except TimeoutError:
                    timed_out = True
                    logger.warning(
                        "1P agent '%s' timed out after %ds — "
                        "will read partial transcripts",
                        spec.name,
                        timeout,
                    )
                except anyio.get_cancelled_exc_class():
                    # inspect_ai's time_limit fires via anyio cancellation,
                    # not TimeoutError.  We still need to read transcripts.
                    cancelled = True
                    logger.warning(
                        "1P agent '%s' cancelled (time_limit) — "
                        "will read partial transcripts in shielded scope",
                        spec.name,
                    )

                if result is not None and result.returncode != 0:
                    stderr = result.stderr or ""
                    stdout = result.stdout or ""
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

                # Shield post-invoke hook and transcript reading from
                # cancellation so the solver can populate AgentState even
                # when the outer time_limit cancelled sbox.exec.
                with anyio.CancelScope(shield=True):
                    if spec.post_invoke_hook is not None:
                        try:
                            await spec.post_invoke_hook(sbox, env, kwargs_str)
                        except Exception:
                            if timed_out or cancelled:
                                logger.warning(
                                    "Post-invoke hook failed after timeout/cancel for '%s'",
                                    spec.name,
                                )
                            else:
                                raise

                    # Read and parse transcript directory
                    if spec.transcript_config is None:
                        msg = f"transcript_config is required for detached runtime '{spec.name}'"
                        raise ValueError(msg)

                    try:
                        messages = await _read_transcript_dir(
                            sbox, spec.transcript_config.path, spec.name
                        )
                    except Exception:
                        logger.error(
                            "Failed to read/parse transcript from '%s' for runtime '%s'",
                            spec.transcript_config.path,
                            spec.name,
                        )
                        messages = []

                    if not messages:
                        logger.warning(
                            "No messages parsed for runtime '%s' — "
                            "scoring will run on empty state",
                            spec.name,
                        )
                    else:
                        try:
                            emit_transcript_events(
                                messages,
                                model_name=f"{spec.name}/detached",
                            )
                        except Exception:
                            logger.warning(
                                "Failed to emit transcript events for '%s'",
                                spec.name,
                                exc_info=True,
                            )

                return AgentState(messages=messages)

            async def _execute_attached(state: AgentState) -> AgentState:
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
                    kwargs_str = {k: str(v) for k, v in outer_kwargs.items()}
                    if spec.pre_invoke_hook is not None:
                        env = await spec.pre_invoke_hook(sbox, env, kwargs_str)

                    # Interpolate command with env defaults + metadata + outer kwargs
                    # (env provides baseline values; metadata and kwargs_str override)
                    cmd = interpolate_command(
                        spec.invoke_command,
                        **{**env, **metadata, **kwargs_str},
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
                        await spec.post_invoke_hook(sbox, env, kwargs_str)

                    # Check tool call limits
                    check_tool_limit()

                    return bridge.state

            return execute

        return as_solver(_firstparty_agent(), limits=[tool_call_limit(max_steps)])

    return create_with_prompts


def _derive_timeout(default: int) -> int:
    """Derive solver timeout from inspect_ai sample limits or fallback.

    Reserves ``_TIMEOUT_HEADROOM`` seconds so that ``sbox.exec()`` times out
    *before* inspect_ai's ``time_limit`` cancels the solver. This lets the
    solver's ``except TimeoutError`` handler run transcript-reading code
    instead of being silently cancelled by the outer cancel scope.

    Args:
        default: Default timeout in seconds.

    Returns:
        Timeout in seconds.
    """
    try:
        from inspect_ai.util import sample_limits

        remaining = sample_limits().time.remaining
        if remaining is not None:
            return max(int(remaining) - _TIMEOUT_HEADROOM, _MIN_TIMEOUT)
        return default
    except RuntimeError:
        return default


async def _read_transcript_dir(
    sbox: SandboxEnvironment,
    dir_path: str,
    runtime_name: str,
) -> list[ChatMessage]:
    """Read all copilot-log JSON files from a directory inside the sandbox.

    Hyenas writes one JSON file per agent conversation in a nested directory
    structure under the copilot-log path.  This function finds all ``.json``
    files, parses each as a :class:`CopilotSession`, and returns the
    aggregated messages in chronological order.

    Args:
        sbox: SandboxEnvironment instance.
        dir_path: Path to the copilot-log directory in the container.
        runtime_name: Runtime name for log messages.

    Returns:
        List of ChatMessage objects aggregated from all sessions.
    """

    # List all JSON files in the directory tree
    find_result = await sbox.exec(  # type: ignore[union-attr]
        ["find", dir_path, "-name", "*.json", "-type", "f"],
        timeout=30,
    )
    if not find_result.success or not find_result.stdout.strip():
        logger.warning(
            "No copilot-log JSON files found in '%s' for runtime '%s'",
            dir_path,
            runtime_name,
        )
        return []

    json_paths = [p for p in find_result.stdout.strip().split("\n") if p.endswith(".json")]
    logger.info(
        "Found %d copilot-log transcript files for runtime '%s'",
        len(json_paths),
        runtime_name,
    )

    all_messages: list[ChatMessage] = []
    # Collect (startedAt, messages) tuples for chronological sorting
    session_entries: list[tuple[str, list[ChatMessage]]] = []

    for path in json_paths:
        try:
            raw = await sbox.read_file(path)  # type: ignore[union-attr]
            data = json.loads(raw)
            session = CopilotSession.model_validate(data)
        except Exception:
            logger.warning("Skipping unparseable transcript: %s", path)
            continue

        msgs = parse_copilot_v3(session)
        if msgs:
            session_entries.append((session.session.startedAt, msgs))

    # Sort sessions by start time, then flatten
    session_entries.sort(key=lambda x: x[0])
    for _, msgs in session_entries:
        all_messages.extend(msgs)

    logger.info(
        "Parsed %d messages from %d sessions for runtime '%s'",
        len(all_messages),
        len(session_entries),
        runtime_name,
    )
    return all_messages
