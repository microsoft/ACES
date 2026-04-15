"""Integration smoke tests for firstparty agent pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry
from saber.agents.registry.firstparty.runtime_spec import RuntimeSpec


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # noqa: PT004
    """Reset the registry before and after each test for isolation."""
    RuntimeRegistry._reset()
    yield  # type: ignore[misc]
    RuntimeRegistry._reset()


def _make_spec(
    name: str = "integration-rt",
    *,
    timeout: int = 600,
    port_base: int = 5000,
    pre_invoke_hook: AsyncMock | None = None,
    post_invoke_hook: AsyncMock | None = None,
) -> RuntimeSpec:
    """Create a RuntimeSpec for integration testing."""
    return RuntimeSpec(
        name=name,
        invoke_command=["run-agent", "--task={task_id}"],
        timeout=timeout,
        port_base=port_base,
        default_model_aliases={"inner": "openai/gpt-4o"},
        pre_invoke_hook=pre_invoke_hook,
        post_invoke_hook=post_invoke_hook,
    )


def _build_bridge_mocks(
    *,
    exec_returncode: int = 0,
    exec_stderr: str = "",
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build mocks for sandbox_agent_bridge, sandbox_env, store, as_solver, tool_call_limit.

    Returns:
        (mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tool_call_limit)
    """
    mock_bridge = MagicMock()
    mock_bridge.port = 9999
    mock_bridge.state = MagicMock(name="final_state")

    @asynccontextmanager
    async def _fake_bridge(*_args: object, **_kwargs: object) -> AsyncIterator[MagicMock]:
        yield mock_bridge

    mock_bridge_ctx = MagicMock(side_effect=_fake_bridge)

    mock_sbox = MagicMock()
    exec_result = MagicMock()
    exec_result.returncode = exec_returncode
    exec_result.stderr = exec_stderr
    mock_sbox.exec = AsyncMock(return_value=exec_result)
    mock_sandbox_env = MagicMock(return_value=mock_sbox)

    _store_data: dict[str, object] = {}
    mock_store_obj = MagicMock()
    mock_store_obj.get = MagicMock(side_effect=lambda k, default=None: _store_data.get(k, default))
    mock_store_obj.set = MagicMock(side_effect=lambda k, v: _store_data.__setitem__(k, v))
    mock_store_fn = MagicMock(return_value=mock_store_obj)

    mock_as_solver = MagicMock(side_effect=lambda agent, **kw: agent)
    mock_tool_call_limit = MagicMock(return_value=MagicMock())

    return mock_bridge_ctx, mock_sandbox_env, mock_store_fn, mock_as_solver, mock_tool_call_limit


def _create_solver_patched(
    spec: RuntimeSpec,
    mock_bridge_ctx: MagicMock,
    mock_sandbox_env: MagicMock,
    mock_store: MagicMock,
    mock_as_solver: MagicMock,
    mock_tool_call_limit: MagicMock,
) -> MagicMock:
    """Create the solver with all inspect_ai imports patched."""
    from saber.agents.registry.firstparty.solver import create_agent

    mock_filter = MagicMock()
    mock_check = MagicMock()

    with (
        patch(
            "saber.agents.registry.firstparty.solver.create_tool_call_limit_filter",
            return_value=(mock_filter, mock_check),
        ),
        patch(
            "saber.agents.registry.firstparty.solver._derive_timeout",
            return_value=600,
        ),
        patch.dict(
            "sys.modules",
            {
                "inspect_ai": MagicMock(),
                "inspect_ai.agent": MagicMock(
                    sandbox_agent_bridge=mock_bridge_ctx,
                    as_solver=mock_as_solver,
                    agent=lambda fn: fn,
                    Agent=MagicMock,
                    AgentState=MagicMock,
                ),
                "inspect_ai.util": MagicMock(
                    sandbox=mock_sandbox_env,
                    store=mock_store,
                    tool_call_limit=mock_tool_call_limit,
                ),
            },
        ),
    ):
        factory = create_agent(runtime=spec.name)
        return factory(max_steps=100)


def _make_state(metadata: dict[str, str] | None = None) -> MagicMock:
    state = MagicMock()
    state.metadata = metadata or {}
    return state


# ---------------------------------------------------------------------------
# Test 1: Full Pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Register a RuntimeSpec, exercise the full two-level factory with mocked bridge."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> None:
        spec = _make_spec("pipeline-rt")
        RuntimeRegistry.register(spec)

        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        # create_agent(runtime=...) returns inner factory; inner factory returns solver
        solver = _create_solver_patched(
            spec, mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl
        )
        assert callable(solver)

        state = _make_state(metadata={"task_id": "smoke-1"})
        result = await solver(state)

        # Bridge is opened
        mock_bridge_ctx.assert_called_once()
        bridge_kwargs = mock_bridge_ctx.call_args[1]
        assert bridge_kwargs["model"] == "inspect"
        assert bridge_kwargs["model_aliases"] == {"inner": "openai/gpt-4o"}

        # Env is built from spec (dict passed to exec)
        exec_call = mock_sbox_env.return_value.exec
        exec_call.assert_awaited_once()
        exec_kwargs = exec_call.call_args[1]
        assert isinstance(exec_kwargs["env"], dict)

        # Command is interpolated
        cmd = exec_call.call_args[0][0]
        assert cmd == ["run-agent", "--task=smoke-1"]

        # sandbox.exec is called
        assert exec_kwargs["timeout"] == 600

        # bridge.state is returned
        assert result is not None


# ---------------------------------------------------------------------------
# Test 2: Auto-Discovery Verification
# ---------------------------------------------------------------------------


class TestAutoDiscovery:
    """Verify firstparty agent is discoverable via AgentRegistry."""

    def test_firstparty_in_agent_registry(self) -> None:
        from saber.agents import AgentRegistry

        # Reset and re-trigger auto-discovery
        AgentRegistry._reset()
        # Re-import to trigger _register_core_agents
        from saber.agents import _register_core_agents

        _register_core_agents()

        assert "firstparty" in AgentRegistry.list_agents()


# ---------------------------------------------------------------------------
# Test 3: Full Pipeline With Hooks Ordering
# ---------------------------------------------------------------------------


class TestHooksOrdering:
    """Verify pre_invoke_hook → sandbox.exec → post_invoke_hook ordering."""

    @pytest.mark.asyncio
    async def test_hooks_fire_in_correct_order(self) -> None:
        call_order: list[str] = []

        async def pre_hook(sbox: object, env: dict[str, str]) -> dict[str, str]:
            call_order.append("pre_invoke_hook")
            return {**env, "PRE": "1"}

        async def post_hook(sbox: object, env: dict[str, str]) -> None:
            call_order.append("post_invoke_hook")

        spec = _make_spec(
            "hooks-order-rt",
            pre_invoke_hook=AsyncMock(side_effect=pre_hook),
            post_invoke_hook=AsyncMock(side_effect=post_hook),
        )
        RuntimeRegistry.register(spec)

        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        # Intercept sandbox.exec to record call ordering
        original_exec = mock_sbox_env.return_value.exec

        async def _tracking_exec(*args: object, **kwargs: object) -> MagicMock:
            call_order.append("sandbox.exec")
            return await original_exec(*args, **kwargs)

        mock_sbox_env.return_value.exec = AsyncMock(side_effect=_tracking_exec)

        solver = _create_solver_patched(
            spec, mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl
        )

        state = _make_state(metadata={"task_id": "order-test"})
        await solver(state)

        assert call_order == ["pre_invoke_hook", "sandbox.exec", "post_invoke_hook"]
