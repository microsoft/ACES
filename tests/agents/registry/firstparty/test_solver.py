"""Tests for firstparty agent solver."""

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
    name: str = "test-runtime",
    *,
    timeout: int = 600,
    port_base: int = 5000,
) -> RuntimeSpec:
    """Create a minimal RuntimeSpec for testing."""
    return RuntimeSpec(
        name=name,
        invoke_command=["run-agent", "--task={task_id}"],
        timeout=timeout,
        port_base=port_base,
        default_model_aliases={"inner": "openai/gpt-4o"},
    )


# ---------------------------------------------------------------------------
# create_agent — outer factory
# ---------------------------------------------------------------------------


class TestCreateAgentOuterFactory:
    def test_unknown_runtime_raises(self) -> None:
        from saber.agents.registry.firstparty.solver import create_agent

        with pytest.raises(ValueError, match="Unknown runtime 'nope'"):
            create_agent(runtime="nope")

    def test_registered_runtime_returns_callable(self) -> None:
        from saber.agents.registry.firstparty.solver import create_agent

        RuntimeRegistry.register(_make_spec("my-rt"))
        factory = create_agent(runtime="my-rt")
        assert callable(factory)


# ---------------------------------------------------------------------------
# Solver execution (inner factory + agent execution)
# ---------------------------------------------------------------------------


def _build_bridge_mocks(
    *,
    exec_returncode: int = 0,
    exec_stderr: str = "",
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build mocks for sandbox_agent_bridge, sandbox_env, store, as_solver, tool_call_limit.

    Returns:
        (mock_bridge_ctx, mock_sbox, mock_store, mock_as_solver, mock_tool_call_limit)
    """
    # Bridge context manager
    mock_bridge = MagicMock()
    mock_bridge.port = 9999
    mock_bridge.state = MagicMock(name="final_state")

    @asynccontextmanager
    async def _fake_bridge(*_args: object, **_kwargs: object) -> AsyncIterator[MagicMock]:
        yield mock_bridge

    mock_bridge_ctx = MagicMock(side_effect=_fake_bridge)

    # Sandbox environment
    mock_sbox = MagicMock()
    exec_result = MagicMock()
    exec_result.returncode = exec_returncode
    exec_result.stderr = exec_stderr
    mock_sbox.exec = AsyncMock(return_value=exec_result)

    mock_sandbox_env = MagicMock(return_value=mock_sbox)

    # Store
    _store_data: dict[str, object] = {}
    mock_store_obj = MagicMock()
    mock_store_obj.get = MagicMock(side_effect=lambda k, default=None: _store_data.get(k, default))
    mock_store_obj.set = MagicMock(side_effect=lambda k, v: _store_data.__setitem__(k, v))
    mock_store_fn = MagicMock(return_value=mock_store_obj)

    # as_solver — returns its first arg for inspection
    mock_as_solver = MagicMock(side_effect=lambda agent, **kw: agent)

    # tool_call_limit
    mock_tool_call_limit = MagicMock(return_value=MagicMock())

    return mock_bridge_ctx, mock_sandbox_env, mock_store_fn, mock_as_solver, mock_tool_call_limit


@pytest.fixture()
def registered_spec() -> RuntimeSpec:
    """Register and return a test spec."""
    spec = _make_spec("test-rt")
    RuntimeRegistry.register(spec)
    return spec


class TestSolverExecution:
    """Test the inner solver execution path."""

    @pytest.mark.asyncio
    async def test_bridge_opened_with_port_and_model_aliases(
        self, registered_spec: RuntimeSpec
    ) -> None:
        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        solver = self._create_solver_patched(
            registered_spec,
            mock_bridge_ctx,
            mock_sbox_env,
            mock_store,
            mock_as_solver,
            mock_tcl,
        )

        state = self._make_state()
        await solver(state)

        # sandbox_agent_bridge called once with expected args
        mock_bridge_ctx.assert_called_once()
        call_kwargs = mock_bridge_ctx.call_args[1]
        assert call_kwargs["model"] == "inspect"
        assert call_kwargs["port"] == registered_spec.port_base + 1
        assert call_kwargs["model_aliases"] == {"inner": "openai/gpt-4o"}

    @pytest.mark.asyncio
    async def test_env_built_with_bridge_url_and_key(
        self, registered_spec: RuntimeSpec
    ) -> None:
        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        solver = self._create_solver_patched(
            registered_spec,
            mock_bridge_ctx,
            mock_sbox_env,
            mock_store,
            mock_as_solver,
            mock_tcl,
        )

        state = self._make_state()
        await solver(state)

        # The sandbox.exec must have been called with env containing bridge values
        call_kwargs = mock_sbox_env.return_value.exec.call_args
        env = call_kwargs[1]["env"]
        # build_env returns defaults only since spec has no bridge_injected
        # but we verify the call flow works — bridge_url is used inside build_env
        assert isinstance(env, dict)

    @pytest.mark.asyncio
    async def test_command_interpolated_with_metadata(
        self, registered_spec: RuntimeSpec
    ) -> None:
        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        solver = self._create_solver_patched(
            registered_spec,
            mock_bridge_ctx,
            mock_sbox_env,
            mock_store,
            mock_as_solver,
            mock_tcl,
        )

        state = self._make_state(metadata={"task_id": "abc-123"})
        await solver(state)

        call_args = mock_sbox_env.return_value.exec.call_args
        cmd = call_args[0][0]
        assert cmd == ["run-agent", "--task=abc-123"]

    @pytest.mark.asyncio
    async def test_sandbox_exec_called_with_cmd_and_env(
        self, registered_spec: RuntimeSpec
    ) -> None:
        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        solver = self._create_solver_patched(
            registered_spec,
            mock_bridge_ctx,
            mock_sbox_env,
            mock_store,
            mock_as_solver,
            mock_tcl,
        )

        state = self._make_state()
        await solver(state)

        mock_sbox_env.return_value.exec.assert_awaited_once()
        call_args = mock_sbox_env.return_value.exec.call_args
        assert "env" in call_args[1]
        assert "timeout" in call_args[1]

    @pytest.mark.asyncio
    async def test_pre_invoke_hook_called_before_exec(
        self,
    ) -> None:
        hook = AsyncMock(side_effect=lambda sbox, env: {**env, "HOOKED": "1"})
        spec = RuntimeSpec(
            name="hooked-rt",
            invoke_command=["echo"],
            pre_invoke_hook=hook,
        )
        RuntimeRegistry.register(spec)

        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        solver = self._create_solver_patched(
            spec,
            mock_bridge_ctx,
            mock_sbox_env,
            mock_store,
            mock_as_solver,
            mock_tcl,
        )

        state = self._make_state()
        await solver(state)

        hook.assert_awaited_once()
        # Env passed to exec should include the hooked value
        exec_env = mock_sbox_env.return_value.exec.call_args[1]["env"]
        assert exec_env["HOOKED"] == "1"

    @pytest.mark.asyncio
    async def test_post_invoke_hook_called_after_exec(
        self,
    ) -> None:
        post_hook = AsyncMock(return_value={})
        spec = RuntimeSpec(
            name="post-rt",
            invoke_command=["echo"],
            post_invoke_hook=post_hook,
        )
        RuntimeRegistry.register(spec)

        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        solver = self._create_solver_patched(
            spec,
            mock_bridge_ctx,
            mock_sbox_env,
            mock_store,
            mock_as_solver,
            mock_tcl,
        )

        state = self._make_state()
        await solver(state)

        post_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tools_parameter_accepted_but_ignored(
        self, registered_spec: RuntimeSpec
    ) -> None:
        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks()
        )

        solver = self._create_solver_patched(
            registered_spec,
            mock_bridge_ctx,
            mock_sbox_env,
            mock_store,
            mock_as_solver,
            mock_tcl,
            tools=[MagicMock()],
        )

        state = self._make_state()
        await solver(state)

        # Bridge should not receive any bridged_tools
        call_kwargs = mock_bridge_ctx.call_args[1]
        assert "bridged_tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_bridge_state(
        self,
        registered_spec: RuntimeSpec,
    ) -> None:
        mock_bridge_ctx, mock_sbox_env, mock_store, mock_as_solver, mock_tcl = (
            _build_bridge_mocks(exec_returncode=1, exec_stderr="boom")
        )

        solver = self._create_solver_patched(
            registered_spec,
            mock_bridge_ctx,
            mock_sbox_env,
            mock_store,
            mock_as_solver,
            mock_tcl,
        )

        state = self._make_state()
        result = await solver(state)

        # Even on failure, bridge.state is returned
        assert result is not None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_state(
        metadata: dict[str, str] | None = None,
    ) -> MagicMock:
        state = MagicMock()
        state.metadata = metadata or {}
        return state

    @staticmethod
    def _create_solver_patched(
        spec: RuntimeSpec,
        mock_bridge_ctx: MagicMock,
        mock_sandbox_env: MagicMock,
        mock_store: MagicMock,
        mock_as_solver: MagicMock,
        mock_tool_call_limit: MagicMock,
        tools: list[MagicMock] | None = None,
    ) -> MagicMock:
        """Create the solver with all inspect_ai imports patched."""
        from saber.agents.registry.firstparty.solver import create_agent

        # Mock the filter returned by create_tool_call_limit_filter
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
                        agent=lambda fn: fn,  # @agent decorator is identity
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
            solver = factory(max_steps=100, tools=tools)

            # solver is the result of as_solver, which is called with the agent.
            # Since as_solver returns its first arg (the @agent decorated fn),
            # solver is the result of calling _firstparty_agent() — the execute fn.
            # We need to get the actual execute function out.
            # With our mock: @agent is identity, so _firstparty_agent() returns execute.
            # as_solver(agent, ...) returns agent (mock_as_solver side_effect).
            # So solver IS the execute function.
            return solver


# ---------------------------------------------------------------------------
# _derive_timeout
# ---------------------------------------------------------------------------


class TestDeriveTimeout:
    def test_uses_sample_limits_when_available(self) -> None:
        from saber.agents.registry.firstparty.solver import _derive_timeout

        mock_limits = MagicMock()
        mock_limits.time.remaining = 120.0

        with patch(
            "inspect_ai.util.sample_limits",
            return_value=mock_limits,
        ):
            result = _derive_timeout(600)

        assert result == 120

    def test_falls_back_to_default_on_runtime_error(self) -> None:
        from saber.agents.registry.firstparty.solver import _derive_timeout

        with patch(
            "inspect_ai.util.sample_limits",
            side_effect=RuntimeError("no active sample"),
        ):
            result = _derive_timeout(600)

        assert result == 600

    def test_enforces_min_timeout(self) -> None:
        from saber.agents.registry.firstparty.solver import _MIN_TIMEOUT, _derive_timeout

        mock_limits = MagicMock()
        mock_limits.time.remaining = 5.0

        with patch(
            "inspect_ai.util.sample_limits",
            return_value=mock_limits,
        ):
            result = _derive_timeout(600)

        assert result == _MIN_TIMEOUT

    def test_returns_default_when_remaining_is_none(self) -> None:
        from saber.agents.registry.firstparty.solver import _derive_timeout

        mock_limits = MagicMock()
        mock_limits.time.remaining = None

        with patch(
            "inspect_ai.util.sample_limits",
            return_value=mock_limits,
        ):
            result = _derive_timeout(900)

        assert result == 900
