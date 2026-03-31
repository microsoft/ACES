"""Tests for AgentRegistry, plugin discovery, resolver, and solver factory."""

from __future__ import annotations

import asyncio
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.solver import TaskState

from saber.agents import AgentNotFoundError, AgentRegistry, _discover_plugin_agents
from saber.agents.models import AgentCapabilities
from saber.agents.resolver import register_agent_package, resolve_agent
from saber.agents.solver_factory import AGENT_CAPABILITIES, _resolve_capabilities, create_saber_solver
from saber.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # noqa: PT004
    """Save and restore registry state around each test."""
    saved = dict(AgentRegistry._agents)
    _resolve_capabilities.cache_clear()
    yield  # type: ignore[misc]
    AgentRegistry._agents = saved
    _resolve_capabilities.cache_clear()


# ---------------------------------------------------------------------------
# AgentRegistry basic operations
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    """AgentRegistry basic operations."""

    def test_register_and_get(self) -> None:
        """Register a factory, get it back."""

        def dummy_factory() -> str:
            return "dummy"

        AgentRegistry.register("test_agent", dummy_factory)
        assert AgentRegistry.get("test_agent") is dummy_factory

    def test_get_nonexistent_returns_none(self) -> None:
        """get() for unknown name returns None."""
        assert AgentRegistry.get("nonexistent_xyz_12345") is None

    def test_register_duplicate_raises(self) -> None:
        """ValueError on duplicate registration."""

        def factory_a() -> str:
            return "a"

        def factory_b() -> str:
            return "b"

        AgentRegistry.register("dup_agent", factory_a)
        with pytest.raises(ValueError, match="already registered"):
            AgentRegistry.register("dup_agent", factory_b)

    def test_register_override_replaces_existing(self) -> None:
        """override=True should replace an existing registration."""

        def factory_a() -> str:
            return "a"

        def factory_b() -> str:
            return "b"

        AgentRegistry.register("override_agent", factory_a)
        AgentRegistry.register("override_agent", factory_b, override=True)
        assert AgentRegistry.get("override_agent") is factory_b

    def test_list_agents_sorted(self) -> None:
        """list_agents returns sorted names."""
        AgentRegistry._reset()

        def noop() -> None: ...

        AgentRegistry.register("zeta", noop)
        AgentRegistry.register("alpha", noop)
        AgentRegistry.register("mid", noop)
        assert AgentRegistry.list_agents() == ["alpha", "mid", "zeta"]

    def test_reset_clears_registry(self) -> None:
        """_reset() clears all agents."""

        def noop() -> None: ...

        AgentRegistry.register("temp_agent", noop)
        assert AgentRegistry.get("temp_agent") is not None
        AgentRegistry._reset()
        assert AgentRegistry.get("temp_agent") is None
        assert AgentRegistry.list_agents() == []

    def test_react_auto_discovered(self) -> None:
        """'react' should be available from auto-discovery at import time."""
        factory = AgentRegistry.get("react")
        assert factory is not None
        assert callable(factory)


# ---------------------------------------------------------------------------
# resolve_agent
# ---------------------------------------------------------------------------


class TestResolveAgent:
    """resolve_agent() function."""

    def test_resolve_known_agent(self) -> None:
        """resolve_agent returns callable for registered agent."""

        def noop() -> None: ...

        AgentRegistry.register("known_test", noop)
        result = resolve_agent("known_test")
        assert result is noop

    def test_resolve_unknown_raises_with_plugin_guidance(self) -> None:
        """Unknown-agent error should explain built-in and plugin registration paths."""
        with pytest.raises(AgentNotFoundError, match="not found") as exc_info:
            resolve_agent("unknown_agent_xyz_99999")

        msg = str(exc_info.value)
        assert "unknown_agent_xyz_99999" in msg
        assert '[project.entry-points."saber.agents"]' in msg
        assert 'your_package.module:create_agent' in msg


# ---------------------------------------------------------------------------
# Plugin discovery and registration
# ---------------------------------------------------------------------------


class TestPluginDiscovery:
    """_discover_plugin_agents() entry-point loading."""

    def test_discover_registers_entry_point(self) -> None:
        """A mock entry point should be registered as an agent."""
        mock_factory = MagicMock()

        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"
        mock_ep.value = "fake_package.module:create_agent"
        mock_ep.load.return_value = mock_factory

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            _discover_plugin_agents()

        assert AgentRegistry.get("test_plugin") is mock_factory

    def test_discover_skips_duplicate_name(self) -> None:
        """Plugin agent with same name as built-in is skipped."""
        original_factory = AgentRegistry.get("react")

        mock_ep = MagicMock()
        mock_ep.name = "react"
        mock_ep.value = "fake_package:create_agent"
        mock_ep.load.return_value = MagicMock()

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            _discover_plugin_agents()

        assert AgentRegistry.get("react") is original_factory

    def test_discover_handles_load_failure(self) -> None:
        """Failed entry point load should not crash discovery."""
        mock_ep = MagicMock()
        mock_ep.name = "broken_plugin"
        mock_ep.value = "broken_package:create_agent"
        mock_ep.load.side_effect = ImportError("no such package")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            _discover_plugin_agents()

        assert AgentRegistry.get("broken_plugin") is None

    def test_discover_skips_non_callable(self) -> None:
        """Entry point resolving to non-callable is skipped."""
        mock_ep = MagicMock()
        mock_ep.name = "noncallable_plugin"
        mock_ep.value = "fake_package:NOT_A_FUNCTION"
        mock_ep.load.return_value = "I am a string, not callable"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            _discover_plugin_agents()

        assert AgentRegistry.get("noncallable_plugin") is None


class TestRegisterAgentPackage:
    """register_agent_package() dynamic module loading."""

    def test_register_from_module(self) -> None:
        """Should register an agent from a module path."""
        mock_module = MagicMock()
        mock_module.create_agent = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            register_agent_package("dynamic_test_agent", "fake_package.adapter")

        factory = AgentRegistry.get("dynamic_test_agent")
        assert factory is mock_module.create_agent

    def test_register_missing_module_raises(self) -> None:
        with pytest.raises(ImportError):
            register_agent_package("bad_agent", "nonexistent.fake.module")

    def test_register_module_without_create_agent_raises(self) -> None:
        with pytest.raises(AttributeError, match="create_agent"):
            register_agent_package("bad_agent", "os")

    def test_register_override_existing_agent(self) -> None:
        """register_agent_package should override a previously registered agent."""
        original_factory = MagicMock(name="original")
        AgentRegistry.register("override_test", original_factory)
        assert AgentRegistry.get("override_test") is original_factory

        new_module = MagicMock()
        new_module.create_agent = MagicMock(name="replacement")
        with patch("importlib.import_module", return_value=new_module):
            register_agent_package("override_test", "fake.override.module")

        assert AgentRegistry.get("override_test") is new_module.create_agent
        assert AgentRegistry.get("override_test") is not original_factory


# ---------------------------------------------------------------------------
# Capabilities and solver behavior
# ---------------------------------------------------------------------------


class TestCapabilitiesResolution:
    """Static AGENT_CAPABILITIES lookup for external agents."""

    def test_plugin_agent_reads_module_caps(self) -> None:
        mock_factory = MagicMock()
        mock_factory.__module__ = "fake_plugin_module"
        AgentRegistry.register("caps_test_plugin", mock_factory)

        mock_module = ModuleType("fake_plugin_module")
        mock_module.AGENT_CAPABILITIES = AgentCapabilities(supports_tools=False)  # type: ignore[attr-defined]

        with patch("saber.agents.solver_factory.importlib.import_module", return_value=mock_module):
            caps = _resolve_capabilities("caps_test_plugin")

        assert caps.supports_tools is False

    def test_unknown_agent_gets_default_caps(self) -> None:
        caps = _resolve_capabilities("nonexistent_agent_xyz")
        assert caps.supports_tools is True

    def test_invalid_plugin_capabilities_fall_back_to_default(self) -> None:
        mock_factory = MagicMock()
        mock_factory.__module__ = "bad_caps_module"
        AgentRegistry.register("bad_caps_agent", mock_factory)

        mock_module = ModuleType("bad_caps_module")
        mock_module.AGENT_CAPABILITIES = {"supports_tools": False}  # type: ignore[attr-defined]

        with patch("saber.agents.solver_factory.importlib.import_module", return_value=mock_module):
            caps = _resolve_capabilities("bad_caps_agent")

        assert caps.supports_tools is True

    def test_capability_cache_is_cleared_on_override_registration(self) -> None:
        first_factory = MagicMock()
        first_factory.__module__ = "first_caps_module"
        second_factory = MagicMock()
        second_factory.__module__ = "second_caps_module"

        first_module = ModuleType("first_caps_module")
        first_module.AGENT_CAPABILITIES = AgentCapabilities(supports_tools=False)  # type: ignore[attr-defined]
        second_module = ModuleType("second_caps_module")
        second_module.AGENT_CAPABILITIES = AgentCapabilities(supports_tools=True)  # type: ignore[attr-defined]

        modules = {
            "first_caps_module": first_module,
            "second_caps_module": second_module,
        }

        with patch(
            "saber.agents.solver_factory.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            AgentRegistry.register("cache_test_plugin", first_factory)
            first_caps = _resolve_capabilities("cache_test_plugin")
            AgentRegistry.register("cache_test_plugin", second_factory, override=True)
            second_caps = _resolve_capabilities("cache_test_plugin")

        assert first_caps.supports_tools is False
        assert second_caps.supports_tools is True

    def test_builtin_capabilities_alias_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            AGENT_CAPABILITIES["custom"] = AgentCapabilities()  # type: ignore[index]


class TestSolverFactory:
    """create_saber_solver() behavior for external agents."""

    @staticmethod
    def _make_state(*, include_tools: bool = True) -> MagicMock:
        state = MagicMock(spec=TaskState)
        state.metadata = {
            "instruction_prompt": "test instruction",
            "assistant_prompt": "test assistant",
            "max_steps": 5,
        }
        if include_tools:
            state.metadata["tools"] = {"bash": {}}
        state.messages = []
        state.output = None
        return state

    def test_solver_omits_tools_when_plugin_declares_no_support(self) -> None:
        captured: dict[str, object] = {}

        def agent_factory(**outer_kwargs: object):
            def create_with_prompts(**inner_kwargs: object):
                captured.update(inner_kwargs)

                async def solve(state: TaskState, generate: object) -> TaskState:
                    return state

                return solve

            return create_with_prompts

        agent_factory.__module__ = "no_tools_plugin_module"  # type: ignore[attr-defined]
        AgentRegistry.register("no_tools_plugin", agent_factory)

        fake_mod = ModuleType("no_tools_plugin_module")
        fake_mod.AGENT_CAPABILITIES = AgentCapabilities(supports_tools=False)  # type: ignore[attr-defined]

        with patch("saber.agents.solver_factory.importlib.import_module", return_value=fake_mod):
            solver = create_saber_solver(
                agent_name="no_tools_plugin",
                agent_factory=agent_factory,
                tool_registry=ToolRegistry(),
            )
            asyncio.run(solver(self._make_state(), AsyncMock()))

        assert captured["max_steps"] == 5
        assert "tools" not in captured

    def test_solver_passes_tools_by_default(self) -> None:
        captured: dict[str, object] = {}

        def agent_factory(**outer_kwargs: object):
            def create_with_prompts(**inner_kwargs: object):
                captured.update(inner_kwargs)

                async def solve(state: TaskState, generate: object) -> TaskState:
                    return state

                return solve

            return create_with_prompts

        agent_factory.__module__ = "missing_caps_module_xyz"  # type: ignore[attr-defined]
        solver = create_saber_solver(
            agent_name="default_tools_plugin",
            agent_factory=agent_factory,
            tool_registry=ToolRegistry(),
        )

        asyncio.run(solver(self._make_state(), AsyncMock()))
        assert captured["max_steps"] == 5
        assert "tools" in captured
        assert isinstance(captured["tools"], list)
        assert len(captured["tools"]) == 1

    def test_uncaught_exception_returns_model_output(self) -> None:
        def agent_factory(**outer_kwargs: object):
            def create_with_prompts(**inner_kwargs: object):
                async def solve(state: TaskState, generate: object) -> TaskState:
                    raise RuntimeError("boom")

                return solve

            return create_with_prompts

        solver = create_saber_solver(
            agent_name="failing_plugin",
            agent_factory=agent_factory,
        )

        result = asyncio.run(solver(self._make_state(include_tools=False), AsyncMock()))
        assert isinstance(result.output, ModelOutput)
        assert "internal error" in result.output.choices[0].message.content

    def test_cancelled_error_is_re_raised(self) -> None:
        def agent_factory(**outer_kwargs: object):
            def create_with_prompts(**inner_kwargs: object):
                async def solve(state: TaskState, generate: object) -> TaskState:
                    raise asyncio.CancelledError()

                return solve

            return create_with_prompts

        solver = create_saber_solver(
            agent_name="cancelled_plugin",
            agent_factory=agent_factory,
        )

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(solver(self._make_state(include_tools=False), AsyncMock()))
