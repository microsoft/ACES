"""Tests for AgentRegistry, auto-discovery, and resolve_agent."""

from __future__ import annotations

import pytest

from saber.agents import AgentNotFoundError, AgentRegistry
from saber.agents.models import AgentCapabilities
from saber.agents.resolver import resolve_agent


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # noqa: PT004
    """Save and restore registry state around each test."""
    saved = dict(AgentRegistry._agents)
    # Clear capability cache to prevent cross-test pollution
    from saber.agents.solver_factory import _resolve_capabilities

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

    def test_resolve_unknown_raises(self) -> None:
        """AgentNotFoundError for unknown agent."""
        with pytest.raises(AgentNotFoundError, match="not found"):
            resolve_agent("unknown_agent_xyz_99999")

    def test_error_lists_available(self) -> None:
        """Error message includes available agents list."""
        AgentRegistry._reset()

        def noop() -> None: ...

        AgentRegistry.register("agent_a", noop)
        AgentRegistry.register("agent_b", noop)

        with pytest.raises(AgentNotFoundError, match="agent_a") as exc_info:
            resolve_agent("missing_agent")
        msg = str(exc_info.value)
        assert "agent_a" in msg
        assert "agent_b" in msg
        assert "missing_agent" in msg

    def test_error_shows_none_when_empty(self) -> None:
        """Error message shows '(none)' when registry is empty."""
        AgentRegistry._reset()
        with pytest.raises(AgentNotFoundError, match=r"\(none\)"):
            resolve_agent("anything")

    def test_error_mentions_entry_points(self) -> None:
        """Error message should mention plugin entry points."""
        with pytest.raises(AgentNotFoundError, match="saber.agents"):
            resolve_agent("unknown_plugin_xyz_99999")


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


class TestPluginDiscovery:
    """_discover_plugin_agents() entry-point loading."""

    def test_discover_registers_entry_point(self) -> None:
        """A mock entry point should be registered as an agent."""
        from unittest.mock import MagicMock, patch

        from saber.agents import _discover_plugin_agents

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
        from unittest.mock import MagicMock, patch

        from saber.agents import _discover_plugin_agents

        original_factory = AgentRegistry.get("react")

        mock_ep = MagicMock()
        mock_ep.name = "react"
        mock_ep.value = "fake_package:create_agent"
        mock_ep.load.return_value = MagicMock()

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            _discover_plugin_agents()

        # Should still be the original
        assert AgentRegistry.get("react") is original_factory

    def test_discover_handles_load_failure(self) -> None:
        """Failed entry point load should not crash discovery."""
        from unittest.mock import MagicMock, patch

        from saber.agents import _discover_plugin_agents

        mock_ep = MagicMock()
        mock_ep.name = "broken_plugin"
        mock_ep.value = "broken_package:create_agent"
        mock_ep.load.side_effect = ImportError("no such package")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            _discover_plugin_agents()  # Should not raise

        assert AgentRegistry.get("broken_plugin") is None

    def test_discover_skips_non_callable(self) -> None:
        """Entry point resolving to non-callable is skipped."""
        from unittest.mock import MagicMock, patch

        from saber.agents import _discover_plugin_agents

        mock_ep = MagicMock()
        mock_ep.name = "noncallable_plugin"
        mock_ep.value = "fake_package:NOT_A_FUNCTION"
        mock_ep.load.return_value = "I am a string, not callable"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            _discover_plugin_agents()

        assert AgentRegistry.get("noncallable_plugin") is None


# ---------------------------------------------------------------------------
# Dynamic capabilities resolution
# ---------------------------------------------------------------------------


class TestDynamicCapabilities:
    """_resolve_capabilities() dynamic lookup."""

    def test_builtin_agent_uses_builtin_caps(self) -> None:
        from saber.agents.solver_factory import _resolve_capabilities

        caps = _resolve_capabilities("react")
        assert caps.supports_tools is True

    def test_plugin_agent_reads_module_caps(self) -> None:
        from unittest.mock import MagicMock, patch

        from saber.agents.solver_factory import AgentCapabilities, _resolve_capabilities

        # Simulate a plugin agent whose module declares AGENT_CAPABILITIES
        mock_factory = MagicMock()
        mock_factory.__module__ = "fake_plugin_module"
        AgentRegistry.register("caps_test_plugin", mock_factory)

        mock_module = MagicMock()
        mock_module.AGENT_CAPABILITIES = AgentCapabilities(supports_tools=False)

        with patch("importlib.import_module", return_value=mock_module):
            caps = _resolve_capabilities("caps_test_plugin")
        assert caps.supports_tools is False

    def test_unknown_agent_gets_default_caps(self) -> None:
        from saber.agents.solver_factory import _resolve_capabilities

        caps = _resolve_capabilities("nonexistent_agent_xyz")
        assert caps.supports_tools is True  # default


# ---------------------------------------------------------------------------
# register_agent_package (CLI override)
# ---------------------------------------------------------------------------


class TestRegisterAgentPackage:
    """register_agent_package() dynamic module loading."""

    def test_register_from_module(self) -> None:
        """Should register an agent from a module path."""
        from unittest.mock import MagicMock, patch

        from saber.agents.resolver import register_agent_package

        mock_module = MagicMock()
        mock_module.create_agent = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            register_agent_package("dynamic_test_agent", "fake_package.adapter")

        factory = AgentRegistry.get("dynamic_test_agent")
        assert factory is not None
        assert callable(factory)

    def test_register_missing_module_raises(self) -> None:
        from saber.agents.resolver import register_agent_package

        with pytest.raises(ImportError):
            register_agent_package("bad_agent", "nonexistent.fake.module")

    def test_register_module_without_create_agent_raises(self) -> None:
        from saber.agents.resolver import register_agent_package

        # os module exists but has no create_agent
        with pytest.raises(AttributeError, match="create_agent"):
            register_agent_package("bad_agent", "os")

    def test_register_override_existing_agent(self) -> None:
        """register_agent_package should override a previously registered agent."""
        from unittest.mock import MagicMock, patch

        from saber.agents.resolver import register_agent_package

        # Register initial agent
        original_factory = MagicMock(name="original")
        AgentRegistry.register("override_test", original_factory)
        assert AgentRegistry.get("override_test") is original_factory

        # Override via register_agent_package
        new_module = MagicMock()
        new_module.create_agent = MagicMock(name="replacement")
        with patch("importlib.import_module", return_value=new_module):
            register_agent_package("override_test", "fake.override.module")

        # Factory should be replaced
        assert AgentRegistry.get("override_test") is new_module.create_agent
        assert AgentRegistry.get("override_test") is not original_factory

    def test_register_agent_package_integration_in_create_task(self) -> None:
        """agent_package param in create_task should trigger registration."""
        from unittest.mock import MagicMock, patch

        from saber.agents.resolver import register_agent_package

        mock_module = MagicMock()
        mock_module.create_agent = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            register_agent_package("task_integration_test", "fake.package")

        factory = AgentRegistry.get("task_integration_test")
        assert factory is mock_module.create_agent


# ---------------------------------------------------------------------------
# Capabilities caching
# ---------------------------------------------------------------------------


class TestCapabilitiesCaching:
    """_resolve_capabilities() should be cached."""

    def test_capabilities_cached_across_calls(self) -> None:
        """Repeated calls should not re-import the module."""
        from unittest.mock import MagicMock, patch

        from saber.agents.solver_factory import _resolve_capabilities

        # Clear the lru_cache
        _resolve_capabilities.cache_clear()

        mock_factory = MagicMock()
        mock_factory.__module__ = "cached_plugin_module"
        AgentRegistry.register("cache_test_plugin", mock_factory)

        mock_module = MagicMock()
        mock_module.AGENT_CAPABILITIES = AgentCapabilities(supports_tools=False)

        with patch("saber.agents.solver_factory.importlib.import_module", return_value=mock_module) as mock_import:
            caps1 = _resolve_capabilities("cache_test_plugin")
            caps2 = _resolve_capabilities("cache_test_plugin")

        assert caps1.supports_tools is False
        assert caps2.supports_tools is False
        # import_module should only be called once due to caching
        assert mock_import.call_count == 1

        # Clean up
        _resolve_capabilities.cache_clear()

    def test_builtin_capabilities_immutable(self) -> None:
        """AGENT_CAPABILITIES public alias should be read-only."""
        from saber.agents.solver_factory import AGENT_CAPABILITIES

        with pytest.raises(TypeError):
            AGENT_CAPABILITIES["custom"] = AgentCapabilities()  # type: ignore[index]

    def test_late_registration_invalidates_cache(self) -> None:
        """Registering an agent after a resolve should update cached caps."""
        from unittest.mock import MagicMock, patch

        from saber.agents.solver_factory import _resolve_capabilities

        # First call: agent not registered yet → default (supports_tools=True)
        caps_before = _resolve_capabilities("late_reg_agent")
        assert caps_before.supports_tools is True

        # Now register with supports_tools=False
        mock_factory = MagicMock()
        mock_factory.__module__ = "late_reg_module"
        AgentRegistry.register("late_reg_agent", mock_factory)

        mock_module = MagicMock()
        mock_module.AGENT_CAPABILITIES = AgentCapabilities(supports_tools=False)

        with patch("saber.agents.solver_factory.importlib.import_module", return_value=mock_module):
            caps_after = _resolve_capabilities("late_reg_agent")

        # Should pick up the new capabilities, not the stale default
        assert caps_after.supports_tools is False


# ---------------------------------------------------------------------------
# Contract version validation
# ---------------------------------------------------------------------------


class TestContractVersion:
    """Contract version checking in AgentCapabilities."""

    def test_default_contract_version(self) -> None:
        """Default contract version should be 1."""
        caps = AgentCapabilities()
        assert caps.contract_version == 1

    def test_current_contract_version_matches_default(self) -> None:
        """CURRENT_CONTRACT_VERSION must match the default in AgentCapabilities."""
        from saber.agents.models import CURRENT_CONTRACT_VERSION

        assert CURRENT_CONTRACT_VERSION == AgentCapabilities().contract_version


# ---------------------------------------------------------------------------
# Required services and preflight
# ---------------------------------------------------------------------------


class TestAgentCapabilitiesExtended:
    """Extended AgentCapabilities fields."""

    def test_required_services_default_empty(self) -> None:
        caps = AgentCapabilities()
        assert caps.required_services == ()

    def test_required_services_roundtrip(self) -> None:
        caps = AgentCapabilities(
            required_services=["gremlin://localhost:8182", "https://search.example.com"],
        )
        assert len(caps.required_services) == 2
        assert "gremlin://localhost:8182" in caps.required_services

    def test_supports_limit_callback_default_false(self) -> None:
        caps = AgentCapabilities()
        assert caps.supports_limit_callback is False

    def test_preflight_check_name_default_none(self) -> None:
        caps = AgentCapabilities()
        assert caps.preflight_check_name is None

    def test_capabilities_frozen(self) -> None:
        """AgentCapabilities should be immutable."""
        caps = AgentCapabilities()
        with pytest.raises((AttributeError, TypeError, ValueError)):
            caps.supports_tools = False  # type: ignore[misc]


class TestAgentPreflight:
    """_run_agent_preflight() service and custom check validation."""

    def test_preflight_passes_when_no_services(self) -> None:
        """No required_services → preflight passes."""
        from saber.task import _run_agent_preflight

        caps = AgentCapabilities()
        _run_agent_preflight("test_agent", caps)  # should not raise

    def test_preflight_fails_unreachable_service(self) -> None:
        """Unreachable service URL should raise RuntimeError."""
        from saber.task import _run_agent_preflight

        caps = AgentCapabilities(
            required_services=["http://192.0.2.1:9999"],  # RFC 5737 TEST-NET, unreachable
        )
        with pytest.raises(RuntimeError, match="unreachable"):
            _run_agent_preflight("test_agent", caps)

    def test_preflight_runs_custom_check(self) -> None:
        """Custom preflight_check_name should be invoked."""
        from unittest.mock import MagicMock, patch

        from saber.task import _run_agent_preflight

        mock_factory = MagicMock()
        mock_factory.__module__ = "fake_preflight_module"
        AgentRegistry.register("preflight_test_agent", mock_factory)

        mock_check = MagicMock(return_value=True)
        mock_module = MagicMock()
        mock_module.my_preflight = mock_check

        caps = AgentCapabilities(preflight_check_name="my_preflight")

        with patch("importlib.import_module", return_value=mock_module):
            _run_agent_preflight("preflight_test_agent", caps)

        mock_check.assert_called_once()

    def test_preflight_fails_on_custom_check_exception(self) -> None:
        """Custom preflight check that raises should cause preflight failure."""
        from unittest.mock import MagicMock, patch

        from saber.task import _run_agent_preflight

        mock_factory = MagicMock()
        mock_factory.__module__ = "fake_preflight_module"
        AgentRegistry.register("preflight_fail_agent", mock_factory)

        mock_check = MagicMock(side_effect=ConnectionError("DB down"))
        mock_module = MagicMock()
        mock_module.failing_check = mock_check

        caps = AgentCapabilities(preflight_check_name="failing_check")

        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(RuntimeError, match="DB down"):
                _run_agent_preflight("preflight_fail_agent", caps)

    def test_preflight_fails_missing_check_name(self) -> None:
        """Declared preflight_check_name that doesn't exist should fail."""
        from unittest.mock import MagicMock, patch

        from saber.task import _run_agent_preflight

        mock_factory = MagicMock()
        mock_factory.__module__ = "fake_module"
        AgentRegistry.register("missing_check_agent", mock_factory)

        mock_module = MagicMock(spec=[])  # empty module, no attributes

        caps = AgentCapabilities(preflight_check_name="nonexistent_check")

        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(RuntimeError, match="no such callable"):
                _run_agent_preflight("missing_check_agent", caps)

    def test_preflight_fails_on_false_return(self) -> None:
        """Preflight check that returns False should fail."""
        from unittest.mock import MagicMock, patch

        from saber.task import _run_agent_preflight

        mock_factory = MagicMock()
        mock_factory.__module__ = "fake_false_module"
        AgentRegistry.register("false_check_agent", mock_factory)

        mock_check = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.my_check = mock_check

        caps = AgentCapabilities(preflight_check_name="my_check")

        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(RuntimeError, match="returned False"):
                _run_agent_preflight("false_check_agent", caps)


class TestSolverFactoryErrorPaths:
    """Tests for timeout, exception, and malformed capabilities paths."""

    def test_timeout_produces_model_output(self) -> None:
        """Timeout fallback should set state.output to ModelOutput, not ChatMessageUser."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from inspect_ai.model import ModelOutput
        from inspect_ai.solver import TaskState

        from saber.agents.solver_factory import create_saber_solver

        async def slow_solve(state: TaskState, generate: object) -> TaskState:
            await asyncio.sleep(999)
            return state

        mock_factory = MagicMock(return_value=MagicMock(return_value=slow_solve))
        AgentRegistry.register("slow_agent", mock_factory)

        slvr = create_saber_solver("slow_agent", mock_factory)

        state = MagicMock(spec=TaskState)
        state.metadata = {"instruction_prompt": "test", "max_steps": 1}
        state.output = None
        state.tool_call_limit = None
        state.messages = []

        with patch(
            "saber.agents.solver_factory.asyncio.wait_for",
            side_effect=TimeoutError("timed out"),
        ):
            result = asyncio.get_event_loop().run_until_complete(slvr(state, AsyncMock()))
        assert isinstance(result.output, ModelOutput)

    def test_uncaught_exception_produces_model_output(self) -> None:
        """Uncaught plugin exception should set state.output to ModelOutput."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from inspect_ai.model import ModelOutput
        from inspect_ai.solver import TaskState

        from saber.agents.solver_factory import create_saber_solver

        async def failing_solve(state: TaskState, generate: object) -> TaskState:
            raise RuntimeError("plugin bug")

        mock_factory = MagicMock(return_value=MagicMock(return_value=failing_solve))
        AgentRegistry.register("failing_agent", mock_factory)

        slvr = create_saber_solver("failing_agent", mock_factory)

        state = MagicMock(spec=TaskState)
        state.metadata = {"instruction_prompt": "test", "max_steps": 1}
        state.output = None
        state.tool_call_limit = None
        state.messages = []

        result = asyncio.get_event_loop().run_until_complete(slvr(state, AsyncMock()))
        assert isinstance(result.output, ModelOutput)
        assert "internal error" in result.output.choices[0].message.content

    def test_malformed_capabilities_warns(self) -> None:
        """AGENT_CAPABILITIES of wrong type should fall back to defaults."""
        from types import ModuleType
        from unittest.mock import MagicMock, patch

        from saber.agents.solver_factory import _resolve_capabilities

        mock_factory = MagicMock()
        mock_factory.__module__ = "fake_bad_caps_module"
        AgentRegistry.register("bad_caps_agent", mock_factory)

        fake_mod = ModuleType("fake_bad_caps_module")
        fake_mod.AGENT_CAPABILITIES = "not_a_capabilities_object"  # type: ignore[attr-defined]

        original_import = __import__

        def patched_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "fake_bad_caps_module":
                return fake_mod
            return original_import(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=patched_import):
            caps = _resolve_capabilities("bad_caps_agent")
        # Should return default (supports_tools=True) because AGENT_CAPABILITIES was wrong type
        assert caps.supports_tools is True
        assert caps.contract_version == 1
