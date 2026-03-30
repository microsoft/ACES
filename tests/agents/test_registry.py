"""Tests for AgentRegistry, auto-discovery, and resolve_agent."""

from __future__ import annotations

import pytest

from saber.agents import AgentNotFoundError, AgentRegistry
from saber.agents.resolver import resolve_agent


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # noqa: PT004
    """Save and restore registry state around each test."""
    saved = dict(AgentRegistry._agents)
    yield  # type: ignore[misc]
    AgentRegistry._agents = saved


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
