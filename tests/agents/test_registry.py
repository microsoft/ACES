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
