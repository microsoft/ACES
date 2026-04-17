"""Integration tests for firstparty plugin registration pattern."""

from __future__ import annotations

import pytest

from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry
from saber.agents.registry.firstparty.runtime_spec import EnvSchema, RuntimeSpec


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Reset the RuntimeRegistry before each test to avoid cross-contamination."""
    RuntimeRegistry._reset()


class TestFirstpartyImportClean:
    """Importing the firstparty package should work without built-in runtimes."""

    def test_import_firstparty_succeeds(self) -> None:
        """Importing firstparty should not raise."""
        import saber.agents.registry.firstparty  # noqa: F401

    def test_no_runtimes_registered_by_default(self) -> None:
        """No runtimes should be registered on a clean import."""
        import saber.agents.registry.firstparty  # noqa: F401

        assert RuntimeRegistry.list_runtimes() == []

    def test_runtime_registry_importable(self) -> None:
        """RuntimeRegistry should be importable from firstparty submodule."""
        from saber.agents.registry.firstparty.runtime_registry import (
            RuntimeRegistry as RR,
        )

        assert RR is RuntimeRegistry


class TestPluginRegistrationPattern:
    """Tests that external plugins can register and resolve RuntimeSpecs."""

    @staticmethod
    def _make_spec(name: str = "fake-runtime") -> RuntimeSpec:
        return RuntimeSpec(
            name=name,
            invoke_command=["echo", "hello"],
            env_schema=EnvSchema(
                bridge_injected={"_BRIDGE_URL": "bridge_url"},
                required={"API_KEY": "A key"},
            ),
        )

    def test_register_and_resolve(self) -> None:
        """A registered RuntimeSpec should be retrievable by name."""
        spec = self._make_spec()
        RuntimeRegistry.register(spec)

        resolved = RuntimeRegistry.get("fake-runtime")
        assert resolved is not None
        assert resolved.name == "fake-runtime"
        assert resolved.invoke_command == ["echo", "hello"]

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same runtime name twice should raise ValueError."""
        RuntimeRegistry.register(self._make_spec())
        with pytest.raises(ValueError, match="already registered"):
            RuntimeRegistry.register(self._make_spec())

    def test_list_runtimes_sorted(self) -> None:
        """list_runtimes returns registered names in sorted order."""
        RuntimeRegistry.register(self._make_spec("zeta"))
        RuntimeRegistry.register(self._make_spec("alpha"))

        assert RuntimeRegistry.list_runtimes() == ["alpha", "zeta"]

    def test_get_unknown_returns_none(self) -> None:
        """Getting an unregistered runtime returns None."""
        assert RuntimeRegistry.get("nonexistent") is None

    def test_create_agent_with_registered_runtime(self) -> None:
        """create_agent should resolve a registered runtime without error."""
        from saber.agents.registry.firstparty import create_agent

        RuntimeRegistry.register(self._make_spec("test-rt"))
        factory = create_agent(runtime="test-rt")
        assert callable(factory)

    def test_create_agent_unknown_runtime_raises(self) -> None:
        """create_agent with unknown runtime should raise ValueError."""
        from saber.agents.registry.firstparty import create_agent

        with pytest.raises(ValueError, match="Unknown runtime"):
            create_agent(runtime="nonexistent")
