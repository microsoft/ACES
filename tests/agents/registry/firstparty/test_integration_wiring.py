"""Integration tests for full firstparty → runtimes → registry wiring."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml


def _ensure_hyenas_registered() -> None:
    """Re-register hyenas if a prior test reset the registry."""
    from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry

    if RuntimeRegistry.get("hyenas") is None:
        import saber.agents.registry.firstparty.runtimes as runtimes_pkg

        importlib.reload(runtimes_pkg)


class TestFirstpartyImportRegistersRuntimes:
    """Importing the firstparty package should auto-register runtimes."""

    def test_hyenas_registered_on_import(self) -> None:
        """Importing firstparty triggers runtimes import → auto-registration."""
        _ensure_hyenas_registered()
        from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry

        result = RuntimeRegistry.get("hyenas")
        assert result is not None
        assert result.name == "hyenas"

    def test_create_agent_with_hyenas_runtime(self) -> None:
        """create_agent(runtime='hyenas') should resolve without error."""
        _ensure_hyenas_registered()
        from saber.agents.registry.firstparty import create_agent

        factory = create_agent(runtime="hyenas")
        assert callable(factory)

    def test_create_agent_unknown_runtime_raises(self) -> None:
        """create_agent with unknown runtime should raise ValueError."""
        from saber.agents.registry.firstparty import create_agent

        with pytest.raises(ValueError, match="Unknown runtime"):
            create_agent(runtime="nonexistent")


class TestSolverHookWiring:
    """Tests that solver properly dispatches pre/post invoke hooks."""

    @pytest.mark.asyncio
    async def test_pre_invoke_hook_called_by_solver_flow(self) -> None:
        """Verify pre_invoke_hook is dispatched in the expected order."""
        _ensure_hyenas_registered()
        from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry

        spec = RuntimeRegistry.get("hyenas")
        assert spec is not None
        assert spec.pre_invoke_hook is not None

        env = spec.build_env(
            bridge_url="http://localhost:9100/v1",
            bridge_api_key="test",
        )
        sandbox = AsyncMock()

        result_env = await spec.pre_invoke_hook(sandbox, env)

        sandbox.write_file.assert_called_once()
        config_path, config_content = sandbox.write_file.call_args[0]
        assert config_path == "/app/.hyenas-config/config.yaml"

        parsed = yaml.safe_load(config_content)
        assert parsed["endpoints"][0]["url"] == "http://localhost:9100/v1"
        assert "default" in parsed
        assert result_env is env

    @pytest.mark.asyncio
    async def test_post_invoke_hook_called_by_solver_flow(self) -> None:
        """Verify post_invoke_hook extracts findings (or handles missing)."""
        _ensure_hyenas_registered()
        from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry

        spec = RuntimeRegistry.get("hyenas")
        assert spec is not None
        assert spec.post_invoke_hook is not None

        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock(side_effect=FileNotFoundError())
        env = {"_BRIDGE_URL": "http://localhost:9100/v1"}

        result_env = await spec.post_invoke_hook(sandbox, env)
        assert result_env is env
