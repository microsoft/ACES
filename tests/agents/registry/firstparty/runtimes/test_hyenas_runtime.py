"""Tests for HYENAS_RUNTIME spec, hooks, and registry integration."""

from __future__ import annotations

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


class TestHyenasRuntimeSpec:
    """Tests for the HYENAS_RUNTIME constant."""

    def test_name(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        assert HYENAS_RUNTIME.name == "hyenas"

    def test_sandbox_compose_path(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        assert HYENAS_RUNTIME.sandbox_compose is not None
        assert "hyenas" in HYENAS_RUNTIME.sandbox_compose

    def test_invoke_command_has_required_flags(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        cmd = HYENAS_RUNTIME.invoke_command
        assert cmd[0] == "bash"
        assert cmd[1] == "-c"
        bash_script = cmd[2]
        assert "--headless" in bash_script
        assert "--no-copilot" in bash_script
        assert "{prove_flag}" in bash_script
        assert "--no-prove" not in bash_script
        assert "--repo" in bash_script
        assert "--config" in bash_script

    def test_agents_empty(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        assert HYENAS_RUNTIME.agents == []

    def test_env_defaults(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        assert HYENAS_RUNTIME.env_schema.defaults.get("HYENAS_HEADLESS") == "true"

    def test_prove_flag_default(self) -> None:
        """prove_flag defaults to --no-prove (prove disabled by default)."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        assert HYENAS_RUNTIME.env_schema.defaults.get("prove_flag") == "--no-prove"

    def test_bridge_injected_url(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        assert "_BRIDGE_URL" in HYENAS_RUNTIME.env_schema.bridge_injected
        # _BRIDGE_PORT should NOT be in bridge_injected (parsed from URL instead)
        assert "_BRIDGE_PORT" not in HYENAS_RUNTIME.env_schema.bridge_injected

    def test_default_model_aliases_map_real_models(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        aliases = HYENAS_RUNTIME.default_model_aliases
        assert aliases["gpt-5.4"] == "copilot/gpt-5.4"
        assert aliases["claude-opus-4.6"] == "copilot/claude-opus-4.6"
        assert aliases["claude-sonnet-4.6"] == "copilot/claude-sonnet-4.6"
        assert aliases["gpt-4.1"] == "copilot/gpt-4.1"

    def test_hooks_are_set(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        assert HYENAS_RUNTIME.pre_invoke_hook is not None
        assert HYENAS_RUNTIME.post_invoke_hook is not None

    def test_timeout(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        assert HYENAS_RUNTIME.timeout == 86400

    def test_no_dead_env_vars(self) -> None:
        """HYENAS_STAGES and HYENAS_NO_COPILOT should NOT exist."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        defaults = HYENAS_RUNTIME.env_schema.defaults
        assert "HYENAS_STAGES" not in defaults
        assert "HYENAS_NO_COPILOT" not in defaults


class TestHyenasPreInvokeHook:
    """Tests for the pre-invoke hook."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_writes_config_yaml(self) -> None:
        """Pre-invoke writes config.yaml to sandbox."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        env = {"_BRIDGE_URL": "http://localhost:9100/v1"}

        await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        # Should have called write_file for config.yaml and token file
        assert sandbox.write_file.call_count == 2
        config_call = sandbox.write_file.call_args_list[0]
        assert config_call[0][0] == "/app/.hyenas-config/config.yaml"
        # Config should be valid YAML
        yaml_content = config_call[0][1]
        parsed = yaml.safe_load(yaml_content)
        assert "endpoints" in parsed
        assert "default" in parsed
        # Port should be extracted from URL
        assert parsed["endpoints"][0]["url"] == "http://localhost:9100/v1"
        # No stage overrides in new config format
        for key in ("prepare-stage", "scan-stage", "validate-stage", "prove-stage"):
            assert key not in parsed

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_returns_env_unchanged(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        env = {"_BRIDGE_URL": "http://localhost:9100/v1", "other_key": "value"}

        result = await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        assert result == env

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_injects_repo_when_tarball_in_env(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        sandbox.exec = AsyncMock(return_value=MagicMock(returncode=0, stderr=""))
        env = {
            "_BRIDGE_URL": "http://localhost:9100/v1",
            "repo_tarball": base64.b64encode(b"fake").decode(),
        }

        await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        # write_file called 3 times: config.yaml + tarball + token file
        assert sandbox.write_file.call_count == 3

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_injects_repo_when_path_in_env(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        sandbox.exec = AsyncMock(return_value=MagicMock(returncode=0, stderr=""))
        env = {
            "_BRIDGE_URL": "http://localhost:9100/v1",
            "repo_path": "/some/host/path.tar.gz",
        }

        with patch("pathlib.Path.read_bytes", return_value=b"fake"):
            await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        # write_file called 3 times: config.yaml + tarball + token file
        assert sandbox.write_file.call_count == 3

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_skips_repo_injection_without_metadata(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        env = {"_BRIDGE_URL": "http://localhost:9100/v1"}

        await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        # config.yaml + token file written, no exec for tar (only install for perms)
        assert sandbox.write_file.call_count == 2
        sandbox.exec.assert_called_once_with(
            ["install", "-m", "600", "/dev/null", "/tmp/.copilot_token"]
        )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_uses_default_bridge_port(self) -> None:
        """Falls back to port 3000 when _BRIDGE_URL is missing from env."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        env: dict[str, str] = {}

        await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        yaml_content = sandbox.write_file.call_args_list[0][0][1]
        parsed = yaml.safe_load(yaml_content)
        endpoint_url = parsed["endpoints"][0]["url"]
        assert "3000" in endpoint_url

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_pre_invoke_after_build_env(self) -> None:
        """Integration test: build_env → pre_invoke_hook with real env dict."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        env = HYENAS_RUNTIME.build_env(
            bridge_url="http://localhost:9100/v1",
            bridge_api_key="test-key",
        )
        sandbox = AsyncMock()

        result = await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        assert result is env
        yaml_content = sandbox.write_file.call_args_list[0][0][1]
        parsed = yaml.safe_load(yaml_content)
        assert parsed["endpoints"][0]["url"] == "http://localhost:9100/v1"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_pre_invoke_passes_extra_config(self) -> None:
        """_HYENAS_EXTRA_CONFIG JSON string is deep-merged into config.yaml."""
        import json

        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        extra = {"prove-stage": {"models": ["gpt-5.4"]}}
        env = {
            "_BRIDGE_URL": "http://localhost:9100/v1",
            "_HYENAS_EXTRA_CONFIG": json.dumps(extra),
        }

        await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        yaml_content = sandbox.write_file.call_args_list[0][0][1]
        parsed = yaml.safe_load(yaml_content)
        assert parsed["prove-stage"] == {"models": ["gpt-5.4"]}

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_pre_invoke_no_extra_config_key(self) -> None:
        """Without _HYENAS_EXTRA_CONFIG, config has no stage overrides."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        env = {"_BRIDGE_URL": "http://localhost:9100/v1"}

        await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        yaml_content = sandbox.write_file.call_args_list[0][0][1]
        parsed = yaml.safe_load(yaml_content)
        assert "prove-stage" not in parsed

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_pre_invoke_rejects_malformed_json(self) -> None:
        """Malformed JSON in _HYENAS_EXTRA_CONFIG raises ValueError."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        env = {
            "_BRIDGE_URL": "http://localhost:9100/v1",
            "_HYENAS_EXTRA_CONFIG": "{bad json",
        }

        with pytest.raises(ValueError, match="Invalid JSON in _HYENAS_EXTRA_CONFIG"):
            await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    async def test_pre_invoke_rejects_non_dict_json(self) -> None:
        """Non-dict JSON in _HYENAS_EXTRA_CONFIG raises TypeError."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        env = {
            "_BRIDGE_URL": "http://localhost:9100/v1",
            "_HYENAS_EXTRA_CONFIG": "[1, 2, 3]",
        }

        with pytest.raises(TypeError, match="must be a JSON object"):
            await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)


class TestHyenasTokenInjection:
    """Tests for secure GITHUB_TOKEN injection."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "gho_secret123"})
    async def test_writes_token_file(self) -> None:
        """Pre-invoke writes GITHUB_TOKEN to /tmp/.copilot_token."""
        from saber.agents.registry.firstparty.runtimes.hyenas import (
            _TOKEN_PATH,
            HYENAS_RUNTIME,
        )

        sandbox = AsyncMock()
        env = {"_BRIDGE_URL": "http://localhost:9100/v1"}

        await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        # Find the write_file call for the token
        token_calls = [
            c for c in sandbox.write_file.call_args_list if c[0][0] == _TOKEN_PATH
        ]
        assert len(token_calls) == 1
        assert token_calls[0][0][1] == "gho_secret123"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "gho_secret123"})
    async def test_install_600_before_write(self) -> None:
        """Pre-invoke pre-creates token file with 600 perms via install."""
        from saber.agents.registry.firstparty.runtimes.hyenas import (
            _TOKEN_PATH,
            HYENAS_RUNTIME,
        )

        sandbox = AsyncMock()
        env = {"_BRIDGE_URL": "http://localhost:9100/v1"}

        await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

        sandbox.exec.assert_any_call(["install", "-m", "600", "/dev/null", _TOKEN_PATH])

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GITHUB_TOKEN": "gho_secret123"})
    async def test_raises_on_install_failure(self) -> None:
        """Pre-invoke raises RuntimeError when install for token file fails."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        # Make the install call fail
        failed_result = MagicMock()
        failed_result.success = False
        failed_result.stderr = "install: permission denied"
        sandbox.exec = AsyncMock(return_value=failed_result)
        env = {"_BRIDGE_URL": "http://localhost:9100/v1"}

        with pytest.raises(RuntimeError, match="Failed to create token file"):
            await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_raises_without_github_token(self) -> None:
        """Pre-invoke raises ValueError when GITHUB_TOKEN not set."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        env = {"_BRIDGE_URL": "http://localhost:9100/v1"}

        with pytest.raises(ValueError, match="GITHUB_TOKEN must be set"):
            await HYENAS_RUNTIME.pre_invoke_hook(sandbox, env)

    def test_invoke_command_reads_token_from_file(self) -> None:
        """invoke_command reads token from file and removes it."""
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        cmd = HYENAS_RUNTIME.invoke_command
        bash_script = cmd[2]
        assert "cat /tmp/.copilot_token" in bash_script
        assert "rm -f /tmp/.copilot_token" in bash_script

    def test_token_path_constant(self) -> None:
        """_TOKEN_PATH is a module-level constant."""
        from saber.agents.registry.firstparty.runtimes.hyenas import _TOKEN_PATH

        assert _TOKEN_PATH == "/tmp/.copilot_token"


class TestHyenasPostInvokeHook:
    """Tests for the post-invoke hook."""

    @pytest.mark.asyncio
    async def test_extracts_findings(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock(
            return_value='{"file_path": "test.py"}\n'
        )
        env: dict[str, str] = {}

        result = await HYENAS_RUNTIME.post_invoke_hook(sandbox, env)

        assert result == env
        sandbox.read_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_env_on_no_findings(self) -> None:
        from saber.agents.registry.firstparty.runtimes.hyenas import HYENAS_RUNTIME

        sandbox = AsyncMock()
        sandbox.read_file = AsyncMock(side_effect=FileNotFoundError())
        env = {"key": "val"}

        result = await HYENAS_RUNTIME.post_invoke_hook(sandbox, env)

        assert result == env


class TestHyenasRuntimeRegistration:
    """Tests for registry integration."""

    def test_registry_resolves_hyenas(self) -> None:
        """After importing runtimes, 'hyenas' is registered."""
        from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry

        # Reset to avoid "already registered" from prior tests
        RuntimeRegistry._reset()

        # Import triggers registration
        import importlib

        import saber.agents.registry.firstparty.runtimes as runtimes_pkg

        importlib.reload(runtimes_pkg)

        result = RuntimeRegistry.get("hyenas")
        assert result is not None
        assert result.name == "hyenas"

    def test_double_import_does_not_raise(self) -> None:
        """Importing runtimes twice should not raise."""
        from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry

        RuntimeRegistry._reset()

        import importlib

        import saber.agents.registry.firstparty.runtimes as runtimes_pkg

        importlib.reload(runtimes_pkg)
        # Second reload exercises the double-import guard
        importlib.reload(runtimes_pkg)
        result = RuntimeRegistry.get("hyenas")
        assert result is not None
