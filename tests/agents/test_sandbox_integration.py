# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Integration tests — verify agent tooling inside the saber/sandbox container.

These tests start a real ``saber/sandbox:latest`` Docker container and
validate that the claude_code and copilot agents can resolve their
dependencies exactly as the solvers expect at runtime.

Run with:
    uv run pytest tests/agents/test_sandbox_integration.py -m integration -v
"""

from __future__ import annotations

import json
import subprocess
import uuid

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTAINER_NAME = f"saber-sandbox-integration-{uuid.uuid4().hex[:8]}"
IMAGE = "saber/sandbox:latest"


def _docker_exec(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command inside the test container."""
    return subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=60,
        check=check,
    )


def _image_exists() -> bool:
    """Return True if the sandbox image is available locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def sandbox_container() -> None:  # type: ignore[misc]
    """Start a saber/sandbox container for the test module, tear down after."""
    if not _image_exists():
        pytest.skip(f"Docker image '{IMAGE}' not found — build it first")

    subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER_NAME, IMAGE],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    yield  # type: ignore[misc]
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        timeout=30,
        check=False,
    )


# ===================================================================
# Base image smoke tests
# ===================================================================


@pytest.mark.integration
class TestBaseImageTools:
    """Verify base system tooling present in the sandbox."""

    def test_python_version(self) -> None:
        r = _docker_exec("python3 --version")
        assert "Python 3.11" in r.stdout

    def test_node_version(self) -> None:
        r = _docker_exec("node --version")
        assert r.stdout.strip().startswith("v")

    def test_npm_available(self) -> None:
        r = _docker_exec("npm --version")
        assert r.returncode == 0

    def test_git_available(self) -> None:
        r = _docker_exec("git --version")
        assert "git version" in r.stdout

    def test_jq_available(self) -> None:
        r = _docker_exec("jq --version")
        assert r.returncode == 0

    def test_uv_available(self) -> None:
        r = _docker_exec("uv --version")
        assert "uv" in r.stdout

    def test_curl_available(self) -> None:
        r = _docker_exec("which curl")
        assert r.returncode == 0

    def test_workdir_is_workspace(self) -> None:
        r = _docker_exec("pwd")
        assert r.stdout.strip() == "/workspace"

    def test_pythonpath_set(self) -> None:
        r = _docker_exec("echo $PYTHONPATH")
        assert "/workspace" in r.stdout


# ===================================================================
# Claude Code CLI integration
# ===================================================================


@pytest.mark.integration
class TestClaudeCodeIntegration:
    """Verify claude CLI works as the solver expects."""

    def test_which_claude(self) -> None:
        """Solver runs ``which claude`` to resolve the binary."""
        r = _docker_exec("which claude")
        assert r.returncode == 0
        assert r.stdout.strip().endswith("/claude")

    def test_claude_version(self) -> None:
        """Verify claude binary is executable and reports a version."""
        r = _docker_exec("claude --version")
        assert r.returncode == 0
        assert "Claude Code" in r.stdout

    def test_claude_print_mode_help(self) -> None:
        """Verify --print flag is recognised (used in non-interactive mode)."""
        r = _docker_exec("claude --help 2>&1 | grep -i print", check=False)
        assert r.returncode == 0

    def test_claude_settings_dir_creatable(self) -> None:
        """Solver writes ~/.claude/settings.json — dir must be writable."""
        r = _docker_exec(
            'mkdir -p "$HOME/.claude" '
            '&& echo \'{"apiKeyHelper": "echo test"}\' > "$HOME/.claude/settings.json" '
            '&& cat "$HOME/.claude/settings.json"'
        )
        parsed = json.loads(r.stdout)
        assert "apiKeyHelper" in parsed

    def test_claude_output_format_stream_json(self) -> None:
        """Solver passes --output-format stream-json — verify flag exists."""
        r = _docker_exec("claude --help 2>&1 | grep -i 'output-format'", check=False)
        assert r.returncode == 0

    def test_claude_session_id_flag(self) -> None:
        """Solver passes --session-id — verify flag exists."""
        r = _docker_exec("claude --help 2>&1 | grep -i 'session-id'", check=False)
        assert r.returncode == 0

    def test_claude_dangerously_skip_permissions_flag(self) -> None:
        """Solver passes --dangerously-skip-permissions."""
        r = _docker_exec(
            "claude --help 2>&1 | grep -i 'dangerously-skip-permissions'",
            check=False,
        )
        assert r.returncode == 0


# ===================================================================
# Copilot SDK integration
# ===================================================================


@pytest.mark.integration
class TestCopilotIntegration:
    """Verify copilot SDK and CLI work as the solver expects."""

    def test_import_copilot(self) -> None:
        """Runner script does ``from copilot import CopilotClient``."""
        r = _docker_exec('python3 -c "from copilot import CopilotClient; print(CopilotClient)"')
        assert r.returncode == 0
        assert "CopilotClient" in r.stdout

    def test_copilot_version(self) -> None:
        """github-copilot-sdk is installed."""
        r = _docker_exec('python3 -c "import copilot; print(copilot.__version__)"')
        assert r.returncode == 0

    def test_bundled_cli_binary_exists(self) -> None:
        """SDK bundles a native copilot binary at copilot/bin/copilot."""
        r = _docker_exec(
            'python3 -c "'
            "from copilot.client import _get_bundled_cli_path; "
            "path = _get_bundled_cli_path(); "
            "print(path); "
            "assert path is not None, 'bundled CLI not found'"
            '"'
        )
        assert r.returncode == 0
        assert "copilot" in r.stdout

    def test_bundled_cli_is_executable(self) -> None:
        """Bundled binary must have execute permission."""
        r = _docker_exec(
            'python3 -c "'
            "import os; "
            "from copilot.client import _get_bundled_cli_path; "
            "path = _get_bundled_cli_path(); "
            "print(os.access(path, os.X_OK))"
            '"'
        )
        assert r.stdout.strip() == "True"

    def test_bundled_cli_version(self) -> None:
        """Bundled CLI binary runs and reports version."""
        r = _docker_exec(
            'python3 -c "'
            "import subprocess; "
            "from copilot.client import _get_bundled_cli_path; "
            "path = _get_bundled_cli_path(); "
            "result = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=10); "
            "print(result.stdout.strip())"
            '"'
        )
        assert r.returncode == 0
        assert "Copilot" in r.stdout or "copilot" in r.stdout

    def test_runner_script_compiles_in_sandbox(self) -> None:
        """RUNNER_SCRIPT from the solver compiles as valid Python in sandbox."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        # Write script to sandbox and compile it
        escaped = RUNNER_SCRIPT.replace("'", "'\\''")
        r = _docker_exec(f"echo '{escaped}' > /tmp/_test_runner.py && python3 -m py_compile /tmp/_test_runner.py")
        assert r.returncode == 0

    def test_runner_env_vars_accessible(self) -> None:
        """Runner reads required env vars — verify they pass through."""
        r = _docker_exec(
            "OPENAI_BASE_URL=http://localhost:13131/v1 "
            "COPILOT_MODEL=test-model "
            "COPILOT_PROMPT='hello' "
            'python3 -c "'
            "import os; "
            "assert os.environ['OPENAI_BASE_URL'] == 'http://localhost:13131/v1'; "
            "assert os.environ['COPILOT_MODEL'] == 'test-model'; "
            "assert os.environ['COPILOT_PROMPT'] == 'hello'; "
            "print('OK')\""
        )
        assert r.stdout.strip() == "OK"

    def test_copilot_sdk_pydantic_available(self) -> None:
        """copilot SDK depends on pydantic — must be available."""
        r = _docker_exec('python3 -c "import pydantic; print(pydantic.VERSION)"')
        assert r.returncode == 0


# ===================================================================
# Bridge infrastructure readiness
# ===================================================================


@pytest.mark.integration
class TestBridgeInfrastructure:
    """Verify the sandbox has infra needed by sandbox_agent_bridge."""

    def test_tmp_writable(self) -> None:
        """Bridge writes runner scripts to /tmp."""
        r = _docker_exec("echo 'test' > /tmp/bridge_test.txt && cat /tmp/bridge_test.txt")
        assert r.stdout.strip() == "test"

    def test_python_subprocess_works(self) -> None:
        """Copilot SDK uses subprocess.Popen — verify it works."""
        r = _docker_exec(
            'python3 -c "'
            "import subprocess; "
            "r = subprocess.run(['echo', 'hello'], capture_output=True, text=True); "
            'print(r.stdout.strip())"'
        )
        assert r.stdout.strip() == "hello"

    def test_localhost_networking(self) -> None:
        """Bridge proxy binds to localhost — verify loopback works."""
        r = _docker_exec(
            'python3 -c "'
            "import socket; "
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
            "s.bind(('127.0.0.1', 0)); "
            "port = s.getsockname()[1]; "
            "s.close(); "
            "print(f'bound port {port}')\""
        )
        assert "bound port" in r.stdout

    def test_requests_available(self) -> None:
        """Bridge proxy expects HTTP client — requests must be importable."""
        r = _docker_exec('python3 -c "import requests; print(requests.__version__)"')
        assert r.returncode == 0

    def test_bash_available(self) -> None:
        """Solvers execute via bash -c."""
        r = _docker_exec("bash --version | head -1")
        assert "bash" in r.stdout.lower()

    def test_sandbox_tools_dir_writable(self) -> None:
        """inspect_ai injects sandbox_tools to /var/tmp — must be writable."""
        r = _docker_exec(
            "mkdir -p /var/tmp/sandbox-services && "
            "echo 'ok' > /var/tmp/sandbox-services/test && "
            "cat /var/tmp/sandbox-services/test"
        )
        assert r.stdout.strip() == "ok"
