"""Tests for the `saber new-eval-workspace` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from saber.cli.app import app
from saber.cli.workspace import SaberDependency, _resolve_current_saber_dependency

runner = CliRunner()


class _FakeDistribution:
    """Minimal distribution stub for workspace dependency resolution tests."""

    def __init__(self, version: str, direct_url_json: str | None = None) -> None:
        self.version = version
        self._direct_url_json = direct_url_json

    def read_text(self, name: str) -> str | None:
        if name == "direct_url.json":
            return self._direct_url_json
        return None


class TestResolveCurrentSaberDependency:
    """Tests for dependency pin resolution used by workspace scaffolding."""

    def test_prefers_local_editable_source_and_copies_inspect_ai_override(self, tmp_path: Path) -> None:
        """Editable local installs become uv source pins in the new workspace."""
        project_root = tmp_path / "saber-source"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text(
            "[project]\n"
            'name = "saber"\n'
            'version = "9.9.9"\n'
            "\n"
            "[tool.uv.sources]\n"
            '"inspect-ai" = { git = "https://example.com/inspect-ai.git", branch = "dev" }\n',
            encoding="utf-8",
        )
        direct_url_json = f'{{"url": "{project_root.resolve().as_uri()}", "dir_info": {{"editable": true}}}}'

        with patch("saber.cli.workspace.distribution", return_value=_FakeDistribution("9.9.9", direct_url_json)):
            resolved = _resolve_current_saber_dependency()

        assert resolved.dependency == "saber"
        assert '"saber" = { path = ' in resolved.uv_sources[0]
        assert "editable = true" in resolved.uv_sources[0]
        assert '"inspect-ai" = { git = "https://example.com/inspect-ai.git", branch = "dev" }' in resolved.uv_sources

    def test_falls_back_to_version_pin_when_no_direct_url_is_available(self) -> None:
        """Installed version is used when there is no direct URL metadata."""
        with patch("saber.cli.workspace.distribution", return_value=_FakeDistribution("1.2.3")):
            resolved = _resolve_current_saber_dependency()

        assert resolved == SaberDependency(dependency="saber==1.2.3")


class TestNewEvalWorkspaceCommand:
    """Tests for the CLI subcommand."""

    def test_creates_workspace_with_expected_files(self, tmp_path: Path) -> None:
        """The command writes a minimal runnable workspace scaffold."""
        target = tmp_path / "demo-workspace"
        dependency = SaberDependency(
            dependency="saber",
            uv_sources=(
                '"saber" = { path = "/opt/saber", editable = true }',
                '"inspect-ai" = { git = "https://example.com/inspect-ai.git", branch = "dev" }',
            ),
        )

        with patch("saber.cli.workspace._resolve_current_saber_dependency", return_value=dependency):
            result = runner.invoke(app, ["new-eval-workspace", str(target)])

        assert result.exit_code == 0
        assert target.is_dir()

        pyproject = (target / "pyproject.toml").read_text(encoding="utf-8")
        assert 'name = "demo-workspace"' in pyproject
        assert 'description = "Vanilla SABER benchmark/evaluation workspace"' in pyproject
        assert '    "saber",' in pyproject
        assert "[tool.uv.sources]" in pyproject
        assert '"saber" = { path = "/opt/saber", editable = true }' in pyproject
        assert '"inspect-ai" = { git = "https://example.com/inspect-ai.git", branch = "dev" }' in pyproject

        readme = (target / "README.md").read_text(encoding="utf-8")
        assert "Create a new domain" in readme
        assert "Add a new task" in readme
        assert "uv run inspect eval domains/starter_demo" in readme

        domain_root = target / "domains" / "starter_demo"
        assert domain_root.is_dir()
        assert (domain_root / "starter_demo.py").is_file()
        assert (domain_root / "prompts" / "instructions" / "starter_demo.j2").is_file()
        assert (domain_root / "prompts" / "assistants" / "starter_assistant.j2").is_file()

        eval_yaml = (domain_root / "eval.yaml").read_text(encoding="utf-8")
        assert "slug: starter_demo" in eval_yaml

        task_yaml = (domain_root / "tasks" / "starter_task.yaml").read_text(encoding="utf-8")
        assert "task_id: starter_demo_task_1" in task_yaml
        assert "aces-demo-host" in task_yaml

    def test_no_demo_domain_flag_skips_starter_scaffold(self, tmp_path: Path) -> None:
        """`--no-demo-domain` leaves the workspace bare but documented."""
        target = tmp_path / "bare-workspace"

        with patch(
            "saber.cli.workspace._resolve_current_saber_dependency",
            return_value=SaberDependency(dependency="saber==2.4.6"),
        ):
            result = runner.invoke(app, ["new-eval-workspace", str(target), "--no-demo-domain"])

        assert result.exit_code == 0
        assert "without the starter demo domain" in result.output
        assert (target / "domains").is_dir()
        assert not (target / "domains" / "starter_demo").exists()

        readme = (target / "README.md").read_text(encoding="utf-8")
        assert "No starter demo domain was generated" in readme
        assert "uv run inspect list tasks | grep <your-domain-slug>" in readme
        assert "domains/my_domain/" in readme

    def test_existing_destination_fails_with_informative_message(self, tmp_path: Path) -> None:
        """Existing directories are rejected instead of being overwritten."""
        target = tmp_path / "existing-workspace"
        target.mkdir()

        result = runner.invoke(app, ["new-eval-workspace", str(target)])

        assert result.exit_code == 1
        assert "already exists" in result.output.lower()

    def test_version_pin_workspace_omits_uv_sources(self, tmp_path: Path) -> None:
        """Version-pinned installs do not emit a `[tool.uv.sources]` block."""
        target = tmp_path / "pinned-workspace"

        with patch(
            "saber.cli.workspace._resolve_current_saber_dependency",
            return_value=SaberDependency(dependency="saber==2.4.6"),
        ):
            result = runner.invoke(app, ["new-eval-workspace", str(target)])

        assert result.exit_code == 0
        pyproject = (target / "pyproject.toml").read_text(encoding="utf-8")
        assert '    "saber==2.4.6",' in pyproject
        assert "[tool.uv.sources]" not in pyproject
