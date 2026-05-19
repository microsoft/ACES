# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for saber start command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from saber.cli.app import app
from tests.cli.conftest import make_domain, make_domain_with_permanent

runner = CliRunner()


class TestStartCommand:
    """Tests for the start command."""

    def test_start_calls_start_permanent(self) -> None:
        """Start command calls _start_permanent_services with correct args."""
        domain = make_domain_with_permanent()
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=domain),
            patch("saber.cli.app._start_permanent_services", new_callable=AsyncMock) as mock_start,
        ):
            result = runner.invoke(app, ["start", "excytin"])
        assert result.exit_code == 0
        mock_start.assert_called_once_with(
            domain.permanent_compose,
            domain.permanent_project,
            project_directory=domain.root,
        )

    def test_start_no_permanent_errors(self) -> None:
        """Exit 1 when domain has no permanent environment."""
        domain = make_domain(slug="plain", name="Plain")
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=domain),
        ):
            result = runner.invoke(app, ["start", "plain"])
        assert result.exit_code == 1
        assert "no permanent environment" in result.output.lower() or "no permanent" in (result.stdout or "").lower()

    def test_start_with_task_loads_config(self, tmp_path: Path) -> None:
        """Start with --task loads the task and shows info."""
        # Create the sandbox compose file on disk so Path.exists() works naturally
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        (compose_dir / "sandbox.compose.yml").write_text("services: {}")

        domain = make_domain_with_permanent(slug="excytin")
        # Override root to use tmp_path so compose file lookup succeeds
        domain = make_domain(
            slug="excytin",
            name="Excytin",
            root=tmp_path,
            permanent_compose=tmp_path / "compose" / "all_incidents.compose.yml",
            permanent_project="excytin-databases",
        )
        mock_task = MagicMock()
        mock_task.task_id = "incident_5_task_1"
        mock_task.title = "Incident 5 Task 1"
        mock_task.description = "Test task"

        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=domain),
            patch("saber.cli.app._start_permanent_services", new_callable=AsyncMock),
            patch("saber.cli.app.ConfigLoader") as mock_loader_cls,
        ):
            mock_loader_cls.return_value.load_tasks.return_value = [mock_task]
            result = runner.invoke(app, ["start", "excytin", "--task", "incident_5_task_1"])
        assert result.exit_code == 0

    def test_start_with_invalid_task_errors(self) -> None:
        """Exit 1 when task not found."""
        domain = make_domain_with_permanent()
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=domain),
            patch("saber.cli.app._start_permanent_services", new_callable=AsyncMock),
            patch("saber.cli.app.ConfigLoader") as mock_loader_cls,
        ):
            mock_loader_cls.return_value.load_tasks.return_value = []
            result = runner.invoke(app, ["start", "excytin", "--task", "nonexistent"])
        assert result.exit_code == 1

    def test_start_permanent_failure_exits_nonzero(self) -> None:
        """Exit 1 when permanent services fail to start."""
        domain = make_domain_with_permanent()
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=domain),
            patch(
                "saber.cli.app._start_permanent_services",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Failed"),
            ),
        ):
            result = runner.invoke(app, ["start", "excytin"])
        assert result.exit_code == 1

    def test_start_domains_root_not_found(self) -> None:
        """Exit 1 when domains root is not found."""
        with patch("saber.cli.app.find_domains_root", side_effect=FileNotFoundError("No domains/")):
            result = runner.invoke(app, ["start", "excytin"])
        assert result.exit_code == 1

    def test_start_task_compose_missing(self, tmp_path: Path) -> None:
        """Exit 1 when sandbox compose file doesn't exist."""
        # No compose/sandbox.compose.yml created — so it won't exist
        domain = make_domain(
            slug="excytin",
            name="Excytin",
            root=tmp_path,
            permanent_compose=tmp_path / "compose" / "all_incidents.compose.yml",
            permanent_project="excytin-databases",
        )
        mock_task = MagicMock()
        mock_task.task_id = "incident_5_task_1"
        mock_task.title = "Incident 5 Task 1"
        mock_task.description = "Test task"

        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=domain),
            patch("saber.cli.app._start_permanent_services", new_callable=AsyncMock),
            patch("saber.cli.app.ConfigLoader") as mock_loader_cls,
        ):
            mock_loader_cls.return_value.load_tasks.return_value = [mock_task]
            result = runner.invoke(app, ["start", "excytin", "--task", "incident_5_task_1"])
        assert result.exit_code == 1

    def test_start_sandbox_failure_exits_nonzero(self, tmp_path: Path) -> None:
        """Exit 1 when sandbox startup fails."""
        # Create the sandbox compose file so it passes the exists() check
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        (compose_dir / "sandbox.compose.yml").write_text("services: {}")

        domain = make_domain(
            slug="excytin",
            name="Excytin",
            root=tmp_path,
            permanent_compose=tmp_path / "compose" / "all_incidents.compose.yml",
            permanent_project="excytin-databases",
        )
        mock_task = MagicMock()
        mock_task.task_id = "incident_5_task_1"
        mock_task.title = "Incident 5 Task 1"
        mock_task.description = "Test task"

        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=domain),
            patch(
                "saber.cli.app._start_permanent_services",
                new_callable=AsyncMock,
                side_effect=[None, RuntimeError("container crashed")],
            ),
            patch("saber.cli.app.ConfigLoader") as mock_loader_cls,
        ):
            mock_loader_cls.return_value.load_tasks.return_value = [mock_task]
            result = runner.invoke(app, ["start", "excytin", "--task", "incident_5_task_1"])
        assert result.exit_code == 1
