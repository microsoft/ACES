"""Tests for saber.environments.resolve_sandbox_spec."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from saber.environments import resolve_sandbox_spec
from saber.environments.images import RebuildMode
from saber.sandbox import SaberSandboxEnvironment


@pytest.fixture(autouse=True)
def _reset_sandbox() -> None:
    """Reset SaberSandboxEnvironment class state between tests."""
    yield  # type: ignore[misc]
    SaberSandboxEnvironment._reset()


class TestResolveSandboxSpec:
    """resolve_sandbox_spec returns correct sandbox type tuple."""

    def test_returns_saber_when_permanent_compose_exists(self, tmp_path: Path) -> None:
        """When sandbox + permanent compose both exist, returns ("saber", path)."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("version: '3'\nservices: {}")

        perm = tmp_path / "compose" / "databases.compose.yml"
        perm.write_text("version: '3'\nservices: {}")

        result = resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose="compose/databases.compose.yml",
            permanent_project="test-permanent",
        )

        assert result is not None
        assert result[0] == "saber"

    def test_returns_saber_when_no_permanent_compose(self, tmp_path: Path) -> None:
        """DEC-009: Always returns ('saber', path), even without permanent."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("version: '3'\nservices: {}")

        result = resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose=None,
            permanent_project="test-permanent",
        )

        assert result is not None
        assert result[0] == "saber"

    def test_returns_saber_when_permanent_compose_none(self, tmp_path: Path) -> None:
        """DEC-009: Always returns ('saber', path) when permanent_compose is None."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("version: '3'\nservices: {}")

        result = resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose=None,
            permanent_project="saber-permanent",
        )

        assert result == ("saber", str(sandbox))

    def test_returns_saber_when_permanent_compose_missing(self, tmp_path: Path) -> None:
        """DEC-009: Returns ('saber', path) even when permanent_compose path doesn't exist."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("version: '3'\nservices: {}")

        with patch.object(SaberSandboxEnvironment, "set_permanent_compose") as mock_set:
            result = resolve_sandbox_spec(
                domain_root=tmp_path,
                sandbox_compose="compose/sandbox.compose.yml",
                permanent_compose="compose/nonexistent.yml",
                permanent_project="test-permanent",
            )

        assert result is not None
        assert result[0] == "saber"
        mock_set.assert_not_called()

    def test_returns_none_when_sandbox_missing(self, tmp_path: Path) -> None:
        """When sandbox compose doesn't exist, returns None."""
        result = resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose=None,
            permanent_project="test-permanent",
        )

        assert result is None

    def test_calls_set_permanent_compose(self, tmp_path: Path) -> None:
        """When permanent compose exists, calls SaberSandboxEnvironment.set_permanent_compose."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("version: '3'\nservices: {}")

        perm = tmp_path / "compose" / "databases.compose.yml"
        perm.write_text("version: '3'\nservices: {}")

        with patch.object(SaberSandboxEnvironment, "set_permanent_compose") as mock_set:
            resolve_sandbox_spec(
                domain_root=tmp_path,
                sandbox_compose="compose/sandbox.compose.yml",
                permanent_compose="compose/databases.compose.yml",
                permanent_project="test-proj",
            )

        mock_set.assert_called_once()

    def test_set_permanent_compose_args(self, tmp_path: Path) -> None:
        """set_permanent_compose called with correct path, project, domain_root."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("version: '3'\nservices: {}")

        perm = tmp_path / "compose" / "databases.compose.yml"
        perm.write_text("version: '3'\nservices: {}")

        with patch.object(SaberSandboxEnvironment, "set_permanent_compose") as mock_set:
            resolve_sandbox_spec(
                domain_root=tmp_path,
                sandbox_compose="compose/sandbox.compose.yml",
                permanent_compose="compose/databases.compose.yml",
                permanent_project="my-project",
            )

        mock_set.assert_called_once_with(
            perm,
            project="my-project",
            domain_root=tmp_path,
        )

    def test_does_not_call_set_permanent_compose_without_permanent(self, tmp_path: Path) -> None:
        """When no permanent compose, set_permanent_compose is NOT called."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("version: '3'\nservices: {}")

        with patch.object(SaberSandboxEnvironment, "set_permanent_compose") as mock_set:
            resolve_sandbox_spec(
                domain_root=tmp_path,
                sandbox_compose="compose/sandbox.compose.yml",
                permanent_compose=None,
                permanent_project="test-permanent",
            )

        mock_set.assert_not_called()

    def test_sandbox_path_is_absolute_string(self, tmp_path: Path) -> None:
        """The returned path is an absolute string path."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("version: '3'\nservices: {}")

        result = resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose=None,
            permanent_project="test-permanent",
        )

        assert result is not None
        sandbox_type, sandbox_path = result
        assert Path(sandbox_path).is_absolute()
        assert sandbox_path == str(sandbox)


class TestModuleExports:
    """Module-level __all__ is defined."""

    def test_all_exports(self) -> None:
        import saber.environments

        assert hasattr(saber.environments, "__all__")
        assert "resolve_sandbox_spec" in saber.environments.__all__


class TestResolveSandboxSpecRebuild:
    """resolve_sandbox_spec with rebuild parameter."""

    def test_always_returns_saber(self, tmp_path: Path) -> None:
        """DEC-009: Always returns 'saber', never 'docker'."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("services: {}")

        result = resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose=None,
            permanent_project="saber-permanent",
        )
        assert result is not None
        assert result[0] == "saber"  # Not "docker"

    def test_calls_set_preflight_config(self, tmp_path: Path) -> None:
        """resolve_sandbox_spec calls set_preflight_config."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("services: {}")

        with patch.object(SaberSandboxEnvironment, "set_preflight_config") as mock_set:
            resolve_sandbox_spec(
                domain_root=tmp_path,
                sandbox_compose="compose/sandbox.compose.yml",
                permanent_compose=None,
                permanent_project="saber-permanent",
                rebuild=RebuildMode.all(),
            )
        mock_set.assert_called_once_with(tmp_path, RebuildMode.all())

    def test_rebuild_none_passes_none(self, tmp_path: Path) -> None:
        """When rebuild not specified, passes None to set_preflight_config."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("services: {}")

        with patch.object(SaberSandboxEnvironment, "set_preflight_config") as mock_set:
            resolve_sandbox_spec(
                domain_root=tmp_path,
                sandbox_compose="compose/sandbox.compose.yml",
                permanent_compose=None,
                permanent_project="saber-permanent",
            )
        mock_set.assert_called_once_with(tmp_path, None)


class TestResolveSandboxSpecKeepPermanent:
    """resolve_sandbox_spec with keep_permanent parameter."""

    def test_keep_permanent_passed_through(self, tmp_path: Path) -> None:
        """keep_permanent flag is forwarded to SaberSandboxEnvironment."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("services: {}")

        perm = tmp_path / "compose" / "databases.compose.yml"
        perm.write_text("services: {}")

        resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose="compose/databases.compose.yml",
            permanent_project="test",
            keep_permanent=True,
        )
        assert SaberSandboxEnvironment._keep_permanent is True

    def test_keep_permanent_false_by_default(self, tmp_path: Path) -> None:
        """keep_permanent defaults to False."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("services: {}")

        resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose=None,
            permanent_project="test",
        )
        assert SaberSandboxEnvironment._keep_permanent is False

    def test_keep_permanent_without_permanent_compose(self, tmp_path: Path) -> None:
        """keep_permanent is set even without permanent_compose (harmless)."""
        sandbox = tmp_path / "compose" / "sandbox.compose.yml"
        sandbox.parent.mkdir(parents=True)
        sandbox.write_text("services: {}")

        resolve_sandbox_spec(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose=None,
            permanent_project="test",
            keep_permanent=True,
        )
        assert SaberSandboxEnvironment._keep_permanent is True
