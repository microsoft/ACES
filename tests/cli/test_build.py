# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for saber build command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from saber.cli.app import app
from saber.environments.images import ImageBuildResult, PreflightResult, RebuildMode
from tests.cli.conftest import make_domain

runner = CliRunner()


def _ok_result(slug: str = "test") -> PreflightResult:
    return PreflightResult(
        domain_slug=slug,
        results=(ImageBuildResult(name="sandbox", tag=f"saber/{slug}/sandbox:latest", action="built"),),
    )


def _failed_result(slug: str = "test") -> PreflightResult:
    return PreflightResult(
        domain_slug=slug,
        results=(ImageBuildResult(name="sandbox", tag=f"saber/{slug}/sandbox:latest", action="failed", error="boom"),),
    )


class TestBuildCommand:
    """Tests for the build command."""

    def test_build_all_domains(self) -> None:
        """Build all domains when no domain specified."""
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.discover_domains", return_value=[make_domain("a"), make_domain("b")]),
            patch("saber.cli.app.build_domain_images", new_callable=AsyncMock, return_value=_ok_result()) as mock_build,
        ):
            result = runner.invoke(app, ["build"])
        assert result.exit_code == 0
        assert mock_build.call_count == 2

    def test_build_single_domain(self) -> None:
        """Build only the specified domain."""
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=make_domain("excytin")),
            patch(
                "saber.cli.app.build_domain_images",
                new_callable=AsyncMock,
                return_value=_ok_result("excytin"),
            ) as mock_build,
        ):
            result = runner.invoke(app, ["build", "excytin"])
        assert result.exit_code == 0
        assert mock_build.call_count == 1
        assert mock_build.call_args.kwargs["domain_root"] == Path("/fake/excytin")

    def test_build_with_rebuild_flag(self) -> None:
        """--rebuild passes RebuildMode.all()."""
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=make_domain("excytin")),
            patch("saber.cli.app.build_domain_images", new_callable=AsyncMock, return_value=_ok_result()) as mock_build,
        ):
            result = runner.invoke(app, ["build", "excytin", "--rebuild"])
        assert result.exit_code == 0
        mode = mock_build.call_args.kwargs["rebuild"]
        assert mode == RebuildMode.all()

    def test_build_specific_image_with_rebuild(self) -> None:
        """--image + --rebuild passes RebuildMode.specific()."""
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=make_domain("excytin")),
            patch("saber.cli.app.build_domain_images", new_callable=AsyncMock, return_value=_ok_result()) as mock_build,
        ):
            result = runner.invoke(app, ["build", "excytin", "--image", "sandbox", "--rebuild"])
        assert result.exit_code == 0
        mode = mock_build.call_args.kwargs["rebuild"]
        assert mode == RebuildMode.specific(frozenset({"sandbox"}))

    def test_build_without_rebuild_uses_none_mode(self) -> None:
        """No --rebuild passes RebuildMode.none()."""
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=make_domain("excytin")),
            patch("saber.cli.app.build_domain_images", new_callable=AsyncMock, return_value=_ok_result()) as mock_build,
        ):
            result = runner.invoke(app, ["build", "excytin"])
        assert result.exit_code == 0
        mode = mock_build.call_args.kwargs["rebuild"]
        assert mode == RebuildMode.none()

    def test_build_failure_exits_nonzero(self) -> None:
        """Exit code 1 when any build fails."""
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.resolve_domain", return_value=make_domain("excytin")),
            patch("saber.cli.app.build_domain_images", new_callable=AsyncMock, return_value=_failed_result()),
        ):
            result = runner.invoke(app, ["build", "excytin"])
        assert result.exit_code == 1

    def test_build_domains_root_not_found(self) -> None:
        """Exit code 1 when domains root is not found."""
        with patch("saber.cli.app.find_domains_root", side_effect=FileNotFoundError("No domains/")):
            result = runner.invoke(app, ["build"])
        assert result.exit_code == 1

    def test_build_no_domains_found(self) -> None:
        """Exit 1 when no domains are discovered."""
        with (
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.discover_domains", return_value=[]),
        ):
            result = runner.invoke(app, ["build"])
        assert result.exit_code == 1
