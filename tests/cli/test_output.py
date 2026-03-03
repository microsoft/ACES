"""Tests for saber.cli.output — Rich formatting helpers."""

from __future__ import annotations

from pathlib import Path

from rich.table import Table

from saber.cli.discovery import DiscoveredDomain
from saber.cli.output import build_results_table, domain_table
from saber.environments.images import ImageBuildResult, PreflightResult


class TestDomainTable:
    """Tests for domain_table()."""

    def test_renders_correct_columns(self) -> None:
        """Table has 4 columns."""
        domains = [
            DiscoveredDomain(
                slug="test",
                name="Test Domain",
                root=Path("/fake/test"),
            ),
        ]
        table = domain_table(domains)
        assert isinstance(table, Table)
        assert len(table.columns) == 4

    def test_renders_all_rows(self) -> None:
        """One row per domain."""
        domains = [
            DiscoveredDomain(slug="a", name="Alpha", root=Path("/a")),
            DiscoveredDomain(slug="b", name="Beta", root=Path("/b")),
        ]
        table = domain_table(domains)
        assert table.row_count == 2


class TestBuildResultsTable:
    """Tests for build_results_table()."""

    def test_renders_mixed_results(self) -> None:
        """Table renders built, skipped, and failed rows."""
        result = PreflightResult(
            domain_slug="test",
            results=(
                ImageBuildResult(name="base", tag="saber/sandbox:latest", action="built"),
                ImageBuildResult(name="db", tag="saber/test/db:latest", action="skipped"),
                ImageBuildResult(name="app", tag="saber/test/app:latest", action="failed", error="boom"),
            ),
        )
        table = build_results_table(result)
        assert isinstance(table, Table)
        assert table.row_count == 3
        assert len(table.columns) == 3
