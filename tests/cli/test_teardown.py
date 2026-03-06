"""Tests for saber teardown command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from saber.cli.app import _filter_by_domain, _is_saber_project, app
from saber.cli.discovery import ComposeProject
from tests.cli.conftest import make_domain

runner = CliRunner()


class TestIsSaberProject:
    """Tests for _is_saber_project filter."""

    def test_matches_saber_in_name(self) -> None:
        assert _is_saber_project("saber-sandbox-abc123", frozenset()) is True

    def test_matches_case_insensitive(self) -> None:
        assert _is_saber_project("SABER-test", frozenset()) is True

    def test_matches_domain_databases(self) -> None:
        assert _is_saber_project("excytin-databases", frozenset({"excytin"})) is True

    def test_rejects_unrelated_project(self) -> None:
        assert _is_saber_project("my-web-app", frozenset({"excytin"})) is False

    def test_rejects_partial_slug_match(self) -> None:
        """Only exact {slug}-databases matches, not partial."""
        assert _is_saber_project("excytin-other", frozenset({"excytin"})) is False

    def test_matches_inspect_sandbox_project(self) -> None:
        """inspect-{slug}-{id} projects from Inspect AI should be recognised."""
        assert _is_saber_project("inspect-excytin-i4hzaht", frozenset({"excytin"})) is True

    def test_matches_inspect_sandbox_various_ids(self) -> None:
        """Different random IDs after inspect-{slug}- should match."""
        assert _is_saber_project("inspect-cybench-iab1234", frozenset({"cybench"})) is True

    def test_rejects_inspect_without_known_slug(self) -> None:
        """inspect-* projects for unknown domains should not match."""
        assert _is_saber_project("inspect-unknown-iab1234", frozenset({"excytin"})) is False

    def test_rejects_inspect_prefix_only(self) -> None:
        """Bare 'inspect-excytin' (no trailing id segment) should not match."""
        assert _is_saber_project("inspect-excytin", frozenset({"excytin"})) is False


class TestFilterByDomain:
    """Tests for _filter_by_domain."""

    def test_filters_to_matching_domain(self) -> None:
        projects = [
            ComposeProject(name="excytin-databases", status="running"),
            ComposeProject(name="cybench-databases", status="running"),
            ComposeProject(name="saber-excytin-sandbox", status="running"),
        ]
        result = _filter_by_domain(projects, "excytin")
        assert len(result) == 2
        names = {p.name for p in result}
        assert "excytin-databases" in names
        assert "saber-excytin-sandbox" in names

    def test_returns_empty_for_no_match(self) -> None:
        projects = [ComposeProject(name="other-project", status="running")]
        result = _filter_by_domain(projects, "excytin")
        assert result == []

    def test_filter_rejects_partial_slug_substring(self) -> None:
        """Substring of slug should NOT match."""
        projects = [ComposeProject(name="excytin-databases", status="running")]
        result = _filter_by_domain(projects, "cy")
        assert result == []


class TestTeardownCommand:
    """Tests for the teardown command."""

    def test_teardown_no_projects_found(self) -> None:
        """Exit 0 with info when no SABER projects exist."""
        with (
            patch("saber.cli.app._list_compose_projects", new_callable=AsyncMock, return_value=[]),
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.discover_domains", return_value=[]),
        ):
            result = runner.invoke(app, ["teardown", "--yes"])
        assert result.exit_code == 0

    def test_teardown_all_with_yes_flag(self) -> None:
        """Tears down all SABER projects when --yes is given."""
        projects = [
            ComposeProject(name="excytin-databases", status="running(2)"),
            ComposeProject(name="saber-sandbox-123", status="running(1)"),
            ComposeProject(name="inspect-excytin-i4hzaht", status="exited(1)"),
            ComposeProject(name="unrelated-app", status="running(1)"),
        ]
        domains = [make_domain(slug="excytin", name="Excytin")]

        with (
            patch("saber.cli.app._list_compose_projects", new_callable=AsyncMock, return_value=projects),
            patch("saber.cli.app._teardown_project", new_callable=AsyncMock, return_value=(True, "")) as mock_td,
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.discover_domains", return_value=domains),
        ):
            result = runner.invoke(app, ["teardown", "--yes"])
        assert result.exit_code == 0
        # Should tear down excytin-databases, saber-sandbox-123, and inspect sandbox
        # but NOT unrelated-app
        assert mock_td.call_count == 3
        torn_down = {call.args[0] for call in mock_td.call_args_list}
        assert "excytin-databases" in torn_down
        assert "saber-sandbox-123" in torn_down
        assert "inspect-excytin-i4hzaht" in torn_down
        assert "unrelated-app" not in torn_down

    def test_teardown_single_domain(self) -> None:
        """Tears down only the specified domain's projects."""
        projects = [
            ComposeProject(name="excytin-databases", status="running(2)"),
            ComposeProject(name="cybench-databases", status="running(1)"),
        ]
        domains = [
            make_domain(slug="excytin", name="Excytin"),
            make_domain(slug="cybench", name="Cybench"),
        ]
        with (
            patch("saber.cli.app._list_compose_projects", new_callable=AsyncMock, return_value=projects),
            patch("saber.cli.app._teardown_project", new_callable=AsyncMock, return_value=(True, "")) as mock_td,
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.discover_domains", return_value=domains),
        ):
            result = runner.invoke(app, ["teardown", "excytin", "--yes"])
        assert result.exit_code == 0
        assert mock_td.call_count == 1
        assert mock_td.call_args.args[0] == "excytin-databases"

    def test_teardown_failure_exits_nonzero(self) -> None:
        """Exit 1 when teardown of a project fails."""
        projects = [ComposeProject(name="saber-broken", status="running(1)")]
        with (
            patch("saber.cli.app._list_compose_projects", new_callable=AsyncMock, return_value=projects),
            patch("saber.cli.app._teardown_project", new_callable=AsyncMock, return_value=(False, "container error")),
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.discover_domains", return_value=[]),
        ):
            result = runner.invoke(app, ["teardown", "--yes"])
        assert result.exit_code == 1

    def test_teardown_prompts_without_yes(self) -> None:
        """Without --yes, prompts for confirmation. 'n' aborts."""
        projects = [ComposeProject(name="saber-sandbox", status="running(1)")]
        with (
            patch("saber.cli.app._list_compose_projects", new_callable=AsyncMock, return_value=projects),
            patch("saber.cli.app._teardown_project", new_callable=AsyncMock) as mock_td,
            patch("saber.cli.app.find_domains_root", return_value=Path("/fake")),
            patch("saber.cli.app.discover_domains", return_value=[]),
        ):
            result = runner.invoke(app, ["teardown"], input="n\n")
        assert mock_td.call_count == 0
        assert result.exit_code != 0
