"""Tests for RebuildMode and parse_rebuild_param."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saber.environments.images import RebuildMode, RebuildScope, parse_rebuild_param


class TestRebuildScope:
    """Basic enum tests for RebuildScope."""

    def test_members(self) -> None:
        assert RebuildScope.NONE is RebuildScope.NONE
        assert RebuildScope.ALL is RebuildScope.ALL
        assert RebuildScope.SPECIFIC is RebuildScope.SPECIFIC
        assert len(RebuildScope) == 3


class TestRebuildModeConstructors:
    """Tests for RebuildMode class methods and validation."""

    def test_none(self) -> None:
        mode = RebuildMode.none()
        assert mode.scope is RebuildScope.NONE
        assert mode.names == frozenset()

    def test_all(self) -> None:
        mode = RebuildMode.all()
        assert mode.scope is RebuildScope.ALL
        assert mode.names == frozenset()

    def test_specific_single(self) -> None:
        mode = RebuildMode.specific(frozenset({"sandbox"}))
        assert mode.scope is RebuildScope.SPECIFIC
        assert mode.names == frozenset({"sandbox"})

    def test_specific_multiple(self) -> None:
        mode = RebuildMode.specific(frozenset({"sandbox", "db"}))
        assert mode.scope is RebuildScope.SPECIFIC
        assert mode.names == frozenset({"sandbox", "db"})

    def test_specific_empty_names_raises(self) -> None:
        with pytest.raises(ValueError, match="SPECIFIC.*requires.*non-empty"):
            RebuildMode.specific(frozenset())

    def test_frozen(self) -> None:
        mode = RebuildMode.none()
        with pytest.raises(ValidationError):
            mode.scope = RebuildScope.ALL  # type: ignore[misc]


class TestShouldRebuild:
    """Tests for RebuildMode.should_rebuild()."""

    def test_none_never_rebuilds(self) -> None:
        mode = RebuildMode.none()
        assert mode.should_rebuild("sandbox") is False
        assert mode.should_rebuild("db") is False

    def test_all_always_rebuilds(self) -> None:
        mode = RebuildMode.all()
        assert mode.should_rebuild("sandbox") is True
        assert mode.should_rebuild("anything") is True

    def test_specific_matches(self) -> None:
        mode = RebuildMode.specific(frozenset({"sandbox", "db"}))
        assert mode.should_rebuild("sandbox") is True
        assert mode.should_rebuild("db") is True
        assert mode.should_rebuild("other") is False


class TestParseRebuildParam:
    """Tests for the parse_rebuild_param function."""

    def test_none_input(self) -> None:
        result = parse_rebuild_param(None)
        assert result.scope is RebuildScope.NONE

    def test_true_string(self) -> None:
        result = parse_rebuild_param("true")
        assert result.scope is RebuildScope.ALL

    def test_false_string(self) -> None:
        result = parse_rebuild_param("false")
        assert result.scope is RebuildScope.NONE

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_rebuild_param("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_rebuild_param("   ")

    def test_commas_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_rebuild_param(",")

    def test_multiple_commas_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_rebuild_param(",,")

    def test_single_name(self) -> None:
        result = parse_rebuild_param("sandbox")
        assert result.scope is RebuildScope.SPECIFIC
        assert result.names == frozenset({"sandbox"})

    def test_multiple_names(self) -> None:
        result = parse_rebuild_param("a,b")
        assert result.scope is RebuildScope.SPECIFIC
        assert result.names == frozenset({"a", "b"})

    def test_strips_whitespace(self) -> None:
        result = parse_rebuild_param(" sandbox , db ")
        assert result.scope is RebuildScope.SPECIFIC
        assert result.names == frozenset({"sandbox", "db"})

    def test_trailing_comma_ignored(self) -> None:
        result = parse_rebuild_param("sandbox,")
        assert result.scope is RebuildScope.SPECIFIC
        assert result.names == frozenset({"sandbox"})

    @pytest.mark.parametrize(
        "raw, expected_scope",
        [
            ("True", RebuildScope.ALL),
            ("TRUE", RebuildScope.ALL),
            ("False", RebuildScope.NONE),
            ("FALSE", RebuildScope.NONE),
        ],
    )
    def test_case_insensitive_booleans(self, raw: str, expected_scope: RebuildScope) -> None:
        result = parse_rebuild_param(raw)
        assert result.scope is expected_scope
