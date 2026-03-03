"""Tests for ScoringStrategyRegistry."""

from __future__ import annotations

import pytest

from saber.scoring.registry import ScoringStrategyRegistry
from saber.scoring.strategies import NoneStrategy, StaticStrategy


class TestDefaultStrategies:
    """Default registry ships with 'static' and 'none'."""

    def test_has_static(self) -> None:
        reg = ScoringStrategyRegistry()
        assert "static" in reg.available()

    def test_has_none(self) -> None:
        reg = ScoringStrategyRegistry()
        assert "none" in reg.available()

    def test_get_static_returns_static_strategy(self) -> None:
        reg = ScoringStrategyRegistry()
        assert isinstance(reg.get("static"), StaticStrategy)

    def test_get_none_returns_none_strategy(self) -> None:
        reg = ScoringStrategyRegistry()
        assert isinstance(reg.get("none"), NoneStrategy)


class TestGetUnknown:
    """get() raises KeyError for unknown strategy names."""

    def test_raises_key_error(self) -> None:
        reg = ScoringStrategyRegistry()
        with pytest.raises(KeyError, match="Unknown scoring strategy"):
            reg.get("nonexistent")

    def test_error_message_includes_name(self) -> None:
        reg = ScoringStrategyRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            reg.get("nonexistent")

    def test_error_message_includes_available(self) -> None:
        reg = ScoringStrategyRegistry()
        with pytest.raises(KeyError, match="none"):
            reg.get("nonexistent")


class TestRegister:
    """register() adds a custom strategy."""

    def test_register_custom(self) -> None:
        reg = ScoringStrategyRegistry()
        custom = NoneStrategy()
        reg.register("custom", custom)
        assert reg.get("custom") is custom

    def test_register_duplicate_raises(self) -> None:
        reg = ScoringStrategyRegistry()
        with pytest.raises(ValueError, match="already registered"):
            reg.register("static", StaticStrategy())

    def test_register_force_overwrite(self) -> None:
        reg = ScoringStrategyRegistry()
        custom = NoneStrategy()
        reg.register("static", custom, force=True)
        assert reg.get("static") is custom


class TestAvailable:
    """available() returns sorted list of registered names."""

    def test_default_sorted(self) -> None:
        reg = ScoringStrategyRegistry()
        assert reg.available() == [
            "llm_judge",
            "none",
            "static",
            "static_jaccard",
            "tool_call",
            "tool_call_count",
        ]

    def test_includes_custom(self) -> None:
        reg = ScoringStrategyRegistry()
        reg.register("zebra", NoneStrategy())
        reg.register("alpha", NoneStrategy())
        names = reg.available()
        assert "alpha" in names
        assert "zebra" in names
        assert "none" in names
        assert "static" in names
        assert names == sorted(names)
