# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for RuntimeRegistry."""

from __future__ import annotations

import pytest

from saber.agents.registry.firstparty.runtime_registry import RuntimeRegistry
from saber.agents.registry.firstparty.runtime_spec import RuntimeSpec


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # noqa: PT004
    """Reset the registry before and after each test for isolation."""
    RuntimeRegistry._reset()
    yield  # type: ignore[misc]
    RuntimeRegistry._reset()


def _make_spec(name: str) -> RuntimeSpec:
    """Create a minimal RuntimeSpec for testing."""
    return RuntimeSpec(name=name, invoke_command=["echo", "hi"])


class TestRuntimeRegistry:
    def test_register_and_get(self) -> None:
        spec = _make_spec("alpha")
        RuntimeRegistry.register(spec)
        assert RuntimeRegistry.get("alpha") is spec

    def test_get_unknown_returns_none(self) -> None:
        assert RuntimeRegistry.get("nonexistent") is None

    def test_list_runtimes_sorted(self) -> None:
        RuntimeRegistry.register(_make_spec("charlie"))
        RuntimeRegistry.register(_make_spec("alpha"))
        RuntimeRegistry.register(_make_spec("bravo"))
        assert RuntimeRegistry.list_runtimes() == ["alpha", "bravo", "charlie"]

    def test_duplicate_register_raises(self) -> None:
        RuntimeRegistry.register(_make_spec("dup"))
        with pytest.raises(ValueError, match="dup"):
            RuntimeRegistry.register(_make_spec("dup"))

    def test_reset_clears_all(self) -> None:
        RuntimeRegistry.register(_make_spec("x"))
        RuntimeRegistry.register(_make_spec("y"))
        RuntimeRegistry._reset()
        assert RuntimeRegistry.list_runtimes() == []
        assert RuntimeRegistry.get("x") is None
