"""Tests for agent model definitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saber.agents.models import AgentCapabilities


class TestAgentCapabilities:
    """AgentCapabilities Pydantic model."""

    def test_default_values(self) -> None:
        """Default capabilities: supports_tools=True."""
        caps = AgentCapabilities()
        assert caps.supports_tools is True

    def test_custom_values(self) -> None:
        """Can override capability flags."""
        caps = AgentCapabilities(supports_tools=False)
        assert caps.supports_tools is False

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute assignment."""
        caps = AgentCapabilities()
        with pytest.raises(ValidationError):
            caps.supports_tools = False  # type: ignore[misc]

    def test_field_access(self) -> None:
        """Fields are accessible by name."""
        caps = AgentCapabilities(supports_tools=False)
        assert caps.model_fields_set == {"supports_tools"}

    def test_model_dump(self) -> None:
        """model_dump returns dict with all fields."""
        caps = AgentCapabilities()
        dumped = caps.model_dump()
        assert dumped == {
            "supports_tools": True,
        }
