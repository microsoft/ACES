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
        assert caps.execution_mode == "inspect_managed"
        assert caps.supports_tools is True

    def test_custom_values(self) -> None:
        """Can override capability flags."""
        caps = AgentCapabilities(execution_mode="self_managed", supports_tools=False)
        assert caps.execution_mode == "self_managed"
        assert caps.supports_tools is False

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute assignment."""
        caps = AgentCapabilities()
        with pytest.raises(ValidationError):
            caps.supports_tools = False  # type: ignore[misc]

    def test_field_access(self) -> None:
        """Fields are accessible by name."""
        caps = AgentCapabilities(execution_mode="self_managed", supports_tools=False)
        assert caps.model_fields_set == {"execution_mode", "supports_tools"}

    def test_model_dump(self) -> None:
        """model_dump returns dict with all fields."""
        caps = AgentCapabilities()
        dumped = caps.model_dump()
        assert dumped == {
            "contract_version": 1,
            "execution_mode": "inspect_managed",
            "supports_tools": True,
            "supports_limit_callback": False,
            "required_services": (),
            "preflight_check_name": None,
        }
