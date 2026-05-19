# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for bridge tracking Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saber.agents.bridge_tracking_models import (
    BridgeSessionSummary,
    CallType,
    GenerationMetadata,
)


class TestCallType:
    """CallType enum values."""

    def test_call_type_enum_values(self) -> None:
        """MAIN == 'main' and SUBAGENT == 'subagent'."""
        assert CallType.MAIN == "main"
        assert CallType.SUBAGENT == "subagent"
        assert CallType.MAIN.value == "main"
        assert CallType.SUBAGENT.value == "subagent"


class TestGenerationMetadata:
    """GenerationMetadata Pydantic model."""

    def test_generation_metadata_frozen(self) -> None:
        """Cannot mutate after creation."""
        meta = GenerationMetadata(
            call_type=CallType.MAIN,
            model_name="gpt-4",
            message_count=5,
            generation_index=0,
            tool_count=2,
        )
        with pytest.raises(ValidationError):
            meta.call_type = CallType.SUBAGENT  # type: ignore[misc]

    def test_generation_metadata_defaults(self) -> None:
        """input_tokens and output_tokens default to None."""
        meta = GenerationMetadata(
            call_type=CallType.SUBAGENT,
            model_name="claude-3",
            message_count=3,
            generation_index=1,
            tool_count=0,
        )
        assert meta.input_tokens is None
        assert meta.output_tokens is None

    def test_generation_metadata_required_fields(self) -> None:
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            GenerationMetadata()  # type: ignore[call-arg]

    def test_generation_metadata_all_fields(self) -> None:
        """All fields set correctly including optional tokens."""
        meta = GenerationMetadata(
            call_type=CallType.MAIN,
            model_name="gpt-4",
            message_count=10,
            generation_index=3,
            tool_count=5,
            input_tokens=100,
            output_tokens=200,
        )
        assert meta.call_type == CallType.MAIN
        assert meta.model_name == "gpt-4"
        assert meta.message_count == 10
        assert meta.generation_index == 3
        assert meta.tool_count == 5
        assert meta.input_tokens == 100
        assert meta.output_tokens == 200

    def test_generation_metadata_serialization(self) -> None:
        """model_dump() produces expected dict."""
        meta = GenerationMetadata(
            call_type=CallType.MAIN,
            model_name="gpt-4",
            message_count=5,
            generation_index=0,
            tool_count=2,
            input_tokens=50,
            output_tokens=100,
        )
        dumped = meta.model_dump()
        assert dumped == {
            "call_type": "main",
            "model_name": "gpt-4",
            "message_count": 5,
            "generation_index": 0,
            "tool_count": 2,
            "delegation_detected": False,
            "input_tokens": 50,
            "output_tokens": 100,
        }


class TestBridgeSessionSummary:
    """BridgeSessionSummary Pydantic model."""

    def test_bridge_session_summary_defaults(self) -> None:
        """All counters default to 0, models_used to ()."""
        summary = BridgeSessionSummary()
        assert summary.total_generations == 0
        assert summary.main_generations == 0
        assert summary.subagent_generations == 0
        assert summary.total_tool_calls == 0
        assert summary.models_used == ()

    def test_bridge_session_summary_frozen(self) -> None:
        """Cannot mutate after creation."""
        summary = BridgeSessionSummary()
        with pytest.raises(ValidationError):
            summary.total_generations = 5  # type: ignore[misc]

    def test_bridge_session_summary_custom_values(self) -> None:
        """Can construct with custom values."""
        summary = BridgeSessionSummary(
            total_generations=10,
            main_generations=7,
            subagent_generations=3,
            total_tool_calls=15,
            models_used=("gpt-4", "claude-3"),
        )
        assert summary.total_generations == 10
        assert summary.main_generations == 7
        assert summary.subagent_generations == 3
        assert summary.total_tool_calls == 15
        assert summary.models_used == ("gpt-4", "claude-3")

    def test_bridge_session_summary_serialization(self) -> None:
        """model_dump() produces expected dict."""
        summary = BridgeSessionSummary(
            total_generations=5,
            main_generations=3,
            subagent_generations=2,
            total_tool_calls=8,
            models_used=("gpt-4",),
        )
        dumped = summary.model_dump()
        assert dumped == {
            "total_generations": 5,
            "main_generations": 3,
            "subagent_generations": 2,
            "total_tool_calls": 8,
            "models_used": ("gpt-4",),
        }
