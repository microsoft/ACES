"""Tests for JudgeMessages model with Union[str, List[str]] support."""

import pytest
from saber.models.rest.evaluation import JudgeMessages


def test_judge_messages_single_string():
    """Test JudgeMessages accepts single string for user_message (backwards compatibility)."""
    judge_messages = JudgeMessages(
        system_message="You are a judge.",
        user_message="Evaluate this submission.",
        model="gpt-4"
    )

    assert judge_messages.system_message == "You are a judge."
    assert judge_messages.user_message == "Evaluate this submission."
    assert judge_messages.model == "gpt-4"
    assert isinstance(judge_messages.user_message, str)


def test_judge_messages_list_of_strings():
    """Test JudgeMessages accepts list of strings for user_message (chunked mode)."""
    judge_messages = JudgeMessages(
        system_message="You are a judge.",
        user_message=[
            "Evaluate chunk 1.",
            "Evaluate chunk 2.",
            "Evaluate chunk 3."
        ],
        model="gpt-4"
    )

    assert judge_messages.system_message == "You are a judge."
    assert isinstance(judge_messages.user_message, list)
    assert len(judge_messages.user_message) == 3
    assert judge_messages.user_message[0] == "Evaluate chunk 1."
    assert judge_messages.model == "gpt-4"


def test_judge_messages_serialization_single():
    """Test serialization works correctly with single string."""
    judge_messages = JudgeMessages(
        system_message="System prompt",
        user_message="User prompt",
        model="gpt-4"
    )

    data = judge_messages.model_dump()
    assert data["system_message"] == "System prompt"
    assert data["user_message"] == "User prompt"
    assert data["model"] == "gpt-4"


def test_judge_messages_serialization_list():
    """Test serialization works correctly with list of strings."""
    judge_messages = JudgeMessages(
        system_message="System prompt",
        user_message=["User prompt 1", "User prompt 2"],
        model="gpt-4"
    )

    data = judge_messages.model_dump()
    assert data["system_message"] == "System prompt"
    assert isinstance(data["user_message"], list)
    assert data["user_message"] == ["User prompt 1", "User prompt 2"]
    assert data["model"] == "gpt-4"


def test_judge_messages_deserialization_single():
    """Test deserialization works correctly with single string."""
    data = {
        "system_message": "System prompt",
        "user_message": "User prompt",
        "model": "gpt-4"
    }

    judge_messages = JudgeMessages(**data)
    assert judge_messages.user_message == "User prompt"
    assert isinstance(judge_messages.user_message, str)


def test_judge_messages_deserialization_list():
    """Test deserialization works correctly with list of strings."""
    data = {
        "system_message": "System prompt",
        "user_message": ["User prompt 1", "User prompt 2"],
        "model": "gpt-4"
    }

    judge_messages = JudgeMessages(**data)
    assert isinstance(judge_messages.user_message, list)
    assert judge_messages.user_message == ["User prompt 1", "User prompt 2"]


def test_judge_messages_json_round_trip_single():
    """Test JSON serialization and deserialization round trip with single string."""
    original = JudgeMessages(
        system_message="System",
        user_message="User",
        model="gpt-4"
    )

    json_str = original.model_dump_json()
    restored = JudgeMessages.model_validate_json(json_str)

    assert restored.system_message == original.system_message
    assert restored.user_message == original.user_message
    assert restored.model == original.model


def test_judge_messages_json_round_trip_list():
    """Test JSON serialization and deserialization round trip with list of strings."""
    original = JudgeMessages(
        system_message="System",
        user_message=["User 1", "User 2", "User 3"],
        model="gpt-4"
    )

    json_str = original.model_dump_json()
    restored = JudgeMessages.model_validate_json(json_str)

    assert restored.system_message == original.system_message
    assert isinstance(restored.user_message, list)
    assert restored.user_message == original.user_message
    assert restored.model == original.model


def test_judge_messages_empty_list():
    """Test that empty list is valid (though not recommended)."""
    judge_messages = JudgeMessages(
        system_message="System",
        user_message=[],
        model="gpt-4"
    )

    assert isinstance(judge_messages.user_message, list)
    assert len(judge_messages.user_message) == 0


def test_judge_messages_single_item_list():
    """Test single-item list (should be normalized to string by caller, but valid)."""
    judge_messages = JudgeMessages(
        system_message="System",
        user_message=["Single user prompt"],
        model="gpt-4"
    )

    assert isinstance(judge_messages.user_message, list)
    assert len(judge_messages.user_message) == 1
    assert judge_messages.user_message[0] == "Single user prompt"
