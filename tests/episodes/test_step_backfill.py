"""Unit tests for StepBackfillService.

Tests cover:
- AssistantContext model (creation, immutability)
- BackfillResult model (all status cases)
- build_tool_call_index() function (single, parallel, multi-turn)
- extract_tool_responses() function
- StepBackfillService.backfill_steps() (success, edge cases, idempotency)
"""

from typing import Any

import pytest
from pydantic import ValidationError

from saber.models.constants import MetadataKeys
from saber.server.base import Action, Episode, EpisodeState, Step
from saber.server.episodes.step_backfill import (
    AssistantContext,
    BackfillResult,
    StepBackfillService,
    build_tool_call_index,
    extract_tool_responses,
)

# =============================================================================
# Test Fixtures
# =============================================================================

SINGLE_TOOL_CALL_TRANSCRIPT: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "List files"},
    {
        "role": "assistant",
        "content": "I'll list the files for you.",
        "reasoning": "User wants directory listing",
        "tool_calls": [{"id": "call_abc123", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
    },
    {"role": "tool", "tool_call_id": "call_abc123", "content": "file1.txt\nfile2.txt"},
]

PARALLEL_TOOL_CALLS_TRANSCRIPT: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Check both config files"},
    {
        "role": "assistant",
        "content": "I'll check both configuration files simultaneously.",
        "tool_calls": [
            {"id": "call_1", "function": {"name": "bash", "arguments": '{"command": "cat config1.yml"}'}},
            {"id": "call_2", "function": {"name": "bash", "arguments": '{"command": "cat config2.yml"}'}},
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "config1 contents"},
    {"role": "tool", "tool_call_id": "call_2", "content": "config2 contents"},
]

MULTI_TURN_TRANSCRIPT: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Find and read the readme"},
    {
        "role": "assistant",
        "content": "First, let me find the readme file.",
        "tool_calls": [{"id": "call_find", "function": {"name": "bash", "arguments": '{"command": "find"}'}}],
    },
    {"role": "tool", "tool_call_id": "call_find", "content": "./README.md"},
    {
        "role": "assistant",
        "content": "Found it. Now I'll read its contents.",
        "tool_calls": [{"id": "call_read", "function": {"name": "bash", "arguments": '{"command": "cat README.md"}'}}],
    },
    {"role": "tool", "tool_call_id": "call_read", "content": "# Project Title\n..."},
]


def create_test_episode(
    transcript: list[dict[str, Any]] | None = None,
    steps: list[Step] | None = None,
) -> Episode:
    """Create a test episode with optional transcript and steps."""
    context: dict[str, Any] = {}
    if transcript is not None:
        context[MetadataKeys.CLIENT_TRANSCRIPT.value] = transcript

    return Episode(
        episode_id="test-episode-123",
        task_id="test-task",
        session_id="test-session",
        state=EpisodeState.COMPLETED,
        steps=steps or [],
        context=context,
    )


def create_step(step_number: int, tool_name: str = "bash", command: str = "ls") -> Step:
    """Create a test step with minimal data."""
    return Step(
        step_number=step_number,
        action=Action(
            tool_name=tool_name,
            parameters={"command": command},
            assistant_message=None,
            reasoning=None,
        ),
        response={"exit_code": 0, "stdout": "output", "stderr": ""},
    )


# =============================================================================
# Phase 1: Data Models Tests
# =============================================================================


class TestAssistantContext:
    """Tests for AssistantContext model."""

    def test_create_with_all_fields(self) -> None:
        """Test creating AssistantContext with all fields populated."""
        context = AssistantContext(
            assistant_message="I'll help you with that.",
            reasoning="User needs file listing",
        )
        assert context.assistant_message == "I'll help you with that."
        assert context.reasoning == "User needs file listing"

    def test_create_with_defaults(self) -> None:
        """Test creating AssistantContext with default None values."""
        context = AssistantContext()
        assert context.assistant_message is None
        assert context.reasoning is None

    def test_create_with_only_message(self) -> None:
        """Test creating AssistantContext with only assistant_message."""
        context = AssistantContext(assistant_message="Here's the result.")
        assert context.assistant_message == "Here's the result."
        assert context.reasoning is None

    def test_create_with_only_reasoning(self) -> None:
        """Test creating AssistantContext with only reasoning."""
        context = AssistantContext(reasoning="Analyzing user intent")
        assert context.assistant_message is None
        assert context.reasoning == "Analyzing user intent"

    def test_immutability(self) -> None:
        """Test that AssistantContext is immutable (frozen=True)."""
        context = AssistantContext(
            assistant_message="Original message",
            reasoning="Original reasoning",
        )
        with pytest.raises(ValidationError):
            context.assistant_message = "Modified message"  # type: ignore[misc]


class TestBackfillResult:
    """Tests for BackfillResult model."""

    def test_create_success_result(self) -> None:
        """Test creating a success BackfillResult."""
        result = BackfillResult(
            status="success",
            steps_updated=5,
            steps_skipped=0,
            steps_already_filled=0,
            total_steps=5,
        )
        assert result.status == "success"
        assert result.steps_updated == 5
        assert result.steps_skipped == 0
        assert result.steps_already_filled == 0
        assert result.total_steps == 5
        assert result.error_message is None

    def test_create_no_transcript_result(self) -> None:
        """Test creating a no_transcript BackfillResult."""
        result = BackfillResult(status="no_transcript")
        assert result.status == "no_transcript"
        assert result.steps_updated == 0
        assert result.total_steps == 0

    def test_create_no_steps_result(self) -> None:
        """Test creating a no_steps BackfillResult."""
        result = BackfillResult(status="no_steps", total_steps=0)
        assert result.status == "no_steps"
        assert result.steps_updated == 0

    def test_create_error_result(self) -> None:
        """Test creating an error BackfillResult."""
        result = BackfillResult(
            status="error",
            error_message="Something went wrong",
        )
        assert result.status == "error"
        assert result.error_message == "Something went wrong"

    def test_immutability(self) -> None:
        """Test that BackfillResult is immutable (frozen=True)."""
        result = BackfillResult(status="success", steps_updated=5, total_steps=5)
        with pytest.raises(ValidationError):
            result.steps_updated = 10  # type: ignore[misc]

    def test_status_literal_validation(self) -> None:
        """Test that status must be one of the allowed literals."""
        with pytest.raises(ValidationError):
            BackfillResult(status="invalid_status")  # type: ignore[arg-type]


# =============================================================================
# Phase 2: Index Builder Tests
# =============================================================================


class TestBuildToolCallIndex:
    """Tests for build_tool_call_index() function."""

    def test_single_tool_call(self) -> None:
        """Test index building with single tool call."""
        index = build_tool_call_index(SINGLE_TOOL_CALL_TRANSCRIPT)

        assert len(index) == 1
        assert "call_abc123" in index

        context = index["call_abc123"]
        assert context.assistant_message == "I'll list the files for you."
        assert context.reasoning == "User wants directory listing"

    def test_parallel_tool_calls(self) -> None:
        """Test index building with parallel tool calls (same assistant message)."""
        index = build_tool_call_index(PARALLEL_TOOL_CALLS_TRANSCRIPT)

        assert len(index) == 2
        assert "call_1" in index
        assert "call_2" in index

        # Both should share the same context (same assistant message)
        assert index["call_1"].assistant_message == "I'll check both configuration files simultaneously."
        assert index["call_2"].assistant_message == "I'll check both configuration files simultaneously."

        # No reasoning in this transcript
        assert index["call_1"].reasoning is None
        assert index["call_2"].reasoning is None

    def test_multi_turn_conversation(self) -> None:
        """Test index building with multi-turn conversation."""
        index = build_tool_call_index(MULTI_TURN_TRANSCRIPT)

        assert len(index) == 2
        assert "call_find" in index
        assert "call_read" in index

        # First turn
        assert index["call_find"].assistant_message == "First, let me find the readme file."
        # Second turn
        assert index["call_read"].assistant_message == "Found it. Now I'll read its contents."

    def test_empty_transcript(self) -> None:
        """Test index building with empty transcript."""
        index = build_tool_call_index([])
        assert len(index) == 0

    def test_transcript_without_tool_calls(self) -> None:
        """Test index building when transcript has no tool calls."""
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        index = build_tool_call_index(transcript)
        assert len(index) == 0

    def test_malformed_tool_call_no_id(self) -> None:
        """Test handling tool_call without id field."""
        transcript: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": "Running command",
                "tool_calls": [
                    {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}  # Missing 'id'
                ],
            },
        ]
        index = build_tool_call_index(transcript)
        # Should skip tool calls without id
        assert len(index) == 0

    def test_assistant_message_without_content(self) -> None:
        """Test handling assistant message with None content."""
        transcript: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_xyz", "function": {"name": "bash"}}],
            },
        ]
        index = build_tool_call_index(transcript)
        assert len(index) == 1
        assert index["call_xyz"].assistant_message is None


class TestExtractToolResponses:
    """Tests for extract_tool_responses() function."""

    def test_single_tool_response(self) -> None:
        """Test extracting single tool response."""
        responses = extract_tool_responses(SINGLE_TOOL_CALL_TRANSCRIPT)
        assert len(responses) == 1
        assert responses[0]["tool_call_id"] == "call_abc123"

    def test_parallel_tool_responses(self) -> None:
        """Test extracting parallel tool responses (preserves order)."""
        responses = extract_tool_responses(PARALLEL_TOOL_CALLS_TRANSCRIPT)
        assert len(responses) == 2
        # Order should be preserved
        assert responses[0]["tool_call_id"] == "call_1"
        assert responses[1]["tool_call_id"] == "call_2"

    def test_multi_turn_tool_responses(self) -> None:
        """Test extracting tool responses from multi-turn conversation."""
        responses = extract_tool_responses(MULTI_TURN_TRANSCRIPT)
        assert len(responses) == 2
        assert responses[0]["tool_call_id"] == "call_find"
        assert responses[1]["tool_call_id"] == "call_read"

    def test_empty_transcript(self) -> None:
        """Test extracting from empty transcript."""
        responses = extract_tool_responses([])
        assert len(responses) == 0

    def test_transcript_without_tool_responses(self) -> None:
        """Test extracting when no tool responses exist."""
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        responses = extract_tool_responses(transcript)
        assert len(responses) == 0


# =============================================================================
# Phase 3: Backfill Service Tests
# =============================================================================


class TestStepBackfillService:
    """Tests for StepBackfillService.backfill_steps() method."""

    def test_successful_backfill_single_step(self) -> None:
        """Test successful backfill with single step."""
        steps = [create_step(0, "bash", "ls")]
        episode = create_test_episode(transcript=SINGLE_TOOL_CALL_TRANSCRIPT, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "success"
        assert result.steps_updated == 1
        assert result.total_steps == 1
        assert result.steps_skipped == 0

        # Verify step was updated
        assert episode.steps[0].action.assistant_message == "I'll list the files for you."
        assert episode.steps[0].action.reasoning == "User wants directory listing"

    def test_successful_backfill_parallel_steps(self) -> None:
        """Test successful backfill with parallel tool calls."""
        steps = [
            create_step(0, "bash", "cat config1.yml"),
            create_step(1, "bash", "cat config2.yml"),
        ]
        episode = create_test_episode(transcript=PARALLEL_TOOL_CALLS_TRANSCRIPT, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "success"
        assert result.steps_updated == 2
        assert result.total_steps == 2

        # Both steps should have the same assistant message
        assert episode.steps[0].action.assistant_message == "I'll check both configuration files simultaneously."
        assert episode.steps[1].action.assistant_message == "I'll check both configuration files simultaneously."

    def test_successful_backfill_multi_turn(self) -> None:
        """Test successful backfill with multi-turn conversation."""
        steps = [
            create_step(0, "bash", "find . -name README*"),
            create_step(1, "bash", "cat README.md"),
        ]
        episode = create_test_episode(transcript=MULTI_TURN_TRANSCRIPT, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "success"
        assert result.steps_updated == 2

        # Each step should have its respective assistant message
        assert episode.steps[0].action.assistant_message == "First, let me find the readme file."
        assert episode.steps[1].action.assistant_message == "Found it. Now I'll read its contents."

    def test_no_transcript(self) -> None:
        """Test backfill when no transcript exists."""
        steps = [create_step(0)]
        episode = create_test_episode(transcript=None, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "no_transcript"
        assert result.steps_updated == 0

    def test_empty_transcript(self) -> None:
        """Test backfill with empty transcript."""
        steps = [create_step(0)]
        episode = create_test_episode(transcript=[], steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "no_transcript"
        assert result.steps_updated == 0

    def test_no_steps(self) -> None:
        """Test backfill when no steps exist."""
        episode = create_test_episode(transcript=SINGLE_TOOL_CALL_TRANSCRIPT, steps=[])

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "no_steps"
        assert result.steps_updated == 0
        assert result.total_steps == 0

    def test_idempotency_does_not_overwrite(self) -> None:
        """Test that second backfill doesn't overwrite already filled steps."""
        steps = [create_step(0, "bash", "ls")]
        episode = create_test_episode(transcript=SINGLE_TOOL_CALL_TRANSCRIPT, steps=steps)

        service = StepBackfillService()

        # First backfill
        result1 = service.backfill_steps(episode)
        assert result1.status == "success"
        assert result1.steps_updated == 1
        assert result1.steps_already_filled == 0

        # Second backfill - should recognize step is already filled
        result2 = service.backfill_steps(episode)
        assert result2.status == "success"
        assert result2.steps_updated == 0
        assert result2.steps_already_filled == 1

    def test_more_steps_than_tool_responses(self) -> None:
        """Test handling when there are more steps than tool responses."""
        # Create 3 steps but transcript only has 1 tool response
        steps = [
            create_step(0, "bash", "ls"),
            create_step(1, "bash", "pwd"),
            create_step(2, "bash", "whoami"),
        ]
        episode = create_test_episode(transcript=SINGLE_TOOL_CALL_TRANSCRIPT, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "success"
        assert result.steps_updated == 1
        assert result.steps_skipped == 2
        assert result.total_steps == 3

        # First step should be updated
        assert episode.steps[0].action.assistant_message == "I'll list the files for you."
        # Other steps should remain None
        assert episode.steps[1].action.assistant_message is None
        assert episode.steps[2].action.assistant_message is None

    def test_fewer_steps_than_tool_responses(self) -> None:
        """Test handling when there are fewer steps than tool responses."""
        # Create 1 step but transcript has 2 tool responses
        steps = [create_step(0, "bash", "cat config1.yml")]
        episode = create_test_episode(transcript=PARALLEL_TOOL_CALLS_TRANSCRIPT, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "success"
        assert result.steps_updated == 1
        assert result.total_steps == 1

        # Step should be matched to first tool response
        assert episode.steps[0].action.assistant_message == "I'll check both configuration files simultaneously."

    def test_tool_response_without_tool_call_id(self) -> None:
        """Test handling tool response missing tool_call_id."""
        transcript: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": "Running command",
                "tool_calls": [{"id": "call_123", "function": {"name": "bash"}}],
            },
            {"role": "tool", "content": "output"},  # Missing tool_call_id
        ]
        steps = [create_step(0)]
        episode = create_test_episode(transcript=transcript, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        # Should skip the step with missing tool_call_id
        assert result.status == "success"
        assert result.steps_skipped == 1
        assert result.steps_updated == 0

    def test_tool_call_id_not_in_index(self) -> None:
        """Test handling when tool_call_id doesn't match any assistant message."""
        transcript: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": "I'll help",
                "tool_calls": [{"id": "call_original", "function": {"name": "bash"}}],
            },
            {"role": "tool", "tool_call_id": "call_different", "content": "output"},  # Different ID
        ]
        steps = [create_step(0)]
        episode = create_test_episode(transcript=transcript, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        # Should skip since tool_call_id doesn't match index
        assert result.status == "success"
        assert result.steps_skipped == 1
        assert result.steps_updated == 0

    def test_partial_backfill_mixed_states(self) -> None:
        """Test backfill with mix of filled and unfilled steps."""
        steps = [
            create_step(0, "bash", "find . -name README*"),
            create_step(1, "bash", "cat README.md"),
        ]
        # Pre-fill first step
        steps[0].action.assistant_message = "Pre-existing message"
        steps[0].action.reasoning = "Pre-existing reasoning"

        episode = create_test_episode(transcript=MULTI_TURN_TRANSCRIPT, steps=steps)

        service = StepBackfillService()
        result = service.backfill_steps(episode)

        assert result.status == "success"
        assert result.steps_updated == 1  # Only second step
        assert result.steps_already_filled == 1  # First step was pre-filled

        # First step should retain original values
        assert episode.steps[0].action.assistant_message == "Pre-existing message"
        assert episode.steps[0].action.reasoning == "Pre-existing reasoning"

        # Second step should be updated
        assert episode.steps[1].action.assistant_message == "Found it. Now I'll read its contents."
