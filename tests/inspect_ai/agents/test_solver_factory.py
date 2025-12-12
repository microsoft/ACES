"""Tests for solver factory."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from inspect_ai.solver import TaskState

from saber.inspect_ai.agents.solver_factory import (
    SABERExecutionContext,
    create_saber_solver,
)
from saber.models.constants import MetadataKeys


@pytest.fixture
def mock_task_state() -> MagicMock:
    """Create a mock TaskState with metadata and store."""
    state = MagicMock(spec=TaskState)
    state.metadata = {}
    state.store = MagicMock()
    # Return empty dict for episode_mapping, None for others
    def store_get(key: str, default: Any = None) -> Any:
        if key == "episode_mapping":
            return {}
        return default
    state.store.get = MagicMock(side_effect=store_get)
    return state


@pytest.fixture
def complete_task_state_metadata() -> dict[str, Any]:
    """Complete metadata fixture with all required prompts."""
    return {
        MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
        MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        MetadataKeys.SUBMIT_PROMPT: "Test submit",
        MetadataKeys.CONTINUE_PROMPT: "Test continue",
        MetadataKeys.SAMPLE_ID: "test-sample",
    }


class TestSABERExecutionContext:
    """Tests for SABERExecutionContext."""

    def test_from_task_state_extracts_all_prompts(
        self, mock_task_state: MagicMock, complete_task_state_metadata: dict[str, Any]
    ) -> None:
        """Test that from_task_state extracts all required prompts."""
        mock_task_state.metadata = complete_task_state_metadata

        context = SABERExecutionContext.from_task_state(mock_task_state)

        assert context.instruction_prompt == "Test instruction"
        assert context.assistant_prompt == "Test assistant"
        assert context.submit_prompt == "Test submit"
        assert context.continue_prompt == "Test continue"

    def test_from_task_state_raises_for_missing_instruction_prompt(
        self, mock_task_state: MagicMock
    ) -> None:
        """Test that from_task_state raises for missing instruction_prompt."""
        mock_task_state.metadata = {
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            MetadataKeys.CONTINUE_PROMPT: "Test continue",
        }

        with pytest.raises(ValueError, match="Missing 'instruction_prompt'"):
            SABERExecutionContext.from_task_state(mock_task_state)

    def test_from_task_state_raises_for_missing_assistant_prompt(
        self, mock_task_state: MagicMock
    ) -> None:
        """Test that from_task_state raises for missing assistant_prompt."""
        mock_task_state.metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            MetadataKeys.CONTINUE_PROMPT: "Test continue",
        }

        with pytest.raises(ValueError, match="Missing 'assistant_prompt'"):
            SABERExecutionContext.from_task_state(mock_task_state)

    def test_from_task_state_raises_for_missing_submit_prompt(
        self, mock_task_state: MagicMock
    ) -> None:
        """Test that from_task_state raises for missing submit_prompt."""
        mock_task_state.metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.CONTINUE_PROMPT: "Test continue",
        }

        with pytest.raises(ValueError, match="Missing 'submit_prompt'"):
            SABERExecutionContext.from_task_state(mock_task_state)

    def test_from_task_state_raises_for_missing_continue_prompt(
        self, mock_task_state: MagicMock
    ) -> None:
        """Test that from_task_state raises for missing continue_prompt."""
        mock_task_state.metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            # continue_prompt missing
        }

        with pytest.raises(ValueError, match="Missing 'continue_prompt'"):
            SABERExecutionContext.from_task_state(mock_task_state)

    def test_has_episode_context_returns_false_when_incomplete(
        self, mock_task_state: MagicMock, complete_task_state_metadata: dict[str, Any]
    ) -> None:
        """Test has_episode_context returns False when context incomplete."""
        mock_task_state.metadata = complete_task_state_metadata

        context = SABERExecutionContext.from_task_state(mock_task_state)

        # Without session_id, episode_id, or rest_url
        assert context.has_episode_context() is False


class TestCreateSaberSolver:
    """Tests for create_saber_solver function."""

    def test_create_saber_solver_returns_solver(self) -> None:
        """Test that create_saber_solver returns a solver."""
        mock_agent = MagicMock()

        def agent_factory():
            def create_with_prompts(**kwargs):
                return mock_agent
            return create_with_prompts

        solver = create_saber_solver("test-agent", agent_factory)

        assert solver is not None


class TestContinuePromptIntegration:
    """Integration tests for continue_prompt functionality."""

    def test_continue_prompt_required_in_metadata(
        self, mock_task_state: MagicMock
    ) -> None:
        """Test that continue_prompt is required in metadata."""
        mock_task_state.metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            # continue_prompt missing
        }

        with pytest.raises(ValueError, match="Missing 'continue_prompt'"):
            SABERExecutionContext.from_task_state(mock_task_state)

    def test_different_continue_prompts_allowed(
        self, mock_task_state: MagicMock
    ) -> None:
        """Test different continue prompts can be set per task."""
        blue_metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Blue instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Blue assistant",
            MetadataKeys.SUBMIT_PROMPT: "Blue submit",
            MetadataKeys.CONTINUE_PROMPT: "Continue without submitting",
        }

        red_metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Red instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Red assistant",
            MetadataKeys.SUBMIT_PROMPT: "Red submit",
            MetadataKeys.CONTINUE_PROMPT: "Continue and call submit() when done",
        }

        mock_task_state.metadata = blue_metadata
        blue_context = SABERExecutionContext.from_task_state(mock_task_state)

        mock_task_state.metadata = red_metadata
        red_context = SABERExecutionContext.from_task_state(mock_task_state)

        assert blue_context.continue_prompt == "Continue without submitting"
        assert red_context.continue_prompt == "Continue and call submit() when done"
        assert blue_context.continue_prompt != red_context.continue_prompt

    def test_context_extracts_continue_prompt_correctly(
        self, mock_task_state: MagicMock
    ) -> None:
        """Test that continue_prompt is extracted correctly from context."""
        custom_continue = "Custom continue prompt message"
        mock_task_state.metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            MetadataKeys.CONTINUE_PROMPT: custom_continue,
        }

        context = SABERExecutionContext.from_task_state(mock_task_state)

        assert context.continue_prompt == custom_continue
