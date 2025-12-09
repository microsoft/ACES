"""Tests for server-side transcript initialization.

Tests that episodes are created with pre-populated transcripts containing
system and user messages constructed from task prompts.

Design Doc: SERVER_SIDE_TRANSCRIPT_INIT_DESIGN.md
"""

import pytest
from unittest.mock import MagicMock

from saber.server.base import Episode, EpisodeState
from saber.server.episodes.episode_manager import EpisodeManager
from saber.models.benchmark_task import SingleEpisodeTask
from saber.models.constants import MetadataKeys
from saber.models.transcript import compute_checksum


class TestServerSideTranscriptInit:
    """Test server-side transcript initialization in start_episode()."""

    def test_start_episode_initializes_transcript_for_single_task(self):
        """Server creates transcript with system and user messages from task prompts."""
        # Arrange
        manager = EpisodeManager()

        task = SingleEpisodeTask(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="Solve this CTF challenge",
            max_steps=10,
            episode_attempts=1,
            instruction_prompt="You are a security agent",
            assistant_prompt="Use these tools to complete the task",
            submit_prompt="Submit your answer with /submit",
        )

        # Act
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            task=task
        )

        # Assert - transcript should be initialized
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert transcript is not None, "Transcript should be initialized"
        assert len(transcript) == 2, "Transcript should have system and user messages"

        # Check system message
        assert transcript[0]["role"] == "system"
        system_content = transcript[0]["content"]
        assert "You are a security agent" in system_content
        assert "Use these tools to complete the task" in system_content
        assert "Submit your answer with /submit" in system_content
        assert "---" in system_content, "Prompts should be separated by ---"

        # Check user message
        assert transcript[1]["role"] == "user"
        assert transcript[1]["content"] == "Solve this CTF challenge"

        # Check metadata
        # Version starts at 1 because the "init" operation has been applied
        # (version 0 would mean no operations, i.e., empty transcript)
        assert episode.context.get(MetadataKeys.TRANSCRIPT_VERSION) == 1
        assert episode.context.get(MetadataKeys.TRANSCRIPT_LAST_OPERATION) == "init"

        # Check checksum
        expected_checksum = compute_checksum(transcript)
        actual_checksum = episode.context.get(MetadataKeys.TRANSCRIPT_CHECKSUM)
        assert actual_checksum == expected_checksum

    def test_start_episode_uses_delimiter_between_prompts(self):
        """System message should use --- delimiter between prompt sections."""
        # Arrange
        manager = EpisodeManager()

        task = SingleEpisodeTask(
            task_id="test_task",
            domain="test_domain",
            title="Test",
            description="Task desc",
            max_steps=5,
            episode_attempts=1,
            instruction_prompt="Instruction part",
            assistant_prompt="Assistant part",
            submit_prompt="Submit part",
        )

        # Act
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            task=task
        )

        # Assert
        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        system_content = transcript[0]["content"]

        # Should have exactly 2 delimiter instances (between 3 parts)
        assert system_content.count("\n\n---\n\n") == 2

        # Parts should be in correct order
        assert system_content.index("Instruction part") < system_content.index("Assistant part")
        assert system_content.index("Assistant part") < system_content.index("Submit part")

    def test_start_episode_without_task_has_no_transcript(self):
        """Episodes created without task data should not have transcript."""
        # Arrange
        manager = EpisodeManager()

        # Act - create episode without task
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            initial_context={"some": "context"}
        )

        # Assert - no transcript initialized
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert transcript is None

    def test_start_episode_preserves_initial_context(self):
        """Transcript initialization should preserve existing context fields."""
        # Arrange
        manager = EpisodeManager()

        task = SingleEpisodeTask(
            task_id="test_task",
            domain="test_domain",
            title="Test",
            description="Desc",
            max_steps=5,
            episode_attempts=1,
            instruction_prompt="Inst",
            assistant_prompt="Asst",
            submit_prompt="Submit",
        )

        initial_context = {
            "custom_field": "custom_value",
            "another_field": 42,
        }

        # Act
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            initial_context=initial_context,
            task=task
        )

        # Assert - initial context preserved
        assert episode.context["custom_field"] == "custom_value"
        assert episode.context["another_field"] == 42

        # And transcript added
        assert MetadataKeys.CLIENT_TRANSCRIPT in episode.context

    def test_start_episode_handles_empty_prompt_parts(self):
        """Should handle tasks with empty prompt components gracefully."""
        # Arrange
        manager = EpisodeManager()

        task = SingleEpisodeTask(
            task_id="test_task",
            domain="test_domain",
            title="Test",
            description="Desc",
            max_steps=5,
            episode_attempts=1,
            instruction_prompt="",  # Empty
            assistant_prompt="Assistant content",
            submit_prompt="",  # Empty
        )

        # Act
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            task=task
        )

        # Assert - should still create transcript
        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 2

        # System message should only have non-empty parts
        system_content = transcript[0]["content"]
        assert "Assistant content" in system_content
        # Empty parts should not create extra delimiters
        assert not system_content.startswith("---")
        assert not system_content.endswith("---")

    def test_start_episode_checksum_matches_content(self):
        """Checksum should match the actual transcript content."""
        # Arrange
        manager = EpisodeManager()

        task = SingleEpisodeTask(
            task_id="test_task",
            domain="test_domain",
            title="Test",
            description="Description",
            max_steps=5,
            episode_attempts=1,
            instruction_prompt="Instruction",
            assistant_prompt="Assistant",
            submit_prompt="Submit",
        )

        # Act
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            task=task
        )

        # Assert - recompute checksum and verify
        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        stored_checksum = episode.context.get("_transcript_checksum")
        computed_checksum = compute_checksum(transcript)

        assert stored_checksum == computed_checksum

    def test_start_episode_version_zero_on_init(self):
        """Initial transcript version should be 0."""
        # Arrange
        manager = EpisodeManager()

        task = SingleEpisodeTask(
            task_id="test_task",
            domain="test_domain",
            title="Test",
            description="Desc",
            max_steps=5,
            episode_attempts=1,
            instruction_prompt="I",
            assistant_prompt="A",
            submit_prompt="S",
        )

        # Act
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            task=task
        )

        # Assert - version is 1 because "init" operation has been applied
        version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION)
        assert version == 1

    def test_start_episode_operation_marked_as_init(self):
        """Last operation should be marked as 'init'."""
        # Arrange
        manager = EpisodeManager()

        task = SingleEpisodeTask(
            task_id="test_task",
            domain="test_domain",
            title="Test",
            description="Desc",
            max_steps=5,
            episode_attempts=1,
            instruction_prompt="I",
            assistant_prompt="A",
            submit_prompt="S",
        )

        # Act
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            task=task
        )

        # Assert
        operation = episode.context.get(MetadataKeys.TRANSCRIPT_LAST_OPERATION)
        assert operation == "init"


class TestOrchestratedTaskTranscriptInit:
    """Test transcript initialization for orchestrated tasks."""

    def test_orchestrated_task_parent_no_transcript(self):
        """Parent orchestrated episode should not have transcript."""
        # Arrange
        manager = EpisodeManager()

        # Act - create parent orchestrated episode (no task object means no transcript)
        episode = manager.start_episode(
            session_id="test_session",
            task_id="orchestrated_parent",
            initial_context={"orchestration_state": "initialized"}
        )

        # Assert - no transcript for orchestration coordinator
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert transcript is None

    def test_sub_task_episode_has_transcript(self):
        """Sub-task episodes should have transcripts initialized from SubTaskDefinition."""
        # Arrange
        from saber.models.benchmark_task import SubTaskDefinition

        manager = EpisodeManager()

        sub_task = SubTaskDefinition(
            task_id="blue_team",
            role="blue",
            order=0,
            domain="cybersecurity",
            title="Blue Team Task",
            description="Defend the network",
            episode_attempts=1,
            max_steps=15,
            instruction_prompt="You are a defensive security agent",
            assistant_prompt="Use defensive tools",
            submit_prompt="Submit your defense strategy",
        )

        # Act
        episode = manager.start_episode(
            session_id="test_session",
            task_id="blue_team",
            task=sub_task
        )

        # Assert - sub-task should have transcript
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert transcript is not None
        assert len(transcript) == 2

        # Check system message contains prompts
        assert "You are a defensive security agent" in transcript[0]["content"]
        assert "Use defensive tools" in transcript[0]["content"]
        assert "Submit your defense strategy" in transcript[0]["content"]

        # Check user message
        assert transcript[1]["role"] == "user"
        assert transcript[1]["content"] == "Defend the network"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
