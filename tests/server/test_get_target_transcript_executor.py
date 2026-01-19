"""
Tests for GetTargetTranscriptExecutor - Red team transcript retrieval.

Tests the red team's ability to retrieve blue team transcripts with
state-aware waiting and efficient delta retrieval.
"""

import json
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState, CommandResult
from saber.server.execution.executors.standard_registry.get_target_transcript_executor import (
    GetTargetTranscriptExecutor,
)
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestGetTargetTranscriptExecutorParameterValidation:
    """Test parameter validation for get_target_transcript executor."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode with transcript."""
        return Episode(
            episode_id="ep-blue-123",
            task_id="blue-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                MetadataKeys.TRANSCRIPT_VERSION: 3,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: "append",
            },
        )

    @pytest.fixture
    def red_episode(self) -> Episode:
        """Create a red team episode."""
        return Episode(
            episode_id="ep-red-456",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-123"],
            },
        )

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment for curl command execution."""
        env = AsyncMock()
        env.execute_command = AsyncMock()
        return env

    @pytest.fixture
    def mock_sandbox_manager(self, mock_environment):
        """Create a mock sandbox manager."""
        manager = MagicMock()
        manager.get_episode_environment = MagicMock(return_value=mock_environment)
        return manager

    @pytest.fixture
    def mock_session_manager(self, blue_episode: Episode, red_episode: Episode):
        """Create a mock session manager."""

        class MockEpisodeManager:
            def __init__(self, blue_ep, red_ep):
                self.episodes = {
                    blue_ep.episode_id: blue_ep,
                    red_ep.episode_id: red_ep,
                }

            def get_episode_by_id(self, episode_id: str):
                return self.episodes.get(episode_id)

        class MockSessionManager:
            def __init__(self, blue_ep, red_ep):
                self.episode_manager = MockEpisodeManager(blue_ep, red_ep)

        return MockSessionManager(blue_episode, red_episode)

    @pytest.mark.asyncio
    async def test_invalid_retrieval_mode_rejected(
        self, mock_sandbox_manager, mock_session_manager, red_episode
    ):
        """Test that invalid retrieval_mode is rejected."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager,
        )

        parameters = {
            "retrieval_mode": "invalid_mode",
        }
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Invalid retrieval_mode" in result.error

    @pytest.mark.asyncio
    async def test_tail_count_out_of_range_rejected(
        self, mock_sandbox_manager, mock_session_manager, red_episode
    ):
        """Test that tail_count outside 1-1000 range is rejected."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager,
        )

        # Test too low
        parameters = {
            "retrieval_mode": "tail",
            "tail_count": 0,
        }
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)
        assert result.success is False
        assert "tail_count must be 1-1000" in result.error

        # Test too high
        parameters["tail_count"] = 1001
        result = await executor.execute(parameters, context)
        assert result.success is False
        assert "tail_count must be 1-1000" in result.error

    @pytest.mark.asyncio
    async def test_missing_episode_id_rejected(self, mock_sandbox_manager, mock_session_manager):
        """Test that missing episode_id in context is rejected."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager,
        )

        parameters = {}
        context = {}  # Missing episode_id

        with pytest.raises(ValueError, match="episode_id is required"):
            await executor.execute(parameters, context)


class TestGetTargetTranscriptExecutorTargetResolution:
    """Test target episode ID resolution from orchestration metadata."""

    @pytest.fixture
    def red_episode_no_target(self) -> Episode:
        """Create a red team episode without orchestration metadata."""
        return Episode(
            episode_id="ep-red-456",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={},  # No ORCHESTRATION_TARGET_EPISODES
        )

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock sandbox manager."""
        manager = MagicMock()
        env = AsyncMock()
        manager.get_episode_environment = MagicMock(return_value=env)
        return manager

    @pytest.fixture
    def mock_session_manager_no_target(self, red_episode_no_target):
        """Create a mock session manager with no target episode."""

        class MockEpisodeManager:
            def get_episode_by_id(self, episode_id: str):
                if episode_id == "ep-red-456":
                    return red_episode_no_target
                return None

        class MockSessionManager:
            def __init__(self):
                self.episode_manager = MockEpisodeManager()

        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_missing_orchestration_metadata_fails(
        self, mock_sandbox_manager, mock_session_manager_no_target
    ):
        """Test that missing orchestration metadata results in error."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager_no_target,
        )

        parameters = {}
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Could not resolve target_episode_id" in result.error


class TestGetTargetTranscriptExecutorFullMode:
    """Test full transcript retrieval mode."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode with transcript."""
        return Episode(
            episode_id="ep-blue-123",
            task_id="blue-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "System message"},
                    {"role": "user", "content": "User message"},
                    {"role": "assistant", "content": "Assistant message"},
                ],
                MetadataKeys.TRANSCRIPT_VERSION: 3,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: "append",
            },
        )

    @pytest.fixture
    def red_episode(self) -> Episode:
        """Create a red team episode."""
        return Episode(
            episode_id="ep-red-456",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-123"],
            },
        )

    @pytest.fixture
    def mock_environment(self, blue_episode):
        """Create a mock environment that simulates daemon response."""
        env = AsyncMock()

        async def mock_execute_command(command, timeout):
            # Simulate daemon returning full transcript
            response = {
                "success": True,
                "messages": blue_episode.context[MetadataKeys.CLIENT_TRANSCRIPT],
                "current_version": 3,
                "full_transcript_length": 3,
                "last_operation": "append",
            }
            result = MagicMock()
            result.exit_code = 0
            result.stdout = json.dumps(response)
            result.stderr = ""
            return result

        env.execute_command = mock_execute_command
        return env

    @pytest.fixture
    def mock_sandbox_manager(self, mock_environment):
        """Create a mock sandbox manager."""
        manager = MagicMock()
        manager.get_episode_environment = MagicMock(return_value=mock_environment)
        return manager

    @pytest.fixture
    def mock_session_manager(self, blue_episode, red_episode):
        """Create a mock session manager."""

        class MockEpisodeManager:
            def __init__(self):
                self.episodes = {
                    "ep-blue-123": blue_episode,
                    "ep-red-456": red_episode,
                }

            def get_episode_by_id(self, episode_id: str):
                return self.episodes.get(episode_id)

        class MockSessionManager:
            def __init__(self):
                self.episode_manager = MockEpisodeManager()

        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_full_mode_retrieves_all_messages(
        self, mock_sandbox_manager, mock_session_manager, blue_episode
    ):
        """Test that full mode retrieves entire transcript."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager,
        )

        parameters = {
            "retrieval_mode": "full",
            "wait_for_user": False,  # Skip waiting for this test
        }
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is True
        assert result.data["retrieval_mode"] == "full"
        assert result.data["message_count"] == 3
        assert len(result.data["messages"]) == 3
        assert result.data["messages"][0]["content"] == "System message"


class TestGetTargetTranscriptExecutorDeltaMode:
    """Test delta transcript retrieval mode."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment that simulates delta response."""
        env = AsyncMock()

        async def mock_execute_command(command, timeout):
            # Simulate daemon returning delta (last 2 messages)
            response = {
                "success": True,
                "messages": [
                    {"role": "user", "content": "New user message"},
                    {"role": "assistant", "content": "New assistant message"},
                ],
                "current_version": 5,
                "full_transcript_length": 5,
                "last_operation": "append",
            }
            result = MagicMock()
            result.exit_code = 0
            result.stdout = json.dumps(response)
            result.stderr = ""
            return result

        env.execute_command = mock_execute_command
        return env

    @pytest.fixture
    def mock_sandbox_manager(self, mock_environment):
        """Create a mock sandbox manager."""
        manager = MagicMock()
        manager.get_episode_environment = MagicMock(return_value=mock_environment)
        return manager

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""

        class MockEpisodeManager:
            def get_episode_by_id(self, episode_id: str):
                if episode_id == "ep-red-456":
                    return Episode(
                        episode_id="ep-red-456",
                        task_id="red-task",
                        session_id="session-789",
                        state=EpisodeState.ACTIVE,
                        context={
                            MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-123"],
                        },
                    )
                return None

        class MockSessionManager:
            def __init__(self):
                self.episode_manager = MockEpisodeManager()

        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_delta_mode_retrieves_new_messages(
        self, mock_sandbox_manager, mock_session_manager
    ):
        """Test that delta mode retrieves only new messages since version."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager,
        )

        parameters = {
            "retrieval_mode": "delta",
            "since_version": 3,
            "wait_for_user": False,
        }
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is True
        assert result.data["retrieval_mode"] == "delta"
        assert result.data["message_count"] == 2
        assert result.metadata["current_version"] == 5


class TestGetTargetTranscriptExecutorTailMode:
    """Test tail transcript retrieval mode."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment that simulates tail response."""
        env = AsyncMock()

        async def mock_execute_command(command, timeout):
            # Simulate daemon returning last 5 messages
            response = {
                "success": True,
                "messages": [
                    {"role": "assistant", "content": f"Message {i}"} for i in range(1, 6)
                ],
                "current_version": 10,
                "full_transcript_length": 10,
                "last_operation": "append",
            }
            result = MagicMock()
            result.exit_code = 0
            result.stdout = json.dumps(response)
            result.stderr = ""
            return result

        env.execute_command = mock_execute_command
        return env

    @pytest.fixture
    def mock_sandbox_manager(self, mock_environment):
        """Create a mock sandbox manager."""
        manager = MagicMock()
        manager.get_episode_environment = MagicMock(return_value=mock_environment)
        return manager

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""

        class MockEpisodeManager:
            def get_episode_by_id(self, episode_id: str):
                if episode_id == "ep-red-456":
                    return Episode(
                        episode_id="ep-red-456",
                        task_id="red-task",
                        session_id="session-789",
                        state=EpisodeState.ACTIVE,
                        context={
                            MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-123"],
                        },
                    )
                return None

        class MockSessionManager:
            def __init__(self):
                self.episode_manager = MockEpisodeManager()

        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_tail_mode_retrieves_last_n_messages(
        self, mock_sandbox_manager, mock_session_manager
    ):
        """Test that tail mode retrieves last N messages."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager,
        )

        parameters = {
            "retrieval_mode": "tail",
            "tail_count": 5,
            "wait_for_user": False,
        }
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is True
        assert result.data["retrieval_mode"] == "tail"
        assert result.data["message_count"] == 5


class TestGetTargetTranscriptExecutorRetryLogic:
    """Test retry logic for transient failures."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""

        class MockEpisodeManager:
            def get_episode_by_id(self, episode_id: str):
                if episode_id == "ep-red-456":
                    return Episode(
                        episode_id="ep-red-456",
                        task_id="red-task",
                        session_id="session-789",
                        state=EpisodeState.ACTIVE,
                        context={
                            MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-123"],
                        },
                    )
                return None

        class MockSessionManager:
            def __init__(self):
                self.episode_manager = MockEpisodeManager()

        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_retries_on_connection_refused(self, mock_session_manager):
        """Test that executor retries on connection refused error."""
        env = AsyncMock()
        call_count = 0

        async def mock_execute_command(command, timeout):
            nonlocal call_count
            call_count += 1

            result = MagicMock()
            if call_count < 2:
                # First call fails with connection refused
                result.exit_code = 1
                result.stderr = "connection refused"
                result.stdout = ""
            else:
                # Second call succeeds
                result.exit_code = 0
                result.stdout = json.dumps(
                    {
                        "success": True,
                        "messages": [],
                        "current_version": 0,
                        "full_transcript_length": 0,
                        "last_operation": "append",
                    }
                )
                result.stderr = ""
            return result

        env.execute_command = mock_execute_command

        manager = MagicMock()
        manager.get_episode_environment = MagicMock(return_value=env)

        executor = GetTargetTranscriptExecutor(
            sandbox_manager=manager,
            session_manager=mock_session_manager,
        )

        parameters = {"wait_for_user": False}
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is True
        assert call_count == 2  # Should have retried once

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, mock_session_manager):
        """Test that executor gives up after MAX_RETRIES attempts."""
        env = AsyncMock()

        async def mock_execute_command(command, timeout):
            # Always fail
            result = MagicMock()
            result.exit_code = 1
            result.stderr = "connection refused"
            result.stdout = ""
            return result

        env.execute_command = mock_execute_command

        manager = MagicMock()
        manager.get_episode_environment = MagicMock(return_value=env)

        executor = GetTargetTranscriptExecutor(
            sandbox_manager=manager,
            session_manager=mock_session_manager,
        )

        parameters = {"wait_for_user": False}
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is False
        assert "connection refused" in result.error


class TestGetTargetTranscriptExecutorMetadata:
    """Test metadata inclusion in results."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment."""
        env = AsyncMock()

        async def mock_execute_command(command, timeout):
            response = {
                "success": True,
                "messages": [{"role": "user", "content": "Test"}],
                "current_version": 5,
                "full_transcript_length": 10,
                "last_operation": "append",
            }
            result = MagicMock()
            result.exit_code = 0
            result.stdout = json.dumps(response)
            result.stderr = ""
            return result

        env.execute_command = mock_execute_command
        return env

    @pytest.fixture
    def mock_sandbox_manager(self, mock_environment):
        """Create a mock sandbox manager."""
        manager = MagicMock()
        manager.get_episode_environment = MagicMock(return_value=mock_environment)
        return manager

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""

        class MockEpisodeManager:
            def get_episode_by_id(self, episode_id: str):
                if episode_id == "ep-red-456":
                    return Episode(
                        episode_id="ep-red-456",
                        task_id="red-task",
                        session_id="session-789",
                        state=EpisodeState.ACTIVE,
                        context={
                            MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-123"],
                        },
                    )
                return None

        class MockSessionManager:
            def __init__(self):
                self.episode_manager = MockEpisodeManager()

        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_includes_metadata_when_requested(
        self, mock_sandbox_manager, mock_session_manager
    ):
        """Test that metadata is included when include_metadata=True."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager,
        )

        parameters = {
            "include_metadata": True,
            "wait_for_user": False,
        }
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is True
        assert "metadata" in result.data
        assert result.data["metadata"]["current_version"] == 5
        assert result.data["metadata"]["full_transcript_length"] == 10
        assert result.data["metadata"]["last_operation"] == "append"

    @pytest.mark.asyncio
    async def test_excludes_metadata_when_not_requested(
        self, mock_sandbox_manager, mock_session_manager
    ):
        """Test that metadata is excluded when include_metadata=False."""
        executor = GetTargetTranscriptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager,
        )

        parameters = {
            "include_metadata": False,
            "wait_for_user": False,
        }
        context = {
            "session_id": "session-789",
            "episode_id": "ep-red-456",
        }

        result = await executor.execute(parameters, context)

        assert result.success is True
        assert "metadata" not in result.data
