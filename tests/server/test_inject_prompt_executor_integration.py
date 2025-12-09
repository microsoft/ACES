"""
Integration tests for InjectPromptExecutor with executor factory.

Tests the registration and configuration of InjectPromptExecutor
through the executor factory system.
"""

import json
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState
from saber.server.execution.executors import get_executor_class, get_available_executors
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


def create_mock_sandbox_manager(execute_response: dict = None):
    """Create a mock sandbox manager that returns proper async responses."""
    if execute_response is None:
        execute_response = {"success": True, "version": 1}

    mock_manager = MagicMock(spec=SandboxEnvironmentManager)

    # Create mock environment with async execute_command
    mock_environment = MagicMock()
    mock_exec_result = MagicMock()
    mock_exec_result.exit_code = 0
    mock_exec_result.stdout = json.dumps(execute_response)
    mock_exec_result.stderr = ""
    mock_environment.execute_command = AsyncMock(return_value=mock_exec_result)

    mock_manager.get_episode_environment = MagicMock(return_value=mock_environment)

    return mock_manager, mock_environment


def create_mock_session_manager(*episodes):
    """Create a mock session manager with episode lookup capability."""

    class MockEpisodeManager:
        def __init__(self, *episodes):
            self.episodes = {ep.episode_id: ep for ep in episodes}

        def get_episode_by_id(self, episode_id: str):
            return self.episodes.get(episode_id)

    class MockSessionManager:
        def __init__(self, *episodes):
            self.episode_manager = MockEpisodeManager(*episodes)

    return MockSessionManager(*episodes)


class TestExecutorRegistration:
    """Test that InjectPromptExecutor is properly registered."""

    def test_executor_class_registered(self):
        """Test that inject_prompt executor is registered."""
        # Act
        executor_class = get_executor_class("inject_prompt")

        # Assert
        assert executor_class is not None
        assert executor_class.__name__ == "InjectPromptExecutor"

    def test_inject_prompt_in_available_executors(self):
        """Test that inject_prompt is in the list of available executors."""
        # Act
        available = get_available_executors()

        # Assert
        assert "inject_prompt" in available

    def test_executor_metadata(self):
        """Test executor metadata is correctly set."""
        # Arrange
        executor_class = get_executor_class("inject_prompt")

        # Act
        metadata = executor_class._executor_metadata

        # Assert
        assert metadata["name"] == "inject_prompt"
        assert "inject" in metadata["description"].lower() or "adversarial" in metadata["description"].lower()


class TestSessionManagerInjection:
    """Test that session manager can be injected into InjectPromptExecutor."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode with initial transcript."""
        push_time = datetime.utcnow().isoformat()
        return Episode(
            episode_id="ep-blue-integration",
            task_id="blue-task",
            session_id="session-integration",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                    {"role": "assistant", "content": "Hello! How can I help?"},
                ],
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.fixture
    def red_episode(self) -> Episode:
        """Create a red team episode."""
        return Episode(
            episode_id="ep-red-integration",
            task_id="red-task",
            session_id="session-integration",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-integration"],
            },
        )

    def test_session_manager_can_be_set(self, blue_episode, red_episode):
        """Test that session_manager can be set on executor instance."""
        # Arrange
        mock_sandbox, _ = create_mock_sandbox_manager()
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor_class = get_executor_class("inject_prompt")
        executor = executor_class.create_with_config(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        # Assert
        assert executor._session_manager is not None
        assert hasattr(executor._session_manager, "episode_manager")

    @pytest.mark.asyncio
    async def test_cross_episode_access_via_session_manager(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test that red team can resolve blue team episode via session manager."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor_class = get_executor_class("inject_prompt")
        executor = executor_class.create_with_config(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        parameters = {
            "message": "Cross-episode test injection",
        }
        context = {
            "session_id": "session-integration",
            "episode_id": "ep-red-integration",
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["target_episode_id"] == "ep-blue-integration"

        # Verify the curl command was called with the correct target
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload_str = curl_cmd[d_index + 1]
        payload = json.loads(payload_str)
        assert payload["target_episode_id"] == "ep-blue-integration"
        assert payload["message"] == "Cross-episode test injection"
