"""
Tests for refactored InjectPromptExecutor - IPC-based WebSocket injection.

These tests verify that the executor sends injections via IPC to the WebSocket
daemon running in the red team container, rather than directly modifying memory.
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch

from saber.server.base import CommandResult, Episode, EpisodeState
from saber.server.execution.executors.standard_registry.inject_prompt_executor import InjectPromptExecutor
from saber.models.constants import MetadataKeys


class TestInjectPromptExecutorIPC:
    """Test InjectPromptExecutor using IPC to WebSocket daemon."""

    @pytest.fixture
    def red_episode(self) -> Episode:
        """Create a red team episode with target metadata."""
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
    def mock_sandbox_manager(self):
        """Create a mock sandbox manager with environment that simulates curl execution."""
        # Create mock environment
        mock_env = Mock()
        mock_env.execute_command = AsyncMock()

        # Create mock manager
        manager = Mock()
        manager.get_episode_environment = Mock(return_value=mock_env)

        return manager

    @pytest.fixture
    def mock_session_manager(self, red_episode: Episode):
        """Create a mock session manager for target resolution."""
        class MockEpisodeManager:
            def get_episode_by_id(self, episode_id: str):
                if episode_id == red_episode.episode_id:
                    return red_episode
                return None

        class MockSessionManager:
            def __init__(self):
                self.episode_manager = MockEpisodeManager()

        return MockSessionManager()

    @pytest.mark.asyncio
    async def test_executor_sends_ipc_request(self, mock_sandbox_manager, mock_session_manager, red_episode):
        """Test executor sends HTTP POST to daemon IPC via curl."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager
        )

        # Get the mock environment
        mock_env = mock_sandbox_manager.get_episode_environment(red_episode.episode_id)

        # Mock successful daemon response
        daemon_response = {
            "success": True,
            "version": 1,
            "message_count": 3
        }
        mock_result = Mock()
        mock_result.stdout = json.dumps(daemon_response)
        mock_result.stderr = ""
        mock_result.exit_code = 0
        mock_env.execute_command.return_value = mock_result

        # Act
        result = await executor.execute(
            parameters={"message": "Test injection", "strategy": "append"},
            context={"episode_id": red_episode.episode_id, "session_id": "session-789"}
        )

        # Assert
        assert result.success is True
        # The result message now shows injection status (inject-and-wait)
        assert "Injection" in result.data["message"] or "success" in str(result.data).lower()

        # Verify execute_command was called (now called twice: wait_for_user + inject_and_wait)
        assert mock_env.execute_command.call_count == 2

        # Get the inject_and_wait call (second call)
        inject_call = mock_env.execute_command.call_args_list[1]
        call_kwargs = inject_call[1]

        assert "curl" in call_kwargs["command"]
        assert "http://localhost:9999/inject_and_wait" in call_kwargs["command"]

        # Verify JSON payload contains injection data
        command_list = call_kwargs["command"]
        command_str = " ".join(str(c) for c in command_list)
        assert "Test injection" in command_str
        assert "ep-blue-123" in command_str  # target episode ID
        assert "append" in command_str

    @pytest.mark.asyncio
    async def test_executor_sends_restart_injection(self, mock_sandbox_manager, mock_session_manager, red_episode):
        """Test executor sends restart injection.

        The restart strategy resets the transcript to the initial state
        (system->user->assistant) and then appends the new message.
        """
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager
        )

        mock_env = mock_sandbox_manager.get_episode_environment(red_episode.episode_id)

        daemon_response = {"success": True, "version": 3, "message_count": 4}
        mock_result = Mock()
        mock_result.stdout = json.dumps(daemon_response)
        mock_result.exit_code = 0
        mock_env.execute_command.return_value = mock_result

        # Act
        result = await executor.execute(
            parameters={"message": "Fresh start!", "strategy": "restart"},
            context={"episode_id": red_episode.episode_id, "session_id": "session-789"}
        )

        # Assert
        assert result.success is True

        # Verify restart strategy was included in payload
        call_kwargs = mock_env.execute_command.call_args[1]
        command_str = " ".join(str(c) for c in call_kwargs["command"])
        assert "restart" in command_str
        assert "Fresh start!" in command_str

    @pytest.mark.asyncio
    async def test_executor_handles_daemon_error(self, mock_sandbox_manager, mock_session_manager, red_episode):
        """Test executor handles daemon error responses."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager
        )

        mock_env = mock_sandbox_manager.get_episode_environment(red_episode.episode_id)

        # Mock daemon returning error
        daemon_response = {"success": False, "error": "Target episode not found"}
        mock_result = Mock()
        mock_result.stdout = json.dumps(daemon_response)
        mock_result.exit_code = 0
        mock_env.execute_command.return_value = mock_result

        # Act
        result = await executor.execute(
            parameters={"message": "Test", "strategy": "append"},
            context={"episode_id": red_episode.episode_id, "session_id": "session-789"}
        )

        # Assert
        assert result.success is False
        assert "Target episode not found" in result.error

    @pytest.mark.asyncio
    async def test_executor_handles_invalid_json_response(self, mock_sandbox_manager, mock_session_manager, red_episode):
        """Test executor handles invalid JSON from daemon."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager
        )

        mock_env = mock_sandbox_manager.get_episode_environment(red_episode.episode_id)

        # Mock daemon returning invalid JSON
        mock_result = Mock()
        mock_result.stdout = "Invalid JSON response"
        mock_result.exit_code = 0
        mock_env.execute_command.return_value = mock_result

        # Act
        result = await executor.execute(
            parameters={"message": "Test", "strategy": "append"},
            context={"episode_id": red_episode.episode_id, "session_id": "session-789"}
        )

        # Assert
        assert result.success is False
        # Error message format changed - now includes "Invalid JSON" from the wait_for_user path
        assert "Invalid JSON" in result.error or "parse" in result.error.lower()

    @pytest.mark.asyncio
    async def test_executor_validates_missing_message(self, mock_sandbox_manager, mock_session_manager, red_episode):
        """Test executor validates required message parameter."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager
        )

        # Act - no message parameter
        result = await executor.execute(
            parameters={"strategy": "append"},
            context={"episode_id": red_episode.episode_id, "session_id": "session-789"}
        )

        # Assert
        assert result.success is False
        assert "Missing required field: message" in result.error

    @pytest.mark.asyncio
    async def test_executor_validates_invalid_strategy(self, mock_sandbox_manager, mock_session_manager, red_episode):
        """Test executor validates strategy parameter."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager
        )

        # Act - invalid strategy
        result = await executor.execute(
            parameters={"message": "Test", "strategy": "invalid_strategy"},
            context={"episode_id": red_episode.episode_id, "session_id": "session-789"}
        )

        # Assert
        assert result.success is False
        assert "Invalid strategy" in result.error

    @pytest.mark.asyncio
    async def test_executor_requires_sandbox_manager(self, mock_session_manager, red_episode):
        """Test executor raises error if sandbox manager is None."""
        # Arrange & Act & Assert
        with pytest.raises(Exception) as exc_info:
            executor = InjectPromptExecutor(
                sandbox_manager=None,
                session_manager=mock_session_manager
            )

        assert "sandbox_manager is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_executor_handles_missing_context(self, mock_sandbox_manager, mock_session_manager):
        """Test executor handles missing episode_id or session_id."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager
        )

        # Act - missing episode_id
        result = await executor.execute(
            parameters={"message": "Test", "strategy": "append"},
            context={"session_id": "session-789"}
        )

        # Assert
        assert result.success is False
        assert "Missing episode_id or session_id" in result.error

    @pytest.mark.asyncio
    async def test_executor_resolves_target_episode_id(self, mock_sandbox_manager, mock_session_manager, red_episode):
        """Test executor resolves target episode ID from orchestration metadata."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox_manager,
            session_manager=mock_session_manager
        )

        mock_env = mock_sandbox_manager.get_episode_environment(red_episode.episode_id)

        daemon_response = {"success": True, "version": 1, "message_count": 1}
        mock_result = Mock()
        mock_result.stdout = json.dumps(daemon_response)
        mock_result.exit_code = 0
        mock_env.execute_command.return_value = mock_result

        # Act
        result = await executor.execute(
            parameters={"message": "Test", "strategy": "append"},
            context={"episode_id": red_episode.episode_id, "session_id": "session-789"}
        )

        # Assert
        assert result.success is True

        # Verify target episode ID was resolved from metadata
        call_kwargs = mock_env.execute_command.call_args[1]
        command_str = " ".join(str(c) for c in call_kwargs["command"])
        assert "ep-blue-123" in command_str  # target from metadata
        command_str = " ".join(call_kwargs["command"])
        assert "ep-blue-123" in command_str  # from ORCHESTRATION_TARGET_EPISODES
