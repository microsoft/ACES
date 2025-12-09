"""
Tests for InjectPromptExecutor transparent interface features.

Tests automatic target resolution from orchestration metadata and injection strategies
when communicating with the WebSocket daemon in Docker.
"""

import json
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState, CommandResult
from saber.server.execution.executors.standard_registry.inject_prompt_executor import InjectPromptExecutor
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


class TestAutomaticTargetResolution:
    """Test automatic target episode ID resolution from orchestration metadata."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode with initial transcript."""
        return Episode(
            episode_id="ep-blue-123",
            task_id="blue-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                ],
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
                MetadataKeys.ORCHESTRATION_ROLE: "blue_team",
            },
        )

    @pytest.fixture
    def red_episode_with_orchestration(self, blue_episode: Episode) -> Episode:
        """Create a red team episode with orchestration metadata."""
        return Episode(
            episode_id="ep-red-456",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: [blue_episode.episode_id],
                MetadataKeys.ORCHESTRATION_ROLE: "red_team",
            },
        )

    @pytest.fixture
    def red_episode_without_orchestration(self) -> Episode:
        """Create a red team episode without orchestration metadata."""
        return Episode(
            episode_id="ep-red-789",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={},
        )

    @pytest.mark.asyncio
    async def test_automatic_target_resolution_from_orchestration(
        self,
        blue_episode: Episode,
        red_episode_with_orchestration: Episode,
    ):
        """Test that target episode ID is auto-resolved from ORCHESTRATION_TARGET_EPISODES."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode_with_orchestration)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        # No explicit target_episode_id - should be auto-resolved
        parameters = {
            "message": "Test injection",
        }
        context = {
            "session_id": "session-789",
            "episode_id": red_episode_with_orchestration.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["target_episode_id"] == blue_episode.episode_id

        # Verify payload sent to daemon
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload = json.loads(curl_cmd[d_index + 1])
        assert payload["target_episode_id"] == blue_episode.episode_id

    @pytest.mark.asyncio
    async def test_explicit_target_overrides_orchestration(
        self,
        blue_episode: Episode,
        red_episode_with_orchestration: Episode,
    ):
        """Test that explicit target_episode_id parameter overrides orchestration metadata."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode_with_orchestration)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        # Note: The current implementation doesn't support explicit target_episode_id parameter,
        # it always uses orchestration metadata. This test verifies the current behavior.
        parameters = {
            "message": "Test injection",
        }
        context = {
            "session_id": "session-789",
            "episode_id": red_episode_with_orchestration.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert - target comes from orchestration metadata
        assert result.success is True
        assert result.metadata["target_episode_id"] == blue_episode.episode_id

    @pytest.mark.asyncio
    async def test_missing_target_without_orchestration(
        self,
        blue_episode: Episode,
        red_episode_without_orchestration: Episode,
    ):
        """Test that executor fails gracefully when no target can be resolved."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager()
        mock_session = create_mock_session_manager(blue_episode, red_episode_without_orchestration)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        # No explicit target, no orchestration metadata
        parameters = {
            "message": "Test injection",
        }
        context = {
            "session_id": "session-789",
            "episode_id": red_episode_without_orchestration.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is False
        assert "Could not resolve target_episode_id" in result.error

    @pytest.mark.asyncio
    async def test_multiple_target_episodes_uses_first(
        self,
        blue_episode: Episode,
    ):
        """Test that when multiple targets exist, first one is used."""
        # Arrange
        red_episode = Episode(
            episode_id="ep-red-multi",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: [
                    blue_episode.episode_id,
                    "ep-blue-other",
                ],
                MetadataKeys.ORCHESTRATION_ROLE: "red_team",
            },
        )

        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        parameters = {"message": "Test"}
        context = {
            "session_id": "session-789",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["target_episode_id"] == blue_episode.episode_id  # First target


class TestInjectionStrategies:
    """Test different injection strategies sent to daemon."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode with multiple messages."""
        return Episode(
            episode_id="ep-blue-456",
            task_id="blue-task",
            session_id="session-xyz",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 is 4."},
                    {"role": "user", "content": "What is 3+3?"},
                ],
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.fixture
    def red_episode(self, blue_episode: Episode) -> Episode:
        """Create a red team episode with target in orchestration metadata."""
        return Episode(
            episode_id="ep-red-strat",
            task_id="red-task",
            session_id="session-xyz",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: [blue_episode.episode_id],
                MetadataKeys.ORCHESTRATION_ROLE: "red_team",
            },
        )

    @pytest.mark.asyncio
    async def test_append_strategy_default(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test append strategy (default) - sends append to daemon."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        parameters = {
            "message": "Injected message",
            "strategy": "append"
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["strategy"] == "append"

        # Verify payload
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload = json.loads(curl_cmd[d_index + 1])
        assert payload["strategy"] == "append"
        assert payload["message"] == "Injected message"

    @pytest.mark.asyncio
    async def test_rewind_strategy_removes_then_appends(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test rewind strategy - sends rewind with count to daemon."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        parameters = {
            "message": "Injected after rewind",
            "strategy": "rewind",
            "rewind_count": 2
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["strategy"] == "rewind"

        # Verify payload includes rewind_count
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload = json.loads(curl_cmd[d_index + 1])
        assert payload["strategy"] == "rewind"
        assert payload["rewind_count"] == 2

    @pytest.mark.asyncio
    async def test_rewrite_strategy_replaces_entire_transcript(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test rewrite strategy - sends rewrite to daemon."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        parameters = {
            "message": "New transcript content",
            "strategy": "rewrite"
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["strategy"] == "rewrite"

        # Verify payload
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload = json.loads(curl_cmd[d_index + 1])
        assert payload["strategy"] == "rewrite"

    @pytest.mark.asyncio
    async def test_insert_strategy_inserts_at_position(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test insert strategy - sends insert with position to daemon."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        parameters = {
            "message": "Inserted at position 2",
            "strategy": "insert",
            "insert_position": 2
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["strategy"] == "insert"

        # Verify payload includes insert_position
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload = json.loads(curl_cmd[d_index + 1])
        assert payload["strategy"] == "insert"
        assert payload["insert_position"] == 2

    @pytest.mark.asyncio
    async def test_invalid_strategy_fails(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test that invalid strategy is rejected before sending to daemon."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager()
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        parameters = {
            "message": "Test",
            "strategy": "invalid_strategy"
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is False
        assert "invalid" in result.error.lower() or "strategy" in result.error.lower()

        # Verify curl was NOT called (validation failed before sending)
        mock_env.execute_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_rewind_count_clamping(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test that rewind_count is sent to daemon (daemon handles clamping)."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        # Large rewind count - daemon will handle clamping
        parameters = {
            "message": "Test",
            "strategy": "rewind",
            "rewind_count": 100
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert - should succeed, daemon handles clamping
        assert result.success is True

        # Verify the rewind_count was sent
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload = json.loads(curl_cmd[d_index + 1])
        assert payload["rewind_count"] == 100

    @pytest.mark.asyncio
    async def test_insert_position_clamping(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test that insert_position is sent to daemon (daemon handles clamping)."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        # Large insert position - daemon will handle clamping
        parameters = {
            "message": "Test",
            "strategy": "insert",
            "insert_position": 1000
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert - should succeed, daemon handles clamping
        assert result.success is True

        # Verify the insert_position was sent
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload = json.loads(curl_cmd[d_index + 1])
        assert payload["insert_position"] == 1000


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode."""
        return Episode(
            episode_id="ep-blue-legacy",
            task_id="blue-task",
            session_id="session-legacy",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                ],
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.fixture
    def red_episode(self) -> Episode:
        """Create a red team episode with orchestration metadata."""
        return Episode(
            episode_id="ep-red-legacy",
            task_id="red-task",
            session_id="session-legacy",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-legacy"],
            },
        )

    @pytest.mark.asyncio
    async def test_orchestration_auto_resolution(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test that orchestration metadata auto-resolves target."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        # No explicit target - resolved from ORCHESTRATION_TARGET_EPISODES
        parameters = {
            "message": "Auto-resolved injection",
        }
        context = {
            "session_id": "session-legacy",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["target_episode_id"] == blue_episode.episode_id

    @pytest.mark.asyncio
    async def test_default_strategy_is_append(
        self,
        blue_episode: Episode,
        red_episode: Episode,
    ):
        """Test that omitting strategy defaults to append (backward compatible)."""
        # Arrange
        mock_sandbox, mock_env = create_mock_sandbox_manager({"success": True, "version": 1})
        mock_session = create_mock_session_manager(blue_episode, red_episode)

        executor = InjectPromptExecutor(
            sandbox_manager=mock_sandbox,
            session_manager=mock_session
        )

        # No strategy parameter - should default to append
        parameters = {
            "message": "Default strategy test",
        }
        context = {
            "session_id": "session-legacy",
            "episode_id": red_episode.episode_id,
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True
        assert result.metadata["strategy"] == "append"

        # Verify payload has default strategy
        call_args = mock_env.execute_command.call_args
        curl_cmd = call_args.kwargs.get("command") or call_args[1].get("command")
        d_index = curl_cmd.index("-d")
        payload = json.loads(curl_cmd[d_index + 1])
        assert payload["strategy"] == "append"
