"""
End-to-end tests for session manager injection through the full stack.

Tests that session_manager flows from SessionManager → ExecutionManager →
ExecutorFactory → InjectPromptExecutor properly.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState
from saber.server.execution.execution_manager import ExecutionManager


class TestSessionManagerInjectionE2E:
    """Test session manager injection through the complete stack."""

    @pytest.fixture
    def execution_manager(self, tmp_path):
        """Create an ExecutionManager instance."""
        config_dir = str(tmp_path / "config")
        return ExecutionManager(config_dir)

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager with episode_manager."""
        from unittest.mock import AsyncMock

        session_mgr = Mock()
        session_mgr.episode_manager = Mock()

        # Mock transcript coordinator with async notify_modification
        transcript_coordinator = Mock()
        transcript_coordinator.notify_modification = AsyncMock()
        session_mgr.episode_manager.transcript_coordinator = transcript_coordinator

        # Create blue and red episodes
        blue_episode = Episode(
            episode_id="ep-blue-e2e",
            task_id="blue-task",
            session_id="session-e2e",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                ],
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: datetime.utcnow().isoformat(),
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

        red_episode = Episode(
            episode_id="ep-red-e2e",
            task_id="red-task",
            session_id="session-e2e",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-e2e"],
            },
        )

        # Mock episode lookup
        def get_episode_by_id(episode_id):
            if episode_id == "ep-blue-e2e":
                return blue_episode
            elif episode_id == "ep-red-e2e":
                return red_episode
            return None

        session_mgr.episode_manager.get_episode_by_id = Mock(side_effect=get_episode_by_id)

        return session_mgr

    def test_session_manager_injection_flow(self, execution_manager, mock_session_manager):
        """Test that session_manager flows through the stack."""
        # Arrange - inject session manager into execution manager
        execution_manager.set_session_manager(mock_session_manager)

        # Verify it was stored
        assert execution_manager._session_manager is mock_session_manager

        # Factory should be cleared to pick up session_manager on next access
        assert execution_manager._executor_factory is None

    def test_executor_receives_session_manager(self, execution_manager, mock_session_manager, tmp_path):
        """Test that executor created by factory receives session_manager."""
        # Arrange
        execution_manager.set_session_manager(mock_session_manager)

        # Need to initialize sandbox manager for factory to work
        from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager

        sandbox_config = {
            "domain": "test",
            "config_dir": str(tmp_path / "config"),
            "logs_dir": str(tmp_path / "logs"),
            "enable_container_logging": False,
        }
        sandbox_manager = SandboxEnvironmentManager(sandbox_config)
        execution_manager._sandbox_environment_manager = sandbox_manager

        # Act - get inject_prompt executor through factory
        executor = execution_manager.get_executor("inject_prompt")

        # Assert - executor should have session_manager injected
        assert hasattr(executor, "_session_manager")
        assert executor._session_manager is mock_session_manager

    @pytest.mark.asyncio
    async def test_full_injection_flow_with_execution(
        self, execution_manager, mock_session_manager, tmp_path
    ):
        """Test full flow: SessionManager → ExecutionManager → Factory → Executor → Episode modification."""
        from unittest.mock import AsyncMock

        # Arrange
        execution_manager.set_session_manager(mock_session_manager)

        # Create a mock sandbox manager that returns a mock orchestrator
        mock_sandbox_manager = Mock()
        mock_orchestrator = Mock()
        # Mock execute_command to return a successful curl response
        mock_orchestrator.execute_command = AsyncMock(return_value=Mock(
            exit_code=0,
            stdout='{"success": true}',
            stderr='',
        ))
        mock_sandbox_manager.get_episode_environment = Mock(return_value=mock_orchestrator)
        execution_manager._sandbox_environment_manager = mock_sandbox_manager

        # Get the executor
        executor = execution_manager.get_executor("inject_prompt")

        # Prepare execution parameters
        parameters = {
            "message": "E2E test injection",
        }
        context = {
            "session_id": "session-e2e",
            "episode_id": "ep-red-e2e",
        }

        # Act - execute the injection
        result = await executor.execute(parameters, context)

        # Assert - injection succeeded
        assert result.success is True
