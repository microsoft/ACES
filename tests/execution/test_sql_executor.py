"""
Tests for the SQL executor.

This module tests the SQLExecutor, focusing on the None-environment error path
when no sandbox environment is available for the episode.
"""

from unittest.mock import MagicMock

import pytest

from saber.server.execution.base import ExecutionContext, SqlParameters
from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.executors.standard_registry.sql_executor import SQLExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestSQLExecutorNoneEnvironment:
    """Tests verifying SQLExecutor handles missing sandbox environment correctly."""

    @pytest.fixture
    def mock_sandbox_manager_no_env(self) -> MagicMock:
        """Create a mock SandboxEnvironmentManager that returns None for get_episode_environment."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {"image": "saber/sandbox:latest"}
        manager.get_episode_environment.return_value = None
        return manager

    @pytest.mark.asyncio
    async def test_sql_execute_returns_error_when_no_environment_and_no_target_container(
        self, mock_sandbox_manager_no_env: MagicMock
    ) -> None:
        """Test that SQLExecutor returns error result when no sandbox environment and no target_container."""
        executor = SQLExecutor(sandbox_manager=mock_sandbox_manager_no_env)
        params = SqlParameters(query="SELECT 1")
        context = ExecutionContext(episode_id="test-episode")

        result = await executor.execute(params=params, context=context)

        assert result.success is False
        assert "No sandbox environment" in str(result.data) or "No sandbox environment" in str(result.metadata)
