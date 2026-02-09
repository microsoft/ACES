"""Tests for ExecutionManager file read functionality.

Tests the read_file_from_episode method that retrieves files from
episode sandbox containers using the SandboxFileCopier.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.execution.execution_manager import ExecutionManager
from saber.server.execution.sandbox.file_copier import FileReadResult


class TestExecutionManagerReadFile:
    """Test cases for ExecutionManager.read_file_from_episode."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        return MagicMock()

    @pytest.fixture
    def execution_manager(self, tmp_path, mock_session_manager):
        """Create an ExecutionManager instance with mocked dependencies."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return ExecutionManager(
            config_dir=str(config_dir),
            session_manager=mock_session_manager,
        )

    @pytest.mark.asyncio
    async def test_read_file_success(self, execution_manager) -> None:
        """Test successful file read from episode."""
        # Setup mock file copier
        mock_file_copier = MagicMock()
        mock_file_copier.read_file_from_container = AsyncMock(
            return_value=FileReadResult(
                success=True,
                file_path="/tmp/test.json",
                content='{"key": "value"}',
                bytes_read=16,
                error_message=None,
            )
        )
        execution_manager._file_copier = mock_file_copier

        # Mock get_execution_container_name
        execution_manager.get_execution_container_name = MagicMock(
            return_value="default-ep-123"
        )

        result = await execution_manager.read_file_from_episode(
            episode_id="ep-123",
            file_path="/tmp/test.json",
        )

        assert result.success is True
        assert result.content == '{"key": "value"}'
        assert result.bytes_read == 16

        mock_file_copier.read_file_from_container.assert_called_once_with(
            episode_id="ep-123",
            file_path="/tmp/test.json",
            container_name="default-ep-123",
            encoding="utf-8",
        )

    @pytest.mark.asyncio
    async def test_read_file_with_custom_container(self, execution_manager) -> None:
        """Test file read with explicit container name."""
        mock_file_copier = MagicMock()
        mock_file_copier.read_file_from_container = AsyncMock(
            return_value=FileReadResult(
                success=True,
                file_path="/app/config.json",
                content='{"setting": true}',
                bytes_read=17,
                error_message=None,
            )
        )
        execution_manager._file_copier = mock_file_copier

        result = await execution_manager.read_file_from_episode(
            episode_id="ep-123",
            file_path="/app/config.json",
            container_name="custom-container",
        )

        assert result.success is True
        mock_file_copier.read_file_from_container.assert_called_once_with(
            episode_id="ep-123",
            file_path="/app/config.json",
            container_name="custom-container",
            encoding="utf-8",
        )

    @pytest.mark.asyncio
    async def test_read_file_with_custom_encoding(self, execution_manager) -> None:
        """Test file read with custom encoding."""
        mock_file_copier = MagicMock()
        mock_file_copier.read_file_from_container = AsyncMock(
            return_value=FileReadResult(
                success=True,
                file_path="/tmp/test.txt",
                content="Héllo",
                bytes_read=5,
                error_message=None,
            )
        )
        execution_manager._file_copier = mock_file_copier
        execution_manager.get_execution_container_name = MagicMock(
            return_value="default-ep-123"
        )

        result = await execution_manager.read_file_from_episode(
            episode_id="ep-123",
            file_path="/tmp/test.txt",
            encoding="latin-1",
        )

        assert result.success is True
        mock_file_copier.read_file_from_container.assert_called_once_with(
            episode_id="ep-123",
            file_path="/tmp/test.txt",
            container_name="default-ep-123",
            encoding="latin-1",
        )

    @pytest.mark.asyncio
    async def test_read_file_copier_not_initialized(self, execution_manager) -> None:
        """Test error when file copier is not initialized."""
        execution_manager._file_copier = None

        with pytest.raises(RuntimeError, match="File copier not initialized"):
            await execution_manager.read_file_from_episode(
                episode_id="ep-123",
                file_path="/tmp/test.json",
            )

    @pytest.mark.asyncio
    async def test_read_file_container_name_cannot_be_determined(
        self, execution_manager
    ) -> None:
        """Test error when container name cannot be determined."""
        mock_file_copier = MagicMock()
        execution_manager._file_copier = mock_file_copier
        execution_manager.get_execution_container_name = MagicMock(return_value=None)

        with pytest.raises(RuntimeError, match="Cannot determine execution container"):
            await execution_manager.read_file_from_episode(
                episode_id="ep-123",
                file_path="/tmp/test.json",
            )

    @pytest.mark.asyncio
    async def test_read_file_file_not_found(self, execution_manager) -> None:
        """Test handling when file is not found."""
        mock_file_copier = MagicMock()
        mock_file_copier.read_file_from_container = AsyncMock(
            return_value=FileReadResult(
                success=False,
                file_path="/tmp/missing.json",
                content="",
                bytes_read=0,
                error_message="File not found: /tmp/missing.json",
            )
        )
        execution_manager._file_copier = mock_file_copier
        execution_manager.get_execution_container_name = MagicMock(
            return_value="default-ep-123"
        )

        result = await execution_manager.read_file_from_episode(
            episode_id="ep-123",
            file_path="/tmp/missing.json",
        )

        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_read_file_propagates_exceptions(self, execution_manager) -> None:
        """Test that exceptions from file copier are propagated."""
        from docker.errors import NotFound

        mock_file_copier = MagicMock()
        mock_file_copier.read_file_from_container = AsyncMock(
            side_effect=NotFound("Container not found")
        )
        execution_manager._file_copier = mock_file_copier
        execution_manager.get_execution_container_name = MagicMock(
            return_value="default-ep-123"
        )

        with pytest.raises(NotFound):
            await execution_manager.read_file_from_episode(
                episode_id="ep-123",
                file_path="/tmp/test.json",
            )
