"""Tests for ExecutionManager.copy_file_to_episode method.

Tests the dynamic file upload functionality integrated into the ExecutionManager.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.execution.execution_manager import ExecutionManager
from saber.server.execution.sandbox.file_copier import FileUploadResult


class TestCopyFileToEpisode:
    """Test cases for ExecutionManager.copy_file_to_episode."""

    @pytest.fixture
    def temp_base_dir(self, tmp_path: Path) -> Path:
        """Create a temporary base directory with test files."""
        base_dir = tmp_path
        # Config dir is a child
        config_dir = base_dir / "config"
        config_dir.mkdir()

        # Create data/ subdirectory with files
        data_dir = base_dir / "data"
        data_dir.mkdir()
        (data_dir / "test_file.txt").write_text("Hello, World!")

        return base_dir

    @pytest.fixture
    def mock_execution_manager(
        self, temp_base_dir: Path
    ) -> ExecutionManager:
        """Create an ExecutionManager instance with mocked dependencies."""
        config_dir = str(temp_base_dir / "config")

        with patch(
            "saber.server.execution.execution_manager.SandboxEnvironmentManager"
        ) as mock_sandbox_class:
            mock_sandbox_instance = MagicMock()
            mock_sandbox_instance.is_ready.return_value = True
            mock_sandbox_instance.get_execution_container_name.return_value = "default-episode-123"
            mock_sandbox_class.return_value = mock_sandbox_instance

            manager = ExecutionManager(config_dir)
            manager._sandbox_environment_manager = mock_sandbox_instance

            return manager

    @pytest.mark.asyncio
    async def test_copy_file_success(
        self,
        mock_execution_manager: ExecutionManager,
    ) -> None:
        """Test successful file copy to episode."""
        # Mock the file copier
        mock_result = FileUploadResult(
            success=True,
            source_path="data/test_file.txt",
            destination_path="/root/test_file.txt",
            bytes_copied=100,
            is_directory=False,
            error_message=None,
        )
        mock_execution_manager._file_copier = MagicMock()
        mock_execution_manager._file_copier.copy_file_to_episode_dynamic = AsyncMock(
            return_value=mock_result
        )

        result = await mock_execution_manager.copy_file_to_episode(
            episode_id="episode-123",
            source_path="data/test_file.txt",
            destination_path="/root/test_file.txt",
        )

        assert result.success is True
        assert result.bytes_copied == 100
        mock_execution_manager._file_copier.copy_file_to_episode_dynamic.assert_called_once_with(
            episode_id="episode-123",
            source_path="data/test_file.txt",
            destination_path="/root/test_file.txt",
            container_name="default-episode-123",
        )

    @pytest.mark.asyncio
    async def test_copy_file_with_custom_container(
        self,
        mock_execution_manager: ExecutionManager,
    ) -> None:
        """Test file copy with custom container name."""
        mock_result = FileUploadResult(
            success=True,
            source_path="data/test_file.txt",
            destination_path="/root/test.txt",
            bytes_copied=50,
            is_directory=False,
            error_message=None,
        )
        mock_execution_manager._file_copier = MagicMock()
        mock_execution_manager._file_copier.copy_file_to_episode_dynamic = AsyncMock(
            return_value=mock_result
        )

        result = await mock_execution_manager.copy_file_to_episode(
            episode_id="episode-456",
            source_path="data/test_file.txt",
            destination_path="/root/test.txt",
            container_name="custom-container",
        )

        assert result.success is True
        mock_execution_manager._file_copier.copy_file_to_episode_dynamic.assert_called_once_with(
            episode_id="episode-456",
            source_path="data/test_file.txt",
            destination_path="/root/test.txt",
            container_name="custom-container",
        )

    @pytest.mark.asyncio
    async def test_copy_file_no_file_copier(
        self,
        mock_execution_manager: ExecutionManager,
    ) -> None:
        """Test error when file copier is not initialized."""
        mock_execution_manager._file_copier = None

        with pytest.raises(RuntimeError, match="File copier not initialized"):
            await mock_execution_manager.copy_file_to_episode(
                episode_id="episode-123",
                source_path="data/test_file.txt",
                destination_path="/root/test.txt",
            )

    @pytest.mark.asyncio
    async def test_copy_file_no_container_name(
        self,
        mock_execution_manager: ExecutionManager,
    ) -> None:
        """Test error when execution container cannot be determined."""
        mock_execution_manager._sandbox_environment_manager.get_execution_container_name.return_value = None
        mock_execution_manager._file_copier = MagicMock()

        with pytest.raises(RuntimeError, match="Cannot determine execution container"):
            await mock_execution_manager.copy_file_to_episode(
                episode_id="episode-999",
                source_path="data/test_file.txt",
                destination_path="/root/test.txt",
            )

    @pytest.mark.asyncio
    async def test_copy_file_propagates_errors(
        self,
        mock_execution_manager: ExecutionManager,
    ) -> None:
        """Test that errors from file copier are propagated."""
        mock_execution_manager._file_copier = MagicMock()
        mock_execution_manager._file_copier.copy_file_to_episode_dynamic = AsyncMock(
            side_effect=FileNotFoundError("Source file not found")
        )

        with pytest.raises(FileNotFoundError, match="Source file not found"):
            await mock_execution_manager.copy_file_to_episode(
                episode_id="episode-123",
                source_path="data/missing.txt",
                destination_path="/root/missing.txt",
            )

    @pytest.mark.asyncio
    async def test_copy_file_propagates_value_errors(
        self,
        mock_execution_manager: ExecutionManager,
    ) -> None:
        """Test that validation errors from file copier are propagated."""
        mock_execution_manager._file_copier = MagicMock()
        mock_execution_manager._file_copier.copy_file_to_episode_dynamic = AsyncMock(
            side_effect=ValueError("Invalid path")
        )

        with pytest.raises(ValueError, match="Invalid path"):
            await mock_execution_manager.copy_file_to_episode(
                episode_id="episode-123",
                source_path="/absolute/path.txt",
                destination_path="/root/test.txt",
            )
