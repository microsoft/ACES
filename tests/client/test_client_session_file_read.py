"""Tests for ClientSessionManager file read functionality.

Tests the read_sandbox_file method that retrieves files from
episode sandbox containers via REST API.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from saber.client.client_session import ClientSessionManager
from saber.client.models import SessionManagerConfig


class TestClientSessionManagerReadFile:
    """Test cases for ClientSessionManager.read_sandbox_file."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SessionManagerConfig(
            base_url="http://localhost:8000",
            mcp_server_url="http://localhost:8001/mcp",
            client_id="test-client",
            rest_timeout=30.0,
            rest_max_retries=3,
        )

    @pytest.fixture
    def client_manager(self, config):
        """Create a ClientSessionManager instance."""
        return ClientSessionManager(config)

    @pytest.mark.asyncio
    async def test_read_file_success(self, client_manager) -> None:
        """Test successful file read."""
        mock_response = {
            "message": "File read successfully",
            "session_id": "sess-123",
            "episode_id": "ep-456",
            "file_path": "/tmp/test.json",
            "content": '{"key": "value"}',
            "bytes_read": 16,
            "encoding": "utf-8",
        }

        with patch.object(client_manager, "_retry_request") as mock_retry:
            mock_retry.return_value = (200, mock_response)

            content = await client_manager.read_sandbox_file(
                session_id="sess-123",
                episode_id="ep-456",
                file_path="/tmp/test.json",
            )

            assert content == '{"key": "value"}'
            mock_retry.assert_called_once()
            call_args = mock_retry.call_args
            assert call_args[0][0] == "read_sandbox_file"

    @pytest.mark.asyncio
    async def test_read_file_with_container_name(self, client_manager) -> None:
        """Test file read with explicit container name."""
        mock_response = {
            "message": "File read successfully",
            "session_id": "sess-123",
            "episode_id": "ep-456",
            "file_path": "/app/config.json",
            "content": "{}",
            "bytes_read": 2,
            "encoding": "utf-8",
        }

        with patch.object(client_manager, "_retry_request") as mock_retry:
            mock_retry.return_value = (200, mock_response)

            content = await client_manager.read_sandbox_file(
                session_id="sess-123",
                episode_id="ep-456",
                file_path="/app/config.json",
                container_name="custom-container",
            )

            assert content == "{}"

    @pytest.mark.asyncio
    async def test_read_file_with_custom_encoding(self, client_manager) -> None:
        """Test file read with custom encoding."""
        mock_response = {
            "message": "File read successfully",
            "session_id": "sess-123",
            "episode_id": "ep-456",
            "file_path": "/tmp/test.txt",
            "content": "Héllo",
            "bytes_read": 5,
            "encoding": "latin-1",
        }

        with patch.object(client_manager, "_retry_request") as mock_retry:
            mock_retry.return_value = (200, mock_response)

            content = await client_manager.read_sandbox_file(
                session_id="sess-123",
                episode_id="ep-456",
                file_path="/tmp/test.txt",
                encoding="latin-1",
            )

            assert content == "Héllo"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, client_manager) -> None:
        """Test FileNotFoundError when file doesn't exist."""
        with patch.object(client_manager, "_retry_request") as mock_retry:
            mock_retry.return_value = (404, {"detail": "File not found: /tmp/missing.json"})

            with pytest.raises(FileNotFoundError, match="missing.json"):
                await client_manager.read_sandbox_file(
                    session_id="sess-123",
                    episode_id="ep-456",
                    file_path="/tmp/missing.json",
                )

    @pytest.mark.asyncio
    async def test_read_file_episode_not_found(self, client_manager) -> None:
        """Test FileNotFoundError when episode doesn't exist."""
        with patch.object(client_manager, "_retry_request") as mock_retry:
            mock_retry.return_value = (404, {"detail": "Episode ep-456 not found"})

            with pytest.raises(FileNotFoundError, match="Episode ep-456 not found"):
                await client_manager.read_sandbox_file(
                    session_id="sess-123",
                    episode_id="ep-456",
                    file_path="/tmp/test.json",
                )

    @pytest.mark.asyncio
    async def test_read_file_episode_not_ready(self, client_manager) -> None:
        """Test exception when episode is not ready."""
        with patch.object(client_manager, "_retry_request") as mock_retry:
            mock_retry.return_value = (422, {"detail": "Episode ep-456 is not ready. Current state: creating"})

            with pytest.raises(Exception, match="not ready"):
                await client_manager.read_sandbox_file(
                    session_id="sess-123",
                    episode_id="ep-456",
                    file_path="/tmp/test.json",
                )

    @pytest.mark.asyncio
    async def test_read_file_size_exceeded(self, client_manager) -> None:
        """Test exception when file size exceeds limit."""
        with patch.object(client_manager, "_retry_request") as mock_retry:
            mock_retry.return_value = (413, {"detail": "File size exceeds maximum allowed"})

            with pytest.raises(Exception, match="exceeds"):
                await client_manager.read_sandbox_file(
                    session_id="sess-123",
                    episode_id="ep-456",
                    file_path="/tmp/large.bin",
                )

    @pytest.mark.asyncio
    async def test_read_file_server_error(self, client_manager) -> None:
        """Test exception on server error."""
        with patch.object(client_manager, "_retry_request") as mock_retry:
            mock_retry.return_value = (500, {"detail": "Internal server error"})

            with pytest.raises(Exception, match="Failed to read file"):
                await client_manager.read_sandbox_file(
                    session_id="sess-123",
                    episode_id="ep-456",
                    file_path="/tmp/test.json",
                )
