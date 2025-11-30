"""Tests for MCP client factory."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.inspect_ai.core.mcp_factory import (
    HEADER_EPISODE_ID,
    HEADER_ORCHESTRATION_ENV,
    HEADER_SESSION_ID,
    HEADER_TASK_ID,
    MCPClientFactory,
)
from saber.models.mcp import OrchestrationEnvironment


@pytest.fixture
def factory():
    """Create MCP client factory."""
    return MCPClientFactory(
        mcp_url_base="http://localhost:8000",
        domain_slug="test-domain",
        timeout=30.0,
    )


@pytest.fixture
def sample_data():
    """Sample test data."""
    return {
        "session_id": "test-session-123",
        "episode_id": "test-episode-456",
        "task_id": "test-task-789",
        "sample_id": "test-sample-abc",
    }


class TestMCPClientFactory:
    """Test MCP client factory."""

    def test_init(self):
        """Test factory initialization."""
        factory = MCPClientFactory(
            mcp_url_base="http://example.com",
            domain_slug="my-domain",
            timeout=60.0,
        )

        assert factory.mcp_url_base == "http://example.com"
        assert factory.domain_slug == "my-domain"
        assert factory.timeout == 60.0

    @pytest.mark.asyncio
    async def test_create_mcp_client_success(self, factory, sample_data):
        """Test successful MCP client creation."""
        with patch("saber.inspect_ai.core.mcp_factory.mcp_server_http") as mock_mcp_server:
            mock_tool = MagicMock()
            mock_mcp_server.return_value = mock_tool

            result = await factory.create_mcp_client(
                session_id=sample_data["session_id"],
                episode_id=sample_data["episode_id"],
                task_id=sample_data["task_id"],
                sample_id=sample_data["sample_id"],
            )

            # Verify mcp_server_http was called with correct parameters
            mock_mcp_server.assert_called_once()
            call_kwargs = mock_mcp_server.call_args[1]

            assert call_kwargs["name"] == f"SABER test-domain Tools - {sample_data['sample_id']}"
            assert call_kwargs["url"] == "http://localhost:8000/mcp"
            assert call_kwargs["timeout"] == 30.0

            # Verify headers
            headers = call_kwargs["headers"]
            assert headers[HEADER_SESSION_ID] == sample_data["session_id"]
            assert headers[HEADER_EPISODE_ID] == sample_data["episode_id"]
            assert headers[HEADER_TASK_ID] == sample_data["task_id"]
            assert headers[HEADER_ORCHESTRATION_ENV] == OrchestrationEnvironment.INSPECT.value

            # Verify result
            assert result == mock_tool

    @pytest.mark.asyncio
    async def test_create_mcp_client_retry_success(self, factory, sample_data):
        """Test MCP client creation succeeds after retry."""
        with patch("saber.inspect_ai.core.mcp_factory.mcp_server_http") as mock_mcp_server:
            mock_tool = MagicMock()
            # First call fails, second succeeds
            mock_mcp_server.side_effect = [Exception("Connection failed"), mock_tool]

            with patch("asyncio.sleep", new_callable=AsyncMock):  # Speed up test
                result = await factory.create_mcp_client(
                    session_id=sample_data["session_id"],
                    episode_id=sample_data["episode_id"],
                    task_id=sample_data["task_id"],
                    sample_id=sample_data["sample_id"],
                    max_retries=3,
                )

            # Should have been called twice (failed once, then succeeded)
            assert mock_mcp_server.call_count == 2
            assert result == mock_tool

    @pytest.mark.asyncio
    async def test_create_mcp_client_retry_failure(self, factory, sample_data):
        """Test MCP client creation fails after all retries."""
        with patch("saber.inspect_ai.core.mcp_factory.mcp_server_http") as mock_mcp_server:
            # All calls fail
            mock_mcp_server.side_effect = Exception("Connection failed")

            with patch("asyncio.sleep", new_callable=AsyncMock):  # Speed up test
                with pytest.raises(Exception) as exc_info:
                    await factory.create_mcp_client(
                        session_id=sample_data["session_id"],
                        episode_id=sample_data["episode_id"],
                        task_id=sample_data["task_id"],
                        sample_id=sample_data["sample_id"],
                        max_retries=3,
                    )

            # Should have been called 3 times
            assert mock_mcp_server.call_count == 3
            assert "Failed to create MCP client after 3 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_mcp_client_exponential_backoff(self, factory, sample_data):
        """Test exponential backoff during retries."""
        with patch("saber.inspect_ai.core.mcp_factory.mcp_server_http") as mock_mcp_server:
            mock_tool = MagicMock()
            # Fail twice, succeed on third
            mock_mcp_server.side_effect = [
                Exception("Fail 1"),
                Exception("Fail 2"),
                mock_tool,
            ]

            sleep_times = []

            async def track_sleep(duration):
                sleep_times.append(duration)

            with patch("asyncio.sleep", side_effect=track_sleep):
                result = await factory.create_mcp_client(
                    session_id=sample_data["session_id"],
                    episode_id=sample_data["episode_id"],
                    task_id=sample_data["task_id"],
                    sample_id=sample_data["sample_id"],
                    max_retries=3,
                )

            # Verify exponential backoff: 2^0=1s, 2^1=2s
            assert sleep_times == [1, 2]
            assert result == mock_tool

    @pytest.mark.asyncio
    async def test_create_mcp_client_unique_naming(self, factory):
        """Test that each sample gets a unique MCP client name."""
        with patch("saber.inspect_ai.core.mcp_factory.mcp_server_http") as mock_mcp_server:
            mock_mcp_server.return_value = MagicMock()

            # Create clients for different samples
            await factory.create_mcp_client("session", "episode", "task", "sample-1")
            await factory.create_mcp_client("session", "episode", "task", "sample-2")

            # Verify different names were used
            calls = mock_mcp_server.call_args_list
            name1 = calls[0][1]["name"]
            name2 = calls[1][1]["name"]

            assert name1 == "SABER test-domain Tools - sample-1"
            assert name2 == "SABER test-domain Tools - sample-2"
            assert name1 != name2

    @pytest.mark.asyncio
    async def test_cleanup_mcp_cache_success(self, factory):
        """Test successful MCP cache cleanup."""
        from inspect_ai.tool._mcp._local import MCPServerLocal

        # Mock the cache and session
        mock_session = MagicMock()
        mock_session._session = MagicMock()
        mock_session.__aexit__ = AsyncMock()

        # Create a fake cache entry
        sample_id = "test-sample-123"
        task_id = 12345
        server_name = f"SABER test-domain Tools - {sample_id}"
        cache_key = f"{task_id}_{server_name}"

        MCPServerLocal._task_sessions[cache_key] = mock_session

        with patch("anyio.get_current_task") as mock_get_task:
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_get_task.return_value = mock_task

            # Cleanup the cache
            await factory.cleanup_mcp_cache(sample_id)

        # Verify cache was cleaned
        assert cache_key not in MCPServerLocal._task_sessions
        mock_session.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_cleanup_mcp_cache_not_in_cache(self, factory):
        """Test cleanup when sample is not in cache."""
        from inspect_ai.tool._mcp._local import MCPServerLocal

        # Ensure cache is empty
        MCPServerLocal._task_sessions.clear()

        sample_id = "non-existent-sample"

        with patch("anyio.get_current_task") as mock_get_task:
            mock_task = MagicMock()
            mock_task.id = 12345
            mock_get_task.return_value = mock_task

            # Should not raise exception
            await factory.cleanup_mcp_cache(sample_id)

        # Cache should still be empty
        assert len(MCPServerLocal._task_sessions) == 0

    @pytest.mark.asyncio
    async def test_cleanup_mcp_cache_session_without_session_attr(self, factory):
        """Test cleanup when cached session has no _session attribute."""
        from inspect_ai.tool._mcp._local import MCPServerLocal

        # Mock the cache with a session that has no _session attribute
        mock_session = MagicMock()  # Basic mock without _session attribute
        delattr(mock_session, '_session')  # Ensure no _session attribute

        sample_id = "test-sample-456"
        task_id = 67890
        server_name = f"SABER test-domain Tools - {sample_id}"
        cache_key = f"{task_id}_{server_name}"

        MCPServerLocal._task_sessions[cache_key] = mock_session

        with patch("anyio.get_current_task") as mock_get_task:
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_get_task.return_value = mock_task

            # Cleanup the cache
            await factory.cleanup_mcp_cache(sample_id)

        # Verify cache was cleaned
        assert cache_key not in MCPServerLocal._task_sessions
        # __aexit__ should not have been called since there's no _session attribute

    @pytest.mark.asyncio
    async def test_cleanup_mcp_cache_error_handling(self, factory):
        """Test that cleanup errors are logged but don't raise."""
        with patch("anyio.get_current_task") as mock_get_task:
            # Simulate an error getting task
            mock_get_task.side_effect = Exception("Task error")

            # Should not raise exception
            await factory.cleanup_mcp_cache("sample-id")

    @pytest.mark.asyncio
    async def test_cleanup_mcp_cache_aexit_error(self, factory):
        """Test cleanup when __aexit__ raises an error."""
        from inspect_ai.tool._mcp._local import MCPServerLocal

        # Mock the cache and session with failing __aexit__
        mock_session = MagicMock()
        mock_session._session = MagicMock()
        mock_session.__aexit__ = AsyncMock(side_effect=Exception("Exit error"))

        sample_id = "test-sample-789"
        task_id = 99999
        server_name = f"SABER test-domain Tools - {sample_id}"
        cache_key = f"{task_id}_{server_name}"

        MCPServerLocal._task_sessions[cache_key] = mock_session

        with patch("anyio.get_current_task") as mock_get_task:
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_get_task.return_value = mock_task

            # Should not raise exception even if __aexit__ fails
            await factory.cleanup_mcp_cache(sample_id)

        # Cache should still be removed despite __aexit__ error
        assert cache_key not in MCPServerLocal._task_sessions
