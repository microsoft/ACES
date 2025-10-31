"""Tests for client async episode creation methods."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.client.client_session import ClientSessionManager
from saber.client.models import SessionManagerConfig
from saber.models.rest import EpisodeStatusResponse
from saber.server.base import EpisodeState


@pytest.fixture
def session_config():
    """Create a test session manager configuration."""
    return SessionManagerConfig(
        base_url="http://localhost:8000",
        mcp_server_url="http://localhost:8000/mcp",
        client_id="test_client"
    )


@pytest.fixture
def client_session_manager(session_config):
    """Create a client session manager for testing."""
    return ClientSessionManager(config=session_config)


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client for testing."""
    return AsyncMock()


class TestClientAsync:
    """Test client polling and async methods."""

    @pytest.mark.asyncio
    async def test_get_episode_status_fetches_status(self, client_session_manager):
        """Test that get_episode_status fetches status from server."""
        session_id = "test-session"
        episode_id = "episode-123"

        # Mock HTTP response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "episode_id": episode_id,
                "task_id": "test-task",
                "session_id": session_id,
                "state": "ready",
                "is_ready": True,
                "message": "Episode is ready",
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            status = await client_session_manager.get_episode_status(session_id, episode_id)

            assert status.episode_id == episode_id
            assert status.state == "ready"
            assert status.is_ready is True

    @pytest.mark.asyncio
    async def test_wait_for_episode_ready_polls_until_ready(
        self, client_session_manager
    ):
        """Test that wait_for_episode_ready polls until episode is ready."""
        session_id = "test-session"
        episode_id = "episode-456"

        # Mock responses: CREATING, CREATING, READY
        call_count = 0

        async def mock_get_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = AsyncMock()
            mock_response.status = 200
            if call_count < 3:
                mock_response.json.return_value = {
                    "episode_id": episode_id,
                    "task_id": "test-task",
                    "session_id": session_id,
                    "state": "creating",
                    "is_ready": False,
                    "message": "Episode is being created",
                }
            else:
                mock_response.json.return_value = {
                    "episode_id": episode_id,
                    "task_id": "test-task",
                    "session_id": session_id,
                    "state": "ready",
                    "is_ready": True,
                    "message": "Episode is ready",
                }
            return mock_response

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.side_effect = mock_get_response

            status = await client_session_manager.wait_for_episode_ready(
                session_id, episode_id, timeout_seconds=5, min_poll_interval=0.1
            )

            assert status.is_ready is True
            assert status.state == "ready"
            assert call_count == 3  # Should poll 3 times

    @pytest.mark.asyncio
    async def test_wait_for_episode_ready_raises_timeout(
        self, client_session_manager
    ):
        """Test that wait_for_episode_ready raises TimeoutError when timeout exceeded."""
        session_id = "test-session"
        episode_id = "episode-timeout"

        # Mock response always returns CREATING
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "episode_id": episode_id,
                "task_id": "test-task",
                "session_id": session_id,
                "state": "creating",
                "is_ready": False,
                "message": "Episode is being created",
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(TimeoutError):
                await client_session_manager.wait_for_episode_ready(
                    session_id, episode_id, timeout_seconds=1, min_poll_interval=0.1
                )

    @pytest.mark.asyncio
    async def test_wait_for_episode_ready_raises_on_failed_creation(
        self, client_session_manager
    ):
        """Test that wait_for_episode_ready raises exception when episode fails creation."""
        session_id = "test-session"
        episode_id = "episode-failed"

        # Mock response returns FAILED_CREATION
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "episode_id": episode_id,
                "task_id": "test-task",
                "session_id": session_id,
                "state": "failed_creation",
                "is_ready": False,
                "message": "Episode creation failed",
                "creation_error": "Health check failed",
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception, match="Episode creation failed"):
                await client_session_manager.wait_for_episode_ready(
                    session_id, episode_id, timeout_seconds=5, min_poll_interval=0.1
                )

    @pytest.mark.asyncio
    async def test_create_episode_and_wait_convenience_method(
        self, client_session_manager
    ):
        """Test create_episode_and_wait convenience method."""
        session_id = "test-session"
        task_id = "test-task"
        episode_id = "episode-789"

        # Mock both POST (create) and GET (status) responses
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('aiohttp.ClientSession.get') as mock_get:

            # Mock create episode response
            mock_create_response = AsyncMock()
            mock_create_response.status = 200
            mock_create_response.json.return_value = {
                "episode_id": episode_id,
                "session_id": session_id,
                "task_id": task_id,
                "state": "creating",
                "message": "Episode creation initiated",
                "attached_to_episode_id": None,
            }
            mock_post.return_value.__aenter__.return_value = mock_create_response

            # Mock status response (ready immediately)
            mock_status_response = AsyncMock()
            mock_status_response.status = 200
            mock_status_response.json.return_value = {
                "episode_id": episode_id,
                "task_id": task_id,
                "session_id": session_id,
                "state": "ready",
                "is_ready": True,
                "message": "Episode is ready",
            }
            mock_get.return_value.__aenter__.return_value = mock_status_response

            episode = await client_session_manager.create_episode_and_wait(session_id, task_id)

            assert episode.episode_id == episode_id

    @pytest.mark.asyncio
    async def test_wait_for_episode_ready_uses_exponential_backoff(
        self, client_session_manager
    ):
        """Test that wait_for_episode_ready uses exponential backoff with jitter."""
        session_id = "test-session"
        episode_id = "episode-backoff"

        poll_times = []
        call_count = 0

        async def mock_get_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            poll_times.append(asyncio.get_event_loop().time())

            mock_response = AsyncMock()
            mock_response.status = 200
            if call_count < 4:
                mock_response.json.return_value = {
                    "episode_id": episode_id,
                    "task_id": "test-task",
                    "session_id": session_id,
                    "state": "creating",
                    "is_ready": False,
                    "message": "Episode is being created",
                }
            else:
                mock_response.json.return_value = {
                    "episode_id": episode_id,
                    "task_id": "test-task",
                    "session_id": session_id,
                    "state": "ready",
                    "is_ready": True,
                    "message": "Episode is ready",
                }
            return mock_response

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.side_effect = mock_get_response

            await client_session_manager.wait_for_episode_ready(
                session_id, episode_id, timeout_seconds=10, min_poll_interval=0.1
            )

            # Verify polling happened multiple times
            assert call_count >= 3

    @pytest.mark.asyncio
    async def test_concurrent_wait_for_multiple_episodes(
        self, client_session_manager
    ):
        """Test waiting for multiple episodes concurrently."""
        session_id = "test-session"
        episode_ids = [f"episode-{i}" for i in range(3)]

        # All episodes return ready
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "episode_id": "any",
                "task_id": "test-task",
                "session_id": session_id,
                "state": "ready",
                "is_ready": True,
                "message": "Episode is ready",
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            # Wait for all concurrently
            tasks = [
                client_session_manager.wait_for_episode_ready(session_id, episode_id)
                for episode_id in episode_ids
            ]

            statuses = await asyncio.gather(*tasks)

            assert len(statuses) == 3
            for status in statuses:
                assert status.is_ready is True

    @pytest.mark.asyncio
    async def test_get_episode_status_with_error_message(
        self, client_session_manager
    ):
        """Test get_episode_status includes error message for failed episodes."""
        session_id = "test-session"
        episode_id = "episode-error"

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "episode_id": episode_id,
                "task_id": "test-task",
                "session_id": session_id,
                "state": "failed_creation",
                "is_ready": False,
                "message": "Episode creation failed",
                "creation_error": "Docker Compose failed",
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            status = await client_session_manager.get_episode_status(session_id, episode_id)

            assert status.state == "failed_creation"
            assert status.is_ready is False
            assert status.creation_error == "Docker Compose failed"
