"""Tests for SessionLifecycleManager in inspect_ai/server/session_manager.py.

These tests cover:
- Async session creation via REST API
- Fire-and-forget synchronous session termination
- Thread-based termination to avoid event loop cancellation
- Error handling for session creation failures
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from inspect_ai._util.error import PrerequisiteError

from saber.inspect_ai.constants import SandboxTimeouts
from saber.inspect_ai.server.session_manager import SessionLifecycleManager
from saber.models import APIEndpoints, ClientIdentifiers


@pytest.fixture
def session_manager():
    """Create SessionLifecycleManager instance for testing."""
    return SessionLifecycleManager(rest_base_url="http://localhost:8000")


class TestSessionLifecycleManagerInit:
    """Test SessionLifecycleManager initialization."""

    def test_init_stores_base_url(self):
        """Test that initialization stores the base URL."""
        manager = SessionLifecycleManager(rest_base_url="http://test:9000")
        assert manager.rest_base_url == "http://test:9000"


class TestCreateSession:
    """Test async session creation via REST API."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, session_manager):
        """Test successful session creation returns session ID."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"session_id": "test-session-123"})

        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_session = MagicMock()
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_session.post.return_value = mock_post_cm
            mock_client_session.return_value.__aenter__.return_value = mock_session

            session_id = await session_manager.create_session("test_task")

            assert session_id == "test-session-123"
            mock_session.post.assert_called_once()

            # Verify URL and params
            call_args = mock_session.post.call_args
            assert call_args[0][0] == f"http://localhost:8000{APIEndpoints.SESSION}"
            assert call_args[1]["params"] == {
                "client_id": f"{ClientIdentifiers.INSPECT_AI_PREFIX}test_task"
            }

    @pytest.mark.asyncio
    async def test_create_session_uses_correct_timeout(self, session_manager):
        """Test that session creation uses configured timeout."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"session_id": "test-session-123"})

        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_session = MagicMock()
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_session.post.return_value = mock_post_cm
            mock_client_session.return_value.__aenter__.return_value = mock_session

            await session_manager.create_session("test_task")

            # Verify timeout parameter
            call_args = mock_session.post.call_args
            timeout = call_args[1]["timeout"]
            assert isinstance(timeout, aiohttp.ClientTimeout)
            assert timeout.total == SandboxTimeouts.SESSION_CREATE_SECONDS

    @pytest.mark.asyncio
    async def test_create_session_failure_non_200_status(self, session_manager):
        """Test that non-200 status raises PrerequisiteError."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_session = MagicMock()
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_session.post.return_value = mock_post_cm
            mock_client_session.return_value.__aenter__.return_value = mock_session

            with pytest.raises(PrerequisiteError) as exc_info:
                await session_manager.create_session("test_task")

            assert "Failed to create SABER session" in str(exc_info.value)
            assert "500" in str(exc_info.value)
            assert "Internal Server Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_session_failure_404(self, session_manager):
        """Test that 404 status raises PrerequisiteError."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")

        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_session = MagicMock()
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_session.post.return_value = mock_post_cm
            mock_client_session.return_value.__aenter__.return_value = mock_session

            with pytest.raises(PrerequisiteError) as exc_info:
                await session_manager.create_session("test_task")

            assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_session_converts_session_id_to_string(self, session_manager):
        """Test that session ID is converted to string (even if numeric)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"session_id": 12345})

        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_session = MagicMock()
            mock_post_cm = AsyncMock()
            mock_post_cm.__aenter__.return_value = mock_response
            mock_session.post.return_value = mock_post_cm
            mock_client_session.return_value.__aenter__.return_value = mock_session

            session_id = await session_manager.create_session("test_task")

            assert session_id == "12345"
            assert isinstance(session_id, str)


class TestTerminateSessionSync:
    """Test synchronous fire-and-forget session termination."""

    def test_terminate_session_sync_success(self, session_manager):
        """Test successful session termination logs success."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response) as mock_delete:
            with patch("threading.Thread") as mock_thread:
                # Mock thread to execute immediately
                def run_target(*args, **kwargs):
                    kwargs["target"]()
                    return Mock(start=Mock(), join=Mock())

                mock_thread.side_effect = run_target

                session_manager.terminate_session_sync("test-session-123")

                # Verify DELETE was called
                expected_url = f"http://localhost:8000{APIEndpoints.SESSION_BY_ID.format(session_id='test-session-123')}"
                mock_delete.assert_called_once_with(expected_url, timeout=2.0)

    def test_terminate_session_sync_non_200_status(self, session_manager):
        """Test that non-200 status is logged but doesn't raise."""
        mock_response = Mock()
        mock_response.status_code = 404

        with patch("requests.delete", return_value=mock_response):
            with patch("threading.Thread") as mock_thread:
                # Mock thread to execute immediately
                def run_target(*args, **kwargs):
                    kwargs["target"]()
                    return Mock(start=Mock(), join=Mock())

                mock_thread.side_effect = run_target

                # Should not raise exception
                session_manager.terminate_session_sync("test-session-123")

    def test_terminate_session_sync_exception_caught(self, session_manager):
        """Test that exceptions during termination are caught and logged."""
        with patch("requests.delete", side_effect=Exception("Network error")):
            with patch("threading.Thread") as mock_thread:
                # Mock thread to execute immediately
                def run_target(*args, **kwargs):
                    kwargs["target"]()
                    return Mock(start=Mock(), join=Mock())

                mock_thread.side_effect = run_target

                # Should not raise exception
                session_manager.terminate_session_sync("test-session-123")

    def test_terminate_session_sync_uses_non_daemon_thread(self, session_manager):
        """Test that termination uses non-daemon thread."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response):
            with patch("threading.Thread") as mock_thread_class:
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread

                session_manager.terminate_session_sync("test-session-123")

                # Verify thread created with daemon=False
                mock_thread_class.assert_called_once()
                call_kwargs = mock_thread_class.call_args[1]
                assert call_kwargs["daemon"] is False

                # Verify thread was started and joined
                mock_thread.start.assert_called_once()
                mock_thread.join.assert_called_once()

    def test_terminate_session_sync_join_timeout(self, session_manager):
        """Test that thread join uses configured timeout."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response):
            with patch("threading.Thread") as mock_thread_class:
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread

                session_manager.terminate_session_sync("test-session-123")

                # Verify join timeout
                mock_thread.join.assert_called_once()
                join_timeout = mock_thread.join.call_args[1]["timeout"]
                assert join_timeout == SandboxTimeouts.SESSION_TERMINATE_THREAD_JOIN_SECONDS

    def test_terminate_session_sync_timeout_in_delete_request(self, session_manager):
        """Test that DELETE request uses 2.0 second timeout."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response) as mock_delete:
            with patch("threading.Thread") as mock_thread:
                # Mock thread to execute immediately
                def run_target(*args, **kwargs):
                    kwargs["target"]()
                    return Mock(start=Mock(), join=Mock())

                mock_thread.side_effect = run_target

                session_manager.terminate_session_sync("test-session-123")

                # Verify DELETE timeout
                call_kwargs = mock_delete.call_args[1]
                assert call_kwargs["timeout"] == 2.0

    def test_terminate_session_sync_constructs_correct_url(self, session_manager):
        """Test that termination constructs correct API endpoint URL."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response) as mock_delete:
            with patch("threading.Thread") as mock_thread:
                # Mock thread to execute immediately
                def run_target(*args, **kwargs):
                    kwargs["target"]()
                    return Mock(start=Mock(), join=Mock())

                mock_thread.side_effect = run_target

                session_manager.terminate_session_sync("abc-123")

                # Verify URL construction
                expected_url = f"http://localhost:8000{APIEndpoints.SESSION_BY_ID.format(session_id='abc-123')}"
                assert mock_delete.call_args[0][0] == expected_url


class TestThreadSafety:
    """Test thread safety of session termination."""

    def test_concurrent_termination_calls(self, session_manager):
        """Test that multiple concurrent termination calls are thread-safe."""
        mock_response = Mock()
        mock_response.status_code = 200

        call_count = 0
        call_lock = threading.Lock()

        def mock_delete(*args, **kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
            return mock_response

        with patch("requests.delete", side_effect=mock_delete):
            threads = []
            num_threads = 5

            for i in range(num_threads):
                t = threading.Thread(
                    target=session_manager.terminate_session_sync,
                    args=(f"session-{i}",)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=5.0)

            # All terminations should have been attempted
            assert call_count == num_threads
