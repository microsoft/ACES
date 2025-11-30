"""Tests for EpisodeLifecycleManager in inspect_ai/server/episode_manager.py.

These tests cover:
- Episode creation with health check waiting and timeout handling
- Happy path cleanup (async with submission support)
- Interrupted cleanup (fire-and-forget with retry logic)
- Episode verification after deletion
- Retry mechanisms with exponential backoff
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from inspect_ai._util.error import PrerequisiteError

from saber.client.client_session import ClientSessionManager
from saber.inspect_ai.constants import InspectStoreKeys, SandboxTimeouts
from saber.inspect_ai.server.episode_manager import EpisodeLifecycleManager, SandboxError
from saber.models import EpisodeCreateResponse
from saber.models import APIEndpoints
from saber.server.episodes.constants import EpisodeTerminationReason


@pytest.fixture
def mock_session_manager():
    """Create mock ClientSessionManager for testing."""
    manager = Mock(spec=ClientSessionManager)
    manager.base_url = "http://localhost:8000"
    manager._current_session_id = "test-session-123"
    return manager


@pytest.fixture
def episode_manager(mock_session_manager):
    """Create EpisodeLifecycleManager instance for testing."""
    return EpisodeLifecycleManager(session_manager=mock_session_manager)


class TestEpisodeLifecycleManagerInit:
    """Test EpisodeLifecycleManager initialization."""

    def test_init_stores_session_manager(self, mock_session_manager):
        """Test that initialization stores the session manager."""
        manager = EpisodeLifecycleManager(session_manager=mock_session_manager)
        assert manager.session_manager is mock_session_manager


class TestCreateEpisode:
    """Test episode creation with health check waiting."""

    @pytest.mark.asyncio
    async def test_create_episode_success(self, episode_manager, mock_session_manager):
        """Test successful episode creation returns episode ID."""
        mock_response = EpisodeCreateResponse(
            episode_id="test-episode-456",
            session_id="test-session-123",
            task_id="test_task",
            state="creating",
            message="Episode created successfully"
        )

        mock_session_manager.create_episode_and_wait = AsyncMock(return_value=mock_response)

        episode_id = await episode_manager.create_episode(
            session_id="test-session-123",
            task_id="test_task"
        )

        assert episode_id == "test-episode-456"
        mock_session_manager.create_episode_and_wait.assert_called_once_with(
            session_id="test-session-123",
            task_id="test_task",
            timeout_seconds=SandboxTimeouts.EPISODE_CREATE_SECONDS,
        )

    @pytest.mark.asyncio
    async def test_create_episode_timeout_error(self, episode_manager, mock_session_manager):
        """Test that episode creation timeout raises PrerequisiteError with diagnostic."""
        mock_session_manager.create_episode_and_wait = AsyncMock(
            side_effect=TimeoutError("Episode creation timed out")
        )

        with pytest.raises(PrerequisiteError) as exc_info:
            await episode_manager.create_episode(
                session_id="test-session-123",
                task_id="test_task"
            )

        error_msg = str(exc_info.value)
        assert "Episode creation timed out" in error_msg
        assert f"{SandboxTimeouts.EPISODE_CREATE_SECONDS}s" in error_msg
        assert "Docker health checks" in error_msg
        assert "test_task" in error_msg

    @pytest.mark.asyncio
    async def test_create_episode_generic_exception(self, episode_manager, mock_session_manager):
        """Test that generic exceptions are wrapped in PrerequisiteError."""
        mock_session_manager.create_episode_and_wait = AsyncMock(
            side_effect=ValueError("Invalid task configuration")
        )

        with pytest.raises(PrerequisiteError) as exc_info:
            await episode_manager.create_episode(
                session_id="test-session-123",
                task_id="test_task"
            )

        error_msg = str(exc_info.value)
        assert "Failed to create episode" in error_msg
        assert "test_task" in error_msg
        assert "ValueError" in error_msg
        assert "Invalid task configuration" in error_msg

    @pytest.mark.asyncio
    async def test_create_episode_no_session_manager_raises(self):
        """Test that missing session manager raises SandboxError."""
        manager = EpisodeLifecycleManager(session_manager=None)

        with pytest.raises(SandboxError) as exc_info:
            await manager.create_episode(
                session_id="test-session-123",
                task_id="test_task"
            )

        assert "Session manager not initialized" in str(exc_info.value)


class TestCleanupEpisodeHappyPath:
    """Test happy path episode cleanup with async retry logic."""

    @pytest.mark.asyncio
    async def test_cleanup_episode_happy_path_success(self, episode_manager, mock_session_manager):
        """Test successful happy path cleanup with submission."""
        submission = {"answer": "test", "score": 1.0}

        mock_session_manager.end_episode_with_retry = AsyncMock(return_value=True)

        await episode_manager.cleanup_episode(
            session_id="test-session-123",
            episode_id="test-episode-456",
            interrupted=False,
            submission=submission,
        )

        mock_session_manager.end_episode_with_retry.assert_called_once_with(
            session_id="test-session-123",
            episode_id="test-episode-456",
            reason=EpisodeTerminationReason.COMPLETED,
            result=submission,
            cascade_end_attached_episodes=False,
            max_retries=5,
            initial_backoff=1.0,
            max_backoff=30.0,
        )

    @pytest.mark.asyncio
    async def test_cleanup_episode_happy_path_no_submission(self, episode_manager, mock_session_manager):
        """Test happy path cleanup without submission (submission=None)."""
        mock_session_manager.end_episode_with_retry = AsyncMock(return_value=True)

        await episode_manager.cleanup_episode(
            session_id="test-session-123",
            episode_id="test-episode-456",
            interrupted=False,
            submission=None,
        )

        # Should still call end_episode_with_retry with result=None
        mock_session_manager.end_episode_with_retry.assert_called_once()
        call_kwargs = mock_session_manager.end_episode_with_retry.call_args[1]
        assert call_kwargs["result"] is None
        assert call_kwargs["reason"] == EpisodeTerminationReason.COMPLETED

    @pytest.mark.asyncio
    async def test_cleanup_episode_happy_path_failure_after_retries(
        self, episode_manager, mock_session_manager
    ):
        """Test that failed cleanup after retries logs error but doesn't raise."""
        mock_session_manager.end_episode_with_retry = AsyncMock(return_value=False)

        # Should not raise exception
        await episode_manager.cleanup_episode(
            session_id="test-session-123",
            episode_id="test-episode-456",
            interrupted=False,
        )

        mock_session_manager.end_episode_with_retry.assert_called_once()


class TestCleanupEpisodeInterruptedPath:
    """Test interrupted path cleanup with fire-and-forget retry logic."""

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_success(self, episode_manager, mock_session_manager):
        """Test interrupted cleanup uses fire-and-forget with retry."""
        mock_delete_response = Mock()
        mock_delete_response.status_code = 200

        mock_verify_response = Mock()
        mock_verify_response.status_code = 404  # Episode deleted

        with patch("requests.delete", return_value=mock_delete_response) as mock_delete:
            with patch("requests.get", return_value=mock_verify_response):
                with patch("threading.Thread") as mock_thread:
                    # Mock thread to execute immediately
                    def run_target(*args, **kwargs):
                        result = kwargs["target"]()
                        return Mock(start=Mock(), join=Mock())

                    mock_thread.side_effect = run_target

                    await episode_manager.cleanup_episode(
                        session_id="test-session-123",
                        episode_id="test-episode-456",
                        interrupted=True,
                    )

                    # Verify DELETE was called with retry logic
                    endpoint = APIEndpoints.EPISODE_BY_ID.format(
                        session_id='test-session-123', episode_id='test-episode-456'
                    )
                    expected_url = f"http://localhost:8000{endpoint}"
                    mock_delete.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_retries_on_failure(
        self, episode_manager, mock_session_manager
    ):
        """Test that interrupted cleanup retries on failure."""
        # Simulate failures then success
        delete_responses = [
            Mock(status_code=500),  # First attempt fails
            Mock(status_code=500),  # Second attempt fails
            Mock(status_code=200),  # Third attempt succeeds
        ]

        mock_verify_response = Mock()
        mock_verify_response.status_code = 404

        call_count = 0
        def mock_delete(*args, **kwargs):
            nonlocal call_count
            response = delete_responses[min(call_count, len(delete_responses) - 1)]
            call_count += 1
            return response

        with patch("requests.delete", side_effect=mock_delete):
            with patch("requests.get", return_value=mock_verify_response):
                with patch("threading.Thread") as mock_thread:
                    # Mock thread to execute immediately
                    def run_target(*args, **kwargs):
                        kwargs["target"]()
                        return Mock(start=Mock(), join=Mock())

                    mock_thread.side_effect = run_target

                    await episode_manager.cleanup_episode(
                        session_id="test-session-123",
                        episode_id="test-episode-456",
                        interrupted=True,
                    )

                    # Should have retried
                    assert call_count >= 2

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_handles_400_already_ended(
        self, episode_manager, mock_session_manager
    ):
        """Test that 400 status (already ended) is treated as success."""
        mock_response = Mock()
        mock_response.status_code = 400  # Already ended

        with patch("requests.delete", return_value=mock_response):
            with patch("threading.Thread") as mock_thread:
                # Mock thread to execute immediately
                def run_target(*args, **kwargs):
                    kwargs["target"]()
                    return Mock(start=Mock(), join=Mock())

                mock_thread.side_effect = run_target

                # Should not raise exception
                await episode_manager.cleanup_episode(
                    session_id="test-session-123",
                    episode_id="test-episode-456",
                    interrupted=True,
                )

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_uses_non_daemon_thread(
        self, episode_manager, mock_session_manager
    ):
        """Test that interrupted cleanup uses non-daemon thread."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response):
            with patch("requests.get", return_value=Mock(status_code=404)):
                with patch("threading.Thread") as mock_thread_class:
                    mock_thread = Mock()
                    mock_thread_class.return_value = mock_thread

                    await episode_manager.cleanup_episode(
                        session_id="test-session-123",
                        episode_id="test-episode-456",
                        interrupted=True,
                    )

                    # Verify non-daemon thread
                    mock_thread_class.assert_called_once()
                    call_kwargs = mock_thread_class.call_args[1]
                    assert call_kwargs["daemon"] is False

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_join_timeout(
        self, episode_manager, mock_session_manager
    ):
        """Test that thread join uses 20s timeout for retries."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response):
            with patch("requests.get", return_value=Mock(status_code=404)):
                with patch("threading.Thread") as mock_thread_class:
                    mock_thread = Mock()
                    mock_thread_class.return_value = mock_thread

                    await episode_manager.cleanup_episode(
                        session_id="test-session-123",
                        episode_id="test-episode-456",
                        interrupted=True,
                    )

                    # Verify join timeout (20s to allow for retries)
                    mock_thread.join.assert_called_once_with(timeout=20.0)


class TestCleanupEpisodeErrorHandling:
    """Test error handling during episode cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_episode_handles_cancellation(
        self, episode_manager, mock_session_manager
    ):
        """Test that cancellation during cleanup is handled gracefully."""
        mock_session_manager.end_episode_with_retry = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        # Should not raise exception
        await episode_manager.cleanup_episode(
            session_id="test-session-123",
            episode_id="test-episode-456",
            interrupted=False,
        )

    @pytest.mark.asyncio
    async def test_cleanup_episode_handles_timeout(
        self, episode_manager, mock_session_manager
    ):
        """Test that timeout during cleanup is handled gracefully."""
        mock_session_manager.end_episode_with_retry = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        # Should not raise exception
        await episode_manager.cleanup_episode(
            session_id="test-session-123",
            episode_id="test-episode-456",
            interrupted=False,
        )

    @pytest.mark.asyncio
    async def test_cleanup_episode_handles_generic_exception(
        self, episode_manager, mock_session_manager
    ):
        """Test that generic exceptions during cleanup are logged but don't raise."""
        mock_session_manager.end_episode_with_retry = AsyncMock(
            side_effect=ValueError("Unexpected error")
        )

        # Should not raise exception
        await episode_manager.cleanup_episode(
            session_id="test-session-123",
            episode_id="test-episode-456",
            interrupted=False,
        )

    @pytest.mark.asyncio
    async def test_cleanup_episode_no_episode_id_skips_cleanup(
        self, episode_manager, mock_session_manager
    ):
        """Test that cleanup with no episode ID skips cleanup logic."""
        mock_session_manager.end_episode_with_retry = AsyncMock()

        await episode_manager.cleanup_episode(
            session_id="test-session-123",
            episode_id=None,  # No episode ID
            interrupted=False,
        )

        # Should not call end_episode_with_retry
        mock_session_manager.end_episode_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_episode_no_session_manager_skips_cleanup(self):
        """Test that cleanup with no session manager skips cleanup logic."""
        manager = EpisodeLifecycleManager(session_manager=None)

        # Should not raise exception
        await manager.cleanup_episode(
            session_id="test-session-123",
            episode_id="test-episode-456",
            interrupted=False,
        )


class TestCleanupEpisodeReasonHandling:
    """Test that cleanup uses correct termination reasons."""

    @pytest.mark.asyncio
    async def test_cleanup_episode_completed_reason(
        self, episode_manager, mock_session_manager
    ):
        """Test that non-interrupted cleanup uses COMPLETED reason."""
        mock_session_manager.end_episode_with_retry = AsyncMock(return_value=True)

        await episode_manager.cleanup_episode(
            session_id="test-session-123",
            episode_id="test-episode-456",
            interrupted=False,
        )

        call_kwargs = mock_session_manager.end_episode_with_retry.call_args[1]
        assert call_kwargs["reason"] == EpisodeTerminationReason.COMPLETED

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_reason(
        self, episode_manager, mock_session_manager
    ):
        """Test that interrupted cleanup uses INTERRUPTED reason."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response) as mock_delete:
            with patch("requests.get", return_value=Mock(status_code=404)):
                with patch("threading.Thread") as mock_thread:
                    # Mock thread to execute immediately
                    def run_target(*args, **kwargs):
                        kwargs["target"]()
                        return Mock(start=Mock(), join=Mock())

                    mock_thread.side_effect = run_target

                    await episode_manager.cleanup_episode(
                        session_id="test-session-123",
                        episode_id="test-episode-456",
                        interrupted=True,
                    )

                    # Verify DELETE was called with INTERRUPTED reason in params
                    call_kwargs = mock_delete.call_args[1]
                    assert call_kwargs["params"]["reason"] == EpisodeTerminationReason.INTERRUPTED


class TestCleanupEpisodeRetryMechanism:
    """Test retry mechanisms with backoff."""

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_max_retries(
        self, episode_manager, mock_session_manager
    ):
        """Test that interrupted cleanup respects max retries (3 attempts)."""
        mock_response = Mock()
        mock_response.status_code = 500  # Always fail

        retry_count = 0
        def mock_delete(*args, **kwargs):
            nonlocal retry_count
            retry_count += 1
            return mock_response

        with patch("requests.delete", side_effect=mock_delete):
            with patch("threading.Thread") as mock_thread:
                # Mock thread to execute immediately
                def run_target(*args, **kwargs):
                    kwargs["target"]()
                    return Mock(start=Mock(), join=Mock())

                mock_thread.side_effect = run_target
                # Mock time.sleep to avoid delays
                with patch("time.sleep"):
                    await episode_manager.cleanup_episode(
                        session_id="test-session-123",
                        episode_id="test-episode-456",
                        interrupted=True,
                    )

                    # Should have tried 3 times (max_retries=3)
                    assert retry_count == 3

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_backoff_timing(
        self, episode_manager, mock_session_manager
    ):
        """Test that interrupted cleanup uses exponential backoff."""
        mock_response = Mock()
        mock_response.status_code = 500

        sleep_times = []
        def mock_sleep(duration):
            sleep_times.append(duration)

        with patch("requests.delete", return_value=mock_response):
            with patch("threading.Thread") as mock_thread:
                # Mock thread to execute immediately
                def run_target(*args, **kwargs):
                    kwargs["target"]()
                    return Mock(start=Mock(), join=Mock())

                mock_thread.side_effect = run_target
                with patch("time.sleep", side_effect=mock_sleep):
                    await episode_manager.cleanup_episode(
                        session_id="test-session-123",
                        episode_id="test-episode-456",
                        interrupted=True,
                    )

                    # Should have 2 sleep calls (after 1st and 2nd attempts)
                    assert len(sleep_times) == 2
                    # Backoff increases: 1.0 * 1, 1.0 * 2
                    assert sleep_times[0] == 1.0
                    assert sleep_times[1] == 2.0


class TestCleanupEpisodeVerification:
    """Test episode verification after deletion."""

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_verifies_deletion(
        self, episode_manager, mock_session_manager
    ):
        """Test that interrupted cleanup verifies episode deletion."""
        mock_delete_response = Mock()
        mock_delete_response.status_code = 200

        mock_verify_response = Mock()
        mock_verify_response.status_code = 404  # Confirmed deleted

        with patch("requests.delete", return_value=mock_delete_response):
            with patch("requests.get", return_value=mock_verify_response) as mock_get:
                with patch("threading.Thread") as mock_thread:
                    # Mock thread to execute immediately
                    def run_target(*args, **kwargs):
                        kwargs["target"]()
                        return Mock(start=Mock(), join=Mock())

                    mock_thread.side_effect = run_target
                    with patch("time.sleep"):  # Mock sleep to avoid delays
                        await episode_manager.cleanup_episode(
                            session_id="test-session-123",
                            episode_id="test-episode-456",
                            interrupted=True,
                        )

                        # Verify GET was called to check episode status
                        verify_endpoint = APIEndpoints.EPISODE_STATUS.format(
                            session_id='test-session-123', episode_id='test-episode-456'
                        )
                        expected_verify_url = f"http://localhost:8000{verify_endpoint}"
                        mock_get.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_episode_interrupted_accepts_unclear_verification(
        self, episode_manager, mock_session_manager
    ):
        """Test that unclear verification status is accepted as success."""
        mock_delete_response = Mock()
        mock_delete_response.status_code = 200

        mock_verify_response = Mock()
        mock_verify_response.status_code = 200  # Still exists? Unclear

        with patch("requests.delete", return_value=mock_delete_response):
            with patch("requests.get", return_value=mock_verify_response):
                with patch("threading.Thread") as mock_thread:
                    # Mock thread to execute immediately
                    def run_target(*args, **kwargs):
                        kwargs["target"]()
                        return Mock(start=Mock(), join=Mock())

                    mock_thread.side_effect = run_target
                    with patch("time.sleep"):
                        # Should not raise exception
                        await episode_manager.cleanup_episode(
                            session_id="test-session-123",
                            episode_id="test-episode-456",
                            interrupted=True,
                        )
