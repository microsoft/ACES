"""
Unit tests for SessionManager core functionality.

Tests session creation, termination, and basic server lifecycle.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.session_manager import ClientSession, SessionManager


class TestSessionManagerCore:
    """Test core SessionManager functionality."""

    @pytest.fixture
    def mock_task_manager(self):
        """Mock TaskManager for testing."""
        mock = MagicMock()
        mock.domain = "test_domain"
        return mock

    @pytest.fixture
    def mock_execution_manager(self):
        """Mock ExecutionManager for testing."""
        mock = MagicMock()
        mock.step = AsyncMock()
        return mock

    @pytest.fixture
    def mock_policy_manager(self):
        """Mock PolicyManager for testing."""
        mock = MagicMock()
        mock.get_policy = AsyncMock()
        return mock

    @pytest.fixture
    def mock_evaluation_manager(self):
        """Mock EvaluationManager for testing."""
        mock = MagicMock()
        mock.log_session_start = AsyncMock()
        mock.log_session_end = AsyncMock()
        mock.log_episode_start = AsyncMock()
        mock.log_episode_end = AsyncMock()
        mock.log_action = AsyncMock()
        return mock

    @pytest.fixture
    def mock_episode_manager(self):
        """Mock EpisodeManager for testing."""
        mock = MagicMock()
        mock.start_episode = MagicMock()
        mock.end_episode = MagicMock()
        mock.get_episode = MagicMock()
        return mock

    @pytest.fixture
    def session_manager(
        self,
        mock_task_manager,
        mock_execution_manager,
        mock_policy_manager,
        mock_evaluation_manager,
        mock_episode_manager,
    ):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.TaskManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
        ):

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8001)
            return manager

    def test_session_manager_initialization(self, session_manager):
        """Test SessionManager initialization."""
        assert session_manager.domain_name == "test_domain"
        assert session_manager.host == "127.0.0.1"
        assert session_manager.port == 8001
        assert session_manager.active_sessions == {}
        assert session_manager.app is not None

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test session creation."""
        client_id = "test_client_123"

        session = await session_manager.create_session(client_id)

        assert isinstance(session, ClientSession)
        assert session.client_id == client_id
        assert session.is_active is True
        assert session.current_episode_id is None
        assert session.session_id in session_manager.active_sessions

        # Check evaluation manager was called
        session_manager.evaluation_manager.log_session_start.assert_called_once_with(session.session_id, client_id)

    @pytest.mark.asyncio
    async def test_create_multiple_sessions(self, session_manager):
        """Test creating multiple sessions."""
        client1 = "client_1"
        client2 = "client_2"

        session1 = await session_manager.create_session(client1)
        session2 = await session_manager.create_session(client2)

        assert len(session_manager.active_sessions) == 2
        assert session1.session_id != session2.session_id
        assert session1.client_id == client1
        assert session2.client_id == client2

    @pytest.mark.asyncio
    async def test_terminate_session(self, session_manager):
        """Test session termination."""
        client_id = "test_client"
        session = await session_manager.create_session(client_id)
        session_id = session.session_id

        await session_manager.terminate_session(session_id)

        assert session_id not in session_manager.active_sessions
        session_manager.evaluation_manager.log_session_end.assert_called_once_with(session_id)

    @pytest.mark.asyncio
    async def test_terminate_session_with_active_episode(self, session_manager):
        """Test terminating session with active episode."""
        client_id = "test_client"
        session = await session_manager.create_session(client_id)
        session.current_episode_id = "episode_123"
        session_id = session.session_id

        await session_manager.terminate_session(session_id)

        # Should call episode manager to end episode
        session_manager.episode_manager.end_episode.assert_called_once_with(session_id, "session_terminated")
        assert session_id not in session_manager.active_sessions

    @pytest.mark.asyncio
    async def test_terminate_nonexistent_session(self, session_manager):
        """Test terminating non-existent session raises error."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await session_manager.terminate_session("nonexistent_id")

        assert exc_info.value.status_code == 404
        assert "Session not found" in str(exc_info.value.detail)

    def test_get_session_valid(self, session_manager):
        """Test getting a valid session."""
        # Add a session directly
        session_id = str(uuid.uuid4())
        client_session = ClientSession(session_id=session_id, client_id="test_client")
        session_manager.active_sessions[session_id] = client_session

        retrieved_session = session_manager._get_session(session_id)
        assert retrieved_session == client_session

    def test_get_session_not_found(self, session_manager):
        """Test getting non-existent session raises error."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            session_manager._get_session("nonexistent_id")

        assert exc_info.value.status_code == 404

    def test_get_session_inactive(self, session_manager):
        """Test getting inactive session raises error."""
        from fastapi import HTTPException

        session_id = str(uuid.uuid4())
        client_session = ClientSession(session_id=session_id, client_id="test_client")
        client_session.is_active = False
        session_manager.active_sessions[session_id] = client_session

        with pytest.raises(HTTPException) as exc_info:
            session_manager._get_session(session_id)

        assert exc_info.value.status_code == 400
        assert "not active" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_shutdown(self, session_manager):
        """Test SessionManager shutdown."""
        # Create some sessions
        session1 = await session_manager.create_session("client1")
        session2 = await session_manager.create_session("client2")

        assert len(session_manager.active_sessions) == 2

        await session_manager.shutdown()

        # All sessions should be terminated
        assert len(session_manager.active_sessions) == 0

    @pytest.mark.asyncio
    async def test_session_timeout_cleanup(self, session_manager):
        """Test that inactive sessions are automatically cleaned up after timeout."""
        # Create test sessions
        session1 = await session_manager.create_session("client_1")
        session2 = await session_manager.create_session("client_2")

        # Verify sessions are active
        assert len(session_manager.active_sessions) == 2

        # Manually set one session to be inactive beyond timeout
        from datetime import datetime, timedelta

        old_time = datetime.utcnow() - timedelta(minutes=session_manager.session_timeout_minutes + 1)
        session1.last_activity = old_time

        # Run cleanup manually (instead of waiting for the periodic task)
        await session_manager._cleanup_inactive_sessions()

        # Verify only the active session remains
        assert len(session_manager.active_sessions) == 1
        assert session2.session_id in session_manager.active_sessions
        assert session1.session_id not in session_manager.active_sessions

    @pytest.mark.asyncio
    async def test_session_stats(self, session_manager):
        """Test session statistics functionality."""
        # Create test sessions
        session1 = await session_manager.create_session("client_1")
        session2 = await session_manager.create_session("client_2")

        # Get stats
        stats = session_manager.get_session_stats()

        # Verify stats structure
        assert "total_sessions" in stats
        assert "timeout_minutes" in stats
        assert "cleanup_interval_minutes" in stats
        assert "sessions" in stats

        assert stats["total_sessions"] == 2
        assert stats["timeout_minutes"] == session_manager.session_timeout_minutes
        assert len(stats["sessions"]) == 2

        # Verify session details in stats
        session_ids = [s["session_id"] for s in stats["sessions"]]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

        # Check required fields for each session
        for session_stat in stats["sessions"]:
            assert "session_id" in session_stat
            assert "client_id" in session_stat
            assert "uptime_seconds" in session_stat
            assert "time_since_activity_seconds" in session_stat
            assert "is_active" in session_stat


class TestClientSession:
    """Test ClientSession model functionality."""

    def test_client_session_creation(self):
        """Test ClientSession creation with defaults."""
        session = ClientSession(session_id="test_id", client_id="client_123")

        assert session.session_id == "test_id"
        assert session.client_id == "client_123"
        assert session.current_episode_id is None
        assert session.is_active is True
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
        assert session.context == {}
