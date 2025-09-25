"""
Tests for new ClientSessionManager features and evaluation methods.

These tests cover the evaluation-related functionality that was added
to the ClientSessionManager after the major refactor.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import aiohttp

from saber.client.client_session import ClientSessionManager
from saber.client.models import SessionManagerConfig
from saber.models.rest.evaluation import (
    EvaluationCriteriaResponse,
    EvaluationOverrideRequest,
    EvaluationResultResponse,
    TaskEvaluationContext,
    JudgeMessages,
)


class TestClientSessionManagerEvaluationFeatures:
    """Test the new evaluation-related features of ClientSessionManager."""

    @pytest.fixture
    def session_config(self):
        """Create a test session manager configuration."""
        return SessionManagerConfig(
            base_url="http://localhost:8000",
            mcp_server_url="http://localhost:3001",
            client_id="test_client",
            rest_timeout=30.0,
        )

    @pytest.fixture
    def session_manager(self, session_config):
        """Create a ClientSessionManager instance for testing."""
        return ClientSessionManager(session_config)

    @pytest.fixture
    def mock_evaluation_criteria(self):
        """Mock evaluation criteria response."""
        return EvaluationCriteriaResponse(
            session_id="sess_123",
            episode_id="ep_123",
            task_id="task_1",
            submission="I found the flag in /etc/passwd: flag{test_flag}",
            task_context=TaskEvaluationContext(
                task_id="task_1",
                title="Find the Flag",
                description="Locate the hidden flag in the system",
                domain="penetration_testing",
                subtasks=[{"id": "task_1", "description": "Main task objective"}]
            ),
            evaluation_config={
                "strategy": "llm_judge",
                "judge_model": "gpt-4",
                "judge_messages": {
                    "system_message": "You are an expert cybersecurity judge.",
                    "user_message": "Evaluate this submission: {submission}",
                    "model": "gpt-4"
                },
                "scoring": {"max_score": 1.0}
            }
        )

    @pytest.fixture
    def mock_evaluation_result(self):
        """Mock evaluation result response."""
        return EvaluationResultResponse(
            episode_id="ep_123",
            task_id="task_1",
            strategy="llm_judge",
            raw_score=1.0,
            max_score=1.0,
            score=1.0,
            success=True,
            details={
                "scorer_type": "saber_client_side",
                "task_completed_at_step": 2
            }
        )

    @pytest.mark.asyncio
    async def test_get_evaluation_criteria_success(self, session_manager, mock_evaluation_criteria):
        """Test successful retrieval of evaluation criteria."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_evaluation_criteria.dict()
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await session_manager.get_evaluation_criteria("sess_123", "ep_123")

            assert isinstance(result, EvaluationCriteriaResponse)
            assert result.session_id == "sess_123"
            assert result.episode_id == "ep_123"
            assert result.task_id == "task_1"
            assert result.evaluation_config["strategy"] == "llm_judge"

    @pytest.mark.asyncio
    async def test_get_evaluation_criteria_not_found(self, session_manager):
        """Test handling of 404 when evaluation criteria not found."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception) as exc_info:
                await session_manager.get_evaluation_criteria("sess_123", "ep_123")

            assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_episode_evaluation_success(self, session_manager, mock_evaluation_result):
        """Test successful retrieval of episode evaluation."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            # Wrap in evaluation_result key as expected by implementation
            mock_response.json.return_value = {"evaluation_result": mock_evaluation_result.dict()}
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await session_manager.get_episode_evaluation("sess_123", "ep_123")

            assert isinstance(result, EvaluationResultResponse)
            assert result.episode_id == "ep_123"
            assert result.task_id == "task_1"
            assert result.score == 1.0
            assert result.success is True

    @pytest.mark.asyncio
    async def test_override_episode_evaluation_success(self, session_manager):
        """Test successful episode evaluation override."""
        override_request = EvaluationOverrideRequest(
            evaluation_data={
                "episode_id": "ep_123",
                "task_id": "task_1",
                "raw_score": 0.5
            },
            strategy="llm_judge",
            raw_score=0.5,
            max_score=1.0,
            score=0.5,
            success=False,
            details={"reason": "test_override"}
        )

        with patch('aiohttp.ClientSession.put') as mock_put:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"success": True}
            mock_put.return_value.__aenter__.return_value = mock_response

            result = await session_manager.override_episode_evaluation(
                "sess_123", "ep_123", override_request
            )

            assert result["success"] is True
            mock_put.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_evaluations_success(self, session_manager):
        """Test successful retrieval of session evaluations."""
        mock_evaluations = {
            "total_count": 1,
            "evaluations": [
                {
                    "episode_id": "ep_123",
                    "task_id": "task_1",
                    "strategy": "llm_judge",
                    "raw_score": 1.0,
                    "max_score": 1.0,
                    "score": 1.0,
                    "success": True
                }
            ]
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_evaluations
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await session_manager.get_session_evaluations("sess_123")

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0].score == 1.0
            assert result[0].success is True

    @pytest.mark.asyncio
    async def test_upload_evaluation_file_success(self, session_manager):
        """Test successful evaluation file upload."""
        with patch('aiohttp.ClientSession.post') as mock_post, \
             patch('builtins.open') as mock_file_open, \
             patch('os.path.exists', return_value=True) as mock_exists:

            # Mock file context manager
            mock_file = Mock()
            mock_file.read.return_value = b"test file content"
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)
            mock_file_open.return_value = mock_file

            # Mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"file_id": "file_123", "uploaded": True}
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await session_manager.upload_evaluation_file("sess_123", "/tmp/test.json")

            assert result["file_id"] == "file_123"
            assert result["uploaded"] is True

    @pytest.mark.asyncio
    async def test_session_cleanup(self, session_manager):
        """Test session cleanup functionality."""
        # Set a current session ID
        session_manager._current_session_id = "sess_123"

        with patch.object(session_manager, 'terminate_session') as mock_terminate:
            await session_manager.cleanup()

            # Verify terminate_session was called
            mock_terminate.assert_called_once_with("sess_123")

            # Note: cleanup doesn't clear the session ID in the actual implementation    @pytest.mark.asyncio
    async def test_session_cleanup_no_session(self, session_manager):
        """Test cleanup when no active session exists."""
        # Ensure no current session
        session_manager._current_session_id = None

        with patch.object(session_manager, 'terminate_session') as mock_terminate:
            await session_manager.cleanup()

            # Verify terminate_session was not called
            mock_terminate.assert_not_called()

    def test_get_current_session_id(self, session_manager):
        """Test getting current session ID."""
        # Test when no session exists
        assert session_manager.get_current_session_id() is None

        # Test when session exists
        session_manager._current_session_id = "sess_123"
        assert session_manager.get_current_session_id() == "sess_123"

    def test_get_or_create_session_id(self, session_manager):
        """Test getting or creating session ID."""
        # Test when no session exists - should raise error
        with pytest.raises(RuntimeError, match="No active session"):
            session_manager.get_or_create_session_id()

        # Test when session exists
        session_manager._current_session_id = "sess_123"
        assert session_manager.get_or_create_session_id() == "sess_123"

    @pytest.mark.asyncio
    async def test_update_episode_status_success(self, session_manager):
        """Test episode status update (currently a placeholder implementation)."""
        # This method currently just sleeps and doesn't make HTTP requests
        # Test passes if no exception is raised
        await session_manager.update_episode_status("ep_123", "completed")
