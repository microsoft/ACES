"""
Integration tests for client-side evaluation retrieval (Phase 2).

Tests the complete evaluation retrieval pipeline from client to server,
following SABER fail-fast principles.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import aiohttp
from aiohttp import web
import json

from saber.client.api.rest_client import SABERRestClient
from saber.client.client_session import ClientSessionManager
from saber.client.exceptions import (
    EvaluationNotFoundError,
    InvalidEvaluationRequestError,
    SessionEvaluationError,
    EvaluationRetrievalError,
)
from saber.models.rest.evaluation import (
    EvaluationResponse,
    EvaluationListResponse,
    EvaluationSummaryResponse,
    EvaluationResultResponse,
)
from saber.client.models import SessionManagerConfig
from datetime import datetime, timezone


class TestSABERRestClientEvaluation:
    """Test SABERRestClient evaluation methods."""

    @pytest.fixture
    def rest_client(self):
        """Create REST client for testing."""
        return SABERRestClient("http://localhost:8000")

    @pytest.fixture
    def mock_evaluation_result(self):
        """Mock evaluation result data."""
        return {
            "episode_id": "ep_123",
            "task_id": "task_1",
            "strategy": "static",
            "raw_score": 0.8,
            "max_score": 1.0,
            "score": 0.8,
            "success": True,
            "timestamp": "2025-09-09T12:00:00Z",
            "details": {"test": "data"}
        }

    @pytest.fixture
    def mock_evaluation_response(self, mock_evaluation_result):
        """Mock evaluation response."""
        return {
            "evaluation_result": mock_evaluation_result,
            "session_id": "session_123"
        }

    @pytest.mark.asyncio
    async def test_get_evaluation_success(self, rest_client, mock_evaluation_response):
        """Test successful evaluation retrieval."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_evaluation_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await rest_client.get_evaluation("session_123", "ep_123")

            assert isinstance(result, EvaluationResponse)
            assert result.session_id == "session_123"
            assert result.evaluation_result.episode_id == "ep_123"
            assert result.evaluation_result.success is True

    @pytest.mark.asyncio
    async def test_get_evaluation_not_found(self, rest_client):
        """Test evaluation not found error (404)."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text = AsyncMock(return_value="Evaluation not found")
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(EvaluationNotFoundError) as exc_info:
                await rest_client.get_evaluation("session_123", "ep_404")

            assert "session_123" in str(exc_info.value)
            assert "ep_404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_evaluation_invalid_request(self, rest_client):
        """Test invalid evaluation request error (422)."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 422
            mock_response.text = AsyncMock(return_value="Invalid session ID format")
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(InvalidEvaluationRequestError) as exc_info:
                await rest_client.get_evaluation("invalid_session", "ep_123")

            assert "Invalid session ID format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_evaluation_session_error(self, rest_client):
        """Test session evaluation error (500)."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Session evaluation service error")
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(SessionEvaluationError) as exc_info:
                await rest_client.get_evaluation("session_123", "ep_123")

            assert "Session evaluation service error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_evaluations_success(self, rest_client, mock_evaluation_result):
        """Test successful evaluation list retrieval."""
        mock_list_response = {
            "evaluations": [mock_evaluation_result],
            "total_count": 1,
            "session_id": "session_123",
            "task_filter": None
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_list_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await rest_client.list_evaluations("session_123")

            assert isinstance(result, EvaluationListResponse)
            assert result.total_count == 1
            assert len(result.evaluations) == 1
            assert result.evaluations[0].episode_id == "ep_123"

    @pytest.mark.asyncio
    async def test_list_evaluations_with_task_filter(self, rest_client, mock_evaluation_result):
        """Test evaluation list with task filter."""
        mock_list_response = {
            "evaluations": [mock_evaluation_result],
            "total_count": 1,
            "session_id": "session_123",
            "task_filter": "task_1"
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_list_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await rest_client.list_evaluations("session_123", task_id="task_1")

            assert result.task_filter == "task_1"
            # Verify the request was made with correct params
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["params"] == {"task_id": "task_1"}

    @pytest.mark.asyncio
    async def test_get_evaluation_summary_success(self, rest_client):
        """Test successful evaluation summary retrieval."""
        mock_summary_response = {
            "session_id": "session_123",
            "total_episodes": 5,
            "successful_episodes": 3,
            "average_score": 0.6,
            "task_summaries": {
                "task_1": {"episodes": 3, "avg_score": 0.8},
                "task_2": {"episodes": 2, "avg_score": 0.3}
            }
        }

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_summary_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await rest_client.get_evaluation_summary("session_123")

            assert isinstance(result, EvaluationSummaryResponse)
            assert result.session_id == "session_123"
            assert result.total_episodes == 5
            assert result.successful_episodes == 3
            assert result.average_score == 0.6


class TestClientSessionManagerEvaluation:
    """Test ClientSessionManager evaluation methods."""

    @pytest.fixture
    def mock_config(self):
        """Mock session manager config."""
        return SessionManagerConfig(
            base_url="http://localhost:8000",
            mcp_url="http://localhost:3001",
            client_id="test_client",
            rest_timeout=30.0,
            mcp_timeout=30.0
        )

    @pytest.fixture
    def session_manager(self, mock_config):
        """Create session manager for testing."""
        return ClientSessionManager(mock_config)

    @pytest.fixture
    def mock_evaluation_result(self):
        """Mock evaluation result response."""
        return EvaluationResultResponse(
            episode_id="ep_123",
            task_id="task_1",
            strategy="static",
            raw_score=0.8,
            max_score=1.0,
            score=0.8,
            success=True,
            timestamp=datetime.now(timezone.utc),
            details={"test": "data"}
        )

    @pytest.mark.asyncio
    async def test_get_episode_evaluation(self, session_manager, mock_evaluation_result):
        """Test get episode evaluation via session manager."""
        mock_response = EvaluationResponse(
            evaluation_result=mock_evaluation_result,
            session_id="session_123"
        )

        with patch.object(session_manager.rest_client, 'get_evaluation', return_value=mock_response) as mock_get:
            result = await session_manager.get_episode_evaluation("session_123", "ep_123")

            assert result == mock_evaluation_result
            mock_get.assert_called_once_with("session_123", "ep_123")

    @pytest.mark.asyncio
    async def test_get_session_evaluations(self, session_manager, mock_evaluation_result):
        """Test get session evaluations via session manager."""
        mock_response = EvaluationListResponse(
            evaluations=[mock_evaluation_result],
            total_count=1,
            session_id="session_123",
            task_filter=None
        )

        with patch.object(session_manager.rest_client, 'list_evaluations', return_value=mock_response) as mock_list:
            result = await session_manager.get_session_evaluations("session_123")

            assert len(result) == 1
            assert result[0] == mock_evaluation_result
            mock_list.assert_called_once_with("session_123", None)

    @pytest.mark.asyncio
    async def test_get_session_evaluations_with_task_filter(self, session_manager, mock_evaluation_result):
        """Test get session evaluations with task filter."""
        mock_response = EvaluationListResponse(
            evaluations=[mock_evaluation_result],
            total_count=1,
            session_id="session_123",
            task_filter="task_1"
        )

        with patch.object(session_manager.rest_client, 'list_evaluations', return_value=mock_response) as mock_list:
            result = await session_manager.get_session_evaluations("session_123", task_id="task_1")

            assert len(result) == 1
            assert result[0] == mock_evaluation_result
            mock_list.assert_called_once_with("session_123", "task_1")

    @pytest.mark.asyncio
    async def test_get_session_evaluation_summary(self, session_manager):
        """Test get session evaluation summary."""
        mock_summary = EvaluationSummaryResponse(
            session_id="session_123",
            total_episodes=5,
            successful_episodes=3,
            average_score=0.6,
            task_summaries={}
        )

        with patch.object(session_manager.rest_client, 'get_evaluation_summary', return_value=mock_summary) as mock_summary_call:
            result = await session_manager.get_session_evaluation_summary("session_123")

            assert result == mock_summary
            mock_summary_call.assert_called_once_with("session_123")

    @pytest.mark.asyncio
    async def test_evaluation_error_propagation(self, session_manager):
        """Test that evaluation errors propagate correctly."""
        with patch.object(session_manager.rest_client, 'get_evaluation', side_effect=EvaluationNotFoundError("Not found")):
            with pytest.raises(EvaluationNotFoundError):
                await session_manager.get_episode_evaluation("session_123", "ep_404")


class TestEvaluationErrorHandling:
    """Test evaluation error handling and fail-fast behavior."""

    @pytest.fixture
    def rest_client(self):
        return SABERRestClient("http://localhost:8000")

    @pytest.mark.asyncio
    async def test_fail_fast_on_unknown_status_code(self, rest_client):
        """Test that unknown status codes raise EvaluationRetrievalError."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 503
            mock_response.text = AsyncMock(return_value="Service unavailable")
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(EvaluationRetrievalError) as exc_info:
                await rest_client.get_evaluation("session_123", "ep_123")

            assert "503" in str(exc_info.value)
            assert "Service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_silent_fallbacks(self, rest_client):
        """Test that there are no silent fallbacks - all errors surface."""
        # Test each error condition to ensure fail-fast behavior
        error_conditions = [
            (404, EvaluationNotFoundError),
            (422, InvalidEvaluationRequestError),
            (500, SessionEvaluationError),
            (503, EvaluationRetrievalError),
        ]

        for status_code, expected_exception in error_conditions:
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_response = AsyncMock()
                mock_response.status = status_code
                mock_response.text = AsyncMock(return_value=f"Error {status_code}")
                mock_get.return_value.__aenter__.return_value = mock_response

                with pytest.raises(expected_exception):
                    await rest_client.get_evaluation("session_123", "ep_123")

    def test_exception_details_include_context(self):
        """Test that exceptions include detailed context for debugging."""
        error = EvaluationNotFoundError(
            "Test error",
            details={"session_id": "session_123", "episode_id": "ep_123"}
        )

        assert error.details["session_id"] == "session_123"
        assert error.details["episode_id"] == "ep_123"
        assert "Test error" in str(error)
