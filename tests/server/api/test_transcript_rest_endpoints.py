"""Tests for transcript retrieval REST endpoints.

Tests GET /transcript and GET /transcript/count endpoints
that expose TranscriptCoordinator's get_transcript() and get_message_count().
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from saber.server.api.session_rest_api import SessionRestAPI

SESSION_ID = "session-001"
EPISODE_ID = "episode-001"


@pytest.fixture
def mock_session_manager() -> MagicMock:
    """Create a mock SessionManager with transcript coordinator."""
    manager = MagicMock()
    manager.domain_name = "test_domain"

    # Coordinator mock
    coordinator = MagicMock()
    coordinator.get_transcript = AsyncMock(return_value=[])
    coordinator.get_message_count = AsyncMock(return_value=0)
    manager.episode_manager.transcript_coordinator = coordinator

    return manager


@pytest.fixture
def test_client(mock_session_manager: MagicMock) -> TestClient:
    """Create a FastAPI TestClient wired to the REST API."""
    api = SessionRestAPI(session_manager=mock_session_manager, host="localhost", port=8000)
    return TestClient(api.app)


def _transcript_url(session_id: str = SESSION_ID, episode_id: str = EPISODE_ID) -> str:
    return f"/api/v1/session/{session_id}/episodes/{episode_id}/transcript"


def _count_url(session_id: str = SESSION_ID, episode_id: str = EPISODE_ID) -> str:
    return f"/api/v1/session/{session_id}/episodes/{episode_id}/transcript/count"


class TestGetTranscript:
    """Tests for GET /transcript endpoint."""

    def test_returns_messages(
        self, test_client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        """GET /transcript returns messages from coordinator."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        coordinator = mock_session_manager.episode_manager.transcript_coordinator
        coordinator.get_transcript = AsyncMock(return_value=messages)
        mock_session_manager.get_episode_by_id.return_value = MagicMock()

        response = test_client.get(_transcript_url())

        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "hello"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "hi there"
        assert data["message_count"] == 2
        assert data["episode_id"] == EPISODE_ID
        coordinator.get_transcript.assert_awaited_once_with(EPISODE_ID, 0)

    def test_with_since_sequence(
        self, test_client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        """GET /transcript?since_sequence=5 passes parameter to coordinator."""
        coordinator = mock_session_manager.episode_manager.transcript_coordinator
        coordinator.get_transcript = AsyncMock(
            return_value=[{"role": "assistant", "content": "later msg"}]
        )
        mock_session_manager.get_episode_by_id.return_value = MagicMock()

        response = test_client.get(_transcript_url(), params={"since_sequence": 5})

        assert response.status_code == 200
        data = response.json()
        assert data["message_count"] == 1
        coordinator.get_transcript.assert_awaited_once_with(EPISODE_ID, 5)

    def test_404_for_missing_episode(
        self, test_client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        """GET /transcript returns 404 when episode not found."""
        mock_session_manager.get_episode_by_id.return_value = None

        response = test_client.get(_transcript_url())

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_empty_list_when_no_repo(
        self, test_client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        """GET /transcript returns empty list when coordinator has no DB repo."""
        coordinator = mock_session_manager.episode_manager.transcript_coordinator
        coordinator.get_transcript = AsyncMock(return_value=[])
        mock_session_manager.get_episode_by_id.return_value = MagicMock()

        response = test_client.get(_transcript_url())

        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []
        assert data["message_count"] == 0


class TestGetTranscriptCount:
    """Tests for GET /transcript/count endpoint."""

    def test_returns_count(
        self, test_client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        """GET /transcript/count returns message count from coordinator."""
        coordinator = mock_session_manager.episode_manager.transcript_coordinator
        coordinator.get_message_count = AsyncMock(return_value=42)
        mock_session_manager.get_episode_by_id.return_value = MagicMock()

        response = test_client.get(_count_url())

        assert response.status_code == 200
        assert response.json() == {"count": 42}
        coordinator.get_message_count.assert_awaited_once_with(EPISODE_ID)

    def test_404_for_missing_episode(
        self, test_client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        """GET /transcript/count returns 404 when episode not found."""
        mock_session_manager.get_episode_by_id.return_value = None

        response = test_client.get(_count_url())

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
