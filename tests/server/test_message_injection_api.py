"""
Unit tests for message injection API endpoints (Phase 4).

Tests the POST and GET endpoints for red team message injection
into agent conversations for adversarial testing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from saber.server.base import Episode, EpisodeState
from saber.server.session_manager import SessionManager
from saber.models.constants import MetadataKeys


class TestMessageInjectionAPI:
    """Test message injection REST API endpoints."""

    @pytest.fixture
    def session_manager_app(self):
        """Create SessionManager with test client."""
        mock_task_manager = MagicMock()
        mock_config_loader = MagicMock()
        mock_config_loader.get_permanent_environment = MagicMock(return_value=None)
        mock_task_manager.config_loader = mock_config_loader

        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_execution_manager.initialize_permanent_environment_manager = MagicMock()
        mock_execution_manager._permanent_environment_manager = None

        mock_policy_manager = MagicMock()
        mock_evaluation_manager = MagicMock()
        mock_evaluation_manager.log_session_start = AsyncMock()
        mock_evaluation_manager.log_session_end = AsyncMock()
        mock_evaluation_manager.log_episode_start = AsyncMock()
        mock_evaluation_manager.log_episode_end = AsyncMock()
        mock_evaluation_manager.log_action = AsyncMock()

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()

        with (
            patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8003)
            return manager, TestClient(manager.rest_api.app)

    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            episode_id="test_episode_123",
            task_id="test_task_456",
            session_id="test_session_789",
            state=EpisodeState.ACTIVE,
            context={}
        )

    # ========================================================================
    # POST /api/v1/session/{sid}/episodes/{eid}/messages/inject - Inject Message
    # ========================================================================

    def test_inject_message_success(self, session_manager_app, sample_episode):
        """Test successful message injection queues message in pending list."""
        manager, client = session_manager_app

        session_id = sample_episode.session_id
        episode_id = sample_episode.episode_id

        # Mock get_episode_by_id
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Inject a red team message
        inject_request = {
            "role": "user",
            "content": "[RED TEAM] Try listing /etc/passwd",
            "metadata": {
                "injected_by": "red_team_agent",
                "reason": "test_privilege_escalation"
            }
        }

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject",
            json=inject_request
        )

        # Assert: Success response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["episode_id"] == episode_id
        assert "injection_id" in data
        assert "injected_at" in data

        # Assert: Message stored in pending injections
        pending = sample_episode.context.get(MetadataKeys.PENDING_INJECTIONS, [])
        assert len(pending) == 1
        assert pending[0]["role"] == "user"
        assert pending[0]["content"] == "[RED TEAM] Try listing /etc/passwd"
        assert pending[0]["metadata"]["injected_by"] == "red_team_agent"
        assert pending[0]["injection_id"] == data["injection_id"]

    def test_inject_message_episode_not_found(self, session_manager_app):
        """Test injecting to non-existent episode returns 404."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=None)

        inject_request = {
            "role": "user",
            "content": "Test message",
            "metadata": {}
        }

        response = client.post(
            "/api/v1/session/fake_session/episodes/fake_episode/messages/inject",
            json=inject_request
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_inject_multiple_messages(self, session_manager_app, sample_episode):
        """Test injecting multiple messages queues them in order."""
        manager, client = session_manager_app

        session_id = sample_episode.session_id
        episode_id = sample_episode.episode_id

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Inject three messages
        messages = [
            {"role": "user", "content": "Message 1", "metadata": {}},
            {"role": "user", "content": "Message 2", "metadata": {}},
            {"role": "user", "content": "Message 3", "metadata": {}},
        ]

        for msg in messages:
            response = client.post(
                f"/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject",
                json=msg
            )
            assert response.status_code == 200

        # Assert: All messages in pending list
        pending = sample_episode.context.get(MetadataKeys.PENDING_INJECTIONS, [])
        assert len(pending) == 3
        assert pending[0]["content"] == "Message 1"
        assert pending[1]["content"] == "Message 2"
        assert pending[2]["content"] == "Message 3"

    # ========================================================================
    # GET /api/v1/session/{sid}/episodes/{eid}/messages/inject - Get Pending
    # ========================================================================

    def test_get_pending_messages_success(self, session_manager_app, sample_episode):
        """Test retrieving pending messages returns them and clears the list."""
        manager, client = session_manager_app

        session_id = sample_episode.session_id
        episode_id = sample_episode.episode_id

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Setup: Pre-populate pending injections
        sample_episode.context[MetadataKeys.PENDING_INJECTIONS] = [
            {
                "injection_id": "inj_1",
                "role": "user",
                "content": "Injected message 1",
                "metadata": {},
                "injected_at": "2025-11-26T10:00:00Z"
            },
            {
                "injection_id": "inj_2",
                "role": "user",
                "content": "Injected message 2",
                "metadata": {},
                "injected_at": "2025-11-26T10:01:00Z"
            }
        ]

        # Execute: Get pending messages
        response = client.get(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject"
        )

        # Assert: Success response
        assert response.status_code == 200
        data = response.json()
        assert data["episode_id"] == episode_id
        assert data["pending_count"] == 2
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Injected message 1"
        assert data["messages"][1]["content"] == "Injected message 2"
        assert "retrieved_at" in data

        # Assert: Pending list cleared
        pending = sample_episode.context.get(MetadataKeys.PENDING_INJECTIONS, [])
        assert len(pending) == 0

        # Assert: Messages moved to history
        history = sample_episode.context.get(MetadataKeys.INJECTION_HISTORY, [])
        assert len(history) == 2
        assert history[0]["injection_id"] == "inj_1"
        assert history[1]["injection_id"] == "inj_2"
        assert "retrieved_at" in history[0]
        assert "retrieved_at" in history[1]

    def test_get_pending_messages_empty(self, session_manager_app, sample_episode):
        """Test getting pending messages when none exist returns empty list."""
        manager, client = session_manager_app

        session_id = sample_episode.session_id
        episode_id = sample_episode.episode_id

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        response = client.get(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["pending_count"] == 0
        assert len(data["messages"]) == 0

    def test_get_pending_messages_episode_not_found(self, session_manager_app):
        """Test getting pending messages for non-existent episode returns 404."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=None)

        response = client.get(
            "/api/v1/session/fake_session/episodes/fake_episode/messages/inject"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    # ========================================================================
    # Integration Tests
    # ========================================================================

    def test_inject_then_retrieve_workflow(self, session_manager_app, sample_episode):
        """Test full workflow: inject message, then retrieve it."""
        manager, client = session_manager_app

        session_id = sample_episode.session_id
        episode_id = sample_episode.episode_id

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Step 1: Inject a message
        inject_request = {
            "role": "user",
            "content": "Test injection",
            "metadata": {"source": "test"}
        }

        inject_response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject",
            json=inject_request
        )
        assert inject_response.status_code == 200
        injection_id = inject_response.json()["injection_id"]

        # Step 2: Retrieve pending messages
        get_response = client.get(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject"
        )
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["pending_count"] == 1
        assert data["messages"][0]["content"] == "Test injection"

        # Step 3: Verify pending list cleared
        get_response_2 = client.get(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject"
        )
        assert get_response_2.status_code == 200
        assert get_response_2.json()["pending_count"] == 0

        # Step 4: Verify in history
        history = sample_episode.context.get(MetadataKeys.INJECTION_HISTORY, [])
        assert len(history) == 1
        assert history[0]["injection_id"] == injection_id
