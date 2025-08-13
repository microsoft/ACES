"""
Unit tests for SessionRestAPI endpoints.

Tests FastAPI routes and HTTP interactions for session management, episodes,
policy, status, and events. Tool execution is tested separately for MCP API.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from saber.server.session_manager import SessionManager
from saber.server.policy.policy_manager import PolicyDocument


class TestSessionRestAPI:
    """Test SessionRestAPI FastAPI endpoints."""

    @pytest.fixture
    def session_manager_app(self):
        """Create SessionManager with test client."""
        mock_task_manager = MagicMock()
        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_policy_manager = MagicMock()
        mock_policy_doc = PolicyDocument(domain="test_domain")
        mock_policy_manager.get_policy = MagicMock(return_value=mock_policy_doc)
        mock_evaluation_manager = MagicMock()
        mock_evaluation_manager.log_session_start = AsyncMock()
        mock_evaluation_manager.log_session_end = AsyncMock()
        mock_evaluation_manager.log_episode_start = AsyncMock()
        mock_evaluation_manager.log_episode_end = AsyncMock()
        mock_evaluation_manager.log_action = AsyncMock()

        with patch('saber.server.session_manager.TaskManager', return_value=mock_task_manager), \
             patch('saber.server.session_manager.ExecutionManager', return_value=mock_execution_manager), \
             patch('saber.server.session_manager.PolicyManager', return_value=mock_policy_manager), \
             patch('saber.server.session_manager.EvaluationManager', return_value=mock_evaluation_manager):

            manager = SessionManager(
                domain_name="test_domain",
                tasks_config_path="/tmp/test_tasks.yaml",
                host="127.0.0.1",
                port=8003
            )

            return manager, TestClient(manager.app)

    def test_health_endpoint(self, session_manager_app):
        """Test health check endpoint."""
        manager, client = session_manager_app

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["domain"] == "test_domain"

    def test_create_session_endpoint(self, session_manager_app):
        """Test session creation endpoint."""
        manager, client = session_manager_app

        response = client.post("/session?client_id=test_client")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["message"] == "Session created successfully"

        # Verify session was actually created
        assert len(manager.active_sessions) == 1

    def test_list_sessions_endpoint(self, session_manager_app):
        """Test listing sessions endpoint."""
        manager, client = session_manager_app

        # Create a few sessions
        client.post("/session?client_id=client1")
        client.post("/session?client_id=client2")

        response = client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data["active_session_count"] == 2
        assert "sessions" in data
        assert len(data["sessions"]) == 2

    def test_terminate_session_endpoint(self, session_manager_app):
        """Test session termination endpoint."""
        manager, client = session_manager_app

        # Create session first
        create_response = client.post("/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Terminate session
        response = client.delete(f"/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session terminated successfully"

        # Verify session was removed
        assert len(manager.active_sessions) == 0

    def test_terminate_nonexistent_session_endpoint(self, session_manager_app):
        """Test terminating non-existent session via endpoint."""
        manager, client = session_manager_app

        response = client.delete("/session/nonexistent_id")

        assert response.status_code == 404

    def test_start_episode_endpoint(self, session_manager_app):
        """Test starting episode endpoint."""
        manager, client = session_manager_app

        # Mock episode
        mock_episode = MagicMock()
        mock_episode.episode_id = "episode_123"
        mock_episode.task_id = "task_456"
        manager.task_manager.start_episode.return_value = mock_episode

        # Create session first
        create_response = client.post("/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Start episode
        response = client.post(f"/session/{session_id}/start-episode?task_id=task_456")

        assert response.status_code == 200
        data = response.json()
        assert data["episode_id"] == "episode_123"
        assert data["task_id"] == "task_456"
        assert data["message"] == "Episode started successfully"

    def test_get_current_task_endpoint(self, session_manager_app):
        """Test get current task endpoint."""
        manager, client = session_manager_app

        # Create session
        create_response = client.post("/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Set up session with episode
        session = list(manager.active_sessions.values())[0]
        session.current_episode_id = "episode_123"

        # Mock episode and task
        mock_episode = MagicMock()
        mock_episode.episode_id = "episode_123"
        mock_episode.task_id = "task_456"
        mock_episode.current_subtask = "subtask_1"
        mock_episode.completed_subtasks = set()
        mock_episode.in_progress_subtasks = {"subtask_1"}
        mock_episode.not_visited_subtasks = {"subtask_2"}

        mock_task = MagicMock()
        mock_task.task_id = "task_456"
        mock_task.title = "Test Task"
        mock_task.description = "Test description"

        manager.task_manager.get_current_episode.return_value = mock_episode
        manager.task_manager.get_task.return_value = mock_task

        # Get current task
        response = client.get(f"/session/{session_id}/current-task")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_456"
        assert data["title"] == "Test Task"
        assert data["episode_id"] == "episode_123"

    def test_get_policy_endpoint(self, session_manager_app):
        """Test get policy endpoint."""
        manager, client = session_manager_app

        # Create session
        create_response = client.post("/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Get policy
        response = client.get(f"/session/{session_id}/policy")

        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "test_domain"
        assert "available_commands" in data
        assert "guidelines" in data
        assert "constraints" in data

    def test_events_endpoint_structure(self, session_manager_app):
        """Test SSE events endpoint structure (without async streaming)."""
        manager, client = session_manager_app

        # Create session
        create_response = client.post("/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Just test that the endpoint exists by checking it's registered
        # We can't easily test SSE streaming with TestClient without hanging
        # Instead, verify the route exists in the app
        routes = [route.path for route in manager.app.routes]
        assert f"/session/{{session_id}}/events" in routes or "/session/{session_id}/events" in [r.path_regex.pattern for r in manager.app.routes if hasattr(r, 'path_regex')]

    def test_invalid_session_endpoints(self, session_manager_app):
        """Test endpoints with invalid session IDs."""
        manager, client = session_manager_app

        invalid_session_id = "invalid_session_123"

        # Test various endpoints with invalid session
        endpoints_to_test = [
            ("DELETE", f"/session/{invalid_session_id}"),
            ("POST", f"/session/{invalid_session_id}/start-episode?task_id=task1"),
            ("POST", f"/session/{invalid_session_id}/step"),
            ("GET", f"/session/{invalid_session_id}/current-task"),
            ("GET", f"/session/{invalid_session_id}/policy"),
            ("GET", f"/session/{invalid_session_id}/events"),
        ]

        for method, endpoint in endpoints_to_test:
            if method == "POST" and "step" in endpoint:
                response = client.request(method, endpoint, json={"command": "test"})
            else:
                response = client.request(method, endpoint)

            assert response.status_code == 404, f"Endpoint {method} {endpoint} should return 404"
