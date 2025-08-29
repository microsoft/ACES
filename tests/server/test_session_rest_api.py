"""
Unit tests for SessionRestAPI endpoints.

Tests FastAPI routes and HTTP interactions for session management, episodes,
policy, status, and events. Tool execution is tested separately for MCP API.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saber.server.policy.policy_manager import PolicyDocument
from saber.server.session_manager import SessionManager


class TestSessionRestAPI:
    """Test SessionRestAPI FastAPI endpoints."""

    @pytest.fixture
    def session_manager_app(self):
        """Create SessionManager with test client."""
        mock_task_manager = MagicMock()
        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_policy_manager = MagicMock()
        mock_policy_doc = PolicyDocument(prompt="Test domain policy prompt")
        mock_policy_manager.get_policy = MagicMock(return_value=mock_policy_doc)
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

        mock_permanent_environment_manager = MagicMock()
        mock_container_cleanup_manager = MagicMock()

        with (
            patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
            patch("saber.server.session_manager.PermanentEnvironmentManager", return_value=mock_permanent_environment_manager),
            patch("saber.server.session_manager.ContainerCleanupManager", return_value=mock_container_cleanup_manager),
        ):

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8003)
            return manager, TestClient(manager.rest_api.app)

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

    def test_session_stats_endpoint(self, session_manager_app):
        """Test session statistics endpoint."""
        manager, client = session_manager_app

        # Create a few sessions
        client.post("/session?client_id=client1")
        client.post("/session?client_id=client2")

        response = client.get("/sessions/stats")

        assert response.status_code == 200
        data = response.json()

        # Verify expected fields
        assert "total_sessions" in data
        assert "timeout_minutes" in data
        assert "cleanup_interval_minutes" in data
        assert "sessions" in data

        assert data["total_sessions"] == 2
        assert len(data["sessions"]) == 2

        # Verify session details
        for session_info in data["sessions"]:
            assert "session_id" in session_info
            assert "client_id" in session_info
            assert "uptime_seconds" in session_info
            assert "time_since_activity_seconds" in session_info
            assert "is_active" in session_info

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
        """Test starting individual episodes endpoint."""
        manager, client = session_manager_app

        # Mock task with proper initial_context
        mock_task = MagicMock()
        mock_task.initial_context = {"initial_data": "test"}
        manager.benchmark_manager.get_task.return_value = mock_task

        # Mock episode
        mock_episode = MagicMock()
        mock_episode.episode_id = "episode_123"

        # Create session first
        create_response = client.post("/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Mock the start_episode method
        with patch.object(manager, 'start_episode', return_value=mock_episode):
            # Start individual episode
            response = client.post(f"/session/{session_id}/start-episode?task_id=task_456")

            assert response.status_code == 200
            data = response.json()
            assert data["episode_id"] == "episode_123"
            assert data["task_id"] == "task_456"
            assert data["session_id"] == session_id
            assert data["message"] == "Episode started successfully"

    def test_get_benchmark_endpoint(self, session_manager_app):
        """Test get benchmark endpoint for client orchestration."""
        manager, client = session_manager_app

        # Import the BenchmarkInfo and TaskInfo classes
        from saber.server.benchmarks.benchmark_info import BenchmarkInfo, TaskInfo

        # Create mock BenchmarkInfo object
        mock_tasks = [
            TaskInfo(
                task_id="task_1",
                title="Test Task 1",
                description="First test task",
                episode_attempts=2,
                subtask_count=3
            ),
            TaskInfo(
                task_id="task_2",
                title="Test Task 2",
                description="Second test task",
                episode_attempts=1,
                subtask_count=2
            )
        ]

        mock_benchmark_info = BenchmarkInfo(
            domain="test_domain",
            tasks=mock_tasks,
            total_tasks=2,
            total_episodes=3
        )

        # Mock the get_benchmark_info method
        with patch.object(manager, 'get_benchmark_info', return_value=mock_benchmark_info):
            # Call get benchmark endpoint
            response = client.get("/get-benchmark")

            assert response.status_code == 200
            data = response.json()

            # Verify the new structure
            assert data["domain"] == "test_domain"
            assert data["total_tasks"] == 2
            assert data["total_episodes"] == 3
            assert len(data["tasks"]) == 2

            # Verify task structure
            task_1 = data["tasks"][0]
            assert task_1["task_id"] == "task_1"
            assert task_1["title"] == "Test Task 1"
            assert task_1["episode_attempts"] == 2
            assert task_1["subtask_count"] == 3

            task_2 = data["tasks"][1]
            assert task_2["task_id"] == "task_2"
            assert task_2["title"] == "Test Task 2"
            assert task_2["episode_attempts"] == 1
            assert task_2["subtask_count"] == 2

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
        mock_task.to_dict.return_value = {
            "task_id": "task_456",
            "title": "Test Task",
            "description": "Test description"
        }

        manager.episode_manager.get_current_episode.return_value = mock_episode
        manager.benchmark_manager.get_task.return_value = mock_task

        # Get current task
        response = client.get(f"/session/{session_id}/current-task")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_456"
        assert data["title"] == "Test Task"
        assert "episode_context" in data
        assert data["episode_context"]["session_id"] == session_id

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
        assert "prompt" in data
        assert data["prompt"] == "Test domain policy prompt"

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
