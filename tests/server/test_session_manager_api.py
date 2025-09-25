"""
Unit tests for SessionManager REST API endpoints.

Tests FastAPI routes and HTTP interactions for session management, episodes,
policy, status, and events. Tool execution is tested separately for MCP API.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saber.server.policy.policy_manager import PolicyDocument
from saber.server.session_manager import SessionManager


class TestSessionManagerAPI:
    """Test SessionManager FastAPI endpoints."""

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

        with (
            patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
        ):

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8003)
            return manager, TestClient(manager.rest_api.app)

    def test_health_endpoint(self, session_manager_app):
        """Test health check endpoint."""
        manager, client = session_manager_app

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["domain"] == "test_domain"

    def test_create_session_endpoint(self, session_manager_app):
        """Test session creation endpoint."""
        manager, client = session_manager_app

        response = client.post("/api/v1/session?client_id=test_client")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["message"] == "Session created successfully"

        # Verify session was actually created
        assert len(manager.active_sessions) == 1



    def test_terminate_session_endpoint(self, session_manager_app):
        """Test session termination endpoint."""
        manager, client = session_manager_app

        # Create session first
        create_response = client.post("/api/v1/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Terminate session
        response = client.delete(f"/api/v1/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session terminated successfully"

        # Verify session was removed
        assert len(manager.active_sessions) == 0

    def test_terminate_nonexistent_session_endpoint(self, session_manager_app):
        """Test terminating non-existent session via endpoint."""
        manager, client = session_manager_app

        response = client.delete("/api/v1/session/nonexistent_id")

        assert response.status_code == 404

    def test_start_episode_endpoint(self, session_manager_app):
        """Test starting individual episodes endpoint."""
        manager, client = session_manager_app

        # Mock task with proper initial_context
        mock_task = MagicMock()
        mock_task.initial_context = {"initial_data": "test"}
        manager.benchmark_manager.get_task.return_value = mock_task

        # Mock episode
        from saber.server.base import EpisodeState
        mock_episode = MagicMock()
        mock_episode.episode_id = "episode_123"
        mock_episode.state = EpisodeState.ACTIVE
        mock_episode.max_steps = 10
        mock_episode.metadata = {"test": "data"}
        mock_episode.attached_to_episode_id = None  # Set proper type for new dependency feature

        # Create session first
        create_response = client.post("/api/v1/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Mock the start_episode method
        with patch.object(manager, 'start_episode', return_value=mock_episode):
            # Start individual episode using correct endpoint
            response = client.post(f"/api/v1/session/{session_id}/episodes?task_id=task_456")

            assert response.status_code == 200
            data = response.json()
            assert data["episode_id"] == "episode_123"
            assert data["task_id"] == "task_456"
            assert data["session_id"] == session_id
            assert data["message"] == "Episode created successfully"

    def test_get_current_task_endpoint(self, session_manager_app):
        """Test get current task endpoint."""
        manager, client = session_manager_app

        # Create session
        create_response = client.post("/api/v1/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Set up session with episode
        session = list(manager.active_sessions.values())[0]
        session.add_active_episode("episode_123")

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
        # Mock episode config to return a proper dictionary
        manager.benchmark_manager.get_episode_config.return_value = {
            "task_timeout": 300,
            "max_steps": 50
        }

        # Mock the get_current_task method with episode_id parameter (episode-first)
        with patch.object(manager, 'get_current_task', return_value=mock_task) as mock_get_current_task, \
             patch.object(manager, 'get_episode_by_id', return_value=mock_episode) as mock_get_episode:
            # Get task for specific episode (episode-first architecture)
            response = client.get(f"/api/v1/session/{session_id}/episodes/episode_123/task")

            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == "task_456"
            assert data["title"] == "Test Task"
            assert "episode_context" in data

            # Verify get_current_task was called with session_id and episode_id
            mock_get_current_task.assert_called_once_with(session_id, "episode_123")

    def test_get_policy_endpoint(self, session_manager_app):
        """Test get policy endpoint."""
        manager, client = session_manager_app

        # Create session
        create_response = client.post("/api/v1/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        # Set up session with episode for episode-first policy access
        session = list(manager.active_sessions.values())[0]
        session.add_active_episode("episode_123")

        # Mock episode and policy for episode-specific policy retrieval
        mock_episode = MagicMock()
        mock_episode.episode_id = "episode_123"

        # Mock policy manager to return policy for episode
        mock_policy = PolicyDocument(prompt="Test domain policy prompt")
        with patch.object(manager, 'get_policy', return_value=mock_policy) as mock_get_policy, \
             patch.object(manager, 'get_episode_by_id', return_value=mock_episode) as mock_get_episode:
            # Get policy for specific episode (episode-first architecture)
            response = client.get(f"/api/v1/session/{session_id}/episodes/episode_123/policy")

            assert response.status_code == 200
            data = response.json()
            assert "prompt" in data
            assert data["prompt"] == "Test domain policy prompt"

            # Verify get_policy was called with session_id and episode_id
            mock_get_policy.assert_called_once_with(session_id, "episode_123")

    def test_invalid_session_endpoints(self, session_manager_app):
        """Test endpoints with invalid session IDs."""
        manager, client = session_manager_app

        invalid_session_id = "invalid_session_123"

        # Test various endpoints with invalid session (updated for current API)
        endpoints_to_test = [
            ("DELETE", f"/api/v1/session/{invalid_session_id}"),
            ("POST", f"/api/v1/session/{invalid_session_id}/episodes?task_id=task1"),
            ("GET", f"/api/v1/session/{invalid_session_id}/episodes/episode_123/task"),
            ("GET", f"/api/v1/session/{invalid_session_id}/episodes/episode_123/policy"),
            ("GET", f"/api/v1/session/{invalid_session_id}/evaluations"),
            ("GET", f"/api/v1/session/{invalid_session_id}/evaluations/episode_123"),
        ]

        for method, endpoint in endpoints_to_test:
            response = client.request(method, endpoint)
            assert response.status_code == 404, f"Endpoint {method} {endpoint} should return 404"

    def test_get_evaluation_criteria_success(self, session_manager_app):
        """Test successful retrieval of evaluation criteria."""
        manager, client = session_manager_app

        # Create a session first
        create_response = client.post("/api/v1/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        episode_id = "test_episode_456"

        # Mock the episode manager to return a completed episode
        from saber.server.base import EpisodeState
        mock_episode = MagicMock()
        mock_episode.episode_id = episode_id
        mock_episode.submission = "test submission"
        mock_episode.is_complete = True
        mock_episode.task_id = "test_task"
        mock_episode.state = EpisodeState.COMPLETED  # Use proper enum
        mock_episode.steps = []
        manager.episode_manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock the benchmark manager to return a task
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.title = "Test Task"
        mock_task.description = "Test Description"
        mock_task.domain = "test"
        mock_task.evaluation_config = {"strategy": "llm_judge", "model": "gpt-4"}
        mock_task.subtasks = []  # Required for task_context construction
        manager.benchmark_manager.get_task = MagicMock(return_value=mock_task)

        # Mock the prompt generator to return judge prompts
        mock_judge_payload = MagicMock()
        mock_judge_payload.messages = [
            {"role": "system", "content": "Judge system prompt"},
            {"role": "user", "content": "Judge user prompt"}
        ]
        mock_judge_payload.model = "gpt-4"
        manager.benchmark_manager.prompt_generator.render_judge_prompt_for_episode = MagicMock(return_value=mock_judge_payload)

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/evaluation-criteria")

        assert response.status_code == 200
        data = response.json()
        assert "judge_messages" in data
        assert "session_id" in data
        assert "episode_id" in data
        assert "task_id" in data
        assert "submission" in data
        assert data["session_id"] == session_id
        assert data["episode_id"] == episode_id
        assert data["task_id"] == "test_task"

    def test_get_evaluation_criteria_session_not_found(self, session_manager_app):
        """Test evaluation criteria endpoint with non-existent episode."""
        manager, client = session_manager_app

        invalid_session_id = "nonexistent_session"
        episode_id = "test_episode"

        # Mock episode_manager to return None for nonexistent episode
        manager.episode_manager.get_episode_by_id = MagicMock(return_value=None)

        response = client.get(f"/api/v1/session/{invalid_session_id}/episodes/{episode_id}/evaluation-criteria")

        assert response.status_code == 404
        assert f"Episode {episode_id} not found" in response.json()["detail"]

    def test_get_evaluation_criteria_template_rendering_error(self, session_manager_app):
        """Test evaluation criteria endpoint when template rendering fails."""
        manager, client = session_manager_app

        # Create a session first
        create_response = client.post("/api/v1/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        episode_id = "test_episode_456"

        # Mock the episode manager to return a completed episode
        from saber.server.base import EpisodeState
        mock_episode = MagicMock()
        mock_episode.episode_id = episode_id
        mock_episode.submission = "test submission"
        mock_episode.is_complete = True
        mock_episode.task_id = "test_task"
        mock_episode.state = EpisodeState.COMPLETED  # Use proper enum
        mock_episode.steps = []
        manager.episode_manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock the benchmark manager to return a task
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.title = "Test Task"
        mock_task.description = "Test Description"
        mock_task.domain = "test"
        mock_task.evaluation_config = {"strategy": "llm_judge", "model": "gpt-4"}
        mock_task.subtasks = []  # Required for task_context construction
        manager.benchmark_manager.get_task = MagicMock(return_value=mock_task)

        # Mock the prompt generator to raise an error
        manager.benchmark_manager.prompt_generator.render_judge_prompt_for_episode = MagicMock(side_effect=Exception("Template rendering failed"))

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/evaluation-criteria")

        # The actual implementation returns 200 with judge_messages=None on template errors
        # The error is logged but doesn't fail the endpoint
        assert response.status_code == 200
        data = response.json()
        assert data["judge_messages"] is None

    def test_get_evaluation_criteria_attribute_error(self, session_manager_app):
        """Test evaluation criteria endpoint when method doesn't exist (regression test)."""
        manager, client = session_manager_app

        # Create a session first
        create_response = client.post("/api/v1/session?client_id=test_client")
        session_id = create_response.json()["session_id"]

        episode_id = "test_episode_456"

        # Mock the episode manager to return a completed episode
        from saber.server.base import EpisodeState
        mock_episode = MagicMock()
        mock_episode.episode_id = episode_id
        mock_episode.submission = "test submission"
        mock_episode.is_complete = True
        mock_episode.task_id = "test_task"
        mock_episode.state = EpisodeState.COMPLETED  # Use proper enum
        mock_episode.steps = []
        manager.episode_manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock the benchmark manager to return a task
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.title = "Test Task"
        mock_task.description = "Test Description"
        mock_task.domain = "test"
        mock_task.evaluation_config = {"strategy": "llm_judge", "model": "gpt-4"}
        mock_task.subtasks = []  # Required for task_context construction
        manager.benchmark_manager.get_task = MagicMock(return_value=mock_task)

        # Mock the prompt generator to raise AttributeError (like the bug we fixed)
        manager.benchmark_manager.prompt_generator.render_judge_prompt_for_episode = MagicMock(side_effect=AttributeError("'PromptGenerator' object has no attribute 'render_judge_prompt_for_task'"))

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/evaluation-criteria")

        # The actual implementation returns 200 with judge_messages=None on template errors
        # The AttributeError is caught and logged but doesn't fail the endpoint
        assert response.status_code == 200
        data = response.json()
        assert data["judge_messages"] is None
