"""
Tests for ServerClient functionality.

Tests the REST API client for SABER server communication.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from saber.client.server_client import ServerClient
from saber.api_models import StepResponse, TaskInfo, PolicyInfo, EpisodeInfo


@pytest.fixture
def server_client():
    """ServerClient fixture."""
    return ServerClient("http://test-server:8000", ui_enabled=False)


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient."""
    mock_client = AsyncMock()
    return mock_client


class TestServerClient:
    """Test cases for ServerClient."""

    def test_init(self):
        """Test ServerClient initialization."""
        client = ServerClient("http://localhost:8000", timeout=45.0)

        assert client.server_url == "http://localhost:8000"
        assert client.timeout == 45.0
        assert client.session_id is None

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from server URL."""
        client = ServerClient("http://localhost:8000/")
        assert client.server_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_create_session_success(self, server_client):
        """Test successful session creation."""
        mock_response = Mock()
        mock_response.json.return_value = {"session_id": "test-session-123"}

        with patch.object(server_client, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            session_id = await server_client.create_session("test-client")

            assert session_id == "test-session-123"
            assert server_client.session_id == "test-session-123"
            mock_client.post.assert_called_once_with(
                "http://test-server:8000/session",
                params={"client_id": "test-client"}
            )
            mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_default_client_id(self, server_client):
        """Test session creation with default client ID."""
        mock_response = Mock()
        mock_response.json.return_value = {"session_id": "test-session-456"}

        with patch.object(server_client, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            session_id = await server_client.create_session()

            assert session_id == "test-session-456"
            mock_client.post.assert_called_once_with(
                "http://test-server:8000/session",
                params={"client_id": "saber_client"}
            )

    @pytest.mark.asyncio
    async def test_create_session_http_error(self, server_client):
        """Test session creation with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=Mock(), response=Mock()
        )

        with patch.object(server_client, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            with pytest.raises(httpx.HTTPStatusError):
                await server_client.create_session()

    @pytest.mark.asyncio
    async def test_start_episode_success(self, server_client):
        """Test successful episode start."""
        server_client.session_id = "test-session"

        mock_response = Mock()
        mock_response.json.return_value = {
            "episode_id": "episode-123",
            "task_id": "task-456",
            "message": "Episode started"
        }

        with patch.object(server_client, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            episode_info = await server_client.start_episode("malware-task")

            assert isinstance(episode_info, EpisodeInfo)
            assert episode_info.episode_id == "episode-123"
            assert episode_info.task_id == "task-456"
            mock_client.post.assert_called_once_with(
                "http://test-server:8000/session/test-session/start-episode",
                params={"task_id": "malware-task"}
            )

    @pytest.mark.asyncio
    async def test_start_episode_no_session(self, server_client):
        """Test episode start without active session."""
        with pytest.raises(ValueError, match="No active session"):
            await server_client.start_episode()

    @pytest.mark.asyncio
    async def test_execute_step_success(self, server_client):
        """Test successful step execution."""
        server_client.session_id = "test-session"

        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "data": {"output": "command output"},
            "step": {"done": False, "info": "step info"},
            "error": None
        }

        with patch.object(server_client, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            step_response = await server_client.execute_step("ls -la")

            assert isinstance(step_response, StepResponse)
            assert step_response.success is True
            assert step_response.output == "command output"
            assert step_response.done is False
            assert step_response.error is None
            mock_client.post.assert_called_once_with(
                "http://test-server:8000/session/test-session/step",
                json={"command": "ls -la", "parameters": {}}
            )

    @pytest.mark.asyncio
    async def test_execute_step_with_error(self, server_client):
        """Test step execution with error."""
        server_client.session_id = "test-session"

        mock_response = Mock()
        mock_response.json.return_value = {
            "success": False,
            "data": {"output": ""},
            "step": {"done": False},
            "error": "Command failed"
        }

        with patch.object(server_client, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            step_response = await server_client.execute_step("invalid-command")

            assert step_response.success is False
            assert step_response.error == "Command failed"

    @pytest.mark.asyncio
    async def test_execute_step_no_session(self, server_client):
        """Test step execution without active session."""
        with pytest.raises(ValueError, match="No active session"):
            await server_client.execute_step("ls -la")

    @pytest.mark.asyncio
    async def test_get_current_task_success(self, server_client):
        """Test successful task retrieval."""
        server_client.session_id = "test-session"

        mock_response = Mock()
        mock_response.json.return_value = {
            "task_id": "task-123",
            "title": "Malware Analysis",
            "description": "Analyze suspicious file",
            "state": "active"
        }

        with patch.object(server_client, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

            task_info = await server_client.get_current_task()

            assert isinstance(task_info, TaskInfo)
            assert task_info.task_id == "task-123"
            assert task_info.title == "Malware Analysis"
            assert task_info.state == "active"
            mock_client.get.assert_called_once_with(
                "http://test-server:8000/session/test-session/current-task"
            )

    @pytest.mark.asyncio
    async def test_get_current_task_no_session(self, server_client):
        """Test task retrieval without active session."""
        with pytest.raises(ValueError, match="No active session"):
            await server_client.get_current_task()

    @pytest.mark.asyncio
    async def test_get_policy_success(self, server_client):
        """Test successful policy retrieval."""
        server_client.session_id = "test-session"

        mock_response = Mock()
        mock_response.json.return_value = {
            "domain": "malware",
            "available_commands": ["file", "strings", "hexdump"],
            "guidelines": "Use multiple analysis techniques",
            "constraints": ["No network access", "Read-only filesystem"]
        }

        with patch.object(server_client, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

            policy_info = await server_client.get_policy()

            assert isinstance(policy_info, PolicyInfo)
            assert policy_info.available_commands == ["file", "strings", "hexdump"]
            assert policy_info.guidelines == "Use multiple analysis techniques"
            assert len(policy_info.constraints) == 2
            mock_client.get.assert_called_once_with(
                "http://test-server:8000/session/test-session/policy"
            )

    @pytest.mark.asyncio
    async def test_get_policy_no_session(self, server_client):
        """Test policy retrieval without active session."""
        with pytest.raises(ValueError, match="No active session"):
            await server_client.get_policy()

    @pytest.mark.asyncio
    async def test_close_session_success(self, server_client):
        """Test successful session closure."""
        server_client.session_id = "test-session"

        mock_response = Mock()

        with patch.object(server_client, '_client') as mock_client:
            mock_client.delete = AsyncMock(return_value=mock_response)

            await server_client.close_session()

            assert server_client.session_id is None
            mock_client.delete.assert_called_once_with(
                "http://test-server:8000/session/test-session"
            )
            mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session_no_session(self, server_client):
        """Test closing session when no session exists."""
        # Should not raise error
        await server_client.close_session()
        assert server_client.session_id is None

    @pytest.mark.asyncio
    async def test_close_session_with_error(self, server_client):
        """Test session closure with HTTP error."""
        server_client.session_id = "test-session"

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=Mock()
        )

        with patch.object(server_client, '_client') as mock_client:
            mock_client.delete = AsyncMock(return_value=mock_response)

            # Should raise the error but still clear session_id due to finally block
            with pytest.raises(httpx.HTTPStatusError):
                await server_client.close_session()
            assert server_client.session_id is None

    @pytest.mark.asyncio
    async def test_health_check_success(self, server_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "healthy",
            "domain": "malware-analysis"
        }

        with patch.object(server_client, '_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

            health_info = await server_client.health_check()

            assert health_info["status"] == "healthy"
            assert health_info["domain"] == "malware-analysis"
            mock_client.get.assert_called_once_with(
                "http://test-server:8000/health"
            )

    @pytest.mark.asyncio
    async def test_context_manager(self, server_client):
        """Test ServerClient as async context manager."""
        server_client.session_id = "test-session"

        with patch.object(server_client, 'close_session') as mock_close:
            with patch.object(server_client, '_client') as mock_client:
                mock_client.aclose = AsyncMock()

                async with server_client as client:
                    assert client is server_client

                mock_close.assert_called_once()
                mock_client.aclose.assert_called_once()


class TestStepResponseParsing:
    """Test edge cases in step response parsing."""

    @pytest.mark.asyncio
    async def test_step_response_minimal_data(self):
        """Test parsing step response with minimal data."""
        client = ServerClient("http://test:8000", ui_enabled=False)
        client.session_id = "test"

        mock_response = Mock()
        mock_response.json.return_value = {}  # Empty response

        with patch.object(client, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            step_response = await client.execute_step("test")

            assert step_response.success is False
            assert step_response.output == ""
            assert step_response.error is None
            assert step_response.done is False
            assert step_response.info == {}

    @pytest.mark.asyncio
    async def test_step_response_nested_data(self):
        """Test parsing step response with nested data structure."""
        client = ServerClient("http://test:8000", ui_enabled=False)
        client.session_id = "test"

        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "output": "nested output",
                "metadata": {"source": "test"}
            },
            "step": {
                "done": True,
                "subtask_id": "sub-123",
                "additional_info": {"score": 95}
            }
        }

        with patch.object(client, '_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            step_response = await client.execute_step("test")

            assert step_response.success is True
            assert step_response.output == "nested output"
            assert step_response.done is True
            assert "subtask_id" in step_response.info
            assert step_response.info["additional_info"]["score"] == 95
