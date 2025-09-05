"""
Tests for MCP Service Application
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from saber.client.mcp_service.main import create_app
from saber.client.mcp_service.agent_registry import AgentSessionRegistry
from saber.client.mcp_service.mcp_proxy import MCPProxy, MCPResponse


@pytest.fixture
def app():
    """Create test FastAPI app."""
    return create_app(saber_mcp_url="http://test-saber:8001")


@pytest.fixture
def client(app):
    """Create test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def simple_client(app):
    """Create simple test client without lifespan context for faster unit tests."""
    return TestClient(app)


@pytest.fixture
def setup_mcp_globals():
    """Set up MCP service global variables for testing."""
    import saber.client.mcp_service.main as main_module

    # Create mock objects
    mock_registry = MagicMock(spec=AgentSessionRegistry)
    mock_proxy = MagicMock(spec=MCPProxy)

    # Store original values
    original_registry = main_module.session_registry
    original_proxy = main_module.mcp_proxy

    # Set up mocks
    main_module.session_registry = mock_registry
    main_module.mcp_proxy = mock_proxy

    yield mock_registry, mock_proxy

    # Restore original values
    main_module.session_registry = original_registry
    main_module.mcp_proxy = original_proxy


class TestHealthEndpoints:
    """Test health and readiness endpoints."""

    @pytest.mark.integration
    def test_health_endpoint_service_not_ready(self, simple_client):
        """Test health endpoint when service is not ready."""
        # Health check should fail when proxy is not initialized
        response = simple_client.get("/health")
        assert response.status_code == 503

    def test_readiness_endpoint_service_not_ready(self, simple_client):
        """Test readiness endpoint when service is not ready."""
        # Without lifespan context, globals are None so service appears not ready
        response = simple_client.get("/ready")
        assert response.status_code == 503


class TestSessionManagement:
    """Test session management endpoints."""

    @patch('saber.client.mcp_service.main.session_registry')
    def test_register_session_success(self, mock_session_registry, client):
        """Test successful session registration."""
        # Mock session manager
        mock_session = MagicMock()
        mock_session.agent_id = "test-agent"
        mock_session.saber_session_id = "saber-123"
        mock_session_registry.register_session = AsyncMock(return_value=mock_session)

        response = client.post("/admin/sessions", json={
            "agent_id": "test-agent",
            "saber_session_id": "saber-123",
            "saber_episode_id": "episode-456",
            "task_id": "task-456"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["agent_id"] == "test-agent"
        assert data["saber_session_id"] == "saber-123"

    @patch('saber.client.mcp_service.main.session_registry')
    def test_register_session_duplicate_error(self, mock_session_registry, client):
        """Test session registration with duplicate agent ID."""
        mock_session_registry.register_session = AsyncMock(
            side_effect=ValueError("Agent already registered")
        )

        response = client.post("/admin/sessions", json={
            "agent_id": "test-agent",
            "saber_session_id": "saber-123",
            "saber_episode_id": "episode-456"
        })

        assert response.status_code == 400
        assert "Agent already registered" in response.json()["detail"]

    @patch('saber.client.mcp_service.main.session_registry')
    def test_unregister_session_success(self, mock_session_registry, client):
        """Test successful session unregistration."""
        mock_session_registry.unregister_session = AsyncMock(return_value=True)

        response = client.delete("/admin/sessions/test-agent")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch('saber.client.mcp_service.main.session_registry')
    def test_unregister_session_not_found(self, mock_session_registry, client):
        """Test unregistering non-existent session."""
        mock_session_registry.unregister_session = AsyncMock(return_value=False)

        response = client.delete("/admin/sessions/nonexistent")

        assert response.status_code == 404

    @patch('saber.client.mcp_service.main.session_registry')
    def test_list_sessions(self, mock_session_registry, client):
        """Test listing sessions."""
        mock_session_registry.list_active_sessions = AsyncMock(return_value={
            "agent-1": MagicMock(
                saber_session_id="saber-123",
                task_id="task-456",
                registered_at=MagicMock(isoformat=lambda: "2023-01-01T00:00:00"),
                last_activity=MagicMock(isoformat=lambda: "2023-01-01T00:01:00"),
                is_active=True
            )
        })

        response = client.get("/admin/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "agent-1" in data["sessions"]


class TestMCPEndpoints:
    """Test MCP protocol endpoints."""

    def test_list_tools_success(self, client, setup_mcp_globals):
        """Test successful list_tools request."""
        mock_registry, mock_proxy = setup_mcp_globals

        # Mock session lookup
        mock_registry.get_agent_id_by_saber_session = AsyncMock(return_value="test-agent")

        # Mock tool registry response
        from dataclasses import dataclass
        from datetime import datetime

        @dataclass
        class MockToolMetadata:
            description: str
            parameters: dict
            timeout: int
            cached_at: datetime = None

        mock_tools = {
            "test_tool": MockToolMetadata(
                description='Test tool',
                parameters={},
                timeout=30,
                cached_at=None
            )
        }

        # Mock tool_registry instead of mcp_proxy since the actual endpoint uses tool_registry
        with patch('saber.client.mcp_service.main.tool_registry') as mock_tool_registry:
            mock_tool_registry.get_tools_for_session_episode = AsyncMock(return_value=mock_tools)
            # Mock the cache_ttl property to avoid JSON serialization issues
            from datetime import timedelta
            mock_tool_registry.cache_ttl = timedelta(seconds=300)

            response = client.get(
                "/tools",
                headers={
                    "X-Saber-Session-Id": "saber-session-123",
                    "X-SABER-Episode-ID": "test-episode-123"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert "tools" in data
            assert "test_tool" in data["tools"]
            assert data["tools"]["test_tool"]["description"] == "Test tool"

    def test_list_tools_error(self, client, setup_mcp_globals):
        """Test list_tools request with error."""
        mock_registry, mock_proxy = setup_mcp_globals

        # Mock session lookup
        mock_registry.get_agent_id_by_saber_session = AsyncMock(return_value="test-agent")

        # Mock tool_registry to raise an exception
        with patch('saber.client.mcp_service.main.tool_registry') as mock_tool_registry:
            mock_tool_registry.get_tools_for_session_episode = AsyncMock(side_effect=Exception("Tool registry error"))

            response = client.get(
                "/tools",
                headers={
                    "X-Saber-Session-Id": "saber-session-123",
                    "X-SABER-Episode-ID": "test-episode-123"
                }
            )

            assert response.status_code == 500
            data = response.json()
            assert data["detail"] == "Internal server error"

    def test_call_tool_success(self, client, setup_mcp_globals):
        """Test successful call_tool request."""
        mock_registry, mock_proxy = setup_mcp_globals

        # Mock session lookup
        mock_registry.get_agent_id_by_saber_session = AsyncMock(return_value="test-agent")

        # Mock tool registry response
        with patch('saber.client.mcp_service.main.tool_registry') as mock_tool_registry:
            mock_tool_registry.execute_tool = AsyncMock(return_value={
                "output": "Command executed successfully",
                "success": True
            })

            response = client.post(
                "/execute_tool",
                headers={
                    "X-Saber-Session-Id": "test-agent",
                    "X-SABER-Episode-ID": "test-episode-123"
                },
                json={
                    "tool_name": "cli_execute",
                    "arguments": {"command": "ls -la"}
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["output"] == "Command executed successfully"
            assert data["success"] == True


    def test_call_tool_missing_name(self, client, setup_mcp_globals):
        """Test call_tool request with missing tool name."""
        mock_registry, mock_proxy = setup_mcp_globals

        # Mock session lookup
        mock_registry.get_agent_id_by_saber_session = AsyncMock(return_value="test-agent")

        response = client.post(
            "/execute_tool",
            headers={
                "X-Saber-Session-Id": "saber-session-123",
                "X-SABER-Episode-ID": "test-episode-123"
            },
            json={
                "arguments": {"command": "ls -la"}
            }
        )

        # Should return 422 for validation error (missing required field)
        assert response.status_code == 422
        data = response.json()
        assert "field required" in str(data).lower() or "missing" in str(data).lower()

    def test_missing_episode_header(self, client, setup_mcp_globals):
        """Test request without required episode header."""
        mock_registry, mock_proxy = setup_mcp_globals

        # Mock session lookup
        mock_registry.get_agent_id_by_saber_session = AsyncMock(return_value="test-agent")

        # Missing episode header should fail
        response = client.post(
            "/execute_tool",
            headers={"X-Saber-Session-Id": "saber-session-123"},
            json={
                "tool_name": "test_tool",
                "arguments": {"test": "value"}
            }
        )

        # Should return 400 for missing episode header
        assert response.status_code == 400
        assert "Missing required header: X-SABER-Episode-ID" in response.json()["detail"]


    # Note: /mcp/ping endpoint was removed in episode-first architecture
    # Tools are executed via /execute_tool endpoint instead


class TestSessionLookup:
    """Test session-based agent lookup functionality."""

    def test_session_lookup_success(self, client, setup_mcp_globals):
        """Test successful session lookup via execute_tool endpoint."""
        mock_registry, mock_proxy = setup_mcp_globals

        # Mock session lookup
        mock_registry.get_agent_id_by_saber_session = AsyncMock(return_value="test-agent-123")

        # Mock tool execution since that's what we're actually testing
        with patch('saber.client.mcp_service.main.tool_registry') as mock_tool_registry:
            mock_tool_registry.execute_tool = AsyncMock(return_value={
                "output": "test result",
                "success": True
            })

            response = client.post(
                "/execute_tool",
                headers={
                    "X-Saber-Session-Id": "saber-session-123",
                    "X-SABER-Episode-ID": "test-episode-123"
                },
                json={
                    "tool_name": "test_tool",
                    "arguments": {"test": "value"}
                }
            )

            # Verify the session lookup was called correctly
            mock_registry.get_agent_id_by_saber_session.assert_called_once_with("saber-session-123")
            # Verify the tool was executed with the resolved agent ID
            mock_tool_registry.execute_tool.assert_called_once()

            assert response.status_code == 200

    def test_session_lookup_not_found(self, client, setup_mcp_globals):
        """Test session lookup when session not found."""
        mock_registry, mock_proxy = setup_mcp_globals

        # Mock session lookup returning None (session not found)
        mock_registry.get_agent_id_by_saber_session = AsyncMock(return_value=None)

        response = client.post(
            "/execute_tool",
            headers={
                "X-Saber-Session-Id": "nonexistent-session",
                "X-SABER-Episode-ID": "test-episode-123"
            },
            json={
                "tool_name": "test_tool",
                "arguments": {"test": "value"}
            }
        )

        # Should return 404 for invalid session
        assert response.status_code == 404
        assert "Session not registered" in response.json()["detail"]

    def test_missing_session_header(self, client, setup_mcp_globals):
        """Test request without required session header."""
        mock_registry, mock_proxy = setup_mcp_globals

        # No session header provided (but episode header is present)
        response = client.post("/execute_tool",
                             headers={"X-SABER-Episode-ID": "test-episode-123"},
                             json={
                                 "tool_name": "test_tool",
                                 "arguments": {"test": "value"}
                             })

        # Should return 400 for missing session header
        assert response.status_code == 400
        assert "Missing required header: X-SABER-Session-ID" in response.json()["detail"]
