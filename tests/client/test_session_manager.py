"""
Unit tests for ClientSessionManager with typed MCP integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from saber.client.client_session import ClientSessionManager
from saber.client.models import SessionManagerConfig
from saber.models.mcp import (
    MCPToolCallRequest,
    MCPToolCallResponse,
    MCPToolListResponse,
    MCPToolSchema,
    MCPInputSchema,
    SessionContext
)


class TestSessionManagerConfig:
    """Test SessionManagerConfig model."""

    def test_session_manager_config_creation(self):
        """Test creating SessionManagerConfig."""
        config = SessionManagerConfig(
            base_url="http://test:8000",
            client_id="test-client",
            mcp_url="http://test:8001"
        )

        assert config.base_url == "http://test:8000"
        assert config.client_id == "test-client"
        assert config.mcp_url == "http://test:8001"
        assert config.rest_timeout == 30.0
        assert config.mcp_timeout == 30

    def test_to_mcp_config(self):
        """Test conversion to MCPConfig."""
        config = SessionManagerConfig(
            base_url="http://test:8000",
            client_id="test-client",
            mcp_url="http://test:8001",
            mcp_timeout=45
        )

        mcp_config = config.to_mcp_config()

        assert mcp_config.base_url == "http://test:8001"
        assert mcp_config.client_id == "test-client"
        assert mcp_config.timeout == 45.0


class TestClientSessionManager:
    """Test ClientSessionManager functionality."""

    @pytest.fixture
    def session_config(self):
        """Create test session manager configuration."""
        return SessionManagerConfig(
            base_url="http://test:8000",
            client_id="test-client",
            mcp_url="http://test:8001"
        )

    @pytest.fixture
    def session_manager(self, session_config):
        """Create session manager for testing."""
        return ClientSessionManager(session_config)

    def test_session_manager_initialization(self, session_manager, session_config):
        """Test session manager initialization."""
        assert session_manager.config == session_config
        assert session_manager.base_url == "http://test:8000"
        assert session_manager.client_id == "test-client"
        assert session_manager.mcp_client is None
        assert not session_manager.is_mcp_connected()

    def test_build_session_context(self, session_manager):
        """Test building session context."""
        # Set up session
        session_manager._current_session_id = "test-session"

        context = session_manager._build_session_context(
            episode_id="test-episode",
            task_id="test-task"
        )

        assert isinstance(context, SessionContext)
        assert context.session_id == "test-session"
        assert context.episode_id == "test-episode"
        assert context.task_id == "test-task"
        assert context.client_id == "test-client"

    def test_build_session_context_no_session(self, session_manager):
        """Test building session context without active session."""
        with pytest.raises(RuntimeError) as exc_info:
            session_manager._build_session_context("test-episode")

        assert "No active session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connect_mcp(self, session_manager):
        """Test MCP connection with episode context."""
        with patch('saber.client.client_session.MCPClient') as mock_mcp_client_class:
            mock_mcp_client = AsyncMock()
            mock_mcp_client.is_connected.return_value = False
            mock_mcp_client_class.return_value = mock_mcp_client

            # Set up session
            session_manager._current_session_id = "test-session"

            await session_manager.connect_mcp_session()

            # Verify MCP client creation and connection
            mock_mcp_client_class.assert_called_once_with(session_manager.mcp_config)
            mock_mcp_client.connect.assert_called_once()

            # Verify session context
            connect_args = mock_mcp_client.connect.call_args[0]
            session_context = connect_args[0]
            assert session_context.session_id == "test-session"
            assert session_context.episode_id is None  # episode_id is added per-request
            assert session_context.task_id is None  # task_id is added per-request

    @pytest.mark.asyncio
    async def test_connect_mcp_reconnect(self, session_manager):
        """Test MCP reconnection when already connected."""
        with patch('saber.client.api.mcp_client.MCPClient') as mock_mcp_client_class:
            mock_mcp_client = AsyncMock()
            mock_mcp_client.is_connected.return_value = True
            mock_mcp_client_class.return_value = mock_mcp_client

            session_manager._current_session_id = "test-session"
            session_manager.mcp_client = mock_mcp_client

            await session_manager.connect_mcp_session()

            # Verify disconnect and reconnect
            mock_mcp_client.disconnect.assert_called_once()
            mock_mcp_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_mcp(self, session_manager):
        """Test MCP disconnection."""
        with patch('saber.client.api.mcp_client.MCPClient') as mock_mcp_client_class:
            mock_mcp_client = AsyncMock()
            mock_mcp_client_class.return_value = mock_mcp_client

            session_manager.mcp_client = mock_mcp_client
            session_manager._current_episode_id = "test-episode"

            await session_manager.disconnect_mcp()

            mock_mcp_client.disconnect.assert_called_once()
            assert session_manager._current_episode_id is None

    def test_is_mcp_connected(self, session_manager):
        """Test MCP connection status check."""
        # Test not connected (no client)
        assert not session_manager.is_mcp_connected()

        # Test not connected (client not connected)
        mock_mcp_client = MagicMock()
        mock_mcp_client.is_connected.return_value = False
        session_manager.mcp_client = mock_mcp_client
        assert not session_manager.is_mcp_connected()

        # Test connected
        mock_mcp_client.is_connected.return_value = True
        assert session_manager.is_mcp_connected()

    @pytest.mark.asyncio
    async def test_list_mcp_tools(self, session_manager):
        """Test listing MCP tools with typed response."""
        mock_mcp_client = AsyncMock()
        # Return a list of MCPTool objects directly
        mock_tools = [
            MCPToolSchema(
                name="test_tool",
                description="Test tool",
                inputSchema=MCPInputSchema(
                    type="object",
                    properties={},
                    required=[]
                )
            )
        ]
        mock_mcp_client.is_connected.return_value = True
        mock_mcp_client.discover_tools.return_value = mock_tools

        session_manager.mcp_client = mock_mcp_client

        result = await session_manager.list_mcp_tools("test-episode")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].name == "test_tool"
        mock_mcp_client.discover_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_mcp_tools_not_connected(self, session_manager):
        """Test listing MCP tools when not connected."""
        with pytest.raises(RuntimeError) as exc_info:
            await session_manager.list_mcp_tools("test-episode")

        assert "not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self, session_manager):
        """Test executing MCP tool with typed request/response."""
        mock_mcp_client = AsyncMock()
        mock_response = MCPToolCallResponse(
            content=[{"type": "text", "text": "success"}],
            isError=False
        )
        mock_mcp_client.is_connected.return_value = True
        mock_mcp_client.execute_tool.return_value = mock_response

        session_manager.mcp_client = mock_mcp_client

        request = MCPToolCallRequest(
            tool_name="test_tool",
            episode_id="test-episode",
            arguments={"param": "value"}
        )

        result = await session_manager.execute_mcp_tool(request)

        assert isinstance(result, MCPToolCallResponse)
        assert not result.isError
        assert result.content[0]["text"] == "success"
        mock_mcp_client.execute_tool.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_not_connected(self, session_manager):
        """Test executing MCP tool when not connected."""
        request = MCPToolCallRequest(
            tool_name="test_tool",
            episode_id="test-episode",
            arguments={}
        )

        with pytest.raises(RuntimeError) as exc_info:
            await session_manager.execute_mcp_tool(request)

        assert "not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_success(self, session_manager):
        """Test execute_mcp_tool method."""
        mock_mcp_client = AsyncMock()
        mock_response = MCPToolCallResponse(
            content=[{"type": "text", "text": "success"}],
            isError=False
        )
        mock_mcp_client.is_connected.return_value = True
        mock_mcp_client.execute_tool.return_value = mock_response

        session_manager.mcp_client = mock_mcp_client

        # Use execute_mcp_tool instead of call_mcp_tool
        request = MCPToolCallRequest(
            tool_name="test_tool",
            episode_id="test-episode",
            arguments={"param": "value"}
        )

        result = await session_manager.execute_mcp_tool(request)

        # Verify response
        assert isinstance(result, MCPToolCallResponse)
        assert not result.isError
        assert result.content[0]["text"] == "success"

        # Verify typed request was called
        mock_mcp_client.execute_tool.assert_called_once()
        request_arg = mock_mcp_client.execute_tool.call_args[0][0]
        assert isinstance(request_arg, MCPToolCallRequest)
        assert request_arg.tool_name == "test_tool"
        assert request_arg.arguments == {"param": "value"}

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_error(self, session_manager):
        """Test execute_mcp_tool method with error."""
        mock_mcp_client = AsyncMock()
        mock_response = MCPToolCallResponse(
            content=[{"type": "text", "text": "Tool failed"}],
            isError=True
        )
        mock_mcp_client.is_connected.return_value = True
        mock_mcp_client.execute_tool.return_value = mock_response

        session_manager.mcp_client = mock_mcp_client

        request = MCPToolCallRequest(
            tool_name="test_tool",
            episode_id="test-episode",
            arguments={}
        )

        result = await session_manager.execute_mcp_tool(request)

        # Verify error response
        assert isinstance(result, MCPToolCallResponse)
        assert result.isError
        assert result.content[0]["text"] == "Tool failed"

    @pytest.mark.asyncio
    async def test_cleanup(self, session_manager):
        """Test session manager cleanup."""
        mock_mcp_client = AsyncMock()
        session_manager.mcp_client = mock_mcp_client
        session_manager._current_session_id = "test-session"

        with patch.object(session_manager, 'terminate_session') as mock_terminate:
            await session_manager.cleanup()

        # Verify MCP disconnect and session termination
        mock_mcp_client.disconnect.assert_called_once()
        mock_terminate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
