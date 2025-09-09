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
    MCPTool,
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
        assert session_manager.mcp_client_pool == {}
        assert session_manager._current_session_id is None

    def test_build_episode_context(self, session_manager):
        """Test building episode context."""
        # Set up session
        session_manager._current_session_id = "test-session"

        context = session_manager._build_episode_context(
            episode_id="test-episode",
            task_id="test-task"
        )

        assert isinstance(context, SessionContext)
        assert context.session_id == "test-session"
        assert context.episode_id == "test-episode"
        assert context.task_id == "test-task"
        assert context.client_id == "test-client"

    def test_build_episode_context_no_session(self, session_manager):
        """Test building episode context without active session."""
        with pytest.raises(RuntimeError) as exc_info:
            session_manager._build_episode_context("test-episode")

        assert "No active session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ensure_episode_mcp_client(self, session_manager):
        """Test creating MCP client for episode."""
        with patch('saber.client.client_session.MCPClient') as mock_mcp_client_class:
            mock_mcp_client = AsyncMock()
            mock_mcp_client_class.return_value = mock_mcp_client

            # Set up session
            session_manager._current_session_id = "test-session"

            await session_manager.ensure_episode_mcp_client("test-episode", "test-task")

            # Verify MCP client creation and connection
            mock_mcp_client_class.assert_called_once_with(session_manager.mcp_config)
            mock_mcp_client.connect.assert_called_once()

            # Verify episode context
            connect_args = mock_mcp_client.connect.call_args[0]
            episode_context = connect_args[0]
            assert episode_context.session_id == "test-session"
            assert episode_context.episode_id == "test-episode"
            assert episode_context.task_id == "test-task"
            assert episode_context.client_id == "test-client"

            # Verify client is stored in pool
            assert "test-episode" in session_manager.mcp_client_pool
            assert session_manager.mcp_client_pool["test-episode"] == mock_mcp_client

    @pytest.mark.asyncio
    async def test_ensure_episode_mcp_client_no_session(self, session_manager):
        """Test episode MCP client creation fails without session."""
        with pytest.raises(RuntimeError) as exc_info:
            await session_manager.ensure_episode_mcp_client("test-episode")

        assert "No active session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup_episode_mcp_client(self, session_manager):
        """Test cleaning up episode MCP client."""
        # Set up episode client
        mock_mcp_client = AsyncMock()
        session_manager.mcp_client_pool["test-episode"] = mock_mcp_client

        await session_manager.cleanup_episode_mcp_client("test-episode")

        mock_mcp_client.disconnect.assert_called_once()
        assert "test-episode" not in session_manager.mcp_client_pool
        # After cleanup, the pool should be empty
        assert len(session_manager.mcp_client_pool) == 0

    @pytest.mark.asyncio
    async def test_list_mcp_tools(self, session_manager):
        """Test listing MCP tools using episode-specific client."""
        # Set up episode MCP client
        mock_mcp_client = AsyncMock()
        mock_tools = [
            MCPTool(
                name="test_tool",
                description="Test tool",
                inputSchema=MCPInputSchema(
                    type="object",
                    properties={},
                    required=[]
                )
            )
        ]
        mock_mcp_client.discover_tools.return_value = mock_tools
        session_manager.mcp_client_pool["test-episode"] = mock_mcp_client

        result = await session_manager.list_mcp_tools("test-episode", "test-task")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].name == "test_tool"
        mock_mcp_client.discover_tools.assert_called_once_with(episode_id="test-episode", task_id="test-task")

    @pytest.mark.asyncio
    async def test_list_mcp_tools_no_client(self, session_manager):
        """Test listing MCP tools when no episode client exists."""
        with pytest.raises(RuntimeError) as exc_info:
            await session_manager.list_mcp_tools("test-episode")

        assert "No MCP client for episode test-episode" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self, session_manager):
        """Test executing MCP tool using episode-specific client."""
        # Set up episode MCP client
        mock_mcp_client = AsyncMock()
        mock_response = MCPToolCallResponse(
            content=[{"type": "text", "text": "success"}],
            isError=False
        )
        mock_mcp_client.execute_tool.return_value = mock_response
        session_manager.mcp_client_pool["test-episode"] = mock_mcp_client

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

        assert isinstance(result, MCPToolCallResponse)
        assert not result.isError
        assert result.content[0]["text"] == "success"
        mock_mcp_client.execute_tool.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_no_client(self, session_manager):
        """Test executing MCP tool when no episode client exists."""
        request = MCPToolCallRequest(
            tool_name="test_tool",
            episode_id="test-episode",
            arguments={}
        )

        with pytest.raises(RuntimeError) as exc_info:
            await session_manager.execute_mcp_tool(request)

        assert "No MCP client for episode test-episode" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_no_episode_id(self, session_manager):
        """Test executing MCP tool without valid episode_id fails fast."""
        # Test with non-existent episode ID
        request = MCPToolCallRequest(
            tool_name="test_tool",
            episode_id="nonexistent-episode",  # Episode that doesn't exist
            arguments={}
        )

        with pytest.raises(RuntimeError) as exc_info:
            await session_manager.execute_mcp_tool(request)

        assert "No MCP client for episode nonexistent-episode" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_error(self, session_manager):
        """Test executing MCP tool with error response."""
        # Set up episode MCP client
        mock_mcp_client = AsyncMock()
        mock_response = MCPToolCallResponse(
            content=[{"type": "text", "text": "Tool failed"}],
            isError=True
        )
        mock_mcp_client.execute_tool.return_value = mock_response
        session_manager.mcp_client_pool["test-episode"] = mock_mcp_client

        request = MCPToolCallRequest(
            tool_name="test_tool",
            episode_id="test-episode",
            arguments={}
        )

        result = await session_manager.execute_mcp_tool(request)

        # Verify error response is returned as-is (no exception)
        assert isinstance(result, MCPToolCallResponse)
        assert result.isError
        assert result.content[0]["text"] == "Tool failed"

    @pytest.mark.asyncio
    async def test_cleanup(self, session_manager):
        """Test session manager cleanup."""
        # Set up multiple episode clients
        mock_mcp_client1 = AsyncMock()
        mock_mcp_client2 = AsyncMock()
        session_manager.mcp_client_pool["episode-1"] = mock_mcp_client1
        session_manager.mcp_client_pool["episode-2"] = mock_mcp_client2
        session_manager._current_session_id = "test-session"

        with patch.object(session_manager, 'terminate_session') as mock_terminate:
            await session_manager.cleanup()

        # Verify all MCP clients disconnected and session terminated
        mock_mcp_client1.disconnect.assert_called_once()
        mock_mcp_client2.disconnect.assert_called_once()
        assert session_manager.mcp_client_pool == {}
        mock_terminate.assert_called_once_with("test-session")


if __name__ == "__main__":
    pytest.main([__file__])
