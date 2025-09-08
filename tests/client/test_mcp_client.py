#!/usr/bin/env python3
"""
Test suite for SABER MCP client with proper mocking.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from saber.client.api.mcp_client import MCPClient
from saber.client.models import MCPConfig
from saber.models.mcp import (
    MCPToolCallRequest,
    MCPToolCallResponse,
    SessionContext,
    MCPConnectionError,
    MCPToolNotFoundError,
    MCPExecutionError,
    MCPTimeoutError
)
from saber.models import HTTPHeaders


class TestMCPClient:
    """Test MCP client functionality with proper mocking."""

    @pytest.fixture
    def mcp_config(self):
        """Create test MCP configuration."""
        return MCPConfig(
            base_url="http://test-server:8001",
            timeout=10.0,
            client_id="test-client"
        )

    @pytest.fixture
    def session_context(self):
        """Create test session context."""
        return SessionContext(
            session_id="test-session",
            episode_id="test-episode",
            task_id="test-task",
            client_id="test-client"
        )

    @pytest.fixture
    def mcp_client(self, mcp_config):
        """Create MCP client for testing."""
        return MCPClient(mcp_config)

    def test_mcp_client_initialization(self, mcp_client, mcp_config):
        """Test MCP client initialization."""
        assert mcp_client.config == mcp_config
        assert mcp_client.client is None
        assert not mcp_client.is_connected()

    @pytest.mark.asyncio
    async def test_connection_success(self, mcp_client, session_context):
        """Test successful connection."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            await mcp_client.connect(session_context)

            assert mcp_client.is_connected()
            assert mcp_client._session_context == session_context
            mock_client.__aenter__.assert_called_once()
            mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_failure(self, mcp_client, session_context):
        """Test connection failure."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_client.__aenter__.side_effect = Exception("Connection failed")
            mock_create.return_value = mock_client

            with pytest.raises(MCPConnectionError):
                await mcp_client.connect(session_context)

            assert not mcp_client.is_connected()

    @pytest.mark.asyncio
    async def test_disconnect(self, mcp_client, session_context):
        """Test disconnect."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            # Connect first
            await mcp_client.connect(session_context)
            assert mcp_client.is_connected()

            # Then disconnect
            await mcp_client.disconnect()
            assert not mcp_client.is_connected()
            mock_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_discovery(self, mcp_client, session_context):
        """Test tool discovery."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            # Setup mock response - should match MCP protocol format
            mock_tools_response = {
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "another_tool",
                        "description": "Another tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                ]
            }
            mock_client.list_tools.return_value = mock_tools_response

            await mcp_client.connect(session_context)
            response = await mcp_client.discover_tools(session_context.episode_id)

            assert len(response) == 2
            assert response[0].name == "test_tool"
            mock_client.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_execution_success(self, mcp_client, session_context):
        """Test successful tool execution."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            # Setup mock response
            mock_result = {"content": [{"type": "text", "text": "Success"}]}
            mock_client.call_tool.return_value = mock_result

            await mcp_client.connect(session_context)

            request = MCPToolCallRequest(
                tool_name="test_tool",
                arguments={"param": "value"},
                episode_id="test-episode",
                task_id="test-task"
            )

            response = await mcp_client.execute_tool(request)

            assert isinstance(response, MCPToolCallResponse)
            assert response.isError is False
            assert response.content == [{"type": "text", "text": "Success"}]

            mock_client.call_tool.assert_called_once_with(
                tool_name="test_tool",
                arguments={"param": "value"},
                headers={
                    'X-SABER-Session-ID': 'test-session',
                    'X-SABER-Client-ID': 'test-client',
                    'X-SABER-Episode-ID': 'test-episode',
                    'X-SABER-Task-ID': 'test-task'
                }
            )

    @pytest.mark.asyncio
    async def test_tool_execution_timeout(self, mcp_client, session_context):
        """Test tool execution timeout."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            # Setup timeout
            mock_client.call_tool.side_effect = asyncio.TimeoutError()

            await mcp_client.connect(session_context)

            request = MCPToolCallRequest(
                tool_name="test_tool",
                arguments={"param": "value"},
                episode_id="test-episode"
            )

            # Should raise MCPTimeoutError
            with pytest.raises(MCPTimeoutError):
                await mcp_client.execute_tool(request)

    @pytest.mark.asyncio
    async def test_tool_not_found(self, mcp_client, session_context):
        """Test tool not found error."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            # Setup error
            mock_client.call_tool.side_effect = Exception("Tool not found")

            await mcp_client.connect(session_context)

            request = MCPToolCallRequest(
                tool_name="nonexistent_tool",
                arguments={},
                episode_id="test-episode"
            )

            # Should raise MCPToolNotFoundError
            with pytest.raises(MCPToolNotFoundError):
                await mcp_client.execute_tool(request)

    @pytest.mark.asyncio
    async def test_not_connected_error(self, mcp_client):
        """Test operations when not connected."""
        request = MCPToolCallRequest(
            tool_name="test_tool",
            arguments={},
            episode_id="test-episode"
        )

        with pytest.raises(MCPConnectionError):
            await mcp_client.execute_tool(request)

    @pytest.mark.asyncio
    async def test_health_check_success(self, mcp_client, session_context):
        """Test successful health check."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            await mcp_client.connect(session_context)
            result = await mcp_client.health_check()

            assert result is True
            mock_client.ping.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mcp_client, session_context):
        """Test health check failure."""
        with patch.object(mcp_client, '_create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            # Connect successfully first
            await mcp_client.connect(session_context)

            # Then make ping fail for health check
            mock_client.ping.side_effect = Exception("Ping failed")
            result = await mcp_client.health_check()

            assert result is False

    def test_session_context_property(self, mcp_client, session_context):
        """Test session context property."""
        assert mcp_client.session_context is None

        mcp_client._session_context = session_context
        assert mcp_client.session_context == session_context
