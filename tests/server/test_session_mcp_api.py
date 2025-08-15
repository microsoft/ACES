"""
Unit tests for SessionMCPAPI.

Tests MCP protocol functionality for tool discovery and execution.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.server.api.session_mcp_api import SessionMCPAPI
from saber.server.base import CommandResult, Action


class TestSessionMCPAPI:
    """Test SessionMCPAPI MCP protocol functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create mock SessionManager for testing."""
        mock_manager = MagicMock()
        mock_manager.domain_name = "test_domain"
        mock_manager.execution_manager = MagicMock()
        mock_manager.execute_command = AsyncMock()
        mock_manager._get_session = MagicMock()
        return mock_manager

    @pytest.fixture
    def mcp_api(self, session_manager):
        """Create SessionMCPAPI instance for testing."""
        return SessionMCPAPI(session_manager, host="127.0.0.1", port=3001)

    def test_init(self, session_manager):
        """Test SessionMCPAPI initialization."""
        mcp_api = SessionMCPAPI(session_manager, host="0.0.0.0", port=3001)

        assert mcp_api.session_manager == session_manager
        assert mcp_api.host == "0.0.0.0"
        assert mcp_api.port == 3001
        assert mcp_api.mcp_server is None
        assert mcp_api.active_mcp_sessions == {}

    @pytest.mark.asyncio
    async def test_handle_list_tools(self, mcp_api):
        """Test MCP tool discovery."""
        # Mock execution manager returning tools
        mock_tools = [
            {"name": "cli", "description": "Command line executor", "inputSchema": {"type": "object"}},
            {"name": "python", "description": "Python executor", "inputSchema": {"type": "object"}}
        ]
        mcp_api.session_manager.execution_manager.to_mcp_tools.return_value = mock_tools

        tools = await mcp_api.handle_list_tools()

        assert tools == mock_tools
        mcp_api.session_manager.execution_manager.to_mcp_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self, mcp_api):
        """Test successful MCP tool execution."""
        # Mock successful command execution
        command_result = CommandResult.success_result(data={"output": "test output"})
        mcp_api.session_manager.execute_command.return_value = command_result

        # Test tool call
        result = await mcp_api.handle_call_tool(
            name="cli",
            arguments={"session_id": "session_123", "command": "ls", "parameters": {}}
        )

        # Verify result format
        assert result["isError"] is False
        assert result["content"][0]["type"] == "text"
        assert "output" in result["content"][0]["text"]

        # Verify execute_command was called correctly
        mcp_api.session_manager.execute_command.assert_called_once()
        call_args = mcp_api.session_manager.execute_command.call_args
        assert call_args[0][0] == "session_123"  # session_id
        assert isinstance(call_args[0][1], Action)  # action

    @pytest.mark.asyncio
    async def test_handle_call_tool_error(self, mcp_api):
        """Test MCP tool execution with error."""
        # Mock failed command execution
        command_result = CommandResult.error_result(error="Command failed")
        mcp_api.session_manager.execute_command.return_value = command_result

        # Test tool call
        result = await mcp_api.handle_call_tool(
            name="cli",
            arguments={"session_id": "session_123", "command": "invalid_command"}
        )

        # Verify error result format
        assert result["isError"] is True
        assert result["content"][0]["type"] == "text"
        assert "Error: Command failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_call_tool_missing_session(self, mcp_api):
        """Test MCP tool execution without session_id."""
        # Test tool call without session_id
        result = await mcp_api.handle_call_tool(
            name="cli",
            arguments={"command": "ls"}
        )

        # Verify error result
        assert result["isError"] is True
        assert "Missing session_id" in result["content"][0]["text"]

    def test_map_session_context(self, mcp_api):
        """Test session context mapping from MCP arguments."""
        # Test direct session_id
        context = mcp_api._map_session_context({"session_id": "session_123", "client_id": "test_client"})
        assert context["session_id"] == "session_123"
        assert context["mcp_client"] == "test_client"

        # Test nested session_id
        context = mcp_api._map_session_context({"context": {"session_id": "session_456"}})
        assert context["session_id"] == "session_456"
        assert context["mcp_client"] == "unknown"

        # Test missing session_id
        context = mcp_api._map_session_context({"command": "ls"})
        assert context["session_id"] is None

    def test_convert_to_action(self, mcp_api):
        """Test conversion from MCP tool call to Action."""
        action = mcp_api._convert_to_action(
            tool_name="cli",
            arguments={
                "session_id": "session_123",
                "command": "ls -la",
                "parameters": {"flag": "-l"},
                "context": {"extra": "data"}
            }
        )

        assert action.tool_name == "cli"
        assert action.command == "ls -la"
        assert action.parameters == {"flag": "-l"}
        # session_id should be filtered out
        assert "session_id" not in action.parameters

    def test_convert_to_mcp_result_success(self, mcp_api):
        """Test conversion of successful CommandResult to MCP format."""
        command_result = CommandResult.success_result(data={"output": "test output"})

        mcp_result = mcp_api._convert_to_mcp_result(command_result)

        assert mcp_result["isError"] is False
        assert mcp_result["content"][0]["type"] == "text"
        assert "output" in mcp_result["content"][0]["text"]

    def test_convert_to_mcp_result_error(self, mcp_api):
        """Test conversion of error CommandResult to MCP format."""
        command_result = CommandResult.error_result(error="Command execution failed")

        mcp_result = mcp_api._convert_to_mcp_result(command_result)

        assert mcp_result["isError"] is True
        assert mcp_result["content"][0]["type"] == "text"
        assert "Error: Command execution failed" in mcp_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_start_and_shutdown_mcp_server(self, mcp_api):
        """Test MCP server startup and shutdown."""
        # Mock FastMCP
        with patch('saber.server.api.session_mcp_api.FastMCP') as mock_fastmcp:
            mock_server = MagicMock()
            mock_server.run_async = AsyncMock()
            mock_server.close = AsyncMock()
            mock_fastmcp.return_value = mock_server

            # Test startup
            await mcp_api.start_mcp_server()
            assert mcp_api.mcp_server == mock_server
            mock_server.run_async.assert_called_once_with(transport="sse", host="127.0.0.1", port=3001)

            # Test shutdown
            await mcp_api.shutdown_mcp_server()
            mock_server.close.assert_called_once()
            assert mcp_api.mcp_server is None
            assert mcp_api.active_mcp_sessions == {}
