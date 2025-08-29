"""
Unit tests for SessionMCPAPI.

Tests MCP protocol functionality for tool discovery and execution.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.api.session_mcp_api import SessionMCPAPI
from saber.server.base import Action, CommandResult


class TestSessionMCPAPI:
    """Test SessionMCPAPI MCP protocol functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create mock SessionManager for testing."""
        mock_manager = MagicMock()
        mock_manager.domain_name = "test_domain"
        mock_manager.execution_manager = MagicMock()
        mock_manager.execute_action = AsyncMock()
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
        assert hasattr(mcp_api, 'tool_generator')

    @pytest.mark.asyncio
    async def test_handle_list_tools(self, mcp_api):
        """Test MCP tool discovery."""
        # Mock execution manager returning tools
        mock_tools = [
            {"name": "cli", "description": "Command line executor", "inputSchema": {"type": "object"}},
            {"name": "python", "description": "Python executor", "inputSchema": {"type": "object"}},
        ]
        mcp_api.session_manager.execution_manager.to_mcp_tools.return_value = mock_tools

        tools = await mcp_api.handle_list_tools()

        # Should include executor tools + hardcoded tools
        assert len(tools) == 3  # 2 executor tools + 1 hardcoded tool (end_episode)

        # Check executor tools are included
        executor_tools = [t for t in tools if t["name"] in ["cli", "python"]]
        assert len(executor_tools) == 2

        # Check hardcoded tools are included
        hardcoded_tools = [t for t in tools if t["name"] in ["end_episode"]]
        assert len(hardcoded_tools) == 1

        # Verify end_episode tool definition
        end_episode_tool = next(t for t in tools if t["name"] == "end_episode")
        assert (
            end_episode_tool["description"]
            == "End the current episode and optionally record a discovered flag/target/objective"
        )
        assert end_episode_tool["inputSchema"]["required"] == []
        assert "submission" in end_episode_tool["inputSchema"]["properties"]

        mcp_api.session_manager.execution_manager.to_mcp_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self, mcp_api):
        """Test successful MCP tool execution."""
        # Mock successful command execution
        command_result = CommandResult.success_result(data={"output": "test output"})
        mcp_api.session_manager.execute_action.return_value = command_result

        # Mock header-based session retrieval
        with patch.object(mcp_api, '_get_session_from_headers', return_value="session_123"):
            # Test tool call (no session_id in arguments - comes from headers)
            result = await mcp_api.handle_call_tool(
                name="cli", arguments={"command": "ls", "parameters": {}}
            )

        # Verify result format
        assert result["isError"] is False
        assert result["content"][0]["type"] == "text"
        assert "output" in result["content"][0]["text"]

        # Verify execute_action was called correctly
        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args
        assert call_args[0][0] == "session_123"  # session_id
        assert isinstance(call_args[0][1], Action)  # action

    @pytest.mark.asyncio
    async def test_handle_call_tool_error(self, mcp_api):
        """Test MCP tool execution with error."""
        # Mock failed command execution
        command_result = CommandResult.error_result(error="Command failed")
        mcp_api.session_manager.execute_action.return_value = command_result

        # Mock header-based session retrieval
        with patch.object(mcp_api, '_get_session_from_headers', return_value="session_123"):
            # Test tool call
            result = await mcp_api.handle_call_tool(
                name="cli", arguments={"command": "invalid_command"}
            )

        # Verify error result format
        assert result["isError"] is True
        assert result["content"][0]["type"] == "text"
        assert "Error: Command failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_call_tool_missing_session(self, mcp_api):
        """Test MCP tool execution without session_id."""
        # Mock no session found in headers
        with patch.object(mcp_api, '_get_session_from_headers', return_value=None):
            # Test tool call without session_id
            result = await mcp_api.handle_call_tool(name="cli", arguments={"command": "ls"})

        # Verify error result
        assert result["isError"] is True
        assert "No SABER session mapped to MCP request" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_end_episode_call_success(self, mcp_api):
        """Test successful end_episode tool call without result."""
        # Mock end_episode method
        mcp_api.session_manager.end_episode = AsyncMock()

        # Test end_episode tool call
        result = await mcp_api._handle_end_episode_call({"session_id": "session_123"})

        # Verify successful result
        assert result["isError"] is False
        assert result["content"][0]["type"] == "text"
        assert "Episode ended successfully" in result["content"][0]["text"]

        # Verify end_episode was called
        mcp_api.session_manager.end_episode.assert_called_once_with("session_123", "agent_completed")

    @pytest.mark.asyncio
    async def test_handle_end_episode_call_with_result(self, mcp_api):
        """Test end_episode tool call with result flag."""
        # Mock methods
        mcp_api.session_manager.end_episode = AsyncMock()
        mcp_api.session_manager.execute_action = AsyncMock(
            return_value=CommandResult.success_result(data="Action recorded")
        )

        # Test end_episode tool call with result
        result = await mcp_api._handle_end_episode_call(
            {"session_id": "session_123", "parameters": {"submission": "flag{test_flag_found}"}}
        )

        # Verify successful result with flag
        assert result["isError"] is False
        assert result["content"][0]["type"] == "text"
        result_text = result["content"][0]["text"]
        assert "Episode ended successfully with result: flag{test_flag_found}" in result_text

        # Verify result action was executed
        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args
        assert call_args[0][0] == "session_123"  # session_id
        action = call_args[0][1]  # action
        assert action.tool_name == "episode_result"
        assert "flag{test_flag_found}" in action.parameters.get("submission", "")
        assert action.parameters["submission"] == "flag{test_flag_found}"
        assert action.parameters["episode_end"] is True

        # Verify end_episode was called
        mcp_api.session_manager.end_episode.assert_called_once_with("session_123", "agent_completed", "flag{test_flag_found}")

    @pytest.mark.asyncio
    async def test_handle_end_episode_call_missing_session(self, mcp_api):
        """Test end_episode tool call without session_id."""
        # Test end_episode call without session_id
        result = await mcp_api._handle_end_episode_call({"result": "some_flag"})

        # Verify error result
        assert result["isError"] is True
        assert "No SABER session mapped to MCP request" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_call_tool_hardcoded_tools(self, mcp_api):
        """Test handle_call_tool routing for end_episode tool."""
        # Mock end_episode for end_episode tool execution
        command_result = CommandResult.success_result(data="Episode completed")
        mcp_api.session_manager.execute_action.return_value = command_result

        # Mock header-based session retrieval
        with patch.object(mcp_api, '_get_session_from_headers', return_value="session_123"):
            # Test end_episode routing through normal execution path
            result = await mcp_api.handle_call_tool("end_episode", {"submission": "flag{test}"})

        assert result["isError"] is False
        assert "Episode completed" in result["content"][0]["text"]

    def test_convert_to_action(self, mcp_api):
        """Test conversion from MCP tool call to Action."""
        action = mcp_api._convert_to_action(
            tool_name="cli",
            arguments={
                "session_id": "session_123",
                "parameters": {"arguments": "ls -la", "flag": "-l"},
                "context": {"extra": "data"},
            },
        )

        assert action.tool_name == "cli"
        assert action.parameters == {"parameters": {"arguments": "ls -la", "flag": "-l"}, "context": {"extra": "data"}}

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
        with patch("saber.server.api.session_mcp_api.FastMCP") as mock_fastmcp:
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
