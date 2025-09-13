"""
Unit tests for SessionMCPAPI.

Tests MCP protocol functionality for tool discovery and execution.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.models.headers import HTTPHeaders
from saber.models.mcp import OrchestrationEnvironment, RequestHeaders
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
            {
                "name": "bash",
                "description": "Command line executor",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "python",
                "description": "Python executor",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        }
                    },
                    "required": ["code"]
                }
            },
        ]
        mcp_api.session_manager.execution_manager.to_mcp_tools.return_value = mock_tools

        # Mock the required orchestration header
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {HTTPHeaders.ORCHESTRATION_ENV: "standalone"}

            response = await mcp_api.handle_list_tools()

        # Should include executor tools + hardcoded tools
        assert len(response.tools) == 3  # 2 executor tools + 1 hardcoded tool (end_episode)

        # Check executor tools are included
        executor_tools = [t for t in response.tools if t.name in ["bash", "python"]]
        assert len(executor_tools) == 2

        # Check hardcoded tools are included
        hardcoded_tools = [t for t in response.tools if t.name in ["end_episode"]]
        assert len(hardcoded_tools) == 1

        # Verify end_episode tool definition
        end_episode_tool = next(t for t in response.tools if t.name == "end_episode")
        assert (
            end_episode_tool.description
            == "End the current episode and optionally record a discovered flag/target/objective"
        )
        assert end_episode_tool.inputSchema.required == []
        assert "submission" in end_episode_tool.inputSchema.properties

        mcp_api.session_manager.execution_manager.to_mcp_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self, mcp_api):
        """Test successful MCP tool execution."""
        # Mock successful command execution
        command_result = CommandResult.success_result(data={"output": "test output"})
        mcp_api.session_manager.execute_action.return_value = command_result

        # Mock episode for episode lookup
        mock_episode = MagicMock()
        mock_episode.task_id = "task_123"
        mock_episode.steps = []
        mock_episode.max_steps = 10
        mcp_api.session_manager.get_episode_by_id.return_value = mock_episode

        # Mock headers with required orchestration and session/episode IDs
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_123",
                HTTPHeaders.EPISODE_ID: "episode_456",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone"
            }

            # Test tool call (session_id and episode_id come from headers)
            result = await mcp_api.handle_call_tool(
                name="bash", arguments={"command": "ls", "parameters": {}}
            )

        # Verify result format
        assert result.isError is False
        assert result.content[0]["type"] == "application/json"
        assert "stdout" in result.content[0]["data"]
        assert "stderr" in result.content[0]["data"]
        assert "exit_code" in result.content[0]["data"]

        # Verify execute_action was called correctly with both session_id and episode_id
        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args
        assert call_args[0][0] == "session_123"  # session_id
        assert call_args[0][1] == "episode_456"  # episode_id
        assert isinstance(call_args[0][2], Action)  # action

    @pytest.mark.asyncio
    async def test_handle_call_tool_error(self, mcp_api):
        """Test MCP tool execution with error."""
        # Mock failed command execution
        command_result = CommandResult.error_result(error="Command failed")
        mcp_api.session_manager.execute_action.return_value = command_result

        # Mock episode for episode lookup
        mock_episode = MagicMock()
        mock_episode.task_id = "task_123"
        mock_episode.steps = []
        mock_episode.max_steps = 10
        mcp_api.session_manager.get_episode_by_id.return_value = mock_episode

        # Mock headers with required orchestration and session/episode IDs
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_123",
                HTTPHeaders.EPISODE_ID: "episode_456",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone"
            }

            # Test tool call
            result = await mcp_api.handle_call_tool(
                name="bash", arguments={"command": "invalid_command"}
            )

        # Verify error result format
        assert result.isError is True
        assert result.content[0]["type"] == "application/json"
        assert "stdout" in result.content[0]["data"]
        assert "stderr" in result.content[0]["data"]
        assert "exit_code" in result.content[0]["data"]

    @pytest.mark.asyncio
    async def test_handle_call_tool_missing_session(self, mcp_api):
        """Test MCP tool execution without session_id."""
        # Mock no session found in headers
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.ORCHESTRATION_ENV: "standalone"  # Only orchestration header, no session
            }

            # Test tool call without session_id
            result = await mcp_api.handle_call_tool(name="bash", arguments={"command": "ls"})

        # Verify error result
        assert result.isError is True
        assert "No SABER session mapped to MCP request" in result.content[0]["text"]

    @pytest.mark.asyncio
    async def test_handle_end_episode_call_success(self, mcp_api):
        """Test successful end_episode tool call without result."""
        # Mock end_episode method
        mcp_api.session_manager.end_episode = AsyncMock()

        # Test end_episode tool call with proper session_id, episode_id, and orchestration_env
        result = await mcp_api._handle_end_episode_call({}, "session_123", "episode_456", OrchestrationEnvironment.STANDALONE)

        # Verify successful result
        assert result.isError is False
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "No submission provided"

        # Verify end_episode was called with both session_id and episode_id
        mcp_api.session_manager.end_episode.assert_called_once_with("session_123", "episode_456", "agent_completed")

    @pytest.mark.asyncio
    async def test_handle_end_episode_call_with_result(self, mcp_api):
        """Test end_episode tool call with result flag."""
        # Mock methods
        mcp_api.session_manager.end_episode = AsyncMock()
        mcp_api.session_manager.execute_action = AsyncMock(
            return_value=CommandResult.success_result(data="Action recorded")
        )

        # Test end_episode tool call with result - provide session_id, episode_id, and orchestration_env
        result = await mcp_api._handle_end_episode_call(
            {"parameters": {"submission": "flag{test_flag_found}"}}, "session_123", "episode_456", OrchestrationEnvironment.STANDALONE
        )

        # Verify successful result with flag
        assert result.isError is False
        assert result.content[0]["type"] == "text"
        result_text = result.content[0]["text"]
        assert result_text == "flag{test_flag_found}"

        # Verify result action was executed with both session_id and episode_id
        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args
        assert call_args[0][0] == "session_123"  # session_id
        assert call_args[0][1] == "episode_456"  # episode_id
        action = call_args[0][2]  # action
        assert action.tool_name == "episode_result"
        assert "flag{test_flag_found}" in action.parameters.get("submission", "")
        assert action.parameters["submission"] == "flag{test_flag_found}"
        assert action.parameters["episode_end"] is True

        # Verify end_episode was called with both session_id and episode_id and EvalSubmission
        mcp_api.session_manager.end_episode.assert_called_once()
        call_args = mcp_api.session_manager.end_episode.call_args
        assert call_args[0][0] == "session_123"  # session_id
        assert call_args[0][1] == "episode_456"  # episode_id
        assert call_args[0][2] == "agent_completed"  # status

        # Verify EvalSubmission object
        eval_submission = call_args[0][3]
        assert eval_submission.episode_id == "episode_456"
        assert eval_submission.submission == "flag{test_flag_found}"
        assert eval_submission.model == "mcp_agent"

    @pytest.mark.asyncio
    async def test_handle_end_episode_call_missing_session(self, mcp_api):
        """Test end_episode tool call without session_id."""
        # Test end_episode call without session_id but with orchestration_env
        result = await mcp_api._handle_end_episode_call({"result": "some_flag"}, None, None, OrchestrationEnvironment.STANDALONE)

        # Verify error result
        assert result.isError is True
        assert "No SABER session mapped to MCP request" in result.content[0]["text"]

    @pytest.mark.asyncio
    async def test_handle_call_tool_hardcoded_tools(self, mcp_api):
        """Test handle_call_tool routing for end_episode tool."""
        # Mock end_episode for end_episode tool execution
        mcp_api.session_manager.end_episode = AsyncMock()
        mcp_api.session_manager.execute_action = AsyncMock(
            return_value=CommandResult.success_result(data={"output": "Episode action completed"})
        )
        # Mock episode lookup
        mock_episode = MagicMock()
        mock_episode.task_id = "task_123"
        mock_episode.steps = []
        mock_episode.max_steps = 10
        mcp_api.session_manager.get_episode_by_id.return_value = mock_episode

        # Mock headers with required orchestration and session/episode IDs
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_123",
                HTTPHeaders.EPISODE_ID: "episode_456",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone"
            }

            # Test end_episode routing through handle_call_tool (goes via execute_action)
            result = await mcp_api.handle_call_tool("end_episode", {"parameters": {"submission": "flag{test}"}})

        # Verify it executes without error and calls execute_action
        assert result.isError is False
        assert result.content[0]["type"] == "application/json"
        assert "stdout" in result.content[0]["data"]

        # Verify execute_action was called with end_episode action
        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args
        assert call_args[0][0] == "session_123"  # session_id
        assert call_args[0][1] == "episode_456"  # episode_id
        action = call_args[0][2]  # action
        assert action.tool_name == "end_episode"

    def test_convert_to_action(self, mcp_api):
        """Test conversion from MCP tool call to Action."""
        action = mcp_api._convert_to_action(
            tool_name="bash",
            arguments={
                "session_id": "session_123",
                "parameters": {"arguments": "ls -la", "flag": "-l"},
                "context": {"extra": "data"},
            },
        )

        assert action.tool_name == "bash"
        assert action.parameters == {"parameters": {"arguments": "ls -la", "flag": "-l"}, "context": {"extra": "data"}}

    def test_convert_to_mcp_result_success(self, mcp_api):
        """Test conversion of successful CommandResult to MCP format."""
        command_result = CommandResult.success_result(data={"output": "test output"})

        mcp_result = mcp_api._convert_to_mcp_result(command_result)

        assert mcp_result.isError is False
        assert mcp_result.content[0]["type"] == "application/json"
        assert "stdout" in mcp_result.content[0]["data"]
        assert "stderr" in mcp_result.content[0]["data"]
        assert "exit_code" in mcp_result.content[0]["data"]

    def test_convert_to_mcp_result_error(self, mcp_api):
        """Test conversion of error CommandResult to MCP format."""
        command_result = CommandResult.error_result(error="Command execution failed")

        mcp_result = mcp_api._convert_to_mcp_result(command_result)

        assert mcp_result.isError is True
        assert mcp_result.content[0]["type"] == "application/json"
        assert "stdout" in mcp_result.content[0]["data"]
        assert "stderr" in mcp_result.content[0]["data"]
        assert "exit_code" in mcp_result.content[0]["data"]

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
            mock_server.run_async.assert_called_once_with(transport="http", host="127.0.0.1", port=3001)

            # Test shutdown
            await mcp_api.shutdown_mcp_server()
            # FastMCP doesn't require explicit cleanup, just sets server to None
            assert mcp_api.mcp_server is None


class TestSessionMCPAPIOrchestration:
    """Test SessionMCPAPI orchestration environment functionality."""

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

    @pytest.fixture
    def mock_executor_tools(self):
        """Mock tools from execution manager."""
        return [
            {
                "name": "bash",
                "description": "Command line executor",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "python",
                "description": "Python executor",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"]
                }
            }
        ]

    def test_get_headers_with_all_headers(self, mcp_api):
        """Test _get_headers method with all header values present."""
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "test-session-123",
                HTTPHeaders.EPISODE_ID: "test-episode-456",
                HTTPHeaders.ORCHESTRATION_ENV: "inspect",
                HTTPHeaders.TASK_ID: "test-task-789",
                HTTPHeaders.CLIENT_ID: "test-client-abc"
            }

            # Use asyncio to call the async method
            import asyncio
            headers = asyncio.run(mcp_api._get_headers())

        assert isinstance(headers, RequestHeaders)
        assert headers.session_id == "test-session-123"
        assert headers.episode_id == "test-episode-456"
        assert headers.orchestration_env == OrchestrationEnvironment.INSPECT
        assert headers.task_id == "test-task-789"
        assert headers.client_id == "test-client-abc"

    def test_get_headers_with_minimal_headers(self, mcp_api):
        """Test _get_headers method with only required orchestration header."""
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.ORCHESTRATION_ENV: "standalone"
            }

            import asyncio
            headers = asyncio.run(mcp_api._get_headers())

        assert isinstance(headers, RequestHeaders)
        assert headers.session_id is None
        assert headers.episode_id is None
        assert headers.orchestration_env == OrchestrationEnvironment.STANDALONE
        assert headers.task_id is None
        assert headers.client_id is None

    def test_get_headers_missing_orchestration_header(self, mcp_api):
        """Test _get_headers method with missing orchestration header."""
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "test-session"
            }

            import asyncio
            with pytest.raises(ValueError, match="MANDATORY header X-SABER-Orchestration-Env not found"):
                asyncio.run(mcp_api._get_headers())

    def test_get_headers_invalid_orchestration_value(self, mcp_api):
        """Test _get_headers method with invalid orchestration value."""
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.ORCHESTRATION_ENV: "invalid_value"
            }

            import asyncio
            with pytest.raises(ValueError, match="Invalid orchestration environment"):
                asyncio.run(mcp_api._get_headers())

    @pytest.mark.asyncio
    async def test_handle_list_tools_standalone_orchestration(self, mcp_api, mock_executor_tools):
        """Test tool discovery with STANDALONE orchestration shows hardcoded tools."""
        # Mock execution manager returning tools
        mcp_api.session_manager.execution_manager.to_mcp_tools.return_value = mock_executor_tools

        # Mock headers to return STANDALONE orchestration
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {HTTPHeaders.ORCHESTRATION_ENV: "standalone"}

            response = await mcp_api.handle_list_tools()

        # Should include executor tools + hardcoded tools for STANDALONE
        assert len(response.tools) == 3  # 2 executor tools + 1 hardcoded tool (end_episode)

        # Check executor tools are included
        executor_tool_names = [t.name for t in response.tools if t.name in ["bash", "python"]]
        assert len(executor_tool_names) == 2

        # Check hardcoded tools are included for STANDALONE
        hardcoded_tool_names = [t.name for t in response.tools if t.name == "end_episode"]
        assert len(hardcoded_tool_names) == 1

        mcp_api.session_manager.execution_manager.to_mcp_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_list_tools_inspect_orchestration(self, mcp_api, mock_executor_tools):
        """Test tool discovery with INSPECT orchestration hides hardcoded tools."""
        # Mock execution manager returning tools
        mcp_api.session_manager.execution_manager.to_mcp_tools.return_value = mock_executor_tools

        # Mock headers to return INSPECT orchestration
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {HTTPHeaders.ORCHESTRATION_ENV: "inspect"}

            response = await mcp_api.handle_list_tools()

        # Should only include executor tools, no hardcoded tools for INSPECT
        assert len(response.tools) == 2  # Only 2 executor tools

        # Check executor tools are included
        executor_tool_names = [t.name for t in response.tools if t.name in ["bash", "python"]]
        assert len(executor_tool_names) == 2

        # Check hardcoded tools are NOT included for INSPECT
        hardcoded_tool_names = [t.name for t in response.tools if t.name == "end_episode"]
        assert len(hardcoded_tool_names) == 0

        mcp_api.session_manager.execution_manager.to_mcp_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_list_tools_missing_orchestration_header(self, mcp_api, mock_executor_tools):
        """Test tool discovery with missing orchestration header."""
        # Mock execution manager returning tools
        mcp_api.session_manager.execution_manager.to_mcp_tools.return_value = mock_executor_tools

        # Mock headers to return no orchestration header
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {}

            response = await mcp_api.handle_list_tools()

        # Should return empty tools list due to error
        assert len(response.tools) == 0

    @pytest.mark.asyncio
    async def test_handle_call_tool_with_orchestration_context(self, mcp_api):
        """Test tool execution with orchestration context in headers."""
        # Mock successful command execution
        command_result = CommandResult.success_result(data={"output": "test output"})
        mcp_api.session_manager.execute_action.return_value = command_result

        # Mock episode for episode lookup
        mock_episode = MagicMock()
        mock_episode.task_id = "task_123"
        mock_episode.steps = []
        mock_episode.max_steps = 10
        mcp_api.session_manager.get_episode_by_id.return_value = mock_episode

        # Mock headers with orchestration environment
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_123",
                HTTPHeaders.EPISODE_ID: "episode_456",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone",
                HTTPHeaders.TASK_ID: "task_789"
            }

            # Test tool call
            result = await mcp_api.handle_call_tool(
                name="bash",
                arguments={"command": "ls", "parameters": {}}
            )

        # Verify result format
        assert result.isError is False
        assert result.content[0]["type"] == "application/json"
        assert "stdout" in result.content[0]["data"]

        # Verify execute_action was called correctly
        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args
        assert call_args[0][0] == "session_123"  # session_id
        assert call_args[0][1] == "episode_456"  # episode_id
        assert isinstance(call_args[0][2], Action)  # action

    @pytest.mark.asyncio
    async def test_end_episode_tool_standalone_orchestration(self, mcp_api):
        """Test end_episode tool with STANDALONE orchestration (direct call)."""
        # Mock end_episode method
        mcp_api.session_manager.end_episode = AsyncMock()

        # Test end_episode tool call with STANDALONE orchestration
        result = await mcp_api._handle_end_episode_call(
            {}, "session_123", "episode_456", OrchestrationEnvironment.STANDALONE
        )

        # Verify successful result
        assert result.isError is False
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "No submission provided"

        # Verify end_episode was called
        mcp_api.session_manager.end_episode.assert_called_once_with("session_123", "episode_456", "agent_completed")

    @pytest.mark.asyncio
    async def test_end_episode_tool_inspect_orchestration(self, mcp_api):
        """Test end_episode tool with INSPECT orchestration (direct call)."""
        # Mock end_episode method
        mcp_api.session_manager.end_episode = AsyncMock()

        # Test end_episode tool call with INSPECT orchestration
        result = await mcp_api._handle_end_episode_call(
            {}, "session_123", "episode_456", OrchestrationEnvironment.INSPECT
        )

        # Verify successful result (functionality should be the same regardless of orchestration)
        assert result.isError is False
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "No submission provided"

        # Verify end_episode was called
        mcp_api.session_manager.end_episode.assert_called_once_with("session_123", "episode_456", "agent_completed")

    def test_orchestration_environment_logging_context(self, mcp_api):
        """Test that orchestration environment appears in logging context."""
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_123",
                HTTPHeaders.ORCHESTRATION_ENV: "inspect",
                HTTPHeaders.TASK_ID: "task_789"
            }

            import asyncio
            headers = asyncio.run(mcp_api._get_headers())
            context_summary = headers.context_summary

            assert "orchestration:inspect" in context_summary
            assert "session:session_123" in context_summary
            assert "task:task_789" in context_summary

    def test_orchestration_environment_validation_edge_cases(self, mcp_api):
        """Test orchestration environment validation edge cases."""
        # Test case-sensitive validation
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {HTTPHeaders.ORCHESTRATION_ENV: "INSPECT"}  # Wrong case

            import asyncio
            with pytest.raises(ValueError, match="Invalid orchestration environment 'INSPECT'"):
                asyncio.run(mcp_api._get_headers())

        # Test empty string
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {HTTPHeaders.ORCHESTRATION_ENV: ""}

            import asyncio
            with pytest.raises(ValueError, match="MANDATORY header X-SABER-Orchestration-Env not found"):
                asyncio.run(mcp_api._get_headers())

        # Test whitespace
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {HTTPHeaders.ORCHESTRATION_ENV: " standalone "}

            import asyncio
            with pytest.raises(ValueError, match="Invalid orchestration environment ' standalone '"):
                asyncio.run(mcp_api._get_headers())
            assert mcp_api.mcp_server is None
