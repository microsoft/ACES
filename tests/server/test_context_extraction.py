"""
Test that context parameters (__saber_assistant_message__ and __saber_reasoning__)
flow correctly from tool arguments through to Action objects.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.models.headers import HTTPHeaders
from saber.server.api.session_mcp_api import SessionMCPAPI
from saber.server.base import Action, CommandResult


class TestContextExtraction:
    """Test context parameter extraction in MCP tool calls."""

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

    @pytest.mark.asyncio
    async def test_context_parameters_extracted_to_action(self, mcp_api):
        """Test that __saber_assistant_message__ and __saber_reasoning__ are extracted to Action."""
        # Mock successful command execution
        command_result = CommandResult.success_result(data={"output": "test output"})
        mcp_api.session_manager.execute_action.return_value = command_result

        # Mock episode for episode lookup
        mock_episode = MagicMock()
        mock_episode.task_id = "task_123"
        mock_episode.steps = []
        mock_episode.max_steps = 10
        mcp_api.session_manager.get_episode_by_id.return_value = mock_episode

        # Mock headers
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_123",
                HTTPHeaders.EPISODE_ID: "episode_456",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone"
            }

            # Test tool call WITH context parameters
            result = await mcp_api.handle_call_tool(
                name="bash",
                arguments={
                    "command": "ls -la",
                    "__saber_assistant_message__": "I need to list files to understand the directory structure.",
                    "__saber_reasoning__": "First step is to gather information about the environment.",
                }
            )

        # Verify result is successful
        assert result.isError is False

        # Verify execute_action was called
        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args

        # Extract the Action object that was passed
        action: Action = call_args[0][2]

        # CRITICAL ASSERTIONS: Verify context was extracted correctly
        print(f"\n🔍 DEBUG: Action fields:")
        print(f"  tool_name: {action.tool_name}")
        print(f"  parameters: {action.parameters}")
        print(f"  assistant_message: {action.assistant_message}")
        print(f"  reasoning: {action.reasoning}")

        # Verify assistant_message was extracted
        assert action.assistant_message is not None, "assistant_message should not be None"
        assert action.assistant_message == "I need to list files to understand the directory structure."

        # Verify reasoning was extracted
        assert action.reasoning is not None, "reasoning should not be None"
        assert action.reasoning == "First step is to gather information about the environment."

        # Verify context parameters were stripped from parameters dict
        assert "__saber_assistant_message__" not in action.parameters
        assert "__saber_reasoning__" not in action.parameters

        # Verify regular parameters are still present
        assert "command" in action.parameters
        assert action.parameters["command"] == "ls -la"

    @pytest.mark.asyncio
    async def test_context_parameters_optional(self, mcp_api):
        """Test that context parameters are optional (tool works without them)."""
        # Mock successful command execution
        command_result = CommandResult.success_result(data={"output": "test output"})
        mcp_api.session_manager.execute_action.return_value = command_result

        # Mock episode
        mock_episode = MagicMock()
        mock_episode.task_id = "task_123"
        mock_episode.steps = []
        mock_episode.max_steps = 10
        mcp_api.session_manager.get_episode_by_id.return_value = mock_episode

        # Mock headers
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_123",
                HTTPHeaders.EPISODE_ID: "episode_456",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone"
            }

            # Test tool call WITHOUT context parameters
            result = await mcp_api.handle_call_tool(
                name="bash",
                arguments={"command": "ls -la"}
            )

        # Verify result is successful
        assert result.isError is False

        # Extract the Action object
        call_args = mcp_api.session_manager.execute_action.call_args
        action: Action = call_args[0][2]

        # Verify context fields are None when not provided
        assert action.assistant_message is None
        assert action.reasoning is None

        # Verify regular parameters still work
        assert action.parameters["command"] == "ls -la"

    @pytest.mark.asyncio
    async def test_convert_to_action_directly(self, mcp_api):
        """Test _convert_to_action method directly."""
        # Test WITH context
        action_with_context = mcp_api._convert_to_action(
            tool_name="bash",
            arguments={
                "command": "pwd",
                "__saber_assistant_message__": "Let me check the current directory.",
                "__saber_reasoning__": "Understanding the working directory is essential.",
            }
        )

        assert action_with_context.tool_name == "bash"
        assert action_with_context.parameters == {"command": "pwd"}
        assert action_with_context.assistant_message == "Let me check the current directory."
        assert action_with_context.reasoning == "Understanding the working directory is essential."

        # Test WITHOUT context
        action_without_context = mcp_api._convert_to_action(
            tool_name="bash",
            arguments={"command": "pwd"}
        )

        assert action_without_context.tool_name == "bash"
        assert action_without_context.parameters == {"command": "pwd"}
        assert action_without_context.assistant_message is None
        assert action_without_context.reasoning is None
