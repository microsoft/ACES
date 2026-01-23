"""Tests for SABER Copilot tool bridge.

TDD tests for converting MCP tools to Copilot SDK tools.
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from typing import Any


class TestMCPToCopilotToolConversion:
    """Test cases for converting MCP tools to Copilot SDK Tool format."""

    def test_convert_simple_mcp_tool_to_copilot_tool(self):
        """Test converting a simple MCP tool with basic parameters."""
        from saber.inspect_ai.integration.copilot_tools import mcp_tool_to_copilot_tool

        # Create mock MCP tool
        mcp_tool = Mock()
        mcp_tool.name = "bash"
        mcp_tool.description = "Execute bash commands"
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"],
        }

        # Mock MCP client for execution
        mock_mcp_client = AsyncMock()
        mock_mcp_client.call_tool = AsyncMock(return_value=Mock(content=[Mock(text="output")]))

        # Convert to Copilot tool
        copilot_tool = mcp_tool_to_copilot_tool(mcp_tool, mock_mcp_client)

        # Verify tool structure
        assert copilot_tool.name == "bash"
        assert copilot_tool.description == "Execute bash commands"
        assert copilot_tool.parameters == mcp_tool.inputSchema
        assert callable(copilot_tool.handler)

    def test_convert_mcp_tool_preserves_schema(self):
        """Test that conversion preserves complex JSON schemas."""
        from saber.inspect_ai.integration.copilot_tools import mcp_tool_to_copilot_tool

        mcp_tool = Mock()
        mcp_tool.name = "file_edit"
        mcp_tool.description = "Edit a file"
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "New content"},
                "mode": {
                    "type": "string",
                    "enum": ["append", "replace"],
                    "default": "replace",
                },
            },
            "required": ["path", "content"],
        }

        mock_mcp_client = AsyncMock()
        copilot_tool = mcp_tool_to_copilot_tool(mcp_tool, mock_mcp_client)

        # Schema should be preserved exactly
        assert copilot_tool.parameters == mcp_tool.inputSchema

    def test_convert_mcp_tool_without_schema(self):
        """Test converting MCP tool with no input schema."""
        from saber.inspect_ai.integration.copilot_tools import mcp_tool_to_copilot_tool

        # Use spec=[] to prevent auto-mocking of attributes
        mcp_tool = Mock(spec=["name", "description", "inputSchema"])
        mcp_tool.name = "get_status"
        mcp_tool.description = "Get current status"
        mcp_tool.inputSchema = None

        mock_mcp_client = AsyncMock()
        copilot_tool = mcp_tool_to_copilot_tool(mcp_tool, mock_mcp_client)

        assert copilot_tool.name == "get_status"
        assert copilot_tool.parameters is None

    @pytest.mark.asyncio
    async def test_tool_handler_calls_mcp_client(self):
        """Test that the Copilot tool handler correctly calls MCP client."""
        from saber.inspect_ai.integration.copilot_tools import mcp_tool_to_copilot_tool

        mcp_tool = Mock()
        mcp_tool.name = "bash"
        mcp_tool.description = "Execute bash commands"
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

        # Mock MCP client response
        mock_content = Mock(spec=["text"])
        mock_content.text = "command executed successfully"
        mock_result = Mock(spec=["content", "isError", "is_error"])
        mock_result.content = [mock_content]
        mock_result.isError = False
        mock_result.is_error = False

        mock_mcp_client = AsyncMock()
        mock_mcp_client.call_tool = AsyncMock(return_value=mock_result)

        copilot_tool = mcp_tool_to_copilot_tool(mcp_tool, mock_mcp_client)

        # Create invocation
        invocation = {
            "session_id": "test-session",
            "tool_call_id": "call-123",
            "tool_name": "bash",
            "arguments": {"command": "ls -la"},
        }

        # Call handler
        result = await copilot_tool.handler(invocation)

        # Verify MCP client was called correctly
        mock_mcp_client.call_tool.assert_called_once_with("bash", {"command": "ls -la"})

        # Verify result format
        assert result["resultType"] == "success"
        assert "command executed successfully" in result["textResultForLlm"]

    @pytest.mark.asyncio
    async def test_tool_handler_handles_mcp_error(self):
        """Test that handler properly handles MCP tool errors."""
        from saber.inspect_ai.integration.copilot_tools import mcp_tool_to_copilot_tool

        mcp_tool = Mock()
        mcp_tool.name = "bash"
        mcp_tool.description = "Execute bash commands"
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        # Mock MCP client error response
        mock_content = Mock()
        mock_content.text = "Permission denied"
        mock_result = Mock()
        mock_result.content = [mock_content]
        mock_result.isError = True

        mock_mcp_client = AsyncMock()
        mock_mcp_client.call_tool = AsyncMock(return_value=mock_result)

        copilot_tool = mcp_tool_to_copilot_tool(mcp_tool, mock_mcp_client)

        invocation = {
            "session_id": "test-session",
            "tool_call_id": "call-123",
            "tool_name": "bash",
            "arguments": {},
        }

        result = await copilot_tool.handler(invocation)

        # Should return failure result
        assert result["resultType"] == "failure"
        assert "Permission denied" in result["textResultForLlm"]

    @pytest.mark.asyncio
    async def test_tool_handler_handles_exception(self):
        """Test that handler catches and reports exceptions."""
        from saber.inspect_ai.integration.copilot_tools import mcp_tool_to_copilot_tool

        mcp_tool = Mock()
        mcp_tool.name = "bash"
        mcp_tool.description = "Execute bash commands"
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        mock_mcp_client = AsyncMock()
        mock_mcp_client.call_tool = AsyncMock(side_effect=ConnectionError("MCP server unavailable"))

        copilot_tool = mcp_tool_to_copilot_tool(mcp_tool, mock_mcp_client)

        invocation = {
            "session_id": "test-session",
            "tool_call_id": "call-123",
            "tool_name": "bash",
            "arguments": {},
        }

        result = await copilot_tool.handler(invocation)

        assert result["resultType"] == "failure"
        # Error message should be sanitized (not exposing internal details)
        assert "error" in result


class TestConvertAllMCPTools:
    """Test cases for batch conversion of MCP tools."""

    @pytest.mark.asyncio
    async def test_convert_multiple_tools(self):
        """Test converting a list of MCP tools."""
        from saber.inspect_ai.integration.copilot_tools import convert_mcp_tools_to_copilot

        # Create multiple mock MCP tools
        tool1 = Mock()
        tool1.name = "bash"
        tool1.description = "Execute bash"
        tool1.inputSchema = {"type": "object", "properties": {}}

        tool2 = Mock()
        tool2.name = "python"
        tool2.description = "Execute python"
        tool2.inputSchema = {"type": "object", "properties": {}}

        mcp_tools = [tool1, tool2]

        mock_mcp_client = AsyncMock()

        copilot_tools = convert_mcp_tools_to_copilot(mcp_tools, mock_mcp_client)

        assert len(copilot_tools) == 2
        assert copilot_tools[0].name == "bash"
        assert copilot_tools[1].name == "python"

    @pytest.mark.asyncio
    async def test_convert_empty_tools_list(self):
        """Test converting an empty list of tools."""
        from saber.inspect_ai.integration.copilot_tools import convert_mcp_tools_to_copilot

        mock_mcp_client = AsyncMock()
        copilot_tools = convert_mcp_tools_to_copilot([], mock_mcp_client)

        assert copilot_tools == []


class TestCreateSubmitTool:
    """Test cases for the submit tool creation."""

    def test_create_submit_tool_returns_tool(self):
        """Test that create_submit_tool returns a valid Tool."""
        from saber.inspect_ai.integration.copilot_tools import create_submit_tool

        tool = create_submit_tool()

        assert tool.name == "submit"
        assert "final answer" in tool.description.lower() or "submit" in tool.description.lower()
        assert callable(tool.handler)

    def test_submit_tool_has_answer_parameter(self):
        """Test that submit tool requires an answer parameter."""
        from saber.inspect_ai.integration.copilot_tools import create_submit_tool

        tool = create_submit_tool()

        assert tool.parameters is not None
        assert "properties" in tool.parameters
        assert "answer" in tool.parameters["properties"]
        assert "answer" in tool.parameters.get("required", [])

    @pytest.mark.asyncio
    async def test_submit_tool_handler_returns_answer(self):
        """Test that submit tool returns the answer directly (inspect_ai pattern)."""
        from saber.inspect_ai.integration.copilot_tools import create_submit_tool

        tool = create_submit_tool()

        invocation = {
            "session_id": "test-session",
            "tool_call_id": "call-123",
            "tool_name": "submit",
            "arguments": {"answer": "The flag is CTF{secret}"},
        }

        result = await tool.handler(invocation)

        # Should return the answer directly (like inspect_ai's native submit)
        assert result["resultType"] == "success"
        assert result["textResultForLlm"] == "The flag is CTF{secret}"

    @pytest.mark.asyncio
    async def test_submit_tool_handler_with_empty_answer(self):
        """Test submit tool handles empty answer gracefully."""
        from saber.inspect_ai.integration.copilot_tools import create_submit_tool

        tool = create_submit_tool()

        invocation = {
            "session_id": "test-session",
            "tool_call_id": "call-123",
            "tool_name": "submit",
            "arguments": {"answer": ""},
        }

        result = await tool.handler(invocation)

        # Should handle empty answer gracefully
        assert result["resultType"] == "success"
        assert result["textResultForLlm"] == ""
