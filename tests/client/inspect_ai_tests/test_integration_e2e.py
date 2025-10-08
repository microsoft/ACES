"""
Integration test to verify V2 context injection works end-to-end.

This test simulates a real SABER agent execution to ensure:
1. Context is captured from assistant messages
2. Context is injected into tool parameters
3. MCP server receives the context
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.model._call_tools import ExecuteToolsResult

from saber.client.inspect_ai.context_injection import (
    _context,
    saber_execute_tools,
    saber_tool_params,
)


@pytest.mark.asyncio
async def test_end_to_end_context_injection():
    """Test complete flow: capture -> store -> inject -> receive."""

    # Setup: Create messages with assistant content
    messages = [
        ChatMessageSystem(content="System"),
        ChatMessageUser(content="Run ls command"),
        ChatMessageAssistant(content="I'll execute the ls command for you")
    ]

    # Track what the MCP tool receives
    received_params = {}

    # Create a mock SABER MCP tool
    async def mock_saber_run_command(**kwargs):
        received_params.update(kwargs)
        return "command executed"

    mock_saber_run_command.__name__ = "saber_run_command"
    mock_saber_run_command.__module__ = "mcp.tools"

    # Mock tool_params to intercept parameter injection
    def mock_tool_params(input_dict, func):
        # Simulate original tool_params behavior
        params = {"command": input_dict.get("command", "ls")}

        # Our saber_tool_params should add context
        return saber_tool_params(input_dict, func)

    # Mock execute_tools to call tool_params and the tool
    async def mock_execute_tools(messages, tools, max_output):
        # Simulate inspect_ai calling tool_params
        tool = tools[0]
        params = mock_tool_params({"command": "ls"}, tool)

        # Call the tool with params
        result = await tool(**params)

        return ExecuteToolsResult(messages=[], output=result)

    # Patch and run
    with patch('saber.client.inspect_ai.context_injection.original_execute_tools', side_effect=mock_execute_tools):
        with patch('saber.client.inspect_ai.context_injection.original_tool_params', return_value={"command": "ls"}):
            result = await saber_execute_tools(messages, [mock_saber_run_command], None)

    # Verify context was injected
    assert "__saber_assistant_message__" in received_params
    assert received_params["__saber_assistant_message__"] == "I'll execute the ls command for you"
    assert "command" in received_params
    assert received_params["command"] == "ls"

    # Verify context was cleared after execution
    assert _context.assistant_message is None
    assert _context.reasoning is None


if __name__ == "__main__":
    asyncio.run(test_end_to_end_context_injection())
    print("✅ End-to-end context injection test passed!")
