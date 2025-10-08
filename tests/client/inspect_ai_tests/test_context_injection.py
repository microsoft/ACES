"""
Unit tests for SABER context injection system.

These tests validate the context capture and injection mechanism that uses
tool_params patching to avoid wrapper bypass issues.

Critical Test Coverage:
1. Assistant message extraction from various content formats
2. Reasoning content extraction from o1/o3 model responses
3. SABER MCP tool detection logic
4. Context storage and retrieval
5. tool_params parameter injection
6. Integration with inspect_ai tool execution pattern
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from inspect_ai._util.content import ContentText, ContentReasoning
from inspect_ai.model._call_tools import ExecuteToolsResult
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

from saber.client.inspect_ai.context_injection import (
    _context,
    extract_assistant_content,
    extract_reasoning_content,
    is_saber_mcp_tool,
    saber_tool_params,
    saber_execute_tools,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_messages():
    """Create simple message list with assistant message."""
    return [
        ChatMessageSystem(content="You are a helpful assistant"),
        ChatMessageUser(content="What is 2+2?"),
        ChatMessageAssistant(content="Let me calculate that for you. 2+2 equals 4.")
    ]


@pytest.fixture
def reasoning_messages():
    """Create message list with reasoning content."""
    reasoning_content = ContentReasoning(reasoning="First, I analyze the problem...")
    assistant_text = ContentText(text="Here's the solution")

    return [
        ChatMessageSystem(content="System"),
        ChatMessageUser(content="Solve this"),
        ChatMessageAssistant(content=[reasoning_content, assistant_text])
    ]


# ============================================================================
# Test: extract_assistant_content()
# ============================================================================


def test_extract_assistant_content_string():
    """Test extraction from simple string content."""
    message = ChatMessageAssistant(content="Simple message")
    result = extract_assistant_content(message)
    assert result == "Simple message"


def test_extract_assistant_content_list_with_text():
    """Test extraction from list of ContentText objects."""
    message = ChatMessageAssistant(content=[
        ContentText(text="Part 1"),
        ContentText(text="Part 2")
    ])
    result = extract_assistant_content(message)
    assert result == "Part 1\nPart 2"


def test_extract_assistant_content_skips_reasoning():
    """Test that ContentReasoning objects are skipped."""
    message = ChatMessageAssistant(content=[
        ContentReasoning(reasoning="Thinking..."),
        ContentText(text="Answer")
    ])
    result = extract_assistant_content(message)
    assert result == "Answer"


def test_extract_assistant_content_empty():
    """Test handling of empty content."""
    message = ChatMessageAssistant(content=[])
    result = extract_assistant_content(message)
    assert result is None


# ============================================================================
# Test: extract_reasoning_content()
# ============================================================================


def test_extract_reasoning_from_attribute():
    """Test extraction from reasoning attribute (o1 models)."""
    message = Mock()
    message.reasoning = "Reasoning content"
    message.content = "Regular content"

    result = extract_reasoning_content(message)
    assert result == "Reasoning content"


def test_extract_reasoning_from_content_list():
    """Test extraction from ContentReasoning in content list."""
    message = ChatMessageAssistant(content=[
        ContentReasoning(reasoning="Reasoning here"),
        ContentText(text="Message")
    ])
    result = extract_reasoning_content(message)
    assert result == "Reasoning here"


def test_extract_reasoning_no_reasoning():
    """Test when no reasoning is present."""
    message = ChatMessageAssistant(content="Just a message")
    result = extract_reasoning_content(message)
    assert result is None


# ============================================================================
# Test: is_saber_mcp_tool()
# ============================================================================


def test_is_saber_mcp_tool_by_name():
    """Test detection by tool name containing 'saber'."""
    async def saber_run_command(**kwargs):
        pass

    assert is_saber_mcp_tool(saber_run_command) is True


def test_is_saber_mcp_tool_by_module():
    """Test detection by module name containing 'mcp'."""
    tool = Mock()
    tool.__name__ = "run_command"
    tool.__module__ = "mcp.tools"

    assert is_saber_mcp_tool(tool) is True


def test_is_saber_mcp_tool_builtin():
    """Test that built-in tools are not detected as SABER MCP."""
    async def web_search(**kwargs):
        pass

    assert is_saber_mcp_tool(web_search) is False


# ============================================================================
# Test: Context Storage (_context)
# ============================================================================


def test_context_storage_set_and_clear():
    """Test context storage set and clear operations."""
    _context.set_context("assistant message", "reasoning")

    assert _context.assistant_message == "assistant message"
    assert _context.reasoning == "reasoning"
    assert _context.has_context() is True

    _context.clear_context()

    assert _context.assistant_message is None
    assert _context.reasoning is None
    assert _context.has_context() is False


def test_context_storage_has_context():
    """Test has_context with various states."""
    _context.clear_context()
    assert _context.has_context() is False

    _context.set_context("message", None)
    assert _context.has_context() is True

    _context.clear_context()
    _context.set_context(None, "reasoning")
    assert _context.has_context() is True

    _context.clear_context()


# ============================================================================
# Test: saber_tool_params()
# ============================================================================


@pytest.mark.asyncio
async def test_saber_tool_params_injects_into_saber_tool():
    """Test that saber_tool_params injects context into SABER tools."""
    _context.clear_context()

    # Create a SABER-like tool
    async def saber_run_command(command: str, **kwargs: Any) -> str:
        return "result"

    # Set context
    _context.set_context("Assistant message", "Reasoning content")

    # Mock the original tool_params to return base params
    with patch('saber.client.inspect_ai.context_injection.original_tool_params') as mock_original:
        mock_original.return_value = {"command": "ls -la"}

        # Call saber_tool_params
        result = saber_tool_params({"command": "ls -la"}, saber_run_command)

        # Should have injected context
        assert "__saber_assistant_message__" in result
        assert result["__saber_assistant_message__"] == "Assistant message"
        assert "__saber_reasoning__" in result
        assert result["__saber_reasoning__"] == "Reasoning content"
        assert result["command"] == "ls -la"

    _context.clear_context()


@pytest.mark.asyncio
async def test_saber_tool_params_skips_non_saber_tools():
    """Test that saber_tool_params doesn't inject into non-SABER tools."""
    _context.clear_context()

    # Create a non-SABER tool
    async def web_search(query: str) -> str:
        return "results"

    # Set context
    _context.set_context("Assistant message", "Reasoning content")

    # Mock the original tool_params
    with patch('saber.client.inspect_ai.context_injection.original_tool_params') as mock_original:
        mock_original.return_value = {"query": "test"}

        # Call saber_tool_params
        result = saber_tool_params({"query": "test"}, web_search)

        # Should NOT have injected context (not a SABER tool)
        assert "__saber_assistant_message__" not in result
        assert "__saber_reasoning__" not in result
        assert result["query"] == "test"

    _context.clear_context()


@pytest.mark.asyncio
async def test_saber_tool_params_no_context():
    """Test that saber_tool_params works when no context is set."""
    _context.clear_context()

    async def saber_run_command(command: str) -> str:
        return "result"

    # Mock the original tool_params
    with patch('saber.client.inspect_ai.context_injection.original_tool_params') as mock_original:
        mock_original.return_value = {"command": "ls"}

        # Call saber_tool_params without context
        result = saber_tool_params({"command": "ls"}, saber_run_command)

        # Should not inject anything
        assert "__saber_assistant_message__" not in result
        assert "__saber_reasoning__" not in result
        assert result["command"] == "ls"


# ============================================================================
# Test: saber_execute_tools()
# ============================================================================


@pytest.mark.asyncio
async def test_saber_execute_tools_stores_context(simple_messages):
    """Test that saber_execute_tools stores context correctly."""
    _context.clear_context()

    # Mock original execute_tools
    with patch('saber.client.inspect_ai.context_injection.original_execute_tools') as mock_execute:
        mock_execute.return_value = ExecuteToolsResult(
            messages=[],
            output=None
        )

        await saber_execute_tools(simple_messages, [], None)

        # Context should have been stored temporarily and then cleared
        # (cleared after execution completes)
        assert _context.assistant_message is None
        assert _context.reasoning is None


@pytest.mark.asyncio
async def test_saber_execute_tools_captures_assistant_message(simple_messages):
    """Test assistant message capture during execution."""
    _context.clear_context()

    # Track what context was stored
    captured_context = {}

    async def mock_execute_with_capture(messages, tools, max_output):
        # Capture context while it's still set
        captured_context['assistant_message'] = _context.assistant_message
        captured_context['reasoning'] = _context.reasoning
        return ExecuteToolsResult(messages=[], output=None)

    with patch('saber.client.inspect_ai.context_injection.original_execute_tools', side_effect=mock_execute_with_capture):
        await saber_execute_tools(simple_messages, [], None)

    # Verify context was captured during execution
    assert captured_context['assistant_message'] == "Let me calculate that for you. 2+2 equals 4."
    assert captured_context['reasoning'] is None


@pytest.mark.asyncio
async def test_saber_execute_tools_clears_context_on_error():
    """Test that context is cleared even when execution fails."""
    _context.clear_context()

    messages = [
        ChatMessageSystem(content="System"),
        ChatMessageUser(content="User"),
        ChatMessageAssistant(content="Assistant")
    ]

    # Set some context first
    _context.set_context("should be cleared", "on error")

    with patch('saber.client.inspect_ai.context_injection.original_execute_tools') as mock_execute:
        mock_execute.side_effect = Exception("Test error")

        # Should handle error, clear context, and re-raise
        try:
            await saber_execute_tools(messages, [], None)
        except Exception:
            pass  # Expected

        # Context should be cleared even though error occurred
        assert _context.assistant_message is None
        assert _context.reasoning is None


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_integration_context_flows_to_tool_params():
    """Test full flow: execute_tools stores context, tool_params injects it."""
    _context.clear_context()

    messages = [
        ChatMessageSystem(content="System"),
        ChatMessageUser(content="User"),
        ChatMessageAssistant(content="I'll run a command")
    ]

    async def saber_run_command(command: str, **kwargs: Any) -> str:
        return f"Executed: {command}"

    # Track what was injected
    injected_params = {}

    async def mock_execute_with_tool_call(messages, tools, max_output):
        # While context is set, simulate tool_params being called
        with patch('saber.client.inspect_ai.context_injection.original_tool_params') as mock_original:
            mock_original.return_value = {"command": "ls"}

            params = saber_tool_params({"command": "ls"}, saber_run_command)
            injected_params.update(params)

        return ExecuteToolsResult(messages=[], output=None)

    with patch('saber.client.inspect_ai.context_injection.original_execute_tools', side_effect=mock_execute_with_tool_call):
        await saber_execute_tools(messages, [], None)

    # Verify context was injected
    assert "__saber_assistant_message__" in injected_params
    assert injected_params["__saber_assistant_message__"] == "I'll run a command"
    assert "command" in injected_params


@pytest.mark.asyncio
async def test_integration_reasoning_extraction_and_injection(reasoning_messages):
    """Test reasoning extraction and injection flow."""
    _context.clear_context()

    async def saber_tool(param: str, **kwargs: Any) -> str:
        return "result"

    injected_params = {}

    async def mock_execute_with_tool_call(messages, tools, max_output):
        with patch('saber.client.inspect_ai.context_injection.original_tool_params') as mock_original:
            mock_original.return_value = {"param": "value"}

            params = saber_tool_params({"param": "value"}, saber_tool)
            injected_params.update(params)

        return ExecuteToolsResult(messages=[], output=None)

    with patch('saber.client.inspect_ai.context_injection.original_execute_tools', side_effect=mock_execute_with_tool_call):
        await saber_execute_tools(reasoning_messages, [], None)

    # Verify both message and reasoning were injected
    assert "__saber_assistant_message__" in injected_params
    assert injected_params["__saber_assistant_message__"] == "Here's the solution"
    assert "__saber_reasoning__" in injected_params
    assert injected_params["__saber_reasoning__"] == "First, I analyze the problem..."
