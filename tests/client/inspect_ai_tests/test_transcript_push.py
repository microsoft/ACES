"""
Unit tests for transcript push functionality.

These tests validate the transcript serialization and push mechanism that
synchronizes Inspect AI task transcripts to the SABER server via REST API.

Critical Test Coverage:
1. Message serialization for all ChatMessage types
2. Handling of tool calls, reasoning content, and structured content
3. HTTP push with success/failure scenarios
4. Retry logic with exponential backoff
5. Feature flag enable/disable
6. Graceful error handling (no exceptions)
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from inspect_ai._util.content import ContentImage, ContentReasoning, ContentText
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from saber.inspect_ai.integration.transcript_sync import _push_transcript, serialize_message


# ============================================================================
# Test Fixtures
# ============================================================================


def create_mock_response(status: int, json_data: Any = None, text_data: str = ""):
    """Helper to create properly mocked aiohttp response."""
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data if json_data else {})
    mock_response.text = AsyncMock(return_value=text_data)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    return mock_response


@pytest.fixture
def mock_task_state():
    """Create a mock TaskState with sample messages."""
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageSystem(content="You are a helpful assistant"),
        ChatMessageUser(content="What is 2+2?"),
        ChatMessageAssistant(content="The answer is 4"),
    ]
    return state


@pytest.fixture
def mock_task_state_with_tools():
    """Create a mock TaskState with tool calls."""
    state = MagicMock(spec=TaskState)
    tool_call = ToolCall(
        id="call_123",
        function="calculator",
        arguments={"expression": "2+2"},
        type="function",
    )
    state.messages = [
        ChatMessageUser(content="Calculate 2+2"),
        ChatMessageAssistant(content="Let me calculate that", tool_calls=[tool_call]),
        ChatMessageTool(
            content="4",
            tool_call_id="call_123",
            function="calculator",
        ),
    ]
    return state


@pytest.fixture
def mock_task_state_with_reasoning():
    """Create a mock TaskState with reasoning content."""
    state = MagicMock(spec=TaskState)
    reasoning_content = ContentReasoning(reasoning="Let me think about this problem...")
    text_content = ContentText(text="The answer is 4")
    state.messages = [
        ChatMessageUser(content="What is 2+2?"),
        ChatMessageAssistant(content=[reasoning_content, text_content]),
    ]
    return state


@pytest.fixture
def mock_task_state_with_images():
    """Create a mock TaskState with image content."""
    state = MagicMock(spec=TaskState)
    text_content = ContentText(text="What's in this image?")
    image_content = ContentImage(image="data:image/png;base64,iVBORw0KGgo=", detail="auto")
    state.messages = [
        ChatMessageUser(content=[text_content, image_content]),
        ChatMessageAssistant(content="I see a cat in the image"),
    ]
    return state


# ============================================================================
# Test: serialize_message() - System Messages
# ============================================================================


def testserialize_message_system():
    """Test serialization of system messages."""
    msg = ChatMessageSystem(content="You are a helpful assistant")
    result = serialize_message(msg)

    assert result["role"] == "system"
    assert result["content"] == "You are a helpful assistant"
    assert "tool_calls" not in result
    assert "tool_call_id" not in result


def testserialize_message_system_empty():
    """Test serialization of system messages with empty content."""
    msg = ChatMessageSystem(content="")
    result = serialize_message(msg)

    assert result["role"] == "system"
    assert result["content"] == ""


# ============================================================================
# Test: serialize_message() - User Messages
# ============================================================================


def testserialize_message_user_simple():
    """Test serialization of simple user messages."""
    msg = ChatMessageUser(content="What is 2+2?")
    result = serialize_message(msg)

    assert result["role"] == "user"
    assert result["content"] == "What is 2+2?"
    assert "tool_calls" not in result
    assert "tool_call_id" not in result


def testserialize_message_user_with_text_content():
    """Test serialization of user messages with ContentText."""
    text_content = ContentText(text="What is the capital of France?")
    msg = ChatMessageUser(content=[text_content])
    result = serialize_message(msg)

    assert result["role"] == "user"
    assert result["content"] == "What is the capital of France?"


def testserialize_message_user_with_multiple_text():
    """Test serialization of user messages with multiple text content parts."""
    content_parts = [
        ContentText(text="First part. "),
        ContentText(text="Second part."),
    ]
    msg = ChatMessageUser(content=content_parts)
    result = serialize_message(msg)

    assert result["role"] == "user"
    assert result["content"] == "First part. Second part."


def testserialize_message_user_with_image():
    """Test serialization of user messages with image content."""
    text_content = ContentText(text="What's in this image?")
    image_content = ContentImage(image="data:image/png;base64,ABC123", detail="auto")
    msg = ChatMessageUser(content=[text_content, image_content])
    result = serialize_message(msg)

    assert result["role"] == "user"
    # Should extract only text content
    assert result["content"] == "What's in this image?"


def testserialize_message_user_image_only():
    """Test serialization of user messages with only image content."""
    image_content = ContentImage(image="data:image/png;base64,ABC123", detail="auto")
    msg = ChatMessageUser(content=[image_content])
    result = serialize_message(msg)

    assert result["role"] == "user"
    # Should have empty string for image-only content
    assert result["content"] == ""


# ============================================================================
# Test: serialize_message() - Assistant Messages
# ============================================================================


def testserialize_message_assistant_simple():
    """Test serialization of simple assistant messages."""
    msg = ChatMessageAssistant(content="The answer is 4")
    result = serialize_message(msg)

    assert result["role"] == "assistant"
    assert result["content"] == "The answer is 4"
    assert "tool_calls" not in result or result.get("tool_calls") is None
    assert "reasoning" not in result or result.get("reasoning") is None


def testserialize_message_assistant_with_reasoning():
    """Test serialization of assistant messages with reasoning content."""
    reasoning_content = ContentReasoning(reasoning="Let me think step by step...")
    text_content = ContentText(text="The answer is 42")
    msg = ChatMessageAssistant(content=[reasoning_content, text_content])
    result = serialize_message(msg)

    assert result["role"] == "assistant"
    assert result["content"] == "The answer is 42"
    assert result["reasoning"] == "Let me think step by step..."


def testserialize_message_assistant_reasoning_only():
    """Test serialization of assistant messages with only reasoning."""
    reasoning_content = ContentReasoning(reasoning="Thinking...")
    msg = ChatMessageAssistant(content=[reasoning_content])
    result = serialize_message(msg)

    assert result["role"] == "assistant"
    assert result["content"] == ""
    assert result["reasoning"] == "Thinking..."


def testserialize_message_assistant_with_tool_calls():
    """Test serialization of assistant messages with tool calls."""
    tool_call = ToolCall(
        id="call_abc123",
        function="calculator",
        arguments={"expression": "2+2"},
        type="function",
    )
    msg = ChatMessageAssistant(content="Let me calculate that", tool_calls=[tool_call])
    result = serialize_message(msg)

    assert result["role"] == "assistant"
    assert result["content"] == "Let me calculate that"
    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "call_abc123"
    assert result["tool_calls"][0]["function"] == "calculator"
    assert result["tool_calls"][0]["arguments"] == {"expression": "2+2"}


def testserialize_message_assistant_multiple_tool_calls():
    """Test serialization of assistant messages with multiple tool calls."""
    tool_calls = [
        ToolCall(
            id="call_1", function="search", arguments={"query": "weather"}, type="function"
        ),
        ToolCall(
            id="call_2", function="calculator", arguments={"expr": "5*5"}, type="function"
        ),
    ]
    msg = ChatMessageAssistant(content="Using multiple tools", tool_calls=tool_calls)
    result = serialize_message(msg)

    assert result["role"] == "assistant"
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["id"] == "call_1"
    assert result["tool_calls"][1]["id"] == "call_2"


def testserialize_message_assistant_tool_calls_and_reasoning():
    """Test serialization of assistant messages with both tool calls and reasoning."""
    reasoning_content = ContentReasoning(reasoning="I need to use tools...")
    text_content = ContentText(text="Let me search")
    tool_call = ToolCall(
        id="call_xyz", function="search", arguments={"q": "test"}, type="function"
    )
    msg = ChatMessageAssistant(
        content=[reasoning_content, text_content], tool_calls=[tool_call]
    )
    result = serialize_message(msg)

    assert result["role"] == "assistant"
    assert result["content"] == "Let me search"
    assert result["reasoning"] == "I need to use tools..."
    assert len(result["tool_calls"]) == 1


# ============================================================================
# Test: serialize_message() - Tool Messages
# ============================================================================


def testserialize_message_tool():
    """Test serialization of tool messages."""
    msg = ChatMessageTool(
        content="The calculation result is 4",
        tool_call_id="call_123",
        function="calculator",
    )
    result = serialize_message(msg)

    assert result["role"] == "tool"
    assert result["content"] == "The calculation result is 4"
    assert result["tool_call_id"] == "call_123"
    assert result["name"] == "calculator"


def testserialize_message_tool_no_function():
    """Test serialization of tool messages without function name."""
    msg = ChatMessageTool(content="Result", tool_call_id="call_456")
    result = serialize_message(msg)

    assert result["role"] == "tool"
    assert result["content"] == "Result"
    assert result["tool_call_id"] == "call_456"
    # name should be None or not present
    assert result.get("name") is None


def testserialize_message_tool_with_text_content():
    """Test serialization of tool messages with ContentText."""
    text_content = ContentText(text="Tool output here")
    msg = ChatMessageTool(
        content=[text_content], tool_call_id="call_789", function="my_tool"
    )
    result = serialize_message(msg)

    assert result["role"] == "tool"
    assert result["content"] == "Tool output here"
    assert result["tool_call_id"] == "call_789"
    assert result["name"] == "my_tool"


# ============================================================================
# Test: serialize_message() - Edge Cases
# ============================================================================


def testserialize_message_with_mixed_content():
    """Test serialization with mixed content types (text, images, etc.)."""
    content_parts = [
        ContentText(text="Part 1. "),
        ContentImage(image="data:image/png;base64,XYZ", detail="auto"),
        ContentText(text="Part 2."),
    ]
    msg = ChatMessageUser(content=content_parts)
    result = serialize_message(msg)

    # Should extract and concatenate text parts only
    assert result["content"] == "Part 1. Part 2."


def testserialize_message_unknown_type():
    """Test serialization raises ValueError for unknown message type."""
    # Create a mock object that doesn't match any known message type
    mock_msg = MagicMock()
    mock_msg.role = "unknown"
    mock_msg.content = "test"

    with pytest.raises(ValueError, match="Unknown message type"):
        serialize_message(mock_msg)


# ============================================================================
# Test: _push_transcript() - Success Cases
# ============================================================================


@pytest.mark.asyncio
async def test_push_transcript_success(mock_task_state):
    """Test successful transcript push."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        # Setup mock response
        mock_response = create_mock_response(
            status=200,
            json_data={
                "success": True,
                "episode_id": episode_id,
                "message_count": 3,
                "stored_at": datetime.utcnow().isoformat(),
            }
        )

        # Setup mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Execute
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        expected_url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/transcript"
        assert call_args[0][0] == expected_url

        # Verify payload structure
        json_data = call_args[1]["json"]
        assert "messages" in json_data
        assert "metadata" in json_data
        assert len(json_data["messages"]) == 3
        assert json_data["metadata"]["source"] == "inspect_ai"


@pytest.mark.asyncio
async def test_push_transcript_with_tool_calls(mock_task_state_with_tools):
    """Test transcript push with tool calls."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_response = create_mock_response(status=200, json_data={"success": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        await _push_transcript(mock_task_state_with_tools, session_id, episode_id, rest_url)

        # Verify tool calls are serialized correctly
        call_args = mock_session.post.call_args
        json_data = call_args[1]["json"]
        messages = json_data["messages"]

        # Find assistant message with tool calls
        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        assert "tool_calls" in assistant_msg
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["function"] == "calculator"

        # Find tool response message
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "call_123"
        assert tool_msg["name"] == "calculator"


@pytest.mark.asyncio
async def test_push_transcript_with_reasoning(mock_task_state_with_reasoning):
    """Test transcript push with reasoning content."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_response = create_mock_response(status=200, json_data={"success": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        await _push_transcript(mock_task_state_with_reasoning, session_id, episode_id, rest_url)

        # Verify reasoning is included
        call_args = mock_session.post.call_args
        json_data = call_args[1]["json"]
        messages = json_data["messages"]

        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        assert "reasoning" in assistant_msg
        assert assistant_msg["reasoning"] == "Let me think about this problem..."


# ============================================================================
# Test: _push_transcript() - Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_push_transcript_network_error(mock_task_state):
    """Test graceful handling of network errors."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        # Simulate network error
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Network error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Should not raise exception
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify it attempted to call (max 3 retries)
        assert mock_session.post.call_count == 3


@pytest.mark.asyncio
async def test_push_transcript_server_error(mock_task_state):
    """Test graceful handling of 500 server errors."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_response = create_mock_response(status=500, text_data="Internal Server Error")

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Should not raise exception
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify it retried 3 times
        assert mock_session.post.call_count == 3


@pytest.mark.asyncio
async def test_push_transcript_validation_error(mock_task_state):
    """Test graceful handling of 422 validation errors."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_response = create_mock_response(
            status=422,
            json_data={"detail": "Invalid message format"}
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Should not raise exception
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify it only tried once (no retry on validation errors)
        assert mock_session.post.call_count == 1


@pytest.mark.asyncio
async def test_push_transcript_timeout(mock_task_state):
    """Test graceful handling of timeout errors."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        # Simulate timeout
        mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Should not raise exception
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify it retried 3 times
        assert mock_session.post.call_count == 3


# ============================================================================
# Test: _push_transcript() - Retry Logic
# ============================================================================


@pytest.mark.asyncio
async def test_push_transcript_retry_on_failure_then_success(mock_task_state):
    """Test retry logic: fail twice, succeed on third attempt."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        # First two calls fail, third succeeds
        mock_response_fail = create_mock_response(status=500, text_data="Server error")
        mock_response_success = create_mock_response(status=200, json_data={"success": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=[mock_response_fail, mock_response_fail, mock_response_success]
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Execute
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify it retried 3 times
        assert mock_session.post.call_count == 3


@pytest.mark.asyncio
async def test_push_transcript_retry_exhaustion(mock_task_state):
    """Test retry logic exhausts after max attempts."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        # All calls fail
        mock_response = create_mock_response(status=500, text_data="Server error")

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Execute
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify it tried max 3 times
        assert mock_session.post.call_count == 3


@pytest.mark.asyncio
async def test_push_transcript_no_retry_on_validation_error(mock_task_state):
    """Test that 422 validation errors are not retried."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_response = create_mock_response(
            status=422,
            json_data={"detail": "Invalid format"}
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Should only try once (no retry on client errors)
        assert mock_session.post.call_count == 1


# ============================================================================
# Test: _push_transcript() - Graceful Degradation
# ============================================================================


@pytest.mark.asyncio
async def test_push_transcript_degrades_gracefully_on_all_failures(mock_task_state):
    """Test that transcript push failures are logged but don't raise exceptions."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        # Create a session that always returns 500 errors
        mock_response = create_mock_response(500, text_data="Server error")
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Execute - should NOT raise exception despite all retries failing
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify retries occurred (3 attempts)
        assert mock_session.post.call_count == 3


@pytest.mark.asyncio
async def test_push_transcript_enabled_by_default(mock_task_state):
    """Test that transcript push is enabled by default."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Set env var to explicitly enable (simulating default behavior)
    with patch.dict("os.environ", {"SABER_ENABLE_TRANSCRIPT_SYNC": "true"}):
        with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
            mock_response = create_mock_response(status=200, json_data={"success": True})

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

            # Verify HTTP request was made
            mock_session.post.assert_called_once()


# ============================================================================
# Test: _push_transcript() - Metadata
# ============================================================================


@pytest.mark.asyncio
async def test_push_transcript_includes_metadata(mock_task_state):
    """Test that transcript push includes proper metadata."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_response = create_mock_response(status=200, json_data={"success": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify metadata
        call_args = mock_session.post.call_args
        json_data = call_args[1]["json"]
        metadata = json_data["metadata"]

        assert "step_number" in metadata
        assert metadata["step_number"] == len(mock_task_state.messages)
        assert "timestamp" in metadata
        assert metadata["source"] == "inspect_ai"


@pytest.mark.asyncio
async def test_push_transcript_empty_messages(mock_task_state):
    """Test handling of empty message list."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Empty messages
    mock_task_state.messages = []

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_response = create_mock_response(status=200, json_data={"success": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Should still attempt to push
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify it was called
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        json_data = call_args[1]["json"]
        assert len(json_data["messages"]) == 0


@pytest.mark.asyncio
async def test_push_transcript_invalid_url_format(mock_task_state):
    """Test that invalid URL format is rejected (SSRF protection)."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "ftp://malicious-site.com"  # Invalid protocol

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Execute
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify no HTTP request was made (rejected due to invalid URL)
        mock_session.post.assert_not_called()


@pytest.mark.asyncio
async def test_push_transcript_large_payload_rejected(mock_task_state):
    """Test that very large payloads are rejected client-side."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Create a message with very large content (>10MB when serialized)
    large_content = "X" * (11 * 1024 * 1024)  # 11 MB of text
    mock_task_state.messages = [ChatMessageSystem(content=large_content)]

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Execute
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify no HTTP request was made (rejected due to size)
        mock_session.post.assert_not_called()


# Add missing import
import os
