"""
Unit tests for differential transcript synchronization.

These tests validate the new differential sync implementation that tracks
message indices and only pushes new messages in append mode to reduce
network traffic and improve performance.

Critical Test Coverage:
1. Delta tracking with state.store
2. _push_transcript_delta() function
3. create_transcript_syncing_generate() wrapper with delta calculation
4. Append mode vs replace mode
5. Index tracking across multiple iterations
6. Error handling with graceful degradation
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.solver import Generate, TaskState, Store
from inspect_ai.tool import ToolCall

from saber.inspect_ai.integration.transcript_sync import (
    _push_transcript,
    _push_transcript_delta,
    create_transcript_syncing_generate,
)


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
    """Create a mock TaskState with sample messages and store."""
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageSystem(content="You are a helpful assistant"),
        ChatMessageUser(content="What is 2+2?"),
    ]
    state.store = Store()
    return state


@pytest.fixture
def mock_task_state_with_history():
    """Create a mock TaskState with conversation history."""
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageSystem(content="You are a helpful assistant"),
        ChatMessageUser(content="What is 2+2?"),
        ChatMessageAssistant(content="The answer is 4"),
        ChatMessageUser(content="What is 3+3?"),
    ]
    state.store = Store()
    state.store.set("_last_transcript_index", 3)  # Already pushed first 3 messages
    return state


# ============================================================================
# Test: _push_transcript_delta() - Basic Functionality
# ============================================================================


@pytest.mark.asyncio
async def test_push_transcript_delta_success():
    """Test successful delta push with append mode."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Create new messages to push
    new_messages = [
        ChatMessageAssistant(content="The answer is 6"),
    ]

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        # Setup mock response
        mock_response = create_mock_response(
            status=200,
            json_data={
                "success": True,
                "episode_id": episode_id,
                "message_count": 1,
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
        await _push_transcript_delta(
            messages=new_messages,
            session_id=session_id,
            episode_id=episode_id,
            rest_url=rest_url,
        )

        # Verify
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        expected_url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/transcript"
        assert call_args[0][0] == expected_url

        # Verify payload structure
        json_data = call_args[1]["json"]
        assert "messages" in json_data
        assert "mode" in json_data
        assert "metadata" in json_data
        assert json_data["mode"] == "append"  # Critical: must be append mode
        assert len(json_data["messages"]) == 1
        assert json_data["messages"][0]["role"] == "assistant"
        assert json_data["messages"][0]["content"] == "The answer is 6"


@pytest.mark.asyncio
async def test_push_transcript_delta_multiple_messages():
    """Test delta push with multiple new messages."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Create multiple new messages (tool call + response)
    tool_call = ToolCall(
        id="call_123",
        function="calculator",
        arguments={"expression": "7+7"},
        type="function",
    )
    new_messages = [
        ChatMessageAssistant(content="Let me calculate", tool_calls=[tool_call]),
        ChatMessageTool(content="14", tool_call_id="call_123", function="calculator"),
        ChatMessageAssistant(content="The answer is 14"),
    ]

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_response = create_mock_response(status=200, json_data={"success": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        await _push_transcript_delta(
            messages=new_messages,
            session_id=session_id,
            episode_id=episode_id,
            rest_url=rest_url,
        )

        # Verify all messages pushed
        call_args = mock_session.post.call_args
        json_data = call_args[1]["json"]
        assert json_data["mode"] == "append"
        assert len(json_data["messages"]) == 3
        assert json_data["messages"][0]["role"] == "assistant"
        assert json_data["messages"][1]["role"] == "tool"
        assert json_data["messages"][2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_push_transcript_delta_empty_list():
    """Test delta push with empty message list (no-op)."""
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

        # Execute with empty list
        await _push_transcript_delta(
            messages=[],
            session_id=session_id,
            episode_id=episode_id,
            rest_url=rest_url,
        )

        # Verify request was still made (server handles empty list)
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        json_data = call_args[1]["json"]
        assert len(json_data["messages"]) == 0


# ============================================================================
# Test: _push_transcript_delta() - Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_push_transcript_delta_network_error():
    """Test delta push gracefully handles network errors."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    new_messages = [ChatMessageAssistant(content="Test")]

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Network error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Should not raise exception
        await _push_transcript_delta(
            messages=new_messages,
            session_id=session_id,
            episode_id=episode_id,
            rest_url=rest_url,
        )

        # Verify it retried 3 times
        assert mock_session.post.call_count == 3


@pytest.mark.asyncio
async def test_push_transcript_delta_invalid_url():
    """Test delta push rejects invalid URL format (SSRF protection)."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "ftp://malicious-site.com"  # Invalid protocol

    new_messages = [ChatMessageAssistant(content="Test")]

    with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Execute
        await _push_transcript_delta(
            messages=new_messages,
            session_id=session_id,
            episode_id=episode_id,
            rest_url=rest_url,
        )

        # Verify no HTTP request was made
        mock_session.post.assert_not_called()


# ============================================================================
# Test: create_transcript_syncing_generate() - Delta Tracking
# ============================================================================


@pytest.mark.asyncio
async def test_wrapped_generate_first_call_pushes_all_messages():
    """Test first call to wrapped generate pushes all messages."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Create original generate that returns state with new messages
    async def original_generate(state: TaskState, **kwargs) -> TaskState:
        # Simulate model adding assistant message
        state.messages.append(
            ChatMessageAssistant(content="The answer is 4", tool_calls=[])
        )
        return state

    # Create initial state with system + user messages
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageSystem(content="You are a helper"),
        ChatMessageUser(content="What is 2+2?"),
    ]
    state.store = Store()  # Fresh store, no previous index

    # Create wrapped generate
    wrapped_generate = create_transcript_syncing_generate(
        original_generate=original_generate,
        session_id=session_id,
        episode_id=episode_id,
        rest_url=rest_url,
    )

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push_delta:
        mock_push_delta.return_value = AsyncMock()

        # Execute
        result = await wrapped_generate(state)

        # Verify delta push was called
        mock_push_delta.assert_called_once()
        call_args = mock_push_delta.call_args

        # Verify all 3 messages pushed (initial state had 2, model added 1)
        pushed_messages = call_args[1]["messages"]
        assert len(pushed_messages) == 3

        # Verify index was updated
        assert result.store.get("_last_transcript_index") == 3


@pytest.mark.asyncio
async def test_wrapped_generate_subsequent_call_pushes_only_new():
    """Test subsequent calls only push new messages since last index."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Create original generate that adds one message
    async def original_generate(state: TaskState, **kwargs) -> TaskState:
        state.messages.append(ChatMessageAssistant(content="New response"))
        return state

    # Create state with history
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageSystem(content="System"),
        ChatMessageUser(content="User 1"),
        ChatMessageAssistant(content="Assistant 1"),
        ChatMessageUser(content="User 2"),
    ]
    state.store = Store()
    state.store.set("_last_transcript_index", 4)  # Already pushed first 4 messages

    # Create wrapped generate
    wrapped_generate = create_transcript_syncing_generate(
        original_generate=original_generate,
        session_id=session_id,
        episode_id=episode_id,
        rest_url=rest_url,
    )

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push_delta:
        mock_push_delta.return_value = AsyncMock()

        # Execute
        result = await wrapped_generate(state)

        # Verify delta push was called
        mock_push_delta.assert_called_once()
        call_args = mock_push_delta.call_args

        # Verify only 1 new message pushed
        pushed_messages = call_args[1]["messages"]
        assert len(pushed_messages) == 1
        assert pushed_messages[0].content == "New response"

        # Verify index was updated to 5
        assert result.store.get("_last_transcript_index") == 5


@pytest.mark.asyncio
async def test_wrapped_generate_no_new_messages_skips_push():
    """Test wrapped generate skips push when no new messages."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Create original generate that doesn't add messages
    async def original_generate(state: TaskState, **kwargs) -> TaskState:
        # No new messages added
        return state

    # Create state
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageSystem(content="System"),
        ChatMessageUser(content="User"),
    ]
    state.store = Store()
    state.store.set("_last_transcript_index", 2)  # Already pushed both messages

    # Create wrapped generate
    wrapped_generate = create_transcript_syncing_generate(
        original_generate=original_generate,
        session_id=session_id,
        episode_id=episode_id,
        rest_url=rest_url,
    )

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push_delta:
        mock_push_delta.return_value = AsyncMock()

        # Execute
        result = await wrapped_generate(state)

        # Verify NO delta push was called
        mock_push_delta.assert_not_called()

        # Index should remain at 2
        assert result.store.get("_last_transcript_index") == 2


@pytest.mark.asyncio
async def test_wrapped_generate_multiple_iterations():
    """Test wrapped generate tracks deltas correctly across multiple iterations."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    iteration_counter = [0]  # Mutable counter

    async def original_generate(state: TaskState, **kwargs) -> TaskState:
        iteration_counter[0] += 1
        # Add different messages per iteration
        if iteration_counter[0] == 1:
            state.messages.append(ChatMessageAssistant(content="Response 1"))
        elif iteration_counter[0] == 2:
            state.messages.append(ChatMessageTool(content="Tool output", tool_call_id="call_1", function="tool"))
            state.messages.append(ChatMessageAssistant(content="Response 2"))
        return state

    # Create initial state
    state = MagicMock(spec=TaskState)
    state.messages = [ChatMessageSystem(content="System")]
    state.store = Store()

    # Create wrapped generate
    wrapped_generate = create_transcript_syncing_generate(
        original_generate=original_generate,
        session_id=session_id,
        episode_id=episode_id,
        rest_url=rest_url,
    )

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push_delta:
        mock_push_delta.return_value = AsyncMock()

        # Iteration 1
        state = await wrapped_generate(state)
        assert mock_push_delta.call_count == 1
        call1_messages = mock_push_delta.call_args[1]["messages"]
        # First push includes all messages: System + Assistant (2 messages)
        assert len(call1_messages) == 2
        assert len(state.messages) == 2
        assert state.store.get("_last_transcript_index") == 2

        # Iteration 2
        state = await wrapped_generate(state)
        assert mock_push_delta.call_count == 2
        call2_messages = mock_push_delta.call_args[1]["messages"]
        assert len(call2_messages) == 2  # Tool + Assistant messages
        assert state.store.get("_last_transcript_index") == 4


@pytest.mark.asyncio
async def test_wrapped_generate_error_doesnt_update_index():
    """Test that push errors don't update the index (will retry next time)."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    async def original_generate(state: TaskState, **kwargs) -> TaskState:
        state.messages.append(ChatMessageAssistant(content="New message"))
        return state

    state = MagicMock(spec=TaskState)
    state.messages = [ChatMessageSystem(content="System")]
    state.store = Store()

    wrapped_generate = create_transcript_syncing_generate(
        original_generate=original_generate,
        session_id=session_id,
        episode_id=episode_id,
        rest_url=rest_url,
    )

    # Mock _push_transcript_delta to raise exception
    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push_delta:
        mock_push_delta.side_effect = Exception("Network error")

        # Execute (should not raise)
        result = await wrapped_generate(state)

        # Verify push was attempted
        mock_push_delta.assert_called_once()

        # Index is updated AFTER push call in try block, so on error exception is caught
        # but index update line is never reached (it's after the await push)
        # Therefore index should NOT be updated on error (stays at 0)
        # This means failed pushes will be retried on next iteration
        assert result.store.get("_last_transcript_index", 0) == 0


# ============================================================================
# Test: _push_transcript() with mode parameter
# ============================================================================


@pytest.mark.asyncio
async def test_push_transcript_with_replace_mode(mock_task_state):
    """Test _push_transcript with replace mode (default)."""
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

        # Execute with default mode (replace)
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url)

        # Verify mode is "replace"
        call_args = mock_session.post.call_args
        json_data = call_args[1]["json"]
        assert json_data["mode"] == "replace"


@pytest.mark.asyncio
async def test_push_transcript_with_append_mode(mock_task_state):
    """Test _push_transcript with explicit append mode."""
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

        # Execute with append mode
        await _push_transcript(mock_task_state, session_id, episode_id, rest_url, mode="append")

        # Verify mode is "append"
        call_args = mock_session.post.call_args
        json_data = call_args[1]["json"]
        assert json_data["mode"] == "append"


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


@pytest.mark.asyncio
async def test_full_agent_loop_differential_sync():
    """Test complete agent loop with differential sync across multiple iterations."""
    session_id = "session_123"
    episode_id = "episode_456"
    rest_url = "http://localhost:8000"

    # Simulate agent loop: 3 iterations
    iteration_responses = [
        "Let me search for that",
        "Let me analyze the results",
        "Here is the final answer"
    ]

    iteration_counter = [0]

    async def original_generate(state: TaskState, **kwargs) -> TaskState:
        # Add assistant message for this iteration
        response = iteration_responses[iteration_counter[0]]
        state.messages.append(ChatMessageAssistant(content=response))
        iteration_counter[0] += 1
        return state

    # Initial state
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageSystem(content="You are a helpful assistant"),
        ChatMessageUser(content="Search for information about Python"),
    ]
    state.store = Store()

    # Create wrapped generate
    wrapped_generate = create_transcript_syncing_generate(
        original_generate=original_generate,
        session_id=session_id,
        episode_id=episode_id,
        rest_url=rest_url,
    )

    pushed_message_counts = []

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push_delta:
        mock_push_delta.return_value = AsyncMock()

        # Iteration 1
        state = await wrapped_generate(state)
        if mock_push_delta.call_count > 0:
            messages = mock_push_delta.call_args[1]["messages"]
            pushed_message_counts.append(len(messages))

        # Iteration 2
        state = await wrapped_generate(state)
        if mock_push_delta.call_count > 1:
            messages = mock_push_delta.call_args[1]["messages"]
            pushed_message_counts.append(len(messages))

        # Iteration 3
        state = await wrapped_generate(state)
        if mock_push_delta.call_count > 2:
            messages = mock_push_delta.call_args[1]["messages"]
            pushed_message_counts.append(len(messages))

    # Verify differential behavior
    # First call: pushes system + user + assistant (3 messages)
    # Second call: pushes only new assistant (1 message)
    # Third call: pushes only new assistant (1 message)
    assert len(pushed_message_counts) == 3
    assert pushed_message_counts[0] == 3  # Initial push
    assert pushed_message_counts[1] == 1  # Delta
    assert pushed_message_counts[2] == 1  # Delta

    # Verify total message count
    assert len(state.messages) == 5  # system + user + 3 assistants
    assert state.store.get("_last_transcript_index") == 5


@pytest.mark.asyncio
async def test_wrapped_generate_missing_context_skips_push():
    """Test wrapped generate gracefully handles missing session/episode context."""
    # Missing episode_id
    wrapped_generate = create_transcript_syncing_generate(
        original_generate=AsyncMock(return_value=MagicMock(spec=TaskState)),
        session_id="session_123",
        episode_id=None,  # Missing
        rest_url="http://localhost:8000",
    )

    state = MagicMock(spec=TaskState)
    state.messages = []
    state.store = Store()

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push_delta:
        mock_push_delta.return_value = AsyncMock()

        # Execute
        await wrapped_generate(state)

        # Verify no push was attempted
        mock_push_delta.assert_not_called()
