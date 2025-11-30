"""Tests for transcript synchronization module.

This test suite ensures the transcript_sync module properly handles:
- Message serialization for all ChatMessage types
- Transcript pushing with retry logic and error handling
- Generate wrapper for per-iteration transcript syncing
- Payload size validation
- Network error handling and graceful degradation
- Message injection (pull_injected_messages)
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from saber.inspect_ai.integration.transcript_sync import (
    _deserialize_message,
    _push_single_message,
    _push_transcript,
    _push_transcript_delta,
    create_transcript_syncing_generate,
    pull_injected_messages,
    push_transcript_if_enabled,
    serialize_message,
)
from saber.models import MetadataKeys, TranscriptSyncConfig


class TestSerializeMessage:
    """Test serialize_message function for all message types."""

    def test_serialize_system_message(self):
        """Test serialization of ChatMessageSystem."""
        msg = ChatMessageSystem(content="System prompt")
        result = serialize_message(msg)

        assert result["role"] == "system"
        assert result["content"] == "System prompt"

    def test_serialize_user_message(self):
        """Test serialization of ChatMessageUser."""
        msg = ChatMessageUser(content="User question")
        result = serialize_message(msg)

        assert result["role"] == "user"
        assert result["content"] == "User question"

    def test_serialize_assistant_message_simple(self):
        """Test serialization of ChatMessageAssistant without tool calls."""
        msg = ChatMessageAssistant(content="Assistant response")
        result = serialize_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Assistant response"
        assert "tool_calls" not in result

    def test_serialize_assistant_message_with_tool_calls(self):
        """Test serialization of ChatMessageAssistant with tool calls."""
        tool_calls = [
            ToolCall(
                id="call_123",
                function="execute_command",
                arguments={"command": "ls"},
                type="function",
            )
        ]
        msg = ChatMessageAssistant(content="Running command", tool_calls=tool_calls)
        result = serialize_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Running command"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["function"] == "execute_command"
        assert result["tool_calls"][0]["arguments"] == {"command": "ls"}

    def test_serialize_tool_message(self):
        """Test serialization of ChatMessageTool."""
        msg = ChatMessageTool(
            content="Command output",
            tool_call_id="call_123",
            function="execute_command",
        )
        result = serialize_message(msg)

        assert result["role"] == "tool"
        assert result["content"] == "Command output"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "execute_command"

    def test_serialize_message_with_content_list(self):
        """Test serialization with Content list (ContentText and ContentReasoning)."""
        content_list = [
            ContentText(text="First part"),
            ContentText(text=" second part"),
        ]
        msg = ChatMessageUser(content=content_list)
        result = serialize_message(msg)

        assert result["content"] == "First part second part"

    def test_serialize_message_with_reasoning(self):
        """Test serialization with ContentReasoning."""
        content_list = [
            ContentText(text="Response text"),
            ContentReasoning(reasoning="Thought process"),
        ]
        msg = ChatMessageAssistant(content=content_list)
        result = serialize_message(msg)

        assert result["content"] == "Response text"
        assert result["reasoning"] == "Thought process"

    def test_serialize_unknown_message_type(self):
        """Test serialization raises ValueError for unknown message type."""
        # Create a mock message that's not a known type
        unknown_msg = Mock(spec=ChatMessage)
        unknown_msg.content = "test"

        with pytest.raises(ValueError, match="Unknown message type"):
            serialize_message(unknown_msg)


class TestDeserializeMessage:
    """Test _deserialize_message function for message reconstruction."""

    def test_deserialize_system_message(self):
        """Test deserialization of system message."""
        msg_data = {"role": "system", "content": "System prompt"}
        result = _deserialize_message(msg_data)

        assert isinstance(result, ChatMessageSystem)
        assert result.content == "System prompt"

    def test_deserialize_user_message(self):
        """Test deserialization of user message."""
        msg_data = {"role": "user", "content": "User question"}
        result = _deserialize_message(msg_data)

        assert isinstance(result, ChatMessageUser)
        assert result.content == "User question"

    def test_deserialize_assistant_message_simple(self):
        """Test deserialization of assistant message without tool calls."""
        msg_data = {"role": "assistant", "content": "Assistant response"}
        result = _deserialize_message(msg_data)

        assert isinstance(result, ChatMessageAssistant)
        assert result.content == "Assistant response"
        assert result.tool_calls is None

    def test_deserialize_assistant_message_with_tool_calls(self):
        """Test deserialization of assistant message with tool calls."""
        msg_data = {
            "role": "assistant",
            "content": "Running command",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "execute_command",
                    "arguments": {"command": "ls"},
                }
            ],
        }
        result = _deserialize_message(msg_data)

        assert isinstance(result, ChatMessageAssistant)
        assert result.content == "Running command"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].function == "execute_command"
        assert result.tool_calls[0].arguments == {"command": "ls"}

    def test_deserialize_tool_message(self):
        """Test deserialization of tool message."""
        msg_data = {
            "role": "tool",
            "content": "Command output",
            "tool_call_id": "call_123",
            "name": "execute_command",
        }
        result = _deserialize_message(msg_data)

        assert isinstance(result, ChatMessageTool)
        assert result.content == "Command output"
        assert result.tool_call_id == "call_123"
        assert result.function == "execute_command"

    def test_deserialize_tool_message_missing_fields(self):
        """Test deserialization raises ValueError for tool message missing required fields."""
        # Missing tool_call_id
        msg_data = {
            "role": "tool",
            "content": "output",
            "name": "execute_command",
        }
        with pytest.raises(ValueError, match="Tool message missing"):
            _deserialize_message(msg_data)

        # Missing name
        msg_data = {
            "role": "tool",
            "content": "output",
            "tool_call_id": "call_123",
        }
        with pytest.raises(ValueError, match="Tool message missing"):
            _deserialize_message(msg_data)

    def test_deserialize_unknown_role(self):
        """Test deserialization raises ValueError for unknown role."""
        msg_data = {"role": "unknown", "content": "test"}
        with pytest.raises(ValueError, match="Unknown message role"):
            _deserialize_message(msg_data)


class TestPushTranscriptDelta:
    """Test _push_transcript_delta function for differential sync."""

    @pytest.mark.asyncio
    async def test_push_delta_success(self):
        """Test successful transcript delta push."""
        messages = [ChatMessageUser(content="Test message")]
        session_id = "session_123"
        episode_id = "episode_456"
        rest_url = "http://localhost:8000"

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await _push_transcript_delta(
                messages=messages,
                session_id=session_id,
                episode_id=episode_id,
                rest_url=rest_url,
            )

            # Verify POST was called
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert call_args[0][0] == f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/transcript"

            # Verify payload structure
            payload = call_args[1]["json"]
            assert payload["mode"] == "append"
            assert len(payload["messages"]) == 1
            assert payload["messages"][0]["role"] == "user"
            assert payload["messages"][0]["content"] == "Test message"

    @pytest.mark.asyncio
    async def test_push_delta_invalid_url(self):
        """Test that invalid URLs are rejected."""
        messages = [ChatMessageUser(content="Test")]
        invalid_urls = ["ftp://localhost:8000", "javascript:alert(1)", "file:///etc/passwd"]

        for url in invalid_urls:
            await _push_transcript_delta(
                messages=messages,
                session_id="session_123",
                episode_id="episode_456",
                rest_url=url,
            )
            # Should log warning and return early without making request

    @pytest.mark.asyncio
    async def test_push_delta_serialization_error(self):
        """Test handling of serialization errors."""
        # Create a message that will fail serialization
        bad_message = Mock(spec=ChatMessage)
        bad_message.content = "test"

        await _push_transcript_delta(
            messages=[bad_message],
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )
        # Should log warning and return early

    @pytest.mark.asyncio
    async def test_push_delta_payload_too_large(self):
        """Test payload size validation."""
        # Create a message that would exceed max payload size
        huge_content = "x" * (TranscriptSyncConfig.MAX_PAYLOAD_SIZE_BYTES + 1000)
        messages = [ChatMessageUser(content=huge_content)]

        with patch("aiohttp.ClientSession") as mock_session_class:
            await _push_transcript_delta(
                messages=messages,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should not make request due to size check
            mock_session_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_delta_validation_error_422(self):
        """Test handling of 422 validation errors (no retry)."""
        messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 422
            mock_response.json = AsyncMock(return_value={"detail": "Invalid payload"})
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await _push_transcript_delta(
                messages=messages,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should be called only once (no retry for 422)
            assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_push_delta_retry_on_server_error(self):
        """Test retry logic on 500 server errors."""
        messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Server error")
            mock_response.__aenter__.return_value = mock_response

            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session.__aenter__.return_value = mock_session

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await _push_transcript_delta(
                    messages=messages,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                )

                # Should retry (3 attempts total)
                assert mock_session.post.call_count == TranscriptSyncConfig.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_push_delta_timeout_retry(self):
        """Test retry logic on timeout errors."""
        messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.__aenter__.side_effect = asyncio.TimeoutError()

            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session.__aenter__.return_value = mock_session

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await _push_transcript_delta(
                    messages=messages,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                )

                # Should retry on timeout
                assert mock_session.post.call_count == TranscriptSyncConfig.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_push_delta_network_error_retry(self):
        """Test retry logic on network errors."""
        messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.__aenter__.side_effect = aiohttp.ClientError("Connection failed")

            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session.__aenter__.return_value = mock_session

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await _push_transcript_delta(
                    messages=messages,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                )

                # Should retry on network error
                assert mock_session.post.call_count == TranscriptSyncConfig.MAX_RETRIES


class TestPushTranscript:
    """Test _push_transcript function for full transcript sync."""

    @pytest.mark.asyncio
    async def test_push_transcript_success_replace_mode(self):
        """Test successful transcript push in replace mode."""
        state = Mock(spec=TaskState)
        state.messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="User"),
        ]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await _push_transcript(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
                mode="replace",
            )

            # Verify payload
            payload = mock_session.post.call_args[1]["json"]
            assert payload["mode"] == "replace"
            assert len(payload["messages"]) == 2

    @pytest.mark.asyncio
    async def test_push_transcript_append_mode(self):
        """Test transcript push in append mode."""
        state = Mock(spec=TaskState)
        state.messages = [ChatMessageUser(content="User")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await _push_transcript(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
                mode="append",
            )

            # Verify mode
            payload = mock_session.post.call_args[1]["json"]
            assert payload["mode"] == "append"


class TestPushSingleMessage:
    """Test _push_single_message convenience wrapper."""

    @pytest.mark.asyncio
    async def test_push_single_message(self):
        """Test pushing a single message."""
        message = ChatMessageAssistant(content="Assistant response")

        with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push:
            await _push_single_message(
                message=message,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            mock_push.assert_called_once()
            call_args = mock_push.call_args
            assert len(call_args[1]["messages"]) == 1
            assert call_args[1]["messages"][0] == message


class TestCreateTranscriptSyncingGenerate:
    """Test create_transcript_syncing_generate wrapper."""

    @pytest.mark.asyncio
    async def test_wrapper_calls_original_and_pushes(self):
        """Test that wrapper calls original generate and pushes transcript."""
        # Mock original generate
        original_generate = AsyncMock()
        mock_state = Mock(spec=TaskState)
        mock_state.messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="User"),
            ChatMessageAssistant(content="Response"),
        ]
        mock_state.store = Mock()
        mock_state.store.get = Mock(return_value=2)  # Last index was 2
        mock_state.store.set = Mock()
        original_generate.return_value = mock_state

        # Create wrapper
        wrapped = create_transcript_syncing_generate(
            original_generate=original_generate,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

        # Call wrapped function
        with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push:
            result = await wrapped(mock_state)

            # Verify original was called
            original_generate.assert_called_once()

            # Verify transcript was pushed (only new message)
            mock_push.assert_called_once()
            call_args = mock_push.call_args
            assert len(call_args[1]["messages"]) == 1
            assert call_args[1]["messages"][0].content == "Response"

            # Verify index was updated
            mock_state.store.set.assert_called_once_with("_last_transcript_index", 3)

    @pytest.mark.asyncio
    async def test_wrapper_skips_if_no_new_messages(self):
        """Test wrapper skips push if no new messages."""
        original_generate = AsyncMock()
        mock_state = Mock(spec=TaskState)
        mock_state.messages = [ChatMessageSystem(content="System")]
        mock_state.store = Mock()
        mock_state.store.get = Mock(return_value=1)  # Already pushed 1 message
        original_generate.return_value = mock_state

        wrapped = create_transcript_syncing_generate(
            original_generate=original_generate,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

        with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push:
            await wrapped(mock_state)

            # Should not push (no new messages)
            mock_push.assert_not_called()

    @pytest.mark.asyncio
    async def test_wrapper_graceful_degradation_on_error(self):
        """Test wrapper continues on push errors."""
        original_generate = AsyncMock()
        mock_state = Mock(spec=TaskState)
        mock_state.messages = [ChatMessageUser(content="User")]
        mock_state.store = Mock()
        mock_state.store.get = Mock(return_value=0)
        original_generate.return_value = mock_state

        wrapped = create_transcript_syncing_generate(
            original_generate=original_generate,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

        with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push:
            mock_push.side_effect = Exception("Network error")

            # Should not raise, just log and continue
            result = await wrapped(mock_state)
            assert result == mock_state

    @pytest.mark.asyncio
    async def test_wrapper_skips_if_no_context(self):
        """Test wrapper skips push if session/episode context missing."""
        original_generate = AsyncMock()
        mock_state = Mock(spec=TaskState)
        mock_state.messages = [ChatMessageUser(content="User")]
        original_generate.return_value = mock_state

        # Create wrapper with missing context
        wrapped = create_transcript_syncing_generate(
            original_generate=original_generate,
            session_id=None,
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

        with patch("saber.inspect_ai.integration.transcript_sync._push_transcript_delta") as mock_push:
            await wrapped(mock_state)

            # Should not push (missing session_id)
            mock_push.assert_not_called()


class TestPushTranscriptIfEnabled:
    """Test push_transcript_if_enabled helper function."""

    @pytest.mark.asyncio
    async def test_push_if_enabled_with_context(self):
        """Test pushing when all context is available."""
        state = Mock(spec=TaskState)
        state.messages = [ChatMessageUser(content="Test")]

        metadata = {
            MetadataKeys.SESSION_ID: "session_123",
            MetadataKeys.EPISODE_ID: "episode_456",
            MetadataKeys.SABER_DOMAIN_SLUG: "test_domain",
        }

        mock_get_active_domain = Mock(
            return_value={"rest_url": "http://localhost:8000"}
        )

        with patch("saber.inspect_ai.integration.transcript_sync._push_transcript") as mock_push:
            await push_transcript_if_enabled(
                state=state,
                metadata=metadata,
                get_active_domain_func=mock_get_active_domain,
            )

            # Should push in append mode
            mock_push.assert_called_once()
            call_args = mock_push.call_args[1]
            assert call_args["mode"] == "append"

    @pytest.mark.asyncio
    async def test_push_if_enabled_missing_session(self):
        """Test skipping when session_id is missing."""
        state = Mock(spec=TaskState)
        metadata = {
            MetadataKeys.EPISODE_ID: "episode_456",
        }

        with patch("saber.inspect_ai.integration.transcript_sync._push_transcript") as mock_push:
            await push_transcript_if_enabled(
                state=state,
                metadata=metadata,
                get_active_domain_func=Mock(),
            )

            # Should not push
            mock_push.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_if_enabled_missing_episode(self):
        """Test skipping when episode_id is missing."""
        state = Mock(spec=TaskState)
        metadata = {
            MetadataKeys.SESSION_ID: "session_123",
        }

        with patch("saber.inspect_ai.integration.transcript_sync._push_transcript") as mock_push:
            await push_transcript_if_enabled(
                state=state,
                metadata=metadata,
                get_active_domain_func=Mock(),
            )

            # Should not push
            mock_push.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_if_enabled_graceful_error_handling(self):
        """Test graceful error handling."""
        state = Mock(spec=TaskState)
        metadata = {
            MetadataKeys.SESSION_ID: "session_123",
            MetadataKeys.EPISODE_ID: "episode_456",
            MetadataKeys.SABER_DOMAIN_SLUG: "test_domain",
        }

        mock_get_active_domain = Mock(side_effect=Exception("Domain lookup failed"))

        # Should not raise
        await push_transcript_if_enabled(
            state=state,
            metadata=metadata,
            get_active_domain_func=mock_get_active_domain,
        )


class TestPullInjectedMessages:
    """Test pull_injected_messages for red team message injection."""

    @pytest.mark.asyncio
    async def test_pull_injected_messages_success(self):
        """Test successful message injection."""
        state = Mock(spec=TaskState)
        state.messages = []

        injected_data = {
            "messages": [
                {"role": "user", "content": "Injected message 1"},
                {"role": "assistant", "content": "Injected response"},
            ]
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=injected_data)
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await pull_injected_messages(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Verify messages were appended
            assert len(state.messages) == 2
            assert isinstance(state.messages[0], ChatMessageUser)
            assert state.messages[0].content == "Injected message 1"
            assert isinstance(state.messages[1], ChatMessageAssistant)
            assert state.messages[1].content == "Injected response"

    @pytest.mark.asyncio
    async def test_pull_injected_messages_404(self):
        """Test handling of 404 (episode not found)."""
        state = Mock(spec=TaskState)
        state.messages = []

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.__aenter__.return_value = mock_response

            mock_session = AsyncMock()
            mock_session.get.return_value = mock_response
            mock_session.__aenter__.return_value = mock_session

            mock_session_class.return_value = mock_session

            await pull_injected_messages(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should not modify state
            assert len(state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_invalid_url(self):
        """Test rejection of invalid URLs."""
        state = Mock(spec=TaskState)
        state.messages = []

        await pull_injected_messages(
            state=state,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="ftp://localhost:8000",
        )

        # Should not modify state
        assert len(state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_timeout(self):
        """Test handling of timeout errors."""
        state = Mock(spec=TaskState)
        state.messages = []

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.__aenter__.side_effect = asyncio.TimeoutError()

            mock_session = AsyncMock()
            mock_session.get.return_value = mock_response
            mock_session.__aenter__.return_value = mock_session

            mock_session_class.return_value = mock_session

            await pull_injected_messages(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should not crash, just log
            assert len(state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_deserialization_error(self):
        """Test handling of deserialization errors for individual messages."""
        state = Mock(spec=TaskState)
        state.messages = []

        injected_data = {
            "messages": [
                {"role": "user", "content": "Good message"},
                {"role": "invalid_role", "content": "Bad message"},
                {"role": "assistant", "content": "Another good message"},
            ]
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=injected_data)
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await pull_injected_messages(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should inject only valid messages
            assert len(state.messages) == 2
            assert state.messages[0].content == "Good message"
            assert state.messages[1].content == "Another good message"


class TestAdditionalEdgeCases:
    """Additional tests to reach 90% coverage."""

    def test_serialize_message_empty_content_list(self):
        """Test serialization with empty content list."""
        msg = ChatMessageUser(content=[])
        result = serialize_message(msg)

        assert result["role"] == "user"
        assert result["content"] == ""

    def test_serialize_message_content_reasoning_only(self):
        """Test serialization with only ContentReasoning (no text)."""
        content_list = [
            ContentReasoning(reasoning="Just thinking"),
        ]
        msg = ChatMessageAssistant(content=content_list)
        result = serialize_message(msg)

        assert result["content"] == ""
        assert result["reasoning"] == "Just thinking"

    @pytest.mark.asyncio
    async def test_push_delta_retry_with_sleep(self):
        """Test that retry delays are actually applied."""
        messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            # First two attempts fail with 500, third succeeds
            mock_responses = []
            for i in range(2):
                mock_resp = AsyncMock()
                mock_resp.status = 500
                mock_resp.text = AsyncMock(return_value="Server error")
                mock_resp.__aenter__.return_value = mock_resp
                mock_resp.__aexit__.return_value = AsyncMock()
                mock_responses.append(mock_resp)

            # Third attempt succeeds
            success_resp = AsyncMock()
            success_resp.status = 200
            success_resp.__aenter__.return_value = success_resp
            success_resp.__aexit__.return_value = AsyncMock()
            mock_responses.append(success_resp)

            mock_session = AsyncMock()
            mock_session.post = MagicMock(side_effect=mock_responses)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await _push_transcript_delta(
                    messages=messages,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                )

                # Should have slept between retries (2 times for 2 failures)
                assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_push_delta_all_retries_exhausted(self):
        """Test logging when all retries are exhausted."""
        messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Server error")
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await _push_transcript_delta(
                    messages=messages,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                )

                # All retries should be exhausted
                assert mock_session.post.call_count == TranscriptSyncConfig.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_push_transcript_retry_with_sleep(self):
        """Test that _push_transcript also retries with delays."""
        state = Mock(spec=TaskState)
        state.messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            # First attempt fails, second succeeds
            fail_resp = AsyncMock()
            fail_resp.status = 503
            fail_resp.text = AsyncMock(return_value="Service unavailable")
            fail_resp.__aenter__.return_value = fail_resp
            fail_resp.__aexit__.return_value = AsyncMock()

            success_resp = AsyncMock()
            success_resp.status = 200
            success_resp.__aenter__.return_value = success_resp
            success_resp.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(side_effect=[fail_resp, success_resp])
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await _push_transcript(
                    state=state,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                    mode="replace",
                )

                # Should have slept once after first failure
                assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_push_transcript_timeout_error(self):
        """Test _push_transcript handling of timeout errors."""
        state = Mock(spec=TaskState)
        state.messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.__aenter__.side_effect = asyncio.TimeoutError()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await _push_transcript(
                    state=state,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                )

                # Should retry on timeout
                assert mock_session.post.call_count == TranscriptSyncConfig.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_push_transcript_network_error(self):
        """Test _push_transcript handling of network errors."""
        state = Mock(spec=TaskState)
        state.messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.__aenter__.side_effect = aiohttp.ClientError("Connection reset")

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await _push_transcript(
                    state=state,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                )

                # Should retry on network error
                assert mock_session.post.call_count == TranscriptSyncConfig.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_push_transcript_generic_exception(self):
        """Test _push_transcript handling of unexpected exceptions."""
        state = Mock(spec=TaskState)
        state.messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.__aenter__.side_effect = RuntimeError("Unexpected error")

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await _push_transcript(
                    state=state,
                    session_id="session_123",
                    episode_id="episode_456",
                    rest_url="http://localhost:8000",
                )

                # Should retry on unexpected errors
                assert mock_session.post.call_count == TranscriptSyncConfig.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_push_transcript_validation_error_no_retry(self):
        """Test that _push_transcript doesn't retry on 422."""
        state = Mock(spec=TaskState)
        state.messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 422
            mock_response.json = AsyncMock(return_value={"detail": "Invalid"})
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await _push_transcript(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should not retry for 422
            assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_push_transcript_invalid_url(self):
        """Test _push_transcript rejects invalid URLs."""
        state = Mock(spec=TaskState)
        state.messages = [ChatMessageUser(content="Test")]

        with patch("aiohttp.ClientSession") as mock_session_class:
            await _push_transcript(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="javascript:alert(1)",
            )

            # Should not make any requests
            mock_session_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_transcript_payload_too_large(self):
        """Test _push_transcript skips when payload is too large."""
        state = Mock(spec=TaskState)
        huge_content = "x" * (TranscriptSyncConfig.MAX_PAYLOAD_SIZE_BYTES + 1000)
        state.messages = [ChatMessageUser(content=huge_content)]

        with patch("aiohttp.ClientSession") as mock_session_class:
            await _push_transcript(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should not make request due to size check
            mock_session_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_transcript_serialization_error(self):
        """Test _push_transcript handles serialization errors."""
        state = Mock(spec=TaskState)
        bad_message = Mock(spec=ChatMessage)
        bad_message.content = "test"
        state.messages = [bad_message]

        with patch("aiohttp.ClientSession") as mock_session_class:
            await _push_transcript(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should not make request due to serialization failure
            mock_session_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_pull_injected_messages_404_not_found(self):
        """Test pull_injected_messages handles 404 gracefully."""
        state = Mock(spec=TaskState)
        state.messages = []

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await pull_injected_messages(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should not crash, just log
            assert len(state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_server_error(self):
        """Test pull_injected_messages handles non-200/404 status."""
        state = Mock(spec=TaskState)
        state.messages = []

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = AsyncMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = AsyncMock()

            mock_session_class.return_value = mock_session

            await pull_injected_messages(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Should not crash
            assert len(state.messages) == 0
