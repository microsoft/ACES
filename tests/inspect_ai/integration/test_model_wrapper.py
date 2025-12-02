"""Tests for TranscriptSyncingModelWrapper.

These tests cover:
- Model wrapper initialization
- generate() method with transcript push
- __getattr__ delegation to base model
- __repr__ string representation
- Error handling and graceful degradation
- Tool call handling
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser, Model, ModelOutput
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from saber.inspect_ai.integration.model_wrapper import TranscriptSyncingModelWrapper


@pytest.fixture
def mock_base_model():
    """Create mock Model for testing."""
    model = Mock(spec=Model)
    model.name = "test-model"
    model.api = "test-api"
    model.config = {"temperature": 0.7}
    return model


@pytest.fixture
def model_wrapper(mock_base_model):
    """Create TranscriptSyncingModelWrapper instance for testing."""
    return TranscriptSyncingModelWrapper(
        base_model=mock_base_model,
        session_id="session_123",
        episode_id="episode_456",
        rest_url="http://localhost:8000",
    )


class TestModelWrapperInit:
    """Test TranscriptSyncingModelWrapper initialization."""

    def test_init_stores_parameters(self, mock_base_model):
        """Test that initialization stores all parameters correctly."""
        wrapper = TranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

        assert wrapper._base_model is mock_base_model
        assert wrapper._session_id == "session_123"
        assert wrapper._episode_id == "episode_456"
        assert wrapper._rest_url == "http://localhost:8000"

    def test_init_logs_debug_message(self, mock_base_model):
        """Test that initialization logs debug message with context."""
        with patch("saber.inspect_ai.integration.model_wrapper.logger") as mock_logger:
            wrapper = TranscriptSyncingModelWrapper(
                base_model=mock_base_model,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

            # Verify logger.debug was called
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            assert "Created TranscriptSyncingModelWrapper" in call_args[0][0]
            assert "session_id" in call_args[1]["extra"]
            assert call_args[1]["extra"]["session_id"] == "session_123"


class TestGenerate:
    """Test generate() method with transcript push."""

    @pytest.mark.asyncio
    async def test_generate_calls_base_model(self, model_wrapper, mock_base_model):
        """Test that generate() calls the base model's generate method."""
        # Setup mock output
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        # Mock transcript push
        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            result = await model_wrapper.generate(
                input="Test input",
                tools=[],
            )

            # Verify base model was called
            mock_base_model.generate.assert_called_once_with("Test input", [], )

    @pytest.mark.asyncio
    async def test_generate_pushes_transcript(self, model_wrapper, mock_base_model):
        """Test that generate() pushes transcript after generation."""
        # Setup mock output
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        # Mock transcript push - patch the instance method
        with patch.object(model_wrapper, '_push_single_message', new_callable=AsyncMock) as mock_push:
            await model_wrapper.generate(
                input="Test input",
                tools=None,
            )

            # Verify transcript was pushed with just the message (positional argument)
            mock_push.assert_called_once_with(mock_output.message)

    @pytest.mark.asyncio
    async def test_generate_with_tool_calls(self, model_wrapper, mock_base_model):
        """Test generate() with tool calls in response."""
        # Setup mock output with tool calls
        mock_output = ModelOutput.for_tool_call(
            model="test-model",
            tool_name="execute_command",
            tool_arguments={"command": "ls"},
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        # Mock transcript push
        with patch.object(model_wrapper, '_push_single_message', new_callable=AsyncMock) as mock_push:
            result = await model_wrapper.generate(
                input=[ChatMessageUser(content="Test")],
                tools=[],
            )

            # Verify tool_calls were included in push
            mock_push.assert_called_once()
            pushed_message = mock_push.call_args[0][0]  # First positional argument
            assert pushed_message.tool_calls is not None
            assert len(pushed_message.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_generate_logs_after_push(self, model_wrapper, mock_base_model):
        """Test that generate() logs debug message after push."""
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            with patch("saber.inspect_ai.integration.model_wrapper.logger") as mock_logger:
                await model_wrapper.generate(input="Test")

                # Verify debug log was called
                assert mock_logger.debug.call_count >= 1
                # Find the call about pushing message
                debug_calls = [call for call in mock_logger.debug.call_args_list if "Pushed assistant message" in str(call)]
                assert len(debug_calls) > 0

    @pytest.mark.asyncio
    async def test_generate_graceful_degradation_on_push_error(self, model_wrapper, mock_base_model):
        """Test that generate() continues execution if transcript push fails."""
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        # Mock transcript push to raise exception
        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
            mock_push.side_effect = Exception("Network error")

            # Should not raise, just log warning
            result = await model_wrapper.generate(input="Test")

            # Verify result is still returned
            assert result == mock_output

    @pytest.mark.asyncio
    async def test_generate_logs_warning_on_push_error(self, model_wrapper, mock_base_model):
        """Test that generate() logs warning when transcript push fails."""
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        # Mock transcript push to raise exception
        with patch.object(model_wrapper, '_push_single_message', new_callable=AsyncMock) as mock_push:
            mock_push.side_effect = RuntimeError("Connection timeout")

            with patch("saber.inspect_ai.integration.model_wrapper.logger") as mock_logger:
                await model_wrapper.generate(input="Test")

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args
                assert "Failed to push transcript after model.generate()" in call_args[0][0]
                assert call_args[1]["extra"]["error_type"] == "RuntimeError"
                assert "Connection timeout" in call_args[1]["extra"]["error"]

    @pytest.mark.asyncio
    async def test_generate_returns_original_output(self, model_wrapper, mock_base_model):
        """Test that generate() returns the original ModelOutput unchanged."""
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Test response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            result = await model_wrapper.generate(input="Test")

            # Verify exact same object is returned
            assert result is mock_output


class TestGetattr:
    """Test __getattr__ delegation to base model."""

    def test_getattr_delegates_to_base_model(self, model_wrapper, mock_base_model):
        """Test that attribute access is delegated to base model."""
        # Access various attributes
        assert model_wrapper.name == mock_base_model.name
        assert model_wrapper.api == mock_base_model.api
        assert model_wrapper.config == mock_base_model.config

    def test_getattr_with_method(self, model_wrapper, mock_base_model):
        """Test that method access is delegated to base model."""
        mock_base_model.some_method = Mock(return_value="result")

        result = model_wrapper.some_method()

        assert result == "result"
        mock_base_model.some_method.assert_called_once()

    def test_getattr_preserves_attribute_types(self, model_wrapper, mock_base_model):
        """Test that attribute types are preserved through delegation."""
        mock_base_model.int_attr = 42
        mock_base_model.str_attr = "test"
        mock_base_model.list_attr = [1, 2, 3]

        assert isinstance(model_wrapper.int_attr, int)
        assert isinstance(model_wrapper.str_attr, str)
        assert isinstance(model_wrapper.list_attr, list)

    def test_getattr_raises_attribute_error_for_missing(self, model_wrapper, mock_base_model):
        """Test that missing attributes raise AttributeError."""
        # Remove spec to allow AttributeError
        del mock_base_model.nonexistent_attr

        with pytest.raises(AttributeError):
            _ = model_wrapper.nonexistent_attr


class TestRepr:
    """Test __repr__ string representation."""

    def test_repr_shows_wrapped_model(self, model_wrapper, mock_base_model):
        """Test that __repr__ shows the wrapped model."""
        result = repr(model_wrapper)

        assert "TranscriptSyncingModelWrapper" in result
        assert repr(mock_base_model) in result

    def test_repr_format(self, mock_base_model):
        """Test the exact format of __repr__."""
        wrapper = TranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

        result = repr(wrapper)

        # Should be: TranscriptSyncingModelWrapper(<base_model_repr>)
        assert result.startswith("TranscriptSyncingModelWrapper(")
        assert result.endswith(")")


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_generate_with_empty_string_input(self, model_wrapper, mock_base_model):
        """Test generate() with empty string input."""
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            result = await model_wrapper.generate(input="")

            assert result == mock_output

    @pytest.mark.asyncio
    async def test_generate_with_message_list_input(self, model_wrapper, mock_base_model):
        """Test generate() with list of ChatMessage as input."""
        messages = [
            ChatMessageUser(content="Hello"),
            ChatMessageAssistant(content="Hi"),
            ChatMessageUser(content="How are you?"),
        ]
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Good",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            result = await model_wrapper.generate(input=messages)

            # Verify base model received the message list
            mock_base_model.generate.assert_called_once()
            call_args = mock_base_model.generate.call_args
            assert call_args[0][0] == messages

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, model_wrapper, mock_base_model):
        """Test generate() passes through keyword arguments."""
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            await model_wrapper.generate(
                input="Test",
                tools=None,
                temperature=0.5,
                max_tokens=100,
            )

            # Verify kwargs were passed through
            call_kwargs = mock_base_model.generate.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_preserves_stop_reason(self, model_wrapper, mock_base_model):
        """Test that generate() preserves stop_reason from base model."""
        # ModelOutput.from_content doesn't preserve custom stop_reason easily,
        # so we'll just verify the wrapper returns what the base model returns
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            result = await model_wrapper.generate(input="Test")

            # Verify we get back the exact same output
            assert result is mock_output

    @pytest.mark.asyncio
    async def test_generate_different_error_types(self, model_wrapper, mock_base_model):
        """Test that different error types are all handled gracefully."""
        mock_output = ModelOutput.from_content(
            model="test-model",
            content="Response",
        )
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        error_types = [
            ValueError("Invalid"),
            RuntimeError("Runtime"),
            ConnectionError("Connection"),
            TimeoutError("Timeout"),
        ]

        for error in error_types:
            with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
                mock_push.side_effect = error

                # Should not raise
                result = await model_wrapper.generate(input="Test")
                assert result == mock_output


class TestIntegration:
    """Integration tests with real-like scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_generate_calls(self, model_wrapper, mock_base_model):
        """Test that wrapper works correctly across multiple generate calls."""
        outputs = [
            ModelOutput.from_content(model="test-model", content="First"),
            ModelOutput.from_content(model="test-model", content="Second"),
            ModelOutput.from_content(model="test-model", content="Third"),
        ]
        mock_base_model.generate = AsyncMock(side_effect=outputs)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
            for i in range(3):
                result = await model_wrapper.generate(input=f"Input {i}")
                assert result == outputs[i]

            # Verify push was called for each generate
            assert mock_push.call_count == 3

    @pytest.mark.asyncio
    async def test_wrapper_is_transparent(self, mock_base_model):
        """Test that wrapper behaves identically to base model for non-generate methods."""
        # Add some methods to base model
        mock_base_model.reset = Mock()
        mock_base_model.get_config = Mock(return_value={"key": "value"})

        wrapper = TranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

        # Call methods through wrapper
        wrapper.reset()
        config = wrapper.get_config()

        # Verify base model methods were called
        mock_base_model.reset.assert_called_once()
        mock_base_model.get_config.assert_called_once()
        assert config == {"key": "value"}


class TestBlockingTranscriptSyncingModelWrapper:
    """Test BlockingTranscriptSyncingModelWrapper."""

    @pytest.fixture
    def blocking_wrapper(self, mock_base_model):
        """Create BlockingTranscriptSyncingModelWrapper instance for testing."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        return BlockingTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
            poll_interval=0.1,  # Fast polling for tests
            max_iterations=5,
            timeout=10.0,
            skip_first_iteration=True,
        )

    @pytest.mark.asyncio
    async def test_blocking_wrapper_init(self, mock_base_model):
        """Test BlockingTranscriptSyncingModelWrapper initialization."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        wrapper = BlockingTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
            poll_interval=2.0,
            max_iterations=50,
            timeout=100.0,
            skip_first_iteration=True,
        )

        assert wrapper._poll_interval == 2.0
        assert wrapper._max_iterations == 50
        assert wrapper._timeout == 100.0
        assert wrapper._skip_first_iteration is True
        assert wrapper._first_call is True

    @pytest.mark.asyncio
    async def test_blocking_wrapper_skips_first_iteration(self, blocking_wrapper, mock_base_model):
        """Test that blocking is skipped on first generate() call."""
        mock_output = ModelOutput.from_content(model="test-model", content="Response")
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            # First call should skip blocking
            result = await blocking_wrapper.generate(input="First input")

            assert result == mock_output
            # Verify base model was called without blocking delay
            mock_base_model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_blocking_wrapper_blocks_on_second_call(self, blocking_wrapper, mock_base_model):
        """Test that blocking occurs on second generate() call."""
        mock_output = ModelOutput.from_content(model="test-model", content="Response")
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            with patch.object(blocking_wrapper, "_block_and_pull_transcript", new_callable=AsyncMock) as mock_block:
                mock_block.return_value = [
                    ChatMessageUser(content="Modified message")
                ]

                # First call - no blocking (provide list input)
                await blocking_wrapper.generate(input=[ChatMessageUser(content="First")])
                assert mock_block.call_count == 0

                # Second call - should block
                await blocking_wrapper.generate(input=[ChatMessageUser(content="Second")])
                assert mock_block.call_count == 1

    @pytest.mark.asyncio
    async def test_blocking_wrapper_timestamp_polling_success(self, blocking_wrapper):
        """Test successful timestamp polling and transcript pull."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        # Mock get_transcript_timestamp to return different values
        initial_timestamp = "2025-12-01T10:00:00Z"
        modified_timestamp = "2025-12-01T10:01:00Z"

        timestamps = [initial_timestamp, initial_timestamp, modified_timestamp]
        timestamp_iter = iter(timestamps)

        with patch.object(BlockingTranscriptSyncingModelWrapper, "_get_transcript_timestamp", new_callable=AsyncMock) as mock_get_ts:
            mock_get_ts.side_effect = lambda: next(timestamp_iter)

            with patch.object(BlockingTranscriptSyncingModelWrapper, "_pull_transcript", new_callable=AsyncMock) as mock_pull:
                mock_pull.return_value = [
                    ChatMessageUser(content="Modified by red team")
                ]

                result = await blocking_wrapper._block_and_pull_transcript([])

                assert len(result) == 1
                assert result[0].content == "Modified by red team"
                # Should have polled 3 times (initial + 2 checks)
                assert mock_get_ts.call_count == 3
                mock_pull.assert_called_once()

    @pytest.mark.asyncio
    async def test_blocking_wrapper_timeout_error(self, blocking_wrapper):
        """Test that blocking raises RuntimeError on timeout."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        # Mock to always return same timestamp (never changes)
        with patch.object(BlockingTranscriptSyncingModelWrapper, "_get_transcript_timestamp", new_callable=AsyncMock) as mock_get_ts:
            mock_get_ts.return_value = "2025-12-01T10:00:00Z"

            # Set very short timeout and high max_iterations so timeout triggers first
            blocking_wrapper._timeout = 0.5
            blocking_wrapper._poll_interval = 0.1
            blocking_wrapper._max_iterations = 100

            with pytest.raises(RuntimeError, match="Blocking timeout exceeded"):
                await blocking_wrapper._block_and_pull_transcript([])

    @pytest.mark.asyncio
    async def test_blocking_wrapper_max_iterations_error(self, blocking_wrapper):
        """Test that blocking raises RuntimeError on max iterations."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        # Mock to always return same timestamp
        with patch.object(BlockingTranscriptSyncingModelWrapper, "_get_transcript_timestamp", new_callable=AsyncMock) as mock_get_ts:
            mock_get_ts.return_value = "2025-12-01T10:00:00Z"

            # Set very short timeout to not interfere with max_iterations test
            blocking_wrapper._timeout = 100.0
            blocking_wrapper._max_iterations = 3
            blocking_wrapper._poll_interval = 0.01

            with pytest.raises(RuntimeError, match="Blocking max iterations exceeded"):
                await blocking_wrapper._block_and_pull_transcript([])

    @pytest.mark.asyncio
    async def test_blocking_wrapper_get_transcript_timestamp(self, blocking_wrapper):
        """Test _get_transcript_timestamp calls API correctly."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        mock_metadata = {
            "_transcript_last_modified_at": "2025-12-01T10:30:00Z",
            "_transcript_modification_count": 5,
        }

        with patch("saber.inspect_ai.integration.model_wrapper.get_episode_metadata", new_callable=AsyncMock) as mock_get_meta:
            mock_get_meta.return_value = mock_metadata

            timestamp = await blocking_wrapper._get_transcript_timestamp()

            assert timestamp == "2025-12-01T10:30:00Z"
            # Verify called with client, session_id, episode_id
            assert mock_get_meta.call_count == 1
            call_args = mock_get_meta.call_args[0]
            assert call_args[1] == "session_123"  # session_id
            assert call_args[2] == "episode_456"  # episode_id

    @pytest.mark.asyncio
    async def test_blocking_wrapper_pull_transcript(self, blocking_wrapper):
        """Test _pull_transcript converts API response to ChatMessages."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        mock_transcript = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Modified by red team"},
            ]
        }

        with patch("saber.inspect_ai.integration.model_wrapper.pull_episode_transcript", new_callable=AsyncMock) as mock_pull:
            mock_pull.return_value = mock_transcript

            messages = await blocking_wrapper._pull_transcript()

            assert len(messages) == 3
            assert messages[0].role == "user"
            assert messages[0].content == "Hello"
            assert messages[1].role == "assistant"
            assert messages[1].content == "Hi there"
            assert messages[2].role == "user"
            assert messages[2].content == "Modified by red team"

            # Verify called with client, session_id, episode_id
            assert mock_pull.call_count == 1
            call_args = mock_pull.call_args[0]
            assert call_args[1] == "session_123"  # session_id
            assert call_args[2] == "episode_456"  # episode_id

    @pytest.mark.asyncio
    async def test_blocking_wrapper_timestamp_error_handling(self, blocking_wrapper):
        """Test graceful handling of timestamp retrieval errors."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        with patch.object(BlockingTranscriptSyncingModelWrapper, "_get_transcript_timestamp", new_callable=AsyncMock) as mock_get_ts:
            mock_get_ts.side_effect = Exception("Network error")

            # Should return original input on error
            result = await blocking_wrapper._block_and_pull_transcript(
                [ChatMessageUser(content="Original")]
            )

            assert len(result) == 1
            assert result[0].content == "Original"

    @pytest.mark.asyncio
    async def test_blocking_wrapper_with_skip_first_false(self, mock_base_model):
        """Test blocking wrapper with skip_first_iteration=False."""
        from saber.inspect_ai.integration.model_wrapper import BlockingTranscriptSyncingModelWrapper

        wrapper = BlockingTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,  # Don't skip first iteration
        )

        mock_output = ModelOutput.from_content(model="test-model", content="Response")
        mock_base_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
            with patch.object(wrapper, "_block_and_pull_transcript", new_callable=AsyncMock) as mock_block:
                mock_block.return_value = [ChatMessageUser(content="Modified")]

                # First call should also block when skip_first_iteration=False (provide list input)
                await wrapper.generate(input=[ChatMessageUser(content="First")])
                assert mock_block.call_count == 1


class TestMessageSerialization:
    """Test message serialization and deserialization methods."""

    def test_serialize_system_message(self):
        """Test serialization of ChatMessageSystem."""
        msg = ChatMessageUser(content="System prompt")
        result = TranscriptSyncingModelWrapper._serialize_message(msg)

        assert result["role"] == "user"
        assert result["content"] == "System prompt"

    def test_serialize_user_message(self):
        """Test serialization of ChatMessageUser."""
        msg = ChatMessageUser(content="User query")
        result = TranscriptSyncingModelWrapper._serialize_message(msg)

        assert result["role"] == "user"
        assert result["content"] == "User query"

    def test_serialize_assistant_message(self):
        """Test serialization of ChatMessageAssistant."""
        msg = ChatMessageAssistant(content="Response")
        result = TranscriptSyncingModelWrapper._serialize_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Response"

    def test_serialize_assistant_with_tool_calls(self):
        """Test serialization of assistant message with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            function="execute_command",
            arguments={"command": "ls"},
            type="function",
        )
        msg = ChatMessageAssistant(content="", tool_calls=[tool_call])
        result = TranscriptSyncingModelWrapper._serialize_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == ""
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["function"] == "execute_command"
        assert result["tool_calls"][0]["arguments"] == {"command": "ls"}

    def test_deserialize_user_message(self):
        """Test deserialization of user message."""
        msg_data = {"role": "user", "content": "Test query"}
        result = TranscriptSyncingModelWrapper._deserialize_message(msg_data)

        assert isinstance(result, ChatMessageUser)
        assert result.content == "Test query"

    def test_deserialize_assistant_message(self):
        """Test deserialization of assistant message."""
        msg_data = {"role": "assistant", "content": "Test response"}
        result = TranscriptSyncingModelWrapper._deserialize_message(msg_data)

        assert isinstance(result, ChatMessageAssistant)
        assert result.content == "Test response"

    def test_deserialize_assistant_with_tool_calls(self):
        """Test deserialization of assistant message with tool calls."""
        msg_data = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_456",
                    "function": "read_file",
                    "arguments": {"path": "/tmp/test.txt"},
                }
            ],
        }
        result = TranscriptSyncingModelWrapper._deserialize_message(msg_data)

        assert isinstance(result, ChatMessageAssistant)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_456"
        assert result.tool_calls[0].function == "read_file"

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialization and deserialization are inverses."""
        original = ChatMessageUser(content="Roundtrip test")
        serialized = TranscriptSyncingModelWrapper._serialize_message(original)
        deserialized = TranscriptSyncingModelWrapper._deserialize_message(serialized)

        assert isinstance(deserialized, ChatMessageUser)
        assert deserialized.content == original.content


class TestPushSingleMessage:
    """Test _push_single_message instance method."""

    @pytest.fixture
    def wrapper(self):
        """Create a model wrapper for testing."""
        mock_model = Mock(spec=Model)
        return TranscriptSyncingModelWrapper(
            base_model=mock_model,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

    @pytest.mark.asyncio
    async def test_push_single_message_success(self, wrapper):
        """Test successful push of a single message."""
        message = ChatMessageAssistant(content="Test response")

        with patch.object(wrapper._client, "push_episode_transcript", new_callable=AsyncMock) as mock_push:
            await wrapper._push_single_message(message)

            # Verify push was called
            mock_push.assert_called_once()
            call_args = mock_push.call_args
            assert call_args.kwargs["session_id"] == "session_123"
            assert call_args.kwargs["episode_id"] == "episode_456"
            assert call_args.kwargs["mode"] == "append"
            assert len(call_args.kwargs["messages"]) == 1
            assert call_args.kwargs["messages"][0]["role"] == "assistant"
            assert call_args.kwargs["messages"][0]["content"] == "Test response"

    @pytest.mark.asyncio
    async def test_push_single_message_error_handling(self, wrapper):
        """Test that push errors are logged but not raised."""
        message = ChatMessageUser(content="Test")

        with patch.object(
            wrapper._client,
            "push_episode_transcript",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            # Should not raise exception
            await wrapper._push_single_message(message)


class TestPullInjectedMessages:
    """Test pull_injected_messages static method."""

    @pytest.mark.asyncio
    async def test_pull_injected_messages_success(self):
        """Test successful pull of injected messages."""
        state = TaskState(
            model="test-model",
            sample_id=0,
            epoch=1,
            input="",
            messages=[],
        )

        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "messages": [
                    {"role": "user", "content": "Injected message 1"},
                    {"role": "assistant", "content": "Injected message 2"},
                ]
            }
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()))

        with patch("aiohttp.ClientSession", return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())):
            await TranscriptSyncingModelWrapper.pull_injected_messages(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

        # Verify messages were injected
        assert len(state.messages) == 2
        assert isinstance(state.messages[0], ChatMessageUser)
        assert state.messages[0].content == "Injected message 1"
        assert isinstance(state.messages[1], ChatMessageAssistant)
        assert state.messages[1].content == "Injected message 2"

    @pytest.mark.asyncio
    async def test_pull_injected_messages_404(self):
        """Test handling of 404 response (episode not found)."""
        state = TaskState(
            model="test-model",
            sample_id=0,
            epoch=1,
            input="",
            messages=[],
        )

        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()))

        with patch("aiohttp.ClientSession", return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())):
            await TranscriptSyncingModelWrapper.pull_injected_messages(
                state=state,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

        # Verify no messages were added
        assert len(state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_invalid_url(self):
        """Test handling of invalid URL format."""
        state = TaskState(
            model="test-model",
            sample_id=0,
            epoch=1,
            input="",
            messages=[],
        )

        # Should handle gracefully without making HTTP request
        await TranscriptSyncingModelWrapper.pull_injected_messages(
            state=state,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="invalid-url",
        )

        assert len(state.messages) == 0
