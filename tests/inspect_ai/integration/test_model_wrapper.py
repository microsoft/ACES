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

        # Mock transcript push
        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
            await model_wrapper.generate(
                input="Test input",
                tools=None,
            )

            # Verify transcript was pushed
            mock_push.assert_called_once_with(
                message=mock_output.message,
                session_id="session_123",
                episode_id="episode_456",
                rest_url="http://localhost:8000",
            )

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
        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
            result = await model_wrapper.generate(
                input=[ChatMessageUser(content="Test")],
                tools=[],
            )

            # Verify tool_calls were included in push
            mock_push.assert_called_once()
            pushed_message = mock_push.call_args[1]["message"]
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
        with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
            mock_push.side_effect = RuntimeError("Connection timeout")

            with patch("saber.inspect_ai.integration.model_wrapper.logger") as mock_logger:
                await model_wrapper.generate(input="Test")

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args
                assert "Failed to push transcript" in call_args[0][0]
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
