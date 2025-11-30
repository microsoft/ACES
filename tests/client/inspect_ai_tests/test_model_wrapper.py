"""
Unit tests for TranscriptSyncingModelWrapper.

These tests validate that the model wrapper correctly intercepts generate() calls
and pushes transcripts to the SABER server, solving the race condition where
the server needs assistant messages before tool execution.

Critical Test Coverage:
1. Wrapper intercepts model.generate() calls
2. Pushes single message after each generate
3. Delegates all other attributes to base model
4. Graceful error handling (transcript push failures don't crash)
5. Works with different model types
6. Maintains original model behavior
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.tool import ToolCall

from saber.inspect_ai.integration.model_wrapper import TranscriptSyncingModelWrapper


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_base_model():
    """Create a mock Model object."""
    model = MagicMock()
    model.generate = AsyncMock()
    model.some_attribute = "test_value"
    model.some_method = Mock(return_value="test_result")
    return model


@pytest.fixture
def mock_model_output():
    """Create a mock ModelOutput with assistant message."""
    output = MagicMock(spec=ModelOutput)
    output.message = ChatMessageAssistant(
        content="I'll help you with that.",
        tool_calls=[
            ToolCall(
                id="call_123",
                function="bash",
                arguments={"command": "ls -la"},
                type="function",
            )
        ],
    )
    return output


@pytest.fixture
def wrapper(mock_base_model):
    """Create a TranscriptSyncingModelWrapper instance."""
    return TranscriptSyncingModelWrapper(
        base_model=mock_base_model,
        session_id="session_123",
        episode_id="episode_456",
        rest_url="http://localhost:8000",
    )


# ============================================================================
# Test: Wrapper Initialization
# ============================================================================


def test_wrapper_initialization(mock_base_model):
    """Test that wrapper initializes with correct attributes."""
    wrapper = TranscriptSyncingModelWrapper(
        base_model=mock_base_model,
        session_id="session_123",
        episode_id="episode_456",
        rest_url="http://localhost:8000",
    )

    assert wrapper._base_model == mock_base_model
    assert wrapper._session_id == "session_123"
    assert wrapper._episode_id == "episode_456"
    assert wrapper._rest_url == "http://localhost:8000"


def test_wrapper_repr(wrapper):
    """Test that wrapper has useful repr."""
    repr_str = repr(wrapper)
    assert "TranscriptSyncingModelWrapper" in repr_str


# ============================================================================
# Test: Generate Interception
# ============================================================================


@pytest.mark.asyncio
async def test_generate_calls_base_model(mock_base_model, mock_model_output, wrapper):
    """Test that wrapper calls base model's generate method."""
    mock_base_model.generate.return_value = mock_model_output
    messages = [ChatMessageUser(content="Hello")]
    tools = []

    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
        result = await wrapper.generate(messages, tools)

    # Verify base model was called
    mock_base_model.generate.assert_called_once_with(messages, tools)
    assert result == mock_model_output


@pytest.mark.asyncio
async def test_generate_pushes_assistant_message(mock_base_model, mock_model_output, wrapper):
    """Test that wrapper pushes assistant message after generate."""
    mock_base_model.generate.return_value = mock_model_output
    messages = [ChatMessageUser(content="Hello")]

    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
        await wrapper.generate(messages, tools=[])

    # Verify message was pushed
    mock_push.assert_called_once_with(
        message=mock_model_output.message,
        session_id="session_123",
        episode_id="episode_456",
        rest_url="http://localhost:8000",
    )


@pytest.mark.asyncio
async def test_generate_with_kwargs(mock_base_model, mock_model_output, wrapper):
    """Test that wrapper passes kwargs to base model."""
    mock_base_model.generate.return_value = mock_model_output
    messages = [ChatMessageUser(content="Hello")]
    tools = []

    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
        await wrapper.generate(messages, tools, temperature=0.7, max_tokens=100)

    # Verify kwargs were passed
    mock_base_model.generate.assert_called_once_with(
        messages, tools, temperature=0.7, max_tokens=100
    )


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_generate_continues_on_push_failure(mock_base_model, mock_model_output, wrapper):
    """Test that generate doesn't crash if transcript push fails."""
    mock_base_model.generate.return_value = mock_model_output
    messages = [ChatMessageUser(content="Hello")]

    # Make push raise an exception
    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
        mock_push.side_effect = Exception("Network error")

        # Should not raise, should return output
        result = await wrapper.generate(messages, tools=[])

    assert result == mock_model_output
    mock_push.assert_called_once()


@pytest.mark.asyncio
async def test_generate_continues_on_timeout(mock_base_model, mock_model_output, wrapper):
    """Test that generate doesn't crash if transcript push times out."""
    import asyncio

    mock_base_model.generate.return_value = mock_model_output
    messages = [ChatMessageUser(content="Hello")]

    # Make push timeout
    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
        mock_push.side_effect = asyncio.TimeoutError("Request timeout")

        # Should not raise, should return output
        result = await wrapper.generate(messages, tools=[])

    assert result == mock_model_output


@pytest.mark.asyncio
async def test_generate_propagates_base_model_errors(mock_base_model, wrapper):
    """Test that wrapper propagates errors from base model."""
    mock_base_model.generate.side_effect = ValueError("Invalid input")
    messages = [ChatMessageUser(content="Hello")]

    # Should raise the base model error
    with pytest.raises(ValueError, match="Invalid input"):
        await wrapper.generate(messages, tools=[])


# ============================================================================
# Test: Attribute Delegation
# ============================================================================


def test_wrapper_delegates_attributes(mock_base_model, wrapper):
    """Test that wrapper delegates attribute access to base model."""
    # Access an attribute from base model
    assert wrapper.some_attribute == "test_value"

    # Call a method from base model
    result = wrapper.some_method()
    assert result == "test_result"
    mock_base_model.some_method.assert_called_once()


def test_wrapper_delegates_to_real_object():
    """Test that wrapper delegates to actual object (not mock)."""
    # Create a real object to wrap instead of a mock
    class RealModel:
        def __init__(self):
            self.real_attr = "real_value"

    real_model = RealModel()
    wrapper = TranscriptSyncingModelWrapper(
        base_model=real_model,
        session_id="session_123",
        episode_id="episode_456",
        rest_url="http://localhost:8000",
    )

    # Access should work
    assert wrapper.real_attr == "real_value"

    # Missing attribute should raise
    with pytest.raises(AttributeError):
        _ = wrapper.nonexistent_attribute


# ============================================================================
# Test: Multiple Generate Calls
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_generate_calls(mock_base_model, wrapper):
    """Test that wrapper handles multiple generate calls correctly."""
    # Create different outputs for each call
    output1 = MagicMock(spec=ModelOutput)
    output1.message = ChatMessageAssistant(content="First response")

    output2 = MagicMock(spec=ModelOutput)
    output2.message = ChatMessageAssistant(content="Second response")

    mock_base_model.generate.side_effect = [output1, output2]

    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
        # First call
        result1 = await wrapper.generate([ChatMessageUser(content="Hello")], tools=[])
        assert result1 == output1

        # Second call
        result2 = await wrapper.generate([ChatMessageUser(content="How are you?")], tools=[])
        assert result2 == output2

    # Verify both messages were pushed
    assert mock_push.call_count == 2
    assert mock_push.call_args_list[0][1]["message"] == output1.message
    assert mock_push.call_args_list[1][1]["message"] == output2.message


# ============================================================================
# Test: String Input Support
# ============================================================================


@pytest.mark.asyncio
async def test_generate_with_string_input(mock_base_model, mock_model_output, wrapper):
    """Test that wrapper handles string input (not just message list)."""
    mock_base_model.generate.return_value = mock_model_output

    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
        result = await wrapper.generate("Hello, world!", tools=[])

    # Verify base model was called with string
    mock_base_model.generate.assert_called_once_with("Hello, world!", [])
    assert result == mock_model_output


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


@pytest.mark.asyncio
async def test_wrapper_in_agent_loop_simulation(mock_base_model, wrapper):
    """Test wrapper in a simulated agent loop (multiple iterations)."""
    # Simulate 3 iterations of an agent loop
    outputs = [
        MagicMock(spec=ModelOutput, message=ChatMessageAssistant(
            content=f"Response {i}",
            tool_calls=[ToolCall(id=f"call_{i}", function="bash", arguments={}, type="function")]
        ))
        for i in range(3)
    ]

    mock_base_model.generate.side_effect = outputs

    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock) as mock_push:
        for i in range(3):
            messages = [ChatMessageUser(content=f"Input {i}")]
            result = await wrapper.generate(messages, tools=[])
            assert result == outputs[i]

    # Verify all messages were pushed
    assert mock_push.call_count == 3
    for i, call in enumerate(mock_push.call_args_list):
        assert call[1]["message"].content == f"Response {i}"
        assert call[1]["session_id"] == "session_123"
        assert call[1]["episode_id"] == "episode_456"


# ============================================================================
# Test: Thread Safety (if applicable)
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_generate_calls(mock_base_model, wrapper):
    """Test that wrapper handles concurrent generate calls."""
    import asyncio

    outputs = [
        MagicMock(spec=ModelOutput, message=ChatMessageAssistant(content=f"Response {i}"))
        for i in range(5)
    ]

    call_count = 0

    async def generate_side_effect(*args, **kwargs):
        nonlocal call_count
        result = outputs[call_count]
        call_count += 1
        await asyncio.sleep(0.01)  # Simulate network delay
        return result

    mock_base_model.generate = AsyncMock(side_effect=generate_side_effect)

    with patch("saber.inspect_ai.integration.model_wrapper._push_single_message", new_callable=AsyncMock):
        # Run 5 concurrent generate calls
        tasks = [
            wrapper.generate([ChatMessageUser(content=f"Input {i}")], tools=[])
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

    # Verify all calls completed
    assert len(results) == 5
    assert mock_base_model.generate.call_count == 5
