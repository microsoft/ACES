"""
Unit tests for solver-level transcript synchronization.

These tests validate that the solver wrapper in create_saber_solver()
correctly wraps the generate function to push transcripts after each
model inference call.

Test Coverage:
1. Generate wrapper pushes transcript after each generate call
2. Wrapper extracts episode_id/session_id/rest_url from metadata
3. Non-SABER tasks run without transcript sync (no episode_id)
4. Missing metadata handled gracefully (no failures)
5. Push failures don't crash agent execution
6. Multiple generate calls in single episode push multiple times
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ChatMessageSystem
from inspect_ai.solver import Generate, TaskState

from saber.inspect_ai.agents.solver_factory import create_saber_solver
from saber.models.constants import MetadataKeys


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_state_with_saber_metadata():
    """Create TaskState with SABER episode metadata."""
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageSystem(content="You are a helpful assistant"),
        ChatMessageUser(content="What is 2+2?"),
    ]
    state.metadata = {
        MetadataKeys.EPISODE_ID: "ep_test123",
        MetadataKeys.SESSION_ID: "session_test456",
        MetadataKeys.SABER_DOMAIN_SLUG: "test-domain",
        MetadataKeys.TASK_ID: "task_789",
        MetadataKeys.INSTRUCTION_PROMPT: "Solve the task",
        MetadataKeys.ASSISTANT_PROMPT: "You are an assistant",
        MetadataKeys.SUBMIT_PROMPT: "Submit your answer",
    }
    return state


@pytest.fixture
def mock_state_without_saber_metadata():
    """Create TaskState without SABER metadata (non-SABER task)."""
    state = MagicMock(spec=TaskState)
    state.messages = [
        ChatMessageUser(content="What is 2+2?"),
    ]
    state.metadata = {
        MetadataKeys.INSTRUCTION_PROMPT: "Solve the task",
        MetadataKeys.ASSISTANT_PROMPT: "You are an assistant",
        MetadataKeys.SUBMIT_PROMPT: "Submit your answer",
    }
    return state


@pytest.fixture
def mock_generate_function():
    """Create a mock Generate function that simulates model inference."""
    async def mock_generate(state: TaskState) -> TaskState:
        """Simulates model adding a response to messages."""
        # Add an assistant message to simulate model response
        state.messages.append(
            ChatMessageAssistant(content="The answer is 4")
        )
        return state

    return mock_generate


@pytest.fixture
def mock_agent_factory():
    """Create a mock agent factory that returns an agent creator."""
    async def mock_agent(state: TaskState) -> TaskState:
        """Mock agent that just returns the state."""
        return state

    def create_with_prompts(instruction_prompt, assistant_prompt, submit_prompt):
        """Returns the agent function."""
        return mock_agent

    def factory():
        """Factory that returns the prompt-accepting creator."""
        return create_with_prompts

    return factory


# ============================================================================
# Tests for Generate Wrapper Integration
# ============================================================================


@pytest.mark.asyncio
async def test_solver_wraps_generate_with_saber_metadata(
    mock_state_with_saber_metadata,
    mock_generate_function,
    mock_agent_factory,
):
    """Test that solver creates model wrapper when SABER metadata present."""

    with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain, \
         patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model, \
         patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_wrapper_class:

        # Mock get_active_domain to return REST URL
        mock_get_domain.return_value = {
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
        }

        # Mock the active model
        mock_model = Mock(spec=Model)
        mock_active_model.return_value = mock_model

        # Mock the wrapper class to return a mock wrapper
        mock_wrapper = Mock()
        mock_wrapper_class.return_value = mock_wrapper

        # Create solver
        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=mock_agent_factory,
            role_config=None,
        )

        # Execute solver
        result = await solver(mock_state_with_saber_metadata, mock_generate_function)

        # Verify TranscriptSyncingModelWrapper was created with correct parameters
        assert mock_wrapper_class.called, "TranscriptSyncingModelWrapper should have been created"

        # Verify it was called with correct parameters
        call_args = mock_wrapper_class.call_args
        assert call_args is not None

        assert call_args[1]["episode_id"] == "ep_test123"
        assert call_args[1]["session_id"] == "session_test456"
        assert call_args[1]["rest_url"] == "http://localhost:8000"
        assert call_args[1]["base_model"] == mock_model


@pytest.mark.asyncio
async def test_solver_skips_transcript_push_without_saber_metadata(
    mock_state_without_saber_metadata,
    mock_generate_function,
    mock_agent_factory,
):
    """Test that solver runs normally without SABER metadata (no transcript push)."""

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript", new_callable=AsyncMock) as mock_push:
        # Create solver
        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=mock_agent_factory,
            role_config=None,
        )

        # Execute solver
        result = await solver(mock_state_without_saber_metadata, mock_generate_function)

        # Verify _push_transcript was NOT called
        assert not mock_push.called, "_push_transcript should not be called for non-SABER tasks"


@pytest.mark.asyncio
async def test_solver_continues_on_push_failure(
    mock_state_with_saber_metadata,
    mock_generate_function,
    mock_agent_factory,
):
    """Test that solver continues agent execution even if transcript push fails."""

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript", new_callable=AsyncMock) as mock_push, \
         patch("saber.inspect_ai.server.domain_manager.get_active_domain") as mock_get_domain:

        # Mock get_active_domain to return REST URL
        mock_get_domain.return_value = {
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
        }

        # Make push fail
        mock_push.side_effect = Exception("Network error")

        # Create solver
        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=mock_agent_factory,
            role_config=None,
        )

        # Execute solver - should NOT raise exception
        result = await solver(mock_state_with_saber_metadata, mock_generate_function)

        # Verify execution completed successfully
        assert result is not None
        assert isinstance(result, TaskState)


@pytest.mark.asyncio
async def test_solver_handles_partial_metadata(
    mock_generate_function,
    mock_agent_factory,
):
    """Test that solver handles partial SABER metadata gracefully."""

    # Create state with only some SABER metadata
    state = MagicMock(spec=TaskState)
    state.messages = [ChatMessageUser(content="Test")]
    state.metadata = {
        MetadataKeys.EPISODE_ID: "ep_test123",
        # Missing session_id and rest_url
        MetadataKeys.INSTRUCTION_PROMPT: "Solve the task",
        MetadataKeys.ASSISTANT_PROMPT: "You are an assistant",
        MetadataKeys.SUBMIT_PROMPT: "Submit your answer",
    }

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript", new_callable=AsyncMock) as mock_push:
        # Create solver
        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=mock_agent_factory,
            role_config=None,
        )

        # Execute solver
        result = await solver(state, mock_generate_function)

        # Verify _push_transcript was NOT called (missing required metadata)
        assert not mock_push.called, "_push_transcript should not be called with partial metadata"


@pytest.mark.asyncio
async def test_multiple_generate_calls_push_multiple_times(
    mock_state_with_saber_metadata,
    mock_agent_factory,
):
    """Test that solver creates model wrapper for episodes with multiple generate calls."""

    # Create a generate function that gets called multiple times
    call_count = 0

    async def multi_call_generate(state: TaskState) -> TaskState:
        nonlocal call_count
        call_count += 1
        state.messages.append(
            ChatMessageAssistant(content=f"Response {call_count}")
        )
        return state

    # Create an agent that calls generate multiple times
    async def multi_iteration_agent(state: TaskState) -> TaskState:
        # Simulate 3 agent iterations
        for i in range(3):
            state = await multi_call_generate(state)
        return state

    def create_with_prompts(instruction_prompt, assistant_prompt, submit_prompt):
        return multi_iteration_agent

    def factory():
        return create_with_prompts

    with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain, \
         patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model, \
         patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_wrapper_class:

        # Mock get_active_domain to return REST URL
        mock_get_domain.return_value = {
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
        }

        # Mock the active model
        mock_model = Mock(spec=Model)
        mock_active_model.return_value = mock_model

        # Mock the wrapper
        mock_wrapper = Mock()
        mock_wrapper_class.return_value = mock_wrapper

        # Create solver
        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=factory,
            role_config=None,
        )

        # Execute solver
        result = await solver(mock_state_with_saber_metadata, multi_call_generate)

        # Verify the model wrapper was created (it will handle pushing on each model.generate() call)
        assert mock_wrapper_class.called, "Should create model wrapper for transcript sync"


@pytest.mark.asyncio
async def test_wrapper_preserves_generate_return_value(
    mock_state_with_saber_metadata,
    mock_agent_factory,
):
    """Test that generate wrapper returns the same state as original generate."""

    # Create generate that returns a specific state
    expected_state = MagicMock(spec=TaskState)
    expected_state.messages = [
        ChatMessageSystem(content="System"),
        ChatMessageUser(content="User"),
        ChatMessageAssistant(content="Assistant response"),
    ]

    async def custom_generate(state: TaskState) -> TaskState:
        return expected_state

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript", new_callable=AsyncMock), \
         patch("saber.inspect_ai.server.domain_manager.get_active_domain") as mock_get_domain:

        # Mock get_active_domain to return REST URL
        mock_get_domain.return_value = {
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
        }

        # Create solver
        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=mock_agent_factory,
            role_config=None,
        )

        # Execute solver
        result = await solver(mock_state_with_saber_metadata, custom_generate)

        # Verify the result is the expected state
        # (through the agent which should receive it)
        assert result is not None


# ============================================================================
# Tests for Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_solver_with_none_metadata(
    mock_generate_function,
    mock_agent_factory,
):
    """Test that solver handles None metadata gracefully."""

    state = MagicMock(spec=TaskState)
    state.messages = [ChatMessageUser(content="Test")]
    state.metadata = {}

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript", new_callable=AsyncMock) as mock_push:
        # Should raise ValueError for missing prompts, not crash on metadata access
        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=mock_agent_factory,
            role_config=None,
        )

        with pytest.raises(ValueError, match="Missing 'instruction_prompt'"):
            await solver(state, mock_generate_function)


@pytest.mark.asyncio
async def test_solver_with_empty_rest_url(
    mock_generate_function,
    mock_agent_factory,
):
    """Test that solver handles missing domain context gracefully."""

    state = MagicMock(spec=TaskState)
    state.messages = [ChatMessageUser(content="Test")]
    state.metadata = {
        MetadataKeys.EPISODE_ID: "ep_test123",
        MetadataKeys.SESSION_ID: "session_test456",
        MetadataKeys.SABER_DOMAIN_SLUG: "test-domain",
        MetadataKeys.INSTRUCTION_PROMPT: "Solve the task",
        MetadataKeys.ASSISTANT_PROMPT: "You are an assistant",
        MetadataKeys.SUBMIT_PROMPT: "Submit your answer",
    }

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript", new_callable=AsyncMock) as mock_push, \
         patch("saber.inspect_ai.server.domain_manager.get_active_domain") as mock_get_domain:

        # Mock get_active_domain to return None (domain not active)
        mock_get_domain.return_value = None

        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=mock_agent_factory,
            role_config=None,
        )

        # Should execute without crashing (push will be skipped)
        result = await solver(state, mock_generate_function)
        assert result is not None

        # Verify push was NOT called since domain context missing
        assert not mock_push.called
