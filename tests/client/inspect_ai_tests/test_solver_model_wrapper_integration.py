"""
Integration tests for solver_factory with TranscriptSyncingModelWrapper.

These tests validate that the solver correctly wraps models with the
TranscriptSyncingModelWrapper and that agents use the wrapped model.

Critical Test Coverage:
1. Solver wraps model when SABER context is available
2. Solver doesn't wrap when context is missing
3. Wrapped model is set as active model
4. Agent receives wrapped model
5. Original model is restored after execution
6. Role-based model selection works with wrapper
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ModelOutput, ChatMessageAssistant

from saber.inspect_ai.agents.solver_factory import create_saber_solver
from saber.models.constants import MetadataKeys


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_metadata():
    """Create mock metadata with SABER context."""
    return {
        MetadataKeys.SESSION_ID: "session_123",
        MetadataKeys.EPISODE_ID: "episode_456",
        MetadataKeys.SABER_DOMAIN_SLUG: "test_domain",
        MetadataKeys.TASK_ID: "task_789",
        MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
        MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        MetadataKeys.SUBMIT_PROMPT: "Test submit",
    }


@pytest.fixture
def mock_metadata_incomplete():
    """Create mock metadata without complete SABER context."""
    return {
        MetadataKeys.TASK_ID: "task_789",
        MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
        MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        MetadataKeys.SUBMIT_PROMPT: "Test submit",
        # Missing session_id, episode_id, domain_slug
    }


@pytest.fixture
def mock_task_state():
    """Create a mock TaskState."""
    state = MagicMock(spec=TaskState)
    state.messages = [ChatMessageUser(content="Hello")]
    state.metadata = {}
    state.output = None
    return state


@pytest.fixture
def mock_agent():
    """Create a mock agent that simulates execution."""
    async def agent_execute(state):
        # Simulate agent adding an assistant message
        state.output = MagicMock(spec=ModelOutput)
        state.output.message = ChatMessageAssistant(content="Done")
        state.output.completion = "Task complete"
        return state

    return agent_execute


@pytest.fixture
def mock_agent_factory(mock_agent):
    """Create a mock agent factory."""
    def factory():
        def create_with_prompts(**kwargs):
            return mock_agent
        return create_with_prompts
    return factory


# ============================================================================
# Test: Model Wrapping with Complete Context
# ============================================================================


@pytest.mark.asyncio
async def test_solver_wraps_model_with_complete_context(
    mock_metadata, mock_task_state, mock_agent_factory
):
    """Test that solver wraps model when SABER context is available."""
    with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain, \
         patch("saber.inspect_ai.agents.solver_factory._pull_injections_if_enabled", new_callable=AsyncMock), \
         patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model, \
         patch("saber.inspect_ai.agents.solver_factory.active_model_context_var") as mock_context_var, \
         patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_wrapper_class:

        # Setup domain context
        mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

        # Setup model
        mock_model = MagicMock()
        mock_active_model.return_value = mock_model

        # Create solver
        solver = create_saber_solver(
            agent_factory=mock_agent_factory,
            agent_name="test_agent",
            role_config=None,
        )

        # Execute solver
        mock_generate = AsyncMock()
        mock_task_state.metadata = mock_metadata
        await solver(mock_task_state, mock_generate)

        # Verify wrapper was created
        mock_wrapper_class.assert_called_once_with(
            base_model=mock_model,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )

        # Verify wrapped model was set as active
        assert mock_context_var.set.called


@pytest.mark.asyncio
async def test_solver_no_wrap_with_incomplete_context(
    mock_metadata_incomplete, mock_task_state, mock_agent_factory
):
    """Test that solver doesn't wrap model when context is incomplete."""
    with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain, \
         patch("saber.inspect_ai.agents.solver_factory._pull_injections_if_enabled", new_callable=AsyncMock), \
         patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model, \
         patch("saber.inspect_ai.agents.solver_factory.active_model_context_var") as mock_context_var, \
         patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_wrapper_class:

        # Setup model
        mock_model = MagicMock()
        mock_active_model.return_value = mock_model

        # Create solver
        solver = create_saber_solver(
            agent_factory=mock_agent_factory,
            agent_name="test_agent",
            role_config=None,
        )

        # Execute solver
        mock_generate = AsyncMock()
        mock_task_state.metadata = mock_metadata_incomplete
        await solver(mock_task_state, mock_generate)

        # Verify wrapper was NOT created
        mock_wrapper_class.assert_not_called()

        # Verify original model was set as active
        mock_context_var.set.assert_called()


# ============================================================================
# Test: Model Restoration
# ============================================================================


@pytest.mark.asyncio
async def test_solver_restores_original_model_after_execution(
    mock_metadata, mock_task_state, mock_agent_factory
):
    """Test that solver restores original model after agent execution."""
    with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain, \
         patch("saber.inspect_ai.agents.solver_factory._pull_injections_if_enabled", new_callable=AsyncMock), \
         patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model, \
         patch("saber.inspect_ai.agents.solver_factory.active_model_context_var") as mock_context_var, \
         patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper"):

        # Setup domain and model
        mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}
        original_model = MagicMock()
        mock_active_model.return_value = original_model

        # Create and execute solver
        solver = create_saber_solver(
            agent_factory=mock_agent_factory,
            agent_name="test_agent",
            role_config=None,
        )

        mock_generate = AsyncMock()
        mock_task_state.metadata = mock_metadata
        await solver(mock_task_state, mock_generate)

        # Verify model was restored (set called at least twice: once to set wrapped, once to restore)
        assert mock_context_var.set.call_count >= 2

        # Last call should restore original model
        last_call_arg = mock_context_var.set.call_args_list[-1][0][0]
        assert last_call_arg == original_model


@pytest.mark.asyncio
async def test_solver_restores_model_on_agent_exception(
    mock_metadata, mock_task_state
):
    """Test that solver restores model even if agent raises exception."""
    # Create an agent that raises
    async def failing_agent(state):
        raise ValueError("Agent failed")

    def failing_factory():
        def create_with_prompts(**kwargs):
            return failing_agent
        return create_with_prompts

    with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain, \
         patch("saber.inspect_ai.agents.solver_factory._pull_injections_if_enabled", new_callable=AsyncMock), \
         patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model, \
         patch("saber.inspect_ai.agents.solver_factory.active_model_context_var") as mock_context_var, \
         patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper"):

        # Setup
        mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}
        original_model = MagicMock()
        mock_active_model.return_value = original_model

        # Create solver
        solver = create_saber_solver(
            agent_factory=failing_factory,
            agent_name="test_agent",
            role_config=None,
        )

        # Execute should raise
        mock_generate = AsyncMock()
        mock_task_state.metadata = mock_metadata

        with pytest.raises(ValueError, match="Agent failed"):
            await solver(mock_task_state, mock_generate)

        # Verify model was still restored
        last_call_arg = mock_context_var.set.call_args_list[-1][0][0]
        assert last_call_arg == original_model


# ============================================================================
# Test: Role-Based Model Selection with Wrapper
# ============================================================================


@pytest.mark.asyncio
async def test_solver_wraps_role_specific_model(
    mock_task_state, mock_agent_factory
):
    """Test that solver wraps role-specific model correctly."""
    # Create metadata with role
    metadata_with_role = {
        MetadataKeys.SESSION_ID: "session_123",
        MetadataKeys.EPISODE_ID: "episode_456",
        MetadataKeys.SABER_DOMAIN_SLUG: "test_domain",
        MetadataKeys.TASK_ID: "task_789",
        MetadataKeys.SUB_TASK_ROLE: "attacker",
        MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
        MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        MetadataKeys.SUBMIT_PROMPT: "Test submit",
    }

    # Mock role config
    mock_role_config = MagicMock()
    mock_agent_config = MagicMock()
    mock_agent_config.model = "gpt-4"
    mock_role_config.get_config_for_role.return_value = mock_agent_config

    with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain, \
         patch("saber.inspect_ai.agents.solver_factory._pull_injections_if_enabled", new_callable=AsyncMock), \
         patch("saber.inspect_ai.agents.solver_factory.get_model") as mock_get_model, \
         patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model, \
         patch("saber.inspect_ai.agents.solver_factory.active_model_context_var") as mock_context_var, \
         patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_wrapper_class:

        # Setup
        mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}
        role_model = MagicMock()
        mock_get_model.return_value = role_model

        # Create solver with role config
        solver = create_saber_solver(
            agent_factory=mock_agent_factory,
            agent_name="test_agent",
            role_config=mock_role_config,
        )

        # Execute
        mock_generate = AsyncMock()
        mock_task_state.metadata = metadata_with_role
        await solver(mock_task_state, mock_generate)

        # Verify role-specific model was requested
        mock_get_model.assert_called_once_with("gpt-4")

        # Verify wrapper was created with role model
        mock_wrapper_class.assert_called_once()
        assert mock_wrapper_class.call_args[1]["base_model"] == role_model


# ============================================================================
# Test: Message Injection Integration
# ============================================================================


@pytest.mark.asyncio
async def test_solver_pulls_injections_before_execution(
    mock_metadata, mock_task_state, mock_agent_factory
):
    """Test that solver pulls injected messages before agent execution."""
    with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain, \
         patch("saber.inspect_ai.agents.solver_factory._pull_injections_if_enabled", new_callable=AsyncMock) as mock_pull, \
         patch("saber.inspect_ai.agents.solver_factory.active_model"), \
         patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"), \
         patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper"):

        # Setup
        mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

        # Create and execute solver
        solver = create_saber_solver(
            agent_factory=mock_agent_factory,
            agent_name="test_agent",
            role_config=None,
        )

        mock_generate = AsyncMock()
        mock_task_state.metadata = mock_metadata
        await solver(mock_task_state, mock_generate)

        # Verify injection pull was called
        mock_pull.assert_called_once()
        call_args = mock_pull.call_args
        assert call_args[0][0] == mock_task_state  # state
        assert call_args[0][1] == mock_metadata  # metadata
