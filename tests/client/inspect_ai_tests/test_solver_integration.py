"""
Tests for Phase 3: Integration in Solver Wrapper

Tests that create_saber_solver() properly wraps the generate() function to push
transcripts to the server after each agent iteration.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from inspect_ai import Task
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.solver import TaskState, Generate

from saber.inspect_ai.agents.solver_factory import create_saber_solver
from saber.inspect_ai.integration.transcript_sync import _push_transcript
from saber.models.constants import MetadataKeys


@pytest.fixture
def mock_episode_metadata():
    """Episode context metadata."""
    return {
        MetadataKeys.SESSION_ID: "test-session-123",
        MetadataKeys.EPISODE_ID: "test-episode-456",
        MetadataKeys.TASK_ID: "test-task-789",
        MetadataKeys.SABER_DOMAIN_SLUG: "test-domain",
        MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
        MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        MetadataKeys.SUBMIT_PROMPT: "Test submit",
    }


@pytest.fixture
def mock_task_state(mock_episode_metadata):
    """Create a mock TaskState with episode metadata."""
    state = Mock(spec=TaskState)
    state.metadata = mock_episode_metadata
    state.messages = [
        ChatMessageSystem(content="System message"),
        ChatMessageUser(content="User message"),
    ]
    state.output = Mock()
    state.output.completion = "Test completion"
    state.store = {}
    return state


@pytest.fixture
def mock_generate():
    """Mock generate function that simulates model output."""
    async def generate(state: TaskState) -> TaskState:
        # Simulate agent adding an assistant message
        state.messages.append(
            ChatMessageAssistant(content="Agent response")
        )
        return state

    return generate


@pytest.mark.asyncio
async def test_solver_wraps_generate_and_pushes_transcript(
    mock_task_state, mock_generate, mock_episode_metadata
):
    """
    Test that the solver wrapper intercepts generate() calls and pushes transcripts.

    This is the core Phase 3 integration test:
    1. Create a SABER solver with transcript sync enabled
    2. Call the solver with a generate function
    3. Verify that _push_transcript was called with correct parameters
    4. Verify the generate function was still executed
    """
    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript") as mock_push, \
         patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:

        mock_push.return_value = None  # async function returns None

        # Mock the active domain to return REST URL
        mock_get_domain.return_value = {
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
        }

        # Create a simple agent factory that returns an agent function
        def agent_factory():
            def create_with_prompts(**prompts):
                async def agent(state: TaskState) -> TaskState:
                    # Agent uses generate internally
                    return await mock_generate(state)
                return agent
            return create_with_prompts

        # Create the solver
        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=agent_factory,
            role_config=None,
        )

        # Execute the solver
        result = await solver(mock_task_state, mock_generate)

        # Verify generate was called (agent executed)
        assert len(result.messages) == 3  # System + User + Assistant
        assert isinstance(result.messages[-1], ChatMessageAssistant)

        # Verify _push_transcript was called
        # It should be called after the generate() call completes
        assert mock_push.called, "Expected _push_transcript to be called"

        # Verify it was called with the correct episode context
        call_args = mock_push.call_args
        assert call_args is not None

        # Check that session_id, episode_id were passed
        assert call_args[1]["session_id"] == "test-session-123"
        assert call_args[1]["episode_id"] == "test-episode-456"
        assert call_args[1]["rest_url"] == "http://localhost:8000"

        # Check that messages were passed (should include the new assistant message)
        # The state parameter should be the result state
        state_arg = call_args[1]["state"]
        assert len(state_arg.messages) >= 3  # At least System + User + Assistant


@pytest.mark.asyncio
async def test_solver_handles_push_failure_gracefully(
    mock_task_state, mock_generate
):
    """
    Test that solver continues execution even if transcript push fails.

    The solver should:
    1. Catch exceptions from _push_transcript
    2. Log the error
    3. Continue with agent execution
    4. Return the result successfully
    """
    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript") as mock_push:
        # Simulate push failure
        mock_push.side_effect = Exception("Network error")

        def agent_factory():
            def create_with_prompts(**prompts):
                async def agent(state: TaskState) -> TaskState:
                    return await mock_generate(state)
                return agent
            return create_with_prompts

        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=agent_factory,
            role_config=None,
        )

        # Should not raise even though push failed
        result = await solver(mock_task_state, mock_generate)

        # Verify agent still executed successfully
        assert len(result.messages) == 3
        assert isinstance(result.messages[-1], ChatMessageAssistant)


@pytest.mark.asyncio
async def test_solver_skips_push_when_no_episode_metadata(
    mock_task_state, mock_generate
):
    """
    Test that solver skips transcript push when episode metadata is missing.

    When session_id or episode_id is not in metadata:
    1. Solver should detect this
    2. Skip calling _push_transcript
    3. Continue with normal agent execution
    """
    # Remove episode metadata
    mock_task_state.metadata = {
        MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
        MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        MetadataKeys.SUBMIT_PROMPT: "Test submit",
    }

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript") as mock_push:
        def agent_factory():
            def create_with_prompts(**prompts):
                async def agent(state: TaskState) -> TaskState:
                    return await mock_generate(state)
                return agent
            return create_with_prompts

        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=agent_factory,
            role_config=None,
        )

        result = await solver(mock_task_state, mock_generate)

        # Verify agent executed
        assert len(result.messages) == 3

        # Verify push was NOT called (no episode context)
        assert not mock_push.called, "Should skip push when episode metadata missing"


@pytest.mark.asyncio
async def test_solver_pushes_transcript_with_all_message_types(
    mock_episode_metadata, mock_generate
):
    """
    Test that solver correctly serializes all message types when pushing transcript.

    Verifies that the wrapper handles:
    - ChatMessageSystem
    - ChatMessageUser
    - ChatMessageAssistant
    - ChatMessageTool
    - Messages with content, tool_calls, etc.
    """
    from inspect_ai.model import ChatMessageTool, ToolCall, ToolInfo

    # Create state with diverse message types
    state = Mock(spec=TaskState)
    state.metadata = mock_episode_metadata
    state.messages = [
        ChatMessageSystem(content="System prompt"),
        ChatMessageUser(content="User question"),
        ChatMessageAssistant(
            content="I'll use a tool",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    function="test_tool",
                    arguments={"arg": "value"},
                    type="function",
                )
            ],
        ),
        ChatMessageTool(
            content="Tool result",
            tool_call_id="call_123",
        ),
        ChatMessageAssistant(content="Final response"),
    ]
    state.output = Mock()
    state.output.completion = "Test"
    state.store = {}

    with patch("saber.inspect_ai.integration.transcript_sync._push_transcript") as mock_push, \
         patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:

        mock_push.return_value = None

        # Mock the active domain to return REST URL
        mock_get_domain.return_value = {
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
        }

        def agent_factory():
            def create_with_prompts(**prompts):
                async def agent(s: TaskState) -> TaskState:
                    return s  # Return as-is
                return agent
            return create_with_prompts

        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=agent_factory,
            role_config=None,
        )

        await solver(state, mock_generate)

        # Verify push was called with all messages
        assert mock_push.called
        call_args = mock_push.call_args

        # Verify the state parameter contains all messages
        state_arg = call_args[1]["state"]

        # Should have all 5 message types
        assert len(state_arg.messages) == 5

        # Verify message types are preserved in serialization
        # (The actual serialization is tested in test_transcript_push.py)


@pytest.mark.asyncio
async def test_solver_pulls_injections_before_agent_execution(mock_task_state, mock_generate):
    """
    Test that the solver pulls injected messages before executing the agent (Phase 4).

    This test verifies:
    1. Solver calls pull_injected_messages() before agent execution
    2. Injected messages are added to state.messages
    3. Agent receives the injected messages
    """
    injected_message = ChatMessageUser(content="[RED TEAM] Injected command")

    async def mock_pull_injections(state, session_id, episode_id, rest_url):
        """Mock pull that injects a message."""
        state.messages.append(injected_message)

    with patch("saber.inspect_ai.agents.solver_factory.pull_injected_messages", new=mock_pull_injections) as mock_pull, \
         patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:

        # Mock the active domain to return REST URL
        mock_domain = Mock()
        mock_domain.rest_url = "http://localhost:8000"
        mock_get_domain.return_value = mock_domain

        # Track agent input
        agent_received_messages = []

        def agent_factory():
            def create_with_prompts(**prompts):
                async def agent(state: TaskState) -> TaskState:
                    # Capture messages the agent sees
                    agent_received_messages.extend(state.messages)
                    return await mock_generate(state)
                return agent
            return create_with_prompts

        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=agent_factory,
            role_config=None,
        )

        # Initial state has 2 messages
        assert len(mock_task_state.messages) == 2

        result = await solver(mock_task_state, mock_generate)

        # Verify injection was added before agent execution
        assert len(agent_received_messages) == 3  # Original 2 + injected 1
        assert injected_message in agent_received_messages
        assert "[RED TEAM]" in agent_received_messages[2].content


@pytest.mark.asyncio
async def test_solver_skips_pull_when_no_episode_context(mock_generate):
    """Test that pull is skipped for non-SABER tasks (no episode context)."""
    # State without episode metadata
    state = Mock(spec=TaskState)
    state.metadata = {
        # Missing SESSION_ID, EPISODE_ID, DOMAIN_SLUG
        MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
        MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        MetadataKeys.SUBMIT_PROMPT: "Test submit",
    }
    state.messages = [ChatMessageUser(content="User message")]
    state.output = Mock()
    state.output.completion = "Test completion"
    state.store = {}

    with patch("saber.inspect_ai.agents.solver_factory.pull_injected_messages") as mock_pull:
        def agent_factory():
            def create_with_prompts(**prompts):
                async def agent(s: TaskState) -> TaskState:
                    return await mock_generate(s)
                return agent
            return create_with_prompts

        solver = create_saber_solver(
            agent_name="test_agent",
            agent_factory=agent_factory,
            role_config=None,
        )

        await solver(state, mock_generate)

        # Verify pull was NOT called (no episode context)
        mock_pull.assert_not_called()
