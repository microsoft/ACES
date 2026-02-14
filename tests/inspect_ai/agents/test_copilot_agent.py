"""Tests for SABER Copilot agent implementation.

TDD tests for the Copilot solver that uses GitHub Copilot CLI.
"""

from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest

# Import the tools module to ensure it's loaded before patching
import saber.inspect_ai.agents.registry.tools  # noqa: F401


# Module-level fixture to mock sandbox() from inspect_ai.util
@pytest.fixture(autouse=True)
def mock_sandbox_function():
    """Auto-mock sandbox() for all tests in this module."""
    mock_sb = Mock()
    mock_sb._sandbox = Mock()
    mock_sb._sandbox._mcp_client = Mock()
    with patch("saber.inspect_ai.agents.registry.tools.sandbox", return_value=mock_sb):
        with patch("saber.inspect_ai.agents.registry.copilot.solver.sandbox", return_value=mock_sb):
            yield mock_sb


class TestCopilotAgentCreation:
    """Test cases for create_agent factory function."""

    def test_create_agent_returns_callable(self):
        """Test that create_agent returns a callable."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        agent_factory = create_agent()
        assert callable(agent_factory)

    def test_create_agent_with_kwargs(self):
        """Test that create_agent accepts kwargs."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        agent_factory = create_agent(max_turns=10, model="gpt-4o")
        assert callable(agent_factory)

    @patch("saber.inspect_ai.agents.registry.copilot.solver.copilot_solver")
    def test_create_agent_calls_copilot_solver_with_prompts(self, mock_copilot_solver):
        """Test that the returned factory calls copilot_solver with prompts."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        mock_solver = Mock()
        mock_copilot_solver.return_value = mock_solver

        agent_factory = create_agent()

        instruction = "You are a security analyst"
        assistant = "I will help analyze security issues"
        submit = "Submit your findings"
        continue_prompt = "Please proceed to the next step"

        result = agent_factory(instruction, assistant, submit, continue_prompt)

        mock_copilot_solver.assert_called_once_with(
            instruction_prompt=instruction,
            assistant_prompt=assistant,
            submit_prompt=submit,
            continue_prompt=continue_prompt,
            transcript_config=None,
            submit=None,
            skill_directories=None,
            agent_persona=None,
        )
        assert result == mock_solver

    @patch("saber.inspect_ai.agents.registry.copilot.solver.copilot_solver")
    def test_create_agent_passes_kwargs_to_solver(self, mock_copilot_solver):
        """Test that create_agent passes kwargs to copilot_solver."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        mock_copilot_solver.return_value = Mock()

        agent_factory = create_agent(max_turns=20, model="gpt-4o")
        agent_factory("instruction", "assistant", "submit", "continue")

        call_kwargs = mock_copilot_solver.call_args.kwargs
        assert call_kwargs.get("max_turns") == 20
        assert call_kwargs.get("model") == "gpt-4o"

    @patch("saber.inspect_ai.agents.registry.copilot.solver.copilot_solver")
    def test_create_agent_with_transcript_config(self, mock_copilot_solver):
        """Test that transcript config is passed through."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        mock_copilot_solver.return_value = Mock()
        agent_factory = create_agent()

        transcript_config = {"websocket": {"push": {"enabled": True}}}
        agent_factory("instruction", "assistant", "submit", "continue", transcript_config=transcript_config)

        call_kwargs = mock_copilot_solver.call_args.kwargs
        assert call_kwargs.get("transcript_config") == transcript_config

    @patch("saber.inspect_ai.agents.registry.copilot.solver.copilot_solver")
    def test_create_agent_with_submit_disabled(self, mock_copilot_solver):
        """Test that submit=False disables the submit tool."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        mock_copilot_solver.return_value = Mock()
        agent_factory = create_agent()

        agent_factory("instruction", "assistant", "submit", "continue", submit=False)

        call_kwargs = mock_copilot_solver.call_args.kwargs
        assert call_kwargs.get("submit") is False


class TestCopilotSolver:
    """Test cases for copilot_solver implementation."""

    @pytest.fixture
    def mock_copilot_client(self):
        """Create a mock CopilotClient wrapper."""
        client = AsyncMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        return client

    @pytest.fixture
    def mock_copilot_session(self):
        """Create a mock CopilotSession."""
        session = AsyncMock()
        session.send = AsyncMock(return_value="msg-123")
        session.send_and_wait = AsyncMock(return_value=None)
        session.destroy = AsyncMock()
        session.on = Mock()
        return session

    @pytest.fixture
    def mock_task_state(self):
        """Create a mock TaskState."""
        state = Mock()
        state.metadata = {
            "task_id": "test-task",
            "sample_id": "sample-1",
        }
        state.messages = []
        # Mock the store with proper _store attribute for logging
        state.store = Mock()
        state.store._store = {}  # Mock _store for logging
        state.store._store = {}  # Empty dict to be iterable
        state.store.get = Mock(return_value=None)
        state.store.set = Mock()
        return state

    def test_copilot_solver_is_solver(self):
        """Test that copilot_solver returns a Solver."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        solver = copilot_solver(
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="test",
        )

        # Should be callable (Solver is a callable)
        assert callable(solver)

    @pytest.mark.asyncio
    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    async def test_copilot_solver_creates_client_and_session(
        self,
        mock_create_submit,
        mock_convert_tools,
        mock_get_mcp_tools,
        mock_client_wrapper_class,
        mock_session_tracker_class,
        mock_task_state,
        mock_copilot_client,
        mock_copilot_session,
    ):
        """Test that solver creates Copilot client and session."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup mocks
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = True  # End loop immediately
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        mock_client_wrapper_class.return_value = mock_copilot_client
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []
        mock_create_submit.return_value = Mock(name="submit_answer")

        solver = copilot_solver(
            instruction_prompt="Analyze the system",
            assistant_prompt="I am a security expert",
            submit_prompt="Submit your findings",
            continue_prompt="Continue analysis",
        )

        await solver(mock_task_state)

        # Verify client lifecycle
        mock_client_wrapper_class.assert_called_once()
        mock_copilot_client.start.assert_called_once()
        mock_copilot_client.create_session.assert_called_once()
        mock_copilot_client.stop.assert_called_once()

    @pytest.mark.asyncio
    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    async def test_copilot_solver_sends_instruction_prompt_first(
        self,
        mock_create_submit,
        mock_convert_tools,
        mock_get_mcp_tools,
        mock_client_wrapper_class,
        mock_session_tracker_class,
        mock_task_state,
        mock_copilot_client,
        mock_copilot_session,
    ):
        """Test that solver sends instruction prompt on first turn."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock - submit after first turn
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()

        # Use a list to track call count
        turn_count = [0]
        def submitted_getter():
            turn_count[0] += 1
            return turn_count[0] > 1
        type(mock_session_tracker).submitted = PropertyMock(side_effect=submitted_getter)

        mock_session_tracker_class.return_value = mock_session_tracker

        mock_client_wrapper_class.return_value = mock_copilot_client
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []
        mock_create_submit.return_value = Mock(name="submit_answer")

        solver = copilot_solver(
            instruction_prompt="Analyze the system",
            assistant_prompt="I am a security expert",
            submit_prompt="Submit your findings",
            continue_prompt="Continue analysis",
        )

        await solver(mock_task_state)

        # First call should be with instruction prompt
        first_call = mock_copilot_session.send.call_args_list[0]
        assert "Analyze the system" in first_call[0][0].get("prompt", "")

    @pytest.mark.asyncio
    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    async def test_copilot_solver_registers_mcp_tools(
        self,
        mock_create_submit,
        mock_convert_tools,
        mock_get_mcp_tools,
        mock_client_wrapper_class,
        mock_session_tracker_class,
        mock_task_state,
        mock_copilot_client,
        mock_copilot_session,
    ):
        """Test that solver registers MCP tools with Copilot session."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = True  # End loop immediately
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        mock_client_wrapper_class.return_value = mock_copilot_client
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        # Mock MCP tools
        mock_mcp_tool = Mock()
        mock_mcp_tool.name = "bash"
        mock_get_mcp_tools.return_value = [mock_mcp_tool]

        mock_copilot_tool = Mock()
        mock_copilot_tool.name = "bash"
        mock_convert_tools.return_value = [mock_copilot_tool]

        mock_submit_tool = Mock()
        mock_submit_tool.name = "submit_answer"
        mock_create_submit.return_value = mock_submit_tool

        mock_task_state.store.get = Mock(return_value=True)  # Immediately submitted

        solver = copilot_solver(
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="test",
        )

        await solver(mock_task_state)

        # Verify tools were passed to create_session
        session_call = mock_copilot_client.create_session.call_args
        tools = session_call[0][0].get("tools", [])
        assert len(tools) >= 2  # MCP tool + submit tool

    @pytest.mark.asyncio
    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    async def test_copilot_solver_stops_on_max_turns(
        self,
        mock_create_submit,
        mock_convert_tools,
        mock_get_mcp_tools,
        mock_client_wrapper_class,
        mock_session_tracker_class,
        mock_task_state,
        mock_copilot_client,
        mock_copilot_session,
    ):
        """Test that solver stops after max_turns."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock - never submitted so hits max_turns
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = False  # Never submit - should hit max_turns
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        mock_client_wrapper_class.return_value = mock_copilot_client
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []
        mock_create_submit.return_value = Mock(name="submit_answer")

        solver = copilot_solver(
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="test",
            max_turns=3,
        )

        await solver(mock_task_state)

        # Should have been called exactly max_turns times
        assert mock_copilot_session.send.call_count == 3

    @pytest.mark.asyncio
    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    async def test_copilot_solver_cleans_up_on_error(
        self,
        mock_create_submit,
        mock_convert_tools,
        mock_get_mcp_tools,
        mock_client_wrapper_class,
        mock_session_tracker_class,
        mock_task_state,
        mock_copilot_client,
        mock_copilot_session,
    ):
        """Test that solver cleans up client even on error."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = True  # End loop immediately
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        mock_client_wrapper_class.return_value = mock_copilot_client
        mock_copilot_client.create_session = AsyncMock(side_effect=RuntimeError("Session failed"))
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []

        solver = copilot_solver(
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="test",
        )

        with pytest.raises(RuntimeError, match="Session failed"):
            await solver(mock_task_state)

        # Client should still be stopped
        mock_copilot_client.stop.assert_called_once()


class TestCopilotSolverConfiguration:
    """Test cases for copilot_solver configuration options."""

    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    @pytest.mark.asyncio
    async def test_copilot_solver_uses_configured_model(
        self,
        mock_create_submit,
        mock_convert_tools,
        mock_get_mcp_tools,
        mock_client_wrapper_class,
        mock_session_tracker_class,
    ):
        """Test that solver uses the configured model."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = True  # End loop immediately
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        mock_client = AsyncMock()
        mock_session = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_wrapper_class.return_value = mock_client
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []
        mock_create_submit.return_value = Mock()

        state = Mock()
        state.store = Mock()
        state.store._store = {}  # Mock _store for logging
        state.store.get = Mock(return_value=True)
        state.messages = []

        solver = copilot_solver(
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="test",
            model="claude-sonnet-4",
        )

        await solver(state)

        # Verify model was passed to session
        session_config = mock_client.create_session.call_args[0][0]
        assert session_config.get("model") == "claude-sonnet-4"

    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    @pytest.mark.asyncio
    async def test_copilot_solver_with_submit_disabled(
        self,
        mock_create_submit,
        mock_convert_tools,
        mock_get_mcp_tools,
        mock_client_wrapper_class,
        mock_session_tracker_class,
    ):
        """Test that submit tool is not added when submit=False."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = True  # End loop immediately
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        mock_client = AsyncMock()
        mock_session = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_wrapper_class.return_value = mock_client
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []

        state = Mock()
        state.store = Mock()
        state.store._store = {}  # Mock _store for logging
        state.store.get = Mock(return_value=False)  # Never submitted
        state.messages = []

        solver = copilot_solver(
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="test",
            submit=False,
            max_turns=1,
        )

        await solver(state)

        # create_submit_tool should not be called
        mock_create_submit.assert_not_called()


class TestCopilotSolverTranscriptHandling:
    """Test cases for conversation transcript handling."""

    @pytest.fixture
    def mock_session_event(self):
        """Create a mock session event."""
        event = Mock()
        event.type = Mock()
        event.type.value = "assistant.message"
        event.data = Mock()
        event.data.content = "I will analyze the system"
        return event

    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    @pytest.mark.asyncio
    async def test_copilot_solver_collects_conversation(
        self,
        mock_create_submit,
        mock_convert_tools,
        mock_get_mcp_tools,
        mock_client_wrapper_class,
        mock_session_tracker_class,
        mock_session_event,
    ):
        """Test that solver collects conversation messages."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = True  # End loop immediately
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        mock_client = AsyncMock()
        mock_session = AsyncMock()
        mock_session.send_and_wait = AsyncMock(return_value=mock_session_event)
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_wrapper_class.return_value = mock_client
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []
        mock_create_submit.return_value = Mock()

        state = Mock()
        state.store = Mock()
        state.store._store = {}  # Mock _store for logging
        call_count = [0]
        def get_side_effect(k):
            if k == "submitted":
                call_count[0] += 1
                return call_count[0] > 1
            return None
        state.store.get = Mock(side_effect=get_side_effect)
        state.store.set = Mock()
        state.messages = []

        solver = copilot_solver(
            instruction_prompt="Analyze",
            assistant_prompt="I am an assistant",
            submit_prompt="Submit",
            continue_prompt="Continue",
        )

        await solver(state)

        # State messages should have been updated
        assert len(state.messages) >= 1


class TestCopilotSolverBYOK:
    """Test cases for BYOK (Bring Your Own Key) provider configuration."""

    @patch("saber.inspect_ai.agents.registry.copilot.solver.copilot_solver")
    def test_create_agent_with_azure_provider(self, mock_copilot_solver):
        """Test that Azure BYOK provider configuration is passed through."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        mock_copilot_solver.return_value = Mock()

        agent_factory = create_agent(
            model="gpt-4o",
            provider_type="azure",
            provider_base_url="https://my-resource.openai.azure.com",
            provider_api_key="test-api-key",
            provider_api_version="2024-02-15-preview",
        )
        agent_factory("instruction", "assistant", "submit", "continue")

        call_kwargs = mock_copilot_solver.call_args.kwargs
        assert call_kwargs.get("provider_type") == "azure"
        assert call_kwargs.get("provider_base_url") == "https://my-resource.openai.azure.com"
        assert call_kwargs.get("provider_api_key") == "test-api-key"
        assert call_kwargs.get("provider_api_version") == "2024-02-15-preview"

    @patch("saber.inspect_ai.agents.registry.copilot.solver.copilot_solver")
    def test_create_agent_with_openai_provider(self, mock_copilot_solver):
        """Test that OpenAI BYOK provider configuration is passed through."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        mock_copilot_solver.return_value = Mock()

        agent_factory = create_agent(
            model="gpt-4o",
            provider_type="openai",
            provider_base_url="https://api.openai.com/v1",
            provider_api_key="sk-test-key",
        )
        agent_factory("instruction", "assistant", "submit", "continue")

        call_kwargs = mock_copilot_solver.call_args.kwargs
        assert call_kwargs.get("provider_type") == "openai"
        assert call_kwargs.get("provider_base_url") == "https://api.openai.com/v1"
        assert call_kwargs.get("provider_api_key") == "sk-test-key"

    @patch("saber.inspect_ai.agents.registry.copilot.solver.copilot_solver")
    def test_create_agent_with_anthropic_provider(self, mock_copilot_solver):
        """Test that Anthropic BYOK provider configuration is passed through."""
        from saber.inspect_ai.agents.registry.copilot import create_agent

        mock_copilot_solver.return_value = Mock()

        agent_factory = create_agent(
            model="claude-sonnet-4",
            provider_type="anthropic",
            provider_base_url="https://api.anthropic.com",
            provider_api_key="sk-ant-test-key",
        )
        agent_factory("instruction", "assistant", "submit", "continue")

        call_kwargs = mock_copilot_solver.call_args.kwargs
        assert call_kwargs.get("provider_type") == "anthropic"
        assert call_kwargs.get("provider_base_url") == "https://api.anthropic.com"
        assert call_kwargs.get("provider_api_key") == "sk-ant-test-key"

    @pytest.mark.asyncio
    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    async def test_copilot_solver_includes_azure_provider_in_session_config(
        self,
        mock_client_wrapper_class,
        mock_get_mcp_tools,
        mock_convert_tools,
        mock_create_submit,
        mock_session_tracker_class,
    ):
        """Test that BYOK provider config is included in session creation."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = True  # End loop immediately
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        # Setup mocks
        mock_client = AsyncMock()
        mock_session = AsyncMock()

        # Create a proper response mock with data.content as a string
        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "I have completed the task"

        mock_session.send_and_wait = AsyncMock(return_value=mock_response)
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_wrapper_class.return_value = mock_client
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []
        mock_create_submit.return_value = Mock()

        state = Mock()
        state.store = Mock()
        state.store._store = {}  # Mock _store for logging
        call_count = [0]
        def get_side_effect(k):
            if k == "submitted":
                call_count[0] += 1
                return call_count[0] > 1  # Second call returns True
            return None
        state.store.get = Mock(side_effect=get_side_effect)
        state.store.set = Mock()
        state.messages = []

        solver = copilot_solver(
            instruction_prompt="Analyze",
            assistant_prompt="I am an assistant",
            submit_prompt="Submit",
            continue_prompt="Continue",
            provider_type="azure",
            provider_base_url="https://my-resource.openai.azure.com",
            provider_api_key="test-key",
            provider_api_version="2024-02-15-preview",
        )

        await solver(state)

        # Verify create_session was called with provider config
        session_config = mock_client.create_session.call_args[0][0]
        assert "provider" in session_config
        assert session_config["provider"]["type"] == "azure"
        assert session_config["provider"]["base_url"] == "https://my-resource.openai.azure.com"
        assert session_config["provider"]["api_key"] == "test-key"
        assert session_config["provider"]["azure"]["api_version"] == "2024-02-15-preview"

    @pytest.mark.asyncio
    @patch("saber.inspect_ai.agents.registry.copilot.solver.SessionTracker")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.create_submit_tool")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.convert_mcp_tools_to_copilot")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.get_saber_mcp_tools")
    @patch("saber.inspect_ai.agents.registry.copilot.solver.CopilotClientWrapper")
    async def test_copilot_solver_without_byok_has_no_provider(
        self,
        mock_client_wrapper_class,
        mock_get_mcp_tools,
        mock_convert_tools,
        mock_create_submit,
        mock_session_tracker_class,
    ):
        """Test that session config has no provider when BYOK is not configured."""
        from saber.inspect_ai.agents.registry.copilot import copilot_solver

        # Setup SessionTracker mock
        mock_session_tracker = Mock()
        mock_session_tracker.on_event = Mock()
        mock_session_tracker.wait_for_idle = AsyncMock()
        mock_session_tracker.submitted = True  # End loop immediately
        mock_session_tracker.submit_answer = None
        mock_session_tracker.session_id = None
        mock_session_tracker.watcher_ctx.stop = AsyncMock()
        mock_session_tracker_class.return_value = mock_session_tracker

        # Setup mocks
        mock_client = AsyncMock()
        mock_session = AsyncMock()

        # Create a proper response mock with data.content as a string
        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Done"

        mock_session.send_and_wait = AsyncMock(return_value=mock_response)
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client_wrapper_class.return_value = mock_client
        mock_get_mcp_tools.return_value = []
        mock_convert_tools.return_value = []
        mock_create_submit.return_value = Mock()

        state = Mock()
        state.store = Mock()
        state.store._store = {}  # Mock _store for logging
        call_count = [0]
        def get_side_effect(k):
            if k == "submitted":
                call_count[0] += 1
                return call_count[0] > 1
            return None
        state.store.get = Mock(side_effect=get_side_effect)
        state.store.set = Mock()
        state.messages = []

        solver = copilot_solver(
            instruction_prompt="Analyze",
            assistant_prompt="I am an assistant",
            submit_prompt="Submit",
            continue_prompt="Continue",
        )

        await solver(state)

        # Verify create_session was called WITHOUT provider config
        session_config = mock_client.create_session.call_args[0][0]
        assert "provider" not in session_config
