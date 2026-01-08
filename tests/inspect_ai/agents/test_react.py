"""Tests for SABER React agent implementation."""

import pytest
from unittest.mock import Mock, patch, ANY

from saber.inspect_ai.agents.registry.react import create_agent


class TestCreateAgent:
    """Test cases for create_agent factory function."""

    def test_create_agent_returns_callable(self):
        """Test that create_agent returns a callable."""
        agent_factory = create_agent()
        assert callable(agent_factory)

    def test_create_agent_with_kwargs(self):
        """Test that create_agent accepts kwargs."""
        agent_factory = create_agent(max_iterations=10, verbose=True)
        assert callable(agent_factory)

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_create_agent_calls_react_with_prompts(self, mock_saber_tools, mock_react):
        """Test that the returned factory calls react with prompts."""
        # Setup mocks
        mock_tools_instance = Mock()
        mock_saber_tools.return_value = mock_tools_instance
        mock_react_agent = Mock()
        mock_react.return_value = mock_react_agent

        # Create agent factory
        agent_factory = create_agent()

        # Call the factory with prompts (now requires 4 prompts including continue)
        instruction = "You are a security analyst"
        assistant = "I will help analyze security issues"
        submit = "Submit your findings"
        continue_prompt = "Please proceed to the next step"

        result = agent_factory(instruction, assistant, submit, continue_prompt)

        # Verify react was called with correct parameters
        mock_react.assert_called_once()
        call_args = mock_react.call_args

        # Check prompt argument
        assert "prompt" in call_args.kwargs
        prompt = call_args.kwargs["prompt"]
        assert prompt.instructions == instruction
        assert prompt.assistant_prompt == assistant
        assert prompt.submit_prompt == submit
        assert prompt.handoff_prompt is None

        # Check tools argument
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == [mock_tools_instance]

        # Check on_continue argument uses the provided continue_prompt
        assert "on_continue" in call_args.kwargs
        assert call_args.kwargs["on_continue"] == continue_prompt

        # Verify saber_tools was called
        mock_saber_tools.assert_called_once()

        # Verify result is the react agent
        assert result == mock_react_agent

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_create_agent_passes_kwargs_to_react(self, mock_saber_tools, mock_react):
        """Test that create_agent passes kwargs to react."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        # Create agent factory with custom kwargs
        agent_factory = create_agent(max_iterations=20, temperature=0.5)

        # Call the factory with all 4 prompts
        agent_factory("instruction", "assistant", "submit", "continue")

        # Verify kwargs were passed to react
        call_kwargs = mock_react.call_args.kwargs
        assert "max_iterations" in call_kwargs
        assert call_kwargs["max_iterations"] == 20
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.5

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_create_agent_with_different_prompts(self, mock_saber_tools, mock_react):
        """Test that different prompts are correctly passed."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Test with various prompt combinations (now with 4 prompts)
        test_cases = [
            ("short", "prompt", "here", "continue"),
            ("Very long instruction prompt" * 10, "Another one", "Submit now", "Keep going"),
            ("", "", "", ""),  # Edge case: empty prompts
        ]

        for instruction, assistant, submit, continue_prompt in test_cases:
            agent_factory(instruction, assistant, submit, continue_prompt)

            call_kwargs = mock_react.call_args.kwargs
            prompt = call_kwargs["prompt"]
            assert prompt.instructions == instruction
            assert prompt.assistant_prompt == assistant
            assert prompt.submit_prompt == submit
            assert call_kwargs["on_continue"] == continue_prompt


class TestContinuePromptBehavior:
    """Test cases for continue_prompt functionality."""

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_blue_team_continue_prompt_no_submit(self, mock_saber_tools, mock_react):
        """Test that blue team agents can use continue prompts without submit instructions."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Blue team continue prompt should NOT mention submit
        blue_continue = "Please proceed to the next step. Do NOT call submit()."
        agent_factory("instruction", "assistant", "submit", blue_continue)

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["on_continue"] == blue_continue
        assert "submit()" not in call_kwargs["on_continue"] or "NOT" in call_kwargs["on_continue"]

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_red_team_continue_prompt_with_submit(self, mock_saber_tools, mock_react):
        """Test that red team agents can use continue prompts with submit instructions."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Red team continue prompt should mention submit
        red_continue = "Please proceed. If done, call submit() with your answer."
        agent_factory("instruction", "assistant", "submit", red_continue)

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["on_continue"] == red_continue
        assert "submit()" in call_kwargs["on_continue"]

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_continue_prompt_is_configurable_per_task(self, mock_saber_tools, mock_react):
        """Test that each task can have its own continue prompt."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Different tasks with different continue prompts
        tasks = [
            ("task1", "Continue working on the database queries."),
            ("task2", "Keep analyzing the security logs."),
            ("task3", "Proceed with your attack strategy. Submit when successful."),
        ]

        for task_name, continue_prompt in tasks:
            agent_factory(f"instruction for {task_name}", "assistant", "submit", continue_prompt)

            call_kwargs = mock_react.call_args.kwargs
            assert call_kwargs["on_continue"] == continue_prompt


class TestSubmitToolConfiguration:
    """Test cases for submit tool enable/disable functionality."""

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_submit_enabled_by_default(self, mock_saber_tools, mock_react):
        """Test that submit tool is enabled by default (submit=None or not passed)."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Call without submit parameter - should default to enabled
        agent_factory("instruction", "assistant", "submit", "continue")

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["submit"] is True
        assert call_kwargs["prompt"].submit_prompt == "submit"

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_submit_enabled_explicitly(self, mock_saber_tools, mock_react):
        """Test that submit tool can be explicitly enabled."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Explicitly enable submit
        agent_factory("instruction", "assistant", "submit", "continue", submit=True)

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["submit"] is True
        assert call_kwargs["prompt"].submit_prompt == "submit"

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_submit_disabled_for_blue_team(self, mock_saber_tools, mock_react):
        """Test that submit tool can be disabled for continuous monitoring agents."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Disable submit for blue team (continuous monitoring)
        agent_factory("instruction", "assistant", "submit", "continue", submit=False)

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["submit"] is False
        # When submit is disabled, submit_prompt should be None
        assert call_kwargs["prompt"].submit_prompt is None

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_submit_none_treated_as_enabled(self, mock_saber_tools, mock_react):
        """Test that submit=None is treated as enabled (default behavior)."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Pass submit=None explicitly
        agent_factory("instruction", "assistant", "submit", "continue", submit=None)

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["submit"] is True
        assert call_kwargs["prompt"].submit_prompt == "submit"

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_submit_disabled_uses_callback_for_on_continue(self, mock_saber_tools, mock_react):
        """Test that submit=False uses a callback function for on_continue instead of string.

        When submit is disabled, inspect_ai's react() requires a callback for on_continue,
        not a string. A string would cause immediate termination when no tool calls are made.
        """
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Disable submit - should use callback instead of string
        agent_factory("instruction", "assistant", "submit", "my continue prompt", submit=False)

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["submit"] is False
        # on_continue should be a callable (async function), not a string
        assert callable(call_kwargs["on_continue"])
        assert call_kwargs["on_continue"].__name__ == "_no_submit_continue"

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_submit_enabled_uses_string_for_on_continue(self, mock_saber_tools, mock_react):
        """Test that submit=True uses the continue prompt string directly."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Enable submit - should use string directly
        agent_factory("instruction", "assistant", "submit", "my continue prompt", submit=True)

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["submit"] is True
        # on_continue should be the string directly
        assert call_kwargs["on_continue"] == "my continue prompt"
        assert isinstance(call_kwargs["on_continue"], str)


class TestNoSubmitContinueCallback:
    """Test cases for the _no_submit_continue callback behavior."""

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    @pytest.mark.asyncio
    async def test_no_submit_continue_returns_true_when_tools_called(self, mock_saber_tools, mock_react):
        """Test that _no_submit_continue returns True when tools were called.

        When tools are called, the model should continue naturally to reason
        about the output without a continue prompt being injected.
        """
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        continue_prompt = "Keep monitoring the SIEM for security events."
        agent_factory("instruction", "assistant", "submit", continue_prompt, submit=False)

        # Get the callback that was passed to react
        call_kwargs = mock_react.call_args.kwargs
        callback = call_kwargs["on_continue"]

        # Create a mock AgentState with tool calls
        mock_state = Mock()
        mock_state.messages = [Mock(), Mock()]
        mock_state.output = Mock()
        mock_state.output.message = Mock()
        mock_state.output.message.tool_calls = [Mock()]  # Has tool calls

        # Call the callback - should return True (natural continuation)
        result = await callback(mock_state)
        assert result is True

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    @pytest.mark.asyncio
    async def test_no_submit_continue_returns_prompt_when_no_tools_called(self, mock_saber_tools, mock_react):
        """Test that _no_submit_continue returns continue prompt when no tools called.

        When no tools are called, the model needs a nudge to keep monitoring
        instead of stopping.
        """
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        continue_prompt = "Keep monitoring the SIEM for security events."
        agent_factory("instruction", "assistant", "submit", continue_prompt, submit=False)

        # Get the callback that was passed to react
        call_kwargs = mock_react.call_args.kwargs
        callback = call_kwargs["on_continue"]

        # Create a mock AgentState without tool calls
        mock_state = Mock()
        mock_state.messages = [Mock(), Mock()]
        mock_state.output = Mock()
        mock_state.output.message = Mock()
        mock_state.output.message.tool_calls = []  # No tool calls

        # Call the callback - should return the continue prompt
        result = await callback(mock_state)
        assert result == continue_prompt

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    @pytest.mark.asyncio
    async def test_no_submit_continue_captures_correct_prompt(self, mock_saber_tools, mock_react):
        """Test that each agent captures its own continue_prompt in the closure."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        # Create multiple agents with different prompts
        prompts_and_callbacks = []
        for i in range(3):
            agent_factory = create_agent()
            prompt = f"Continue prompt {i}"
            agent_factory("instruction", "assistant", "submit", prompt, submit=False)
            callback = mock_react.call_args.kwargs["on_continue"]
            prompts_and_callbacks.append((prompt, callback))

        # Verify each callback returns its own prompt (when no tools called)
        mock_state = Mock()
        mock_state.messages = []
        mock_state.output = Mock()
        mock_state.output.message = Mock()
        mock_state.output.message.tool_calls = []  # No tool calls

        for expected_prompt, callback in prompts_and_callbacks:
            result = await callback(mock_state)
            assert result == expected_prompt
