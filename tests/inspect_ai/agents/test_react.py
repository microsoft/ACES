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
