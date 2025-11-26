"""Tests for SABER React agent implementation."""

import pytest
from unittest.mock import Mock, patch, ANY

from saber.inspect_ai.agents.registry.react import create_agent, MODIFIED_CONTINUE_PROMPT


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

        # Call the factory with prompts
        instruction = "You are a security analyst"
        assistant = "I will help analyze security issues"
        submit = "Submit your findings"

        result = agent_factory(instruction, assistant, submit)

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

        # Check on_continue argument
        assert "on_continue" in call_args.kwargs
        assert call_args.kwargs["on_continue"] == MODIFIED_CONTINUE_PROMPT

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

        # Call the factory
        agent_factory("instruction", "assistant", "submit")

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

        # Test with various prompt combinations
        test_cases = [
            ("short", "prompt", "here"),
            ("Very long instruction prompt" * 10, "Another one", "Submit now"),
            ("", "", ""),  # Edge case: empty prompts
        ]

        for instruction, assistant, submit in test_cases:
            agent_factory(instruction, assistant, submit)

            call_kwargs = mock_react.call_args.kwargs
            prompt = call_kwargs["prompt"]
            assert prompt.instructions == instruction
            assert prompt.assistant_prompt == assistant
            assert prompt.submit_prompt == submit


class TestModifiedContinuePrompt:
    """Test cases for MODIFIED_CONTINUE_PROMPT constant."""

    def test_continue_prompt_contains_submit_placeholder(self):
        """Test that the continue prompt contains the {submit} placeholder."""
        assert "{submit}" in MODIFIED_CONTINUE_PROMPT

    def test_continue_prompt_mentions_completion(self):
        """Test that the continue prompt mentions task completion."""
        assert "completed" in MODIFIED_CONTINUE_PROMPT.lower()

    def test_continue_prompt_mentions_giving_up(self):
        """Test that the continue prompt mentions giving up."""
        assert "given up" in MODIFIED_CONTINUE_PROMPT.lower()
