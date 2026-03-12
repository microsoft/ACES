"""Tests for saber.agents.registry.react — React agent factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestReactCreateAgent:
    """create_agent returns a factory with expected behavior."""

    def test_create_agent_returns_callable(self) -> None:
        """create_agent returns a callable factory."""
        from saber.agents.registry.react import create_agent

        factory = create_agent()
        assert callable(factory)


class TestReactCreateWithPrompts:
    """create_with_prompts respects max_steps parameter."""

    @patch("inspect_ai.agent.react")
    @patch("inspect_ai.agent.as_solver")
    def test_without_max_steps_model_is_none(
        self, mock_as_solver: MagicMock, mock_react: MagicMock
    ) -> None:
        """Without max_steps, react() is called with model=None."""
        from saber.agents.registry.react import create_agent

        factory = create_agent()
        factory(instruction_prompt="do X", assistant_prompt="you are Y")

        mock_react.assert_called_once()
        call_kwargs = mock_react.call_args[1]
        assert call_kwargs.get("model") is None

    @patch("inspect_ai.agent.react")
    @patch("inspect_ai.agent.as_solver")
    def test_with_max_steps_model_is_agent(
        self, mock_as_solver: MagicMock, mock_react: MagicMock
    ) -> None:
        """With max_steps=25, react() is called with a model agent (not None)."""
        from saber.agents.registry.react import create_agent

        factory = create_agent()
        factory(instruction_prompt="do X", max_steps=25)

        mock_react.assert_called_once()
        call_kwargs = mock_react.call_args[1]
        assert call_kwargs.get("model") is not None

    @patch("inspect_ai.agent.react")
    @patch("inspect_ai.agent.as_solver")
    def test_with_max_steps_zero_model_is_none(
        self, mock_as_solver: MagicMock, mock_react: MagicMock
    ) -> None:
        """With max_steps=0, react() is called with model=None."""
        from saber.agents.registry.react import create_agent

        factory = create_agent()
        factory(instruction_prompt="do X", max_steps=0)

        mock_react.assert_called_once()
        call_kwargs = mock_react.call_args[1]
        assert call_kwargs.get("model") is None

    @patch("inspect_ai.agent.react")
    @patch("inspect_ai.agent.as_solver")
    def test_with_max_steps_negative_model_is_none(
        self, mock_as_solver: MagicMock, mock_react: MagicMock
    ) -> None:
        """With max_steps=-5, react() is called with model=None."""
        from saber.agents.registry.react import create_agent

        factory = create_agent()
        factory(instruction_prompt="do X", max_steps=-5)

        mock_react.assert_called_once()
        call_kwargs = mock_react.call_args[1]
        assert call_kwargs.get("model") is None
