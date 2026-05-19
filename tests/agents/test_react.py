# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
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
    """create_with_prompts requires max_steps as keyword-only parameter."""

    @patch("inspect_ai.agent.react")
    @patch("inspect_ai.agent.as_solver")
    def test_without_max_steps_raises_type_error(
        self, mock_as_solver: MagicMock, mock_react: MagicMock
    ) -> None:
        """Without max_steps keyword arg, TypeError is raised."""
        from saber.agents.registry.react import create_agent

        factory = create_agent()
        with pytest.raises(TypeError):
            factory(instruction_prompt="do X", assistant_prompt="you are Y")

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
