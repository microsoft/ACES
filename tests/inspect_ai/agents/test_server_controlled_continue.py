"""Tests for server-controlled continue functionality in SABER React agent.

Tests the AgentContinue callback that enables server-side transcript injection
for dual-agent scenarios (e.g., AI red team attacking blue team).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import asyncio

from saber.inspect_ai.agents.registry.react import (
    create_agent,
    _server_controlled_on_continue,
)


class TestServerControlledOnContinue:
    """Test cases for _server_controlled_on_continue callback."""

    @pytest.mark.asyncio
    async def test_calls_wait_and_sync_on_wrapper(self):
        """Test that callback calls wait_and_sync_transcript on the model wrapper."""
        from inspect_ai.agent._agent import AgentState
        from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
        from saber.inspect_ai.integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper

        # Create mock wrapper
        mock_wrapper = MagicMock(spec=WebSocketTranscriptSyncingModelWrapper)
        synced_messages = [
            ChatMessageUser(content="Initial user message"),
            ChatMessageAssistant(content="Assistant response"),
            ChatMessageUser(content="Server-injected continue prompt"),
        ]
        mock_wrapper.wait_and_sync_transcript = AsyncMock(return_value=synced_messages)

        # Create initial state
        initial_messages = [
            ChatMessageUser(content="Initial user message"),
            ChatMessageAssistant(content="Assistant response"),
        ]
        state = AgentState(messages=initial_messages)

        # Patch active_model to return our mock wrapper
        with patch("saber.inspect_ai.agents.registry.react.active_model", return_value=mock_wrapper):
            result = await _server_controlled_on_continue(state)

        # Verify wait_and_sync_transcript was called
        mock_wrapper.wait_and_sync_transcript.assert_called_once()

        # Verify result is an AgentState with synced messages
        assert isinstance(result, AgentState)
        assert len(result.messages) == 3
        assert result.messages[-1].content == "Server-injected continue prompt"

    @pytest.mark.asyncio
    async def test_returns_original_state_when_not_wrapper(self):
        """Test that callback returns original state when model is not WebSocket wrapper."""
        from inspect_ai.agent._agent import AgentState
        from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

        # Create mock regular model (not a wrapper)
        mock_model = MagicMock()

        initial_messages = [
            ChatMessageUser(content="User message"),
            ChatMessageAssistant(content="Assistant response"),
        ]
        state = AgentState(messages=initial_messages)

        with patch("saber.inspect_ai.agents.registry.react.active_model", return_value=mock_model):
            result = await _server_controlled_on_continue(state)

        # Should return original state unchanged
        assert result is state

    @pytest.mark.asyncio
    async def test_returns_original_state_when_model_is_none(self):
        """Test that callback returns original state when active_model returns None."""
        from inspect_ai.agent._agent import AgentState
        from inspect_ai.model import ChatMessageUser

        state = AgentState(messages=[ChatMessageUser(content="Test")])

        with patch("saber.inspect_ai.agents.registry.react.active_model", return_value=None):
            result = await _server_controlled_on_continue(state)

        assert result is state

    @pytest.mark.asyncio
    async def test_preserves_output_from_original_state(self):
        """Test that callback preserves the output from original state."""
        from inspect_ai.agent._agent import AgentState
        from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ModelOutput, ChatCompletionChoice
        from saber.inspect_ai.integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper

        mock_wrapper = MagicMock(spec=WebSocketTranscriptSyncingModelWrapper)
        synced_messages = [ChatMessageUser(content="Synced")]
        mock_wrapper.wait_and_sync_transcript = AsyncMock(return_value=synced_messages)

        # Create state with explicit output
        state = AgentState(messages=[ChatMessageUser(content="Original")])
        mock_output = ModelOutput(
            model="test-model",
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content="Original output"),
                stop_reason="stop"
            )]
        )
        state.output = mock_output

        with patch("saber.inspect_ai.agents.registry.react.active_model", return_value=mock_wrapper):
            result = await _server_controlled_on_continue(state)

        # Output should be preserved
        assert result.output.model == "test-model"

    @pytest.mark.asyncio
    async def test_handles_sync_error_gracefully(self):
        """Test that callback handles errors from sync and returns original state."""
        from inspect_ai.agent._agent import AgentState
        from inspect_ai.model import ChatMessageUser
        from saber.inspect_ai.integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper

        mock_wrapper = MagicMock(spec=WebSocketTranscriptSyncingModelWrapper)
        mock_wrapper.wait_and_sync_transcript = AsyncMock(side_effect=RuntimeError("Sync failed"))

        state = AgentState(messages=[ChatMessageUser(content="Test")])

        with patch("saber.inspect_ai.agents.registry.react.active_model", return_value=mock_wrapper):
            result = await _server_controlled_on_continue(state)

        # Should return original state on error
        assert result is state


class TestCreateAgentWithTranscriptConfig:
    """Test cases for create_agent with transcript_config parameter."""

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_uses_server_continue_when_pull_enabled(self, mock_saber_tools, mock_react):
        """Test that server-controlled continue is used when pull.enabled=True."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Call with transcript_config that has pull.enabled=True
        transcript_config = {
            "websocket": {
                "pull": {"enabled": True}
            }
        }

        agent_factory(
            instruction_prompt="instruction",
            assistant_prompt="assistant",
            submit_prompt="submit",
            continue_prompt="continue string (should be ignored)",
            transcript_config=transcript_config,
        )

        # Verify on_continue is a callable (the _server_controlled_on_continue function)
        call_kwargs = mock_react.call_args.kwargs
        assert callable(call_kwargs["on_continue"])
        assert call_kwargs["on_continue"] == _server_controlled_on_continue

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_uses_string_continue_when_pull_disabled(self, mock_saber_tools, mock_react):
        """Test that string continue prompt is used when pull.enabled=False."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Call with transcript_config that has pull.enabled=False
        transcript_config = {
            "websocket": {
                "pull": {"enabled": False}
            }
        }

        continue_prompt = "Keep going with the task"
        agent_factory(
            instruction_prompt="instruction",
            assistant_prompt="assistant",
            submit_prompt="submit",
            continue_prompt=continue_prompt,
            transcript_config=transcript_config,
        )

        # Verify on_continue is the string
        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["on_continue"] == continue_prompt

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_uses_string_continue_when_no_transcript_config(self, mock_saber_tools, mock_react):
        """Test that string continue prompt is used when transcript_config is None."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        continue_prompt = "Keep going"
        agent_factory(
            instruction_prompt="instruction",
            assistant_prompt="assistant",
            submit_prompt="submit",
            continue_prompt=continue_prompt,
            transcript_config=None,  # No config
        )

        # Verify on_continue is the string
        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["on_continue"] == continue_prompt

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_uses_string_continue_when_websocket_not_in_config(self, mock_saber_tools, mock_react):
        """Test that string continue is used when websocket key is missing."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        continue_prompt = "Continue"
        agent_factory(
            instruction_prompt="instruction",
            assistant_prompt="assistant",
            submit_prompt="submit",
            continue_prompt=continue_prompt,
            transcript_config={},  # Empty config
        )

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["on_continue"] == continue_prompt

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_backward_compatible_without_transcript_config(self, mock_saber_tools, mock_react):
        """Test backward compatibility when called without transcript_config."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        continue_prompt = "Keep working"
        # Call with positional args only (old style, no transcript_config)
        agent_factory("instruction", "assistant", "submit", continue_prompt)

        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["on_continue"] == continue_prompt


class TestDualAgentScenario:
    """Test cases for dual-agent (blue team vs red team) scenarios."""

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_blue_team_uses_server_continue(self, mock_saber_tools, mock_react):
        """Test that blue team (pull.enabled=True) uses server-controlled continue."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Blue team config from baseline.yaml
        blue_config = {
            "websocket": {
                "pull": {"enabled": True}
            }
        }

        agent_factory(
            instruction_prompt="You are a guardrailed assistant",
            assistant_prompt="I help with database queries",
            submit_prompt="Submit your work",
            continue_prompt="Continue without submitting",  # This should be ignored
            transcript_config=blue_config,
        )

        call_kwargs = mock_react.call_args.kwargs
        # Blue team should use server-controlled continue
        assert callable(call_kwargs["on_continue"])

    @patch("saber.inspect_ai.agents.registry.react.react")
    @patch("saber.inspect_ai.agents.registry.react.saber_tools")
    def test_red_team_uses_client_continue(self, mock_saber_tools, mock_react):
        """Test that red team (pull.enabled=False or default) uses client-controlled continue."""
        mock_saber_tools.return_value = Mock()
        mock_react.return_value = Mock()

        agent_factory = create_agent()

        # Red team config - no explicit pull config (defaults or pull disabled)
        red_config = {
            "websocket": {
                "push": {"enabled": True}
            }
        }

        red_continue = "Continue your attack. Call submit() when successful."
        agent_factory(
            instruction_prompt="You are a red team attacker",
            assistant_prompt="I will attempt prompt injection",
            submit_prompt="Submit when attack succeeds",
            continue_prompt=red_continue,
            transcript_config=red_config,
        )

        call_kwargs = mock_react.call_args.kwargs
        # Red team should use string continue (client-controlled)
        assert call_kwargs["on_continue"] == red_continue
