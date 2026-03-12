"""Tests for saber.agents.react_limit — graceful tool-call-limit agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCreateReactLimitAgent:
    """create_react_limit_agent returns a callable agent wrapper."""

    def test_returns_callable(self) -> None:
        """Factory returns a callable."""
        from saber.agents.react_limit import create_react_limit_agent

        agent = create_react_limit_agent(tool_call_limit=10)
        assert callable(agent)

    @pytest.mark.asyncio
    async def test_below_limit_generates_with_tools(self) -> None:
        """When count < limit, generate is called with full tool set."""
        from inspect_ai.agent import AgentState
        from inspect_ai.model import ChatMessageUser
        from inspect_ai.model._model_output import ModelOutput

        from saber.agents.react_limit import create_react_limit_agent

        agent = create_react_limit_agent(tool_call_limit=10)

        state = AgentState(messages=[ChatMessageUser(content="hello")])
        mock_tools = [MagicMock(), MagicMock()]

        fake_output = ModelOutput.from_content(
            model="test", content="response", stop_reason="stop"
        )

        with patch("saber.agents.react_limit.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            result = await agent(state, mock_tools)

        # generate called with tools (not empty)
        mock_model.generate.assert_awaited_once()
        call_args = mock_model.generate.call_args
        assert call_args[1].get("tools") is mock_tools  # keyword arg = tools

        # state updated correctly
        assert result.output is fake_output
        # assistant message appended
        assert result.messages[-1] is fake_output.message

    @pytest.mark.asyncio
    async def test_at_limit_generates_without_tools(self) -> None:
        """When count == limit, generate is called with tools=[]."""
        from inspect_ai.agent import AgentState
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.tool import ToolCall

        from saber.agents.react_limit import create_react_limit_agent

        limit = 2
        agent = create_react_limit_agent(tool_call_limit=limit)

        # Set up messages with exactly `limit` tool calls
        messages = [
            ChatMessageUser(content="do something"),
            ChatMessageAssistant(
                content="running",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                    ToolCall(id="c2", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="ok", tool_call_id="c1"),
            ChatMessageTool(content="ok", tool_call_id="c2"),
        ]
        state = AgentState(messages=messages)

        fake_output = ModelOutput.from_content(
            model="test", content="final answer", stop_reason="stop"
        )

        with patch("saber.agents.react_limit.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            result = await agent(state, [MagicMock()])

        # generate called with tools=[] (no tools)
        call_args = mock_model.generate.call_args
        assert call_args[1].get("tools") == []

        assert result.output is fake_output

    @pytest.mark.asyncio
    async def test_above_limit_generates_without_tools(self) -> None:
        """When count > limit, generate is called with tools=[]."""
        from inspect_ai.agent import AgentState
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.tool import ToolCall

        from saber.agents.react_limit import create_react_limit_agent

        limit = 1
        agent = create_react_limit_agent(tool_call_limit=limit)

        # Set up messages with more than `limit` tool calls
        messages = [
            ChatMessageUser(content="do something"),
            ChatMessageAssistant(
                content="step 1",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="ok", tool_call_id="c1"),
            ChatMessageAssistant(
                content="step 2",
                tool_calls=[
                    ToolCall(id="c2", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="ok", tool_call_id="c2"),
        ]
        state = AgentState(messages=messages)

        fake_output = ModelOutput.from_content(
            model="test", content="final answer", stop_reason="stop"
        )

        with patch("saber.agents.react_limit.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            result = await agent(state, [MagicMock()])

        call_args = mock_model.generate.call_args
        assert call_args[1].get("tools") == []
        assert result.output is fake_output

    @pytest.mark.asyncio
    async def test_patches_orphaned_tool_calls_at_limit(self) -> None:
        """When at limit, orphaned tool calls get dummy results."""
        from inspect_ai.agent import AgentState
        from inspect_ai.model import (
            ChatMessageAssistant,
            ChatMessageTool,
            ChatMessageUser,
        )
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.tool import ToolCall

        from saber.agents.react_limit import create_react_limit_agent

        agent = create_react_limit_agent(tool_call_limit=2)

        # c1 has a result, c2 is orphaned
        messages = [
            ChatMessageUser(content="go"),
            ChatMessageAssistant(
                content="running",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                    ToolCall(id="c2", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="ok", tool_call_id="c1"),
            # c2 is orphaned
        ]
        state = AgentState(messages=messages)

        fake_output = ModelOutput.from_content(
            model="test", content="answer", stop_reason="stop"
        )

        with patch("saber.agents.react_limit.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            await agent(state, [MagicMock()])

        # c2 should now have a dummy tool result
        tool_msgs = [m for m in state.messages if isinstance(m, ChatMessageTool)]
        tool_ids = {m.tool_call_id for m in tool_msgs}
        assert "c2" in tool_ids

    @pytest.mark.asyncio
    async def test_injects_limit_message_at_limit(self) -> None:
        """When at limit, a user message with TOOL_CALL_LIMIT_MESSAGE is injected."""
        from inspect_ai.agent import AgentState
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import TOOL_CALL_LIMIT_MESSAGE
        from saber.agents.react_limit import create_react_limit_agent

        agent = create_react_limit_agent(tool_call_limit=1)

        messages = [
            ChatMessageUser(content="go"),
            ChatMessageAssistant(
                content="running",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="ok", tool_call_id="c1"),
        ]
        state = AgentState(messages=messages)

        fake_output = ModelOutput.from_content(
            model="test", content="done", stop_reason="stop"
        )

        with patch("saber.agents.react_limit.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            await agent(state, [MagicMock()])

        # Find the injected limit message
        user_limit_msgs = [
            m
            for m in state.messages
            if isinstance(m, ChatMessageUser) and m.content == TOOL_CALL_LIMIT_MESSAGE
        ]
        assert len(user_limit_msgs) == 1

    @pytest.mark.asyncio
    async def test_output_has_no_tool_calls_at_limit(self) -> None:
        """After graceful generation, output should have no tool_calls."""
        from inspect_ai.agent import AgentState
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.tool import ToolCall

        from saber.agents.react_limit import create_react_limit_agent

        agent = create_react_limit_agent(tool_call_limit=1)

        messages = [
            ChatMessageUser(content="go"),
            ChatMessageAssistant(
                content="running",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="ok", tool_call_id="c1"),
        ]
        state = AgentState(messages=messages)

        fake_output = ModelOutput.from_content(
            model="test", content="my answer", stop_reason="stop"
        )

        with patch("saber.agents.react_limit.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            result = await agent(state, [MagicMock()])

        # The output should not have tool_calls (so react loop breaks)
        assert not result.output.message.tool_calls

    @pytest.mark.asyncio
    async def test_state_output_and_message_appended(self) -> None:
        """Both state.output and the assistant message are set correctly."""
        from inspect_ai.agent import AgentState
        from inspect_ai.model import ChatMessageUser
        from inspect_ai.model._model_output import ModelOutput

        from saber.agents.react_limit import create_react_limit_agent

        agent = create_react_limit_agent(tool_call_limit=100)

        state = AgentState(messages=[ChatMessageUser(content="hi")])

        fake_output = ModelOutput.from_content(
            model="test", content="hello", stop_reason="stop"
        )

        with patch("saber.agents.react_limit.get_model") as mock_get_model:
            mock_model = AsyncMock()
            mock_model.generate = AsyncMock(return_value=fake_output)
            mock_get_model.return_value = mock_model

            result = await agent(state, [])

        assert result.output is fake_output
        assert result.messages[-1] is fake_output.message
