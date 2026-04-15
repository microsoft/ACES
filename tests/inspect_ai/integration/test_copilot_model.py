"""Tests for Copilot model conversion functions.

Tests the four pure conversion functions used by the Copilot model provider:
- _messages_to_prompt: ChatMessage list → prompt string
- _tool_info_to_sdk_tool: ToolInfo → CopilotToolDef
- _sdk_response_to_model_output: SDK response → ModelOutput
- _extract_usage: SDK events → ModelUsage
"""

from types import SimpleNamespace

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall, ToolInfo
from inspect_ai.tool._tool_params import ToolParams

from saber.inspect_ai.integration.copilot_model import (
    CopilotToolDef,
    _extract_usage,
    _messages_to_prompt,
    _sdk_response_to_model_output,
    _tool_info_to_sdk_tool,
)


# ---------------------------------------------------------------------------
# _messages_to_prompt
# ---------------------------------------------------------------------------
class TestMessagesToPrompt:
    """Tests for _messages_to_prompt."""

    def test_single_system_message(self) -> None:
        messages = [ChatMessageSystem(content="You are a helpful assistant.")]
        result = _messages_to_prompt(messages)
        assert result == "[system]\nYou are a helpful assistant."

    def test_multi_turn(self) -> None:
        messages = [
            ChatMessageSystem(content="System prompt"),
            ChatMessageUser(content="Hello"),
            ChatMessageAssistant(content="Hi there"),
            ChatMessageUser(content="How are you?"),
        ]
        result = _messages_to_prompt(messages)
        expected = (
            "[system]\nSystem prompt\n\n"
            "[user]\nHello\n\n"
            "[assistant]\nHi there\n\n"
            "[user]\nHow are you?"
        )
        assert result == expected

    def test_tool_message(self) -> None:
        messages = [
            ChatMessageTool(content="result value", function="run_command"),
        ]
        result = _messages_to_prompt(messages)
        assert result == "[tool: run_command]\nresult value"

    def test_tool_message_no_function(self) -> None:
        messages = [
            ChatMessageTool(content="result value"),
        ]
        result = _messages_to_prompt(messages)
        assert result == "[tool]\nresult value"

    def test_assistant_with_tool_calls(self) -> None:
        tool_calls = [
            ToolCall(
                id="tc_1",
                function="get_weather",
                arguments={"city": "Seattle"},
            ),
        ]
        messages = [
            ChatMessageAssistant(
                content="Let me check the weather.", tool_calls=tool_calls
            ),
        ]
        result = _messages_to_prompt(messages)
        expected = (
            "[assistant]\nLet me check the weather.\n"
            '[tool_call: get_weather({"city":"Seattle"})]'
        )
        assert result == expected

    def test_assistant_with_multiple_tool_calls(self) -> None:
        tool_calls = [
            ToolCall(id="tc_1", function="func_a", arguments={"x": 1}),
            ToolCall(id="tc_2", function="func_b", arguments={"y": 2}),
        ]
        messages = [
            ChatMessageAssistant(content="Calling tools.", tool_calls=tool_calls),
        ]
        result = _messages_to_prompt(messages)
        assert '[tool_call: func_a({"x":1})]' in result
        assert '[tool_call: func_b({"y":2})]' in result

    def test_empty_list(self) -> None:
        assert _messages_to_prompt([]) == ""


# ---------------------------------------------------------------------------
# _tool_info_to_sdk_tool
# ---------------------------------------------------------------------------
class TestToolInfoToSdkTool:
    """Tests for _tool_info_to_sdk_tool."""

    def test_basic_tool(self) -> None:
        params = ToolParams(
            properties={
                "path": {
                    "type": "string",
                    "description": "File path",
                }
            },
            required=["path"],
        )
        tool_info = ToolInfo(
            name="read_file",
            description="Read a file from disk",
            parameters=params,
        )
        result = _tool_info_to_sdk_tool(tool_info)
        assert isinstance(result, CopilotToolDef)
        assert result.name == "read_file"
        assert result.description == "Read a file from disk"
        assert result.parameters["type"] == "object"
        assert "path" in result.parameters["properties"]
        assert result.parameters["required"] == ["path"]

    def test_tool_no_description(self) -> None:
        tool_info = ToolInfo(name="noop", description="")
        result = _tool_info_to_sdk_tool(tool_info)
        assert result.name == "noop"
        assert result.description == ""
        assert result.parameters["type"] == "object"

    def test_tool_def_is_frozen(self) -> None:
        tool_info = ToolInfo(name="t", description="d")
        result = _tool_info_to_sdk_tool(tool_info)
        with pytest.raises(AttributeError):
            result.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _sdk_response_to_model_output
# ---------------------------------------------------------------------------
class TestSdkResponseToModelOutput:
    """Tests for _sdk_response_to_model_output."""

    def test_none_response(self) -> None:
        output = _sdk_response_to_model_output(
            response_data=None,
            model_name="copilot/gpt-4",
            usage=None,
        )
        assert isinstance(output, ModelOutput)
        assert output.model == "copilot/gpt-4"
        assert output.stop_reason == "unknown"
        assert output.choices[0].message.content == ""

    def test_text_only(self) -> None:
        response = SimpleNamespace(
            content="Hello, world!",
            tool_requests=None,
        )
        output = _sdk_response_to_model_output(
            response_data=response,
            model_name="copilot/gpt-4",
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        assert output.stop_reason == "stop"
        assert output.choices[0].message.text == "Hello, world!"
        assert output.usage is not None
        assert output.usage.input_tokens == 10
        assert output.usage.output_tokens == 5

    def test_with_tool_calls(self) -> None:
        tool_req = SimpleNamespace(
            tool_call_id="tc_abc",
            name="run_command",
            arguments={"cmd": "ls -la"},
        )
        response = SimpleNamespace(
            content="Running command.",
            tool_requests=[tool_req],
        )
        output = _sdk_response_to_model_output(
            response_data=response,
            model_name="copilot/gpt-4",
            usage=None,
        )
        assert output.stop_reason == "tool_calls"
        msg = output.choices[0].message
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc.id == "tc_abc"
        assert tc.function == "run_command"
        assert tc.arguments == {"cmd": "ls -la"}

    def test_arguments_as_string(self) -> None:
        """JSON string arguments should be parsed to dict."""
        tool_req = SimpleNamespace(
            tool_call_id="tc_1",
            name="search",
            arguments='{"query": "test"}',
        )
        response = SimpleNamespace(
            content="",
            tool_requests=[tool_req],
        )
        output = _sdk_response_to_model_output(
            response_data=response,
            model_name="copilot/gpt-4",
            usage=None,
        )
        tc = output.choices[0].message.tool_calls[0]  # type: ignore[index]
        assert tc.arguments == {"query": "test"}

    def test_empty_content(self) -> None:
        response = SimpleNamespace(content="", tool_requests=[])
        output = _sdk_response_to_model_output(
            response_data=response,
            model_name="copilot/gpt-4",
            usage=None,
        )
        assert output.stop_reason == "stop"
        assert output.choices[0].message.text == ""

    def test_empty_tool_requests_is_stop(self) -> None:
        response = SimpleNamespace(content="Done.", tool_requests=[])
        output = _sdk_response_to_model_output(
            response_data=response,
            model_name="copilot/gpt-4",
            usage=None,
        )
        assert output.stop_reason == "stop"


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------
class TestExtractUsage:
    """Tests for _extract_usage."""

    def test_single_usage_event(self) -> None:
        events = [
            SimpleNamespace(
                type="usage",
                data=SimpleNamespace(input_tokens=100, output_tokens=50),
            ),
        ]
        usage = _extract_usage(events)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_multiple_events_sum(self) -> None:
        events = [
            SimpleNamespace(
                type="usage",
                data=SimpleNamespace(input_tokens=100, output_tokens=50),
            ),
            SimpleNamespace(
                type="other",
                data=SimpleNamespace(message="hello"),
            ),
            SimpleNamespace(
                type="usage",
                data=SimpleNamespace(input_tokens=200, output_tokens=75),
            ),
        ]
        usage = _extract_usage(events)
        assert usage is not None
        assert usage.input_tokens == 300
        assert usage.output_tokens == 125
        assert usage.total_tokens == 425

    def test_none_tokens(self) -> None:
        """None token values should be treated as 0."""
        events = [
            SimpleNamespace(
                type="usage",
                data=SimpleNamespace(input_tokens=None, output_tokens=None),
            ),
        ]
        usage = _extract_usage(events)
        assert usage is not None
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_no_usage_events(self) -> None:
        events = [
            SimpleNamespace(type="other", data=SimpleNamespace(message="hello")),
        ]
        usage = _extract_usage(events)
        assert usage is None

    def test_empty_events(self) -> None:
        assert _extract_usage([]) is None
