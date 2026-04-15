"""Tests for Copilot model conversion functions and CopilotModelAPI.

Tests the pure conversion functions and the CopilotModelAPI class:
- _messages_to_prompt: ChatMessage list → prompt string
- _tool_info_to_sdk_tool: ToolInfo → CopilotToolDef
- _sdk_response_to_model_output: SDK response → ModelOutput
- _extract_usage: SDK events → ModelUsage
- CopilotModelAPI: ModelAPI subclass with singleton client management
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
    CopilotModelAPI,
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


# ---------------------------------------------------------------------------
# CopilotModelAPI – fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_copilot_state():
    """Reset class-level shared state between tests."""
    CopilotModelAPI._client = None
    CopilotModelAPI._client_lock = None
    CopilotModelAPI._client_refcount = 0
    CopilotModelAPI._github_token = None
    yield
    CopilotModelAPI._client = None
    CopilotModelAPI._client_lock = None
    CopilotModelAPI._client_refcount = 0
    CopilotModelAPI._github_token = None


# ---------------------------------------------------------------------------
# CopilotModelAPI.__init__
# ---------------------------------------------------------------------------
class TestCopilotModelAPIInit:
    """Tests for CopilotModelAPI constructor."""

    def test_init_with_github_token_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        api = CopilotModelAPI(model_name="gpt-4o")
        assert api._timeout == 120
        assert CopilotModelAPI._github_token == "test-token"
        assert CopilotModelAPI._client_refcount == 1

    def test_init_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        api = CopilotModelAPI(model_name="gpt-4o", api_key="my-key")
        assert CopilotModelAPI._github_token == "my-key"
        assert api._timeout == 120

    def test_init_custom_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        api = CopilotModelAPI(model_name="gpt-4o", timeout="60")
        assert api._timeout == 60

    def test_init_without_token_or_gh_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("shutil.which", return_value=None):
            with pytest.raises(Exception, match="Copilot model requires"):
                CopilotModelAPI(model_name="gpt-4o")

    def test_init_with_gh_cli_no_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gh CLI present but no GITHUB_TOKEN should succeed with token=None."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("shutil.which", return_value="/usr/bin/gh"):
            api = CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._github_token is None
        assert api._timeout == 120

    def test_init_increments_refcount(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o")
        CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._client_refcount == 2

    def test_init_api_key_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "env-token")
        CopilotModelAPI(model_name="gpt-4o", api_key="key-token")
        assert CopilotModelAPI._github_token == "key-token"


# ---------------------------------------------------------------------------
# CopilotModelAPI._get_or_create_client
# ---------------------------------------------------------------------------
class TestCopilotModelAPIGetOrCreateClient:
    """Tests for the singleton client factory."""

    async def test_creates_client_on_first_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o")

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        mock_copilot_client_cls = MagicMock(return_value=mock_client_instance)
        mock_subprocess_config_cls = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "copilot": MagicMock(CopilotClient=mock_copilot_client_cls),
                    "copilot.client": MagicMock(
                        SubprocessConfig=mock_subprocess_config_cls
                    ),
                },
            ),
        ):
            client = await CopilotModelAPI._get_or_create_client()
            assert client is mock_client_instance
            assert CopilotModelAPI._client is mock_client_instance

    async def test_returns_existing_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o")

        sentinel = object()
        CopilotModelAPI._client = sentinel

        client = await CopilotModelAPI._get_or_create_client()
        assert client is sentinel


# ---------------------------------------------------------------------------
# CopilotModelAPI.generate
# ---------------------------------------------------------------------------
class TestCopilotModelAPIGenerate:
    """Tests for the generate method."""

    def _make_api(self, monkeypatch: pytest.MonkeyPatch) -> CopilotModelAPI:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        return CopilotModelAPI(model_name="gpt-4o")

    async def test_generate_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = self._make_api(monkeypatch)

        mock_session = AsyncMock()
        mock_session.session_id = "sess-1"
        mock_session.send_and_wait = AsyncMock(
            return_value=SimpleNamespace(
                data=SimpleNamespace(content="Hello!", tool_requests=None)
            )
        )
        mock_session.get_messages = AsyncMock(
            return_value=[
                SimpleNamespace(
                    data=SimpleNamespace(input_tokens=10, output_tokens=5)
                ),
            ]
        )

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()

        CopilotModelAPI._client = mock_client

        # Mock the SDK imports used inside generate
        mock_permission_handler = MagicMock()
        mock_permission_handler.approve_all = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(),
                "copilot.session": MagicMock(
                    PermissionHandler=mock_permission_handler
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=[ChatMessageUser(content="Hi")],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output.model == "gpt-4o"
        assert output.choices[0].message.text == "Hello!"
        assert output.usage is not None
        assert output.usage.input_tokens == 10
        mock_client.delete_session.assert_awaited_once_with("sess-1")

    async def test_generate_timeout_returns_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = self._make_api(monkeypatch)

        mock_session = AsyncMock()
        mock_session.session_id = "sess-2"
        mock_session.send_and_wait = AsyncMock(side_effect=TimeoutError)
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()

        CopilotModelAPI._client = mock_client

        mock_permission_handler = MagicMock()
        mock_permission_handler.approve_all = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(),
                "copilot.session": MagicMock(
                    PermissionHandler=mock_permission_handler
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=[ChatMessageUser(content="Hi")],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output.stop_reason == "unknown"
        assert output.choices[0].message.content == ""
        mock_client.delete_session.assert_awaited_once_with("sess-2")

    async def test_generate_session_cleanup_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = self._make_api(monkeypatch)

        mock_session = AsyncMock()
        mock_session.session_id = "sess-3"
        mock_session.send_and_wait = AsyncMock(
            side_effect=RuntimeError("SDK error")
        )

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()

        CopilotModelAPI._client = mock_client

        mock_permission_handler = MagicMock()
        mock_permission_handler.approve_all = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(),
                "copilot.session": MagicMock(
                    PermissionHandler=mock_permission_handler
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            with pytest.raises(RuntimeError, match="SDK error"):
                await api.generate(
                    input=[ChatMessageUser(content="Hi")],
                    tools=[],
                    tool_choice="auto",
                    config=GenerateConfig(),
                )

        # Session cleanup should still happen
        mock_client.delete_session.assert_awaited_once_with("sess-3")

    async def test_generate_with_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = self._make_api(monkeypatch)

        mock_session = AsyncMock()
        mock_session.session_id = "sess-4"
        mock_session.send_and_wait = AsyncMock(
            return_value=SimpleNamespace(
                data=SimpleNamespace(
                    content="Calling tool.",
                    tool_requests=[
                        SimpleNamespace(
                            tool_call_id="tc_1",
                            name="read_file",
                            arguments={"path": "/tmp/f"},
                        )
                    ],
                )
            )
        )
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()

        CopilotModelAPI._client = mock_client

        mock_permission_handler = MagicMock()
        mock_permission_handler.approve_all = MagicMock()

        # Create a mock CopilotSdkTool class
        mock_sdk_tool_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(),
                "copilot.session": MagicMock(
                    PermissionHandler=mock_permission_handler
                ),
                "copilot.tools": MagicMock(Tool=mock_sdk_tool_cls),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig
            from inspect_ai.tool import ToolInfo

            tool = ToolInfo(name="read_file", description="Read a file")
            output = await api.generate(
                input=[ChatMessageUser(content="Read it")],
                tools=[tool],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output.stop_reason == "tool_calls"
        tc = output.choices[0].message.tool_calls[0]
        assert tc.function == "read_file"

    async def test_generate_delete_session_failure_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If delete_session fails, generate should still return normally."""
        api = self._make_api(monkeypatch)

        mock_session = AsyncMock()
        mock_session.session_id = "sess-5"
        mock_session.send_and_wait = AsyncMock(
            return_value=SimpleNamespace(
                data=SimpleNamespace(content="OK", tool_requests=None)
            )
        )
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock(
            side_effect=RuntimeError("cleanup fail")
        )

        CopilotModelAPI._client = mock_client

        mock_permission_handler = MagicMock()
        mock_permission_handler.approve_all = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(),
                "copilot.session": MagicMock(
                    PermissionHandler=mock_permission_handler
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=[ChatMessageUser(content="Hi")],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        # Should succeed despite delete_session error
        assert output.choices[0].message.text == "OK"


# ---------------------------------------------------------------------------
# CopilotModelAPI.aclose
# ---------------------------------------------------------------------------
class TestCopilotModelAPIClose:
    """Tests for aclose lifecycle management."""

    async def test_aclose_decrements_refcount(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api1 = CopilotModelAPI(model_name="gpt-4o")
        CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._client_refcount == 2

        await api1.aclose()
        assert CopilotModelAPI._client_refcount == 1

    async def test_aclose_last_instance_closes_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api = CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._client_refcount == 1

        mock_client = AsyncMock()
        mock_client.__aexit__ = AsyncMock(return_value=False)
        CopilotModelAPI._client = mock_client

        await api.aclose()
        assert CopilotModelAPI._client_refcount == 0
        assert CopilotModelAPI._client is None
        mock_client.__aexit__.assert_awaited_once()

    async def test_aclose_client_exit_error_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api = CopilotModelAPI(model_name="gpt-4o")

        mock_client = AsyncMock()
        mock_client.__aexit__ = AsyncMock(side_effect=RuntimeError("boom"))
        CopilotModelAPI._client = mock_client

        # Should not raise
        await api.aclose()
        assert CopilotModelAPI._client is None
        assert CopilotModelAPI._client_refcount == 0

    async def test_aclose_no_client_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api = CopilotModelAPI(model_name="gpt-4o")
        CopilotModelAPI._client = None

        # Should not raise when client is None
        await api.aclose()
        assert CopilotModelAPI._client_refcount == 0
