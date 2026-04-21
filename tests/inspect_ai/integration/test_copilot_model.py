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
    _conversation_key,
    _convert_tools_for_sdk,
    _extract_first_user_content,
    _extract_system_content,
    _extract_usage,
    _format_tool_results,
    _make_tool_handler,
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
    CopilotModelAPI._pool = None
    CopilotModelAPI._pool_size = 4
    CopilotModelAPI._all_clients = []
    CopilotModelAPI._instance_count = 0
    CopilotModelAPI._github_token = None
    CopilotModelAPI._shutting_down = False
    CopilotModelAPI._conversations = {}
    yield
    CopilotModelAPI._pool = None
    CopilotModelAPI._pool_size = 4
    CopilotModelAPI._all_clients = []
    CopilotModelAPI._instance_count = 0
    CopilotModelAPI._github_token = None
    CopilotModelAPI._shutting_down = False
    CopilotModelAPI._conversations = {}


# ---------------------------------------------------------------------------
# CopilotModelAPI.__init__
# ---------------------------------------------------------------------------
class TestCopilotModelAPIInit:
    """Tests for CopilotModelAPI constructor."""

    def test_init_with_github_token_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        api = CopilotModelAPI(model_name="gpt-4o")
        assert api._timeout == 300
        assert CopilotModelAPI._github_token == "test-token"
        assert CopilotModelAPI._instance_count == 1

    def test_init_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        api = CopilotModelAPI(model_name="gpt-4o", api_key="my-key")
        assert CopilotModelAPI._github_token == "my-key"
        assert api._timeout == 300

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
        assert api._timeout == 300

    def test_init_increments_instance_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o")
        CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._instance_count == 2

    def test_init_custom_pool_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o", pool_size="8")
        assert CopilotModelAPI._pool_size == 8

    def test_init_api_key_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "env-token")
        CopilotModelAPI(model_name="gpt-4o", api_key="key-token")
        assert CopilotModelAPI._github_token == "key-token"


# ---------------------------------------------------------------------------
# CopilotModelAPI._create_single_client
# ---------------------------------------------------------------------------
class TestCopilotModelAPICreateSingleClient:
    """Tests for the per-client factory."""

    async def test_creates_and_starts_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o")

        mock_client_instance = AsyncMock()
        mock_copilot_client_cls = MagicMock(return_value=mock_client_instance)
        mock_subprocess_config_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(CopilotClient=mock_copilot_client_cls),
                "copilot.client": MagicMock(
                    SubprocessConfig=mock_subprocess_config_cls
                ),
            },
        ):
            client = await CopilotModelAPI._create_single_client()
            assert client is mock_client_instance
            mock_client_instance.start.assert_awaited_once()

    async def test_creates_client_with_github_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o")

        mock_client_instance = AsyncMock()
        mock_copilot_client_cls = MagicMock(return_value=mock_client_instance)
        mock_subprocess_config_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(CopilotClient=mock_copilot_client_cls),
                "copilot.client": MagicMock(
                    SubprocessConfig=mock_subprocess_config_cls
                ),
            },
        ):
            await CopilotModelAPI._create_single_client()
            mock_subprocess_config_cls.assert_called_once_with(github_token="tok")

    async def test_creates_client_without_token(self) -> None:
        CopilotModelAPI._github_token = None
        mock_client_instance = AsyncMock()
        mock_copilot_client_cls = MagicMock(return_value=mock_client_instance)
        mock_subprocess_config_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(CopilotClient=mock_copilot_client_cls),
                "copilot.client": MagicMock(
                    SubprocessConfig=mock_subprocess_config_cls
                ),
            },
        ):
            await CopilotModelAPI._create_single_client()
            mock_subprocess_config_cls.assert_called_once_with()


# ---------------------------------------------------------------------------
# CopilotModelAPI.generate
# ---------------------------------------------------------------------------
class TestCopilotModelAPIGenerate:
    """Tests for the generate method."""

    def _make_api(self, monkeypatch: pytest.MonkeyPatch) -> CopilotModelAPI:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        return CopilotModelAPI(model_name="gpt-4o")

    def _prepopulate_pool(self, mock_client: object) -> None:
        """Pre-populate the class-level pool with a single mock client."""
        import asyncio

        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = [mock_client]

    def _make_session_with_response(
        self,
        session_id: str,
        content: str,
        tool_requests: list[object] | None = None,
        usage_events: list[object] | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        """Create a mock session that fires the response via on-event callback."""
        mock_session_event_type = MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = session_id

        captured_callback = None

        def on_side_effect(callback: object) -> MagicMock:
            nonlocal captured_callback
            captured_callback = callback
            return MagicMock()  # unsubscribe function

        mock_session.on = MagicMock(side_effect=on_side_effect)

        response_data = SimpleNamespace(
            content=content,
            tool_requests=tool_requests or [],
        )

        async def send_side_effect(prompt: str) -> None:
            event = SimpleNamespace(
                type=mock_session_event_type.ASSISTANT_MESSAGE,
                data=response_data,
            )
            if captured_callback is not None:
                captured_callback(event)

        mock_session.send = AsyncMock(side_effect=send_side_effect)
        mock_session.get_messages = AsyncMock(return_value=usage_events or [])

        return mock_session, mock_session_event_type

    async def test_generate_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = self._make_api(monkeypatch)

        usage_events = [
            SimpleNamespace(
                type="usage",
                data=SimpleNamespace(input_tokens=10, output_tokens=5),
            ),
        ]
        mock_session, mock_session_event_type = self._make_session_with_response(
            "sess-1", "Hello!", usage_events=usage_events
        )

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()
        self._prepopulate_pool(mock_client)

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
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
        # Session is now cached (multi-turn), NOT deleted after generate()
        mock_client.delete_session.assert_not_awaited()
        assert len(CopilotModelAPI._conversations) == 1

    async def test_generate_timeout_returns_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = self._make_api(monkeypatch)
        api._timeout = 0.05  # Very short timeout

        mock_session_event_type = MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-2"
        mock_session.on = MagicMock(return_value=MagicMock())
        # Don't trigger callback — let it time out
        mock_session.send = AsyncMock()
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()
        self._prepopulate_pool(mock_client)

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
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
        # Session is cached even on timeout (may recover next turn)
        mock_client.delete_session.assert_not_awaited()
        assert len(CopilotModelAPI._conversations) == 1

    async def test_generate_session_cleanup_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = self._make_api(monkeypatch)

        mock_session_event_type = MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-3"
        mock_session.on = MagicMock(return_value=MagicMock())
        mock_session.send = AsyncMock(side_effect=RuntimeError("SDK error"))
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()
        self._prepopulate_pool(mock_client)

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            with pytest.raises(RuntimeError, match="SDK error"):
                await api.generate(
                    input=[ChatMessageUser(content="Error test")],
                    tools=[],
                    tool_choice="auto",
                    config=GenerateConfig(),
                )

        # On error, session should NOT be cached (conversation not created)
        assert len(CopilotModelAPI._conversations) == 0

    async def test_generate_with_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = self._make_api(monkeypatch)

        tool_requests = [
            SimpleNamespace(
                tool_call_id="tc_1",
                name="read_file",
                arguments={"path": "/tmp/f"},
            )
        ]
        mock_session, mock_session_event_type = self._make_session_with_response(
            "sess-4", "Calling tool.", tool_requests=tool_requests
        )

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()
        self._prepopulate_pool(mock_client)

        mock_sdk_tool_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
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
        """If evict_conversation's delete_session fails, it should be swallowed."""
        api = self._make_api(monkeypatch)

        mock_session, mock_session_event_type = self._make_session_with_response(
            "sess-5", "OK"
        )

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock(
            side_effect=RuntimeError("cleanup fail")
        )
        self._prepopulate_pool(mock_client)

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=[ChatMessageUser(content="Delete fail test")],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        # Should succeed — session is cached
        assert output.choices[0].message.text == "OK"
        assert len(CopilotModelAPI._conversations) == 1

        # Eviction should swallow the delete_session error
        conv_key = list(CopilotModelAPI._conversations.keys())[0]
        await CopilotModelAPI._evict_conversation(conv_key, replace_client=False)
        assert len(CopilotModelAPI._conversations) == 0

    async def test_generate_delete_session_timeout_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If evict_conversation's delete_session hangs, it should time out gracefully."""
        import asyncio

        api = self._make_api(monkeypatch)

        async def _hang_forever(session_id: str) -> None:
            await asyncio.sleep(999)

        mock_session, mock_session_event_type = self._make_session_with_response(
            "sess-6", "OK"
        )

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = _hang_forever
        self._prepopulate_pool(mock_client)

        # Use a very short timeout so the test doesn't actually wait
        monkeypatch.setattr(
            "saber.inspect_ai.integration.copilot_model._SESSION_DELETE_TIMEOUT",
            0.05,
        )

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=[ChatMessageUser(content="Timeout test")],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        # Should succeed — session is cached
        assert output.choices[0].message.text == "OK"

        # Eviction should swallow the timeout
        conv_key = list(CopilotModelAPI._conversations.keys())[0]
        await CopilotModelAPI._evict_conversation(conv_key, replace_client=False)
        assert len(CopilotModelAPI._conversations) == 0


# ---------------------------------------------------------------------------
# CopilotModelAPI.aclose
# ---------------------------------------------------------------------------
class TestCopilotModelAPIClose:
    """Tests for aclose lifecycle management."""

    async def test_aclose_decrements_instance_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api1 = CopilotModelAPI(model_name="gpt-4o")
        CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._instance_count == 2

        await api1.aclose()
        assert CopilotModelAPI._instance_count == 1

    async def test_aclose_last_instance_stops_all_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api = CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._instance_count == 1

        mock_client = AsyncMock()
        mock_client.stop = AsyncMock()
        CopilotModelAPI._all_clients = [mock_client]
        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool

        await api.aclose()
        assert CopilotModelAPI._instance_count == 0
        assert CopilotModelAPI._pool is None
        assert CopilotModelAPI._all_clients == []
        mock_client.stop.assert_awaited_once()

    async def test_aclose_client_exit_error_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api = CopilotModelAPI(model_name="gpt-4o")

        mock_client = AsyncMock()
        mock_client.stop = AsyncMock(side_effect=RuntimeError("boom"))
        CopilotModelAPI._all_clients = [mock_client]
        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool

        # Should not raise
        await api.aclose()
        assert CopilotModelAPI._pool is None
        assert CopilotModelAPI._instance_count == 0

    async def test_aclose_no_pool_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api = CopilotModelAPI(model_name="gpt-4o")
        CopilotModelAPI._pool = None
        CopilotModelAPI._all_clients = []

        # Should not raise when pool is None
        await api.aclose()
        assert CopilotModelAPI._instance_count == 0

    async def test_aclose_stop_timeout_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If client.stop() hangs past timeout, aclose should still complete."""
        import asyncio

        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api = CopilotModelAPI(model_name="gpt-4o")

        async def _hang_forever() -> None:
            await asyncio.sleep(999)

        mock_client = AsyncMock()
        mock_client.stop = _hang_forever
        CopilotModelAPI._all_clients = [mock_client]
        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool

        # Patch the timeout to be very short so the test runs quickly
        monkeypatch.setattr(
            "saber.inspect_ai.integration.copilot_model._CLIENT_STOP_TIMEOUT",
            0.05,
        )

        # Should not hang — the timeout should fire
        await api.aclose()
        assert CopilotModelAPI._pool is None
        assert CopilotModelAPI._instance_count == 0


# ---------------------------------------------------------------------------
# CopilotModelAPI – CancelledError handling
# ---------------------------------------------------------------------------
class TestCopilotModelAPICancelledError:
    """Tests for CancelledError handling in generate (time-limit cancellation)."""

    def _make_api(self, monkeypatch: pytest.MonkeyPatch) -> CopilotModelAPI:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        return CopilotModelAPI(model_name="gpt-4o")

    def _prepopulate_pool(self, mock_client: object) -> None:
        """Pre-populate the class-level pool with a single mock client."""
        import asyncio

        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = [mock_client]

    async def test_generate_handles_cancelled_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CancelledError during response wait returns empty ModelOutput."""
        import asyncio

        api = self._make_api(monkeypatch)

        mock_session = MagicMock()
        mock_session.session_id = "sess-cancel-1"
        mock_session.on = MagicMock(return_value=MagicMock())
        mock_session.send = AsyncMock()
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()

        self._prepopulate_pool(mock_client)

        real_wait_for = asyncio.wait_for
        call_count = 0

        async def selective_wait_for(
            fut: object, *, timeout: float | None = None
        ) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # first_response wait (1=pool.get, 2=create_session)
                raise asyncio.CancelledError()
            return await real_wait_for(fut, timeout=timeout)

        monkeypatch.setattr(asyncio, "wait_for", selective_wait_for)

        mock_session_event_type = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
                ),
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

        assert isinstance(output, ModelOutput)
        assert output.choices[0].message.content == ""
        assert output.stop_reason == "unknown"

    async def test_generate_cancelled_still_cleans_up_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session should not be cached when CancelledError occurs."""
        import asyncio

        api = self._make_api(monkeypatch)

        mock_session = MagicMock()
        mock_session.session_id = "sess-cancel-2"
        mock_session.on = MagicMock(return_value=MagicMock())
        mock_session.send = AsyncMock()
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()

        self._prepopulate_pool(mock_client)

        real_wait_for = asyncio.wait_for
        call_count = 0

        async def selective_wait_for(
            fut: object, *, timeout: float | None = None
        ) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # first_response wait (1=pool.get, 2=create_session)
                raise asyncio.CancelledError()
            return await real_wait_for(fut, timeout=timeout)

        monkeypatch.setattr(asyncio, "wait_for", selective_wait_for)

        mock_session_event_type = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
                ),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=[ChatMessageUser(content="Cancel test")],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        # Cancelled generate should return empty output; session is cached
        # (CancelledError is caught in _send_and_collect, not session-fatal)
        assert isinstance(output, ModelOutput)
        assert output.choices[0].message.content == ""


# ---------------------------------------------------------------------------
# Conversation helper functions
# ---------------------------------------------------------------------------
class TestConversationKey:
    """Tests for _conversation_key stability and uniqueness."""

    def test_stable_across_growing_message_list(self) -> None:
        """Key must be identical for turn-1 and turn-N of the same conversation."""
        turn1 = [
            ChatMessageSystem(content="You are helpful."),
            ChatMessageUser(content="Scan for vulns"),
        ]
        turn2 = [
            ChatMessageSystem(content="You are helpful."),
            ChatMessageUser(content="Scan for vulns"),
            ChatMessageAssistant(
                content="Calling tool.",
                tool_calls=[ToolCall(id="tc_1", function="bash", arguments={"cmd": "ls"})],
            ),
            ChatMessageTool(content="file1.py\nfile2.py", tool_call_id="tc_1", function="bash"),
        ]
        assert _conversation_key(turn1) == _conversation_key(turn2)

    def test_different_tasks_produce_different_keys(self) -> None:
        msgs_a = [
            ChatMessageSystem(content="Task A"),
            ChatMessageUser(content="Do A"),
        ]
        msgs_b = [
            ChatMessageSystem(content="Task B"),
            ChatMessageUser(content="Do A"),
        ]
        assert _conversation_key(msgs_a) != _conversation_key(msgs_b)

    def test_different_user_prompts_produce_different_keys(self) -> None:
        msgs_a = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="User prompt A"),
        ]
        msgs_b = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="User prompt B"),
        ]
        assert _conversation_key(msgs_a) != _conversation_key(msgs_b)

    def test_no_system_message(self) -> None:
        """Works when there's no system message."""
        msgs = [ChatMessageUser(content="Hello")]
        key = _conversation_key(msgs)
        assert len(key) == 16  # 16 hex chars

    def test_empty_messages(self) -> None:
        key = _conversation_key([])
        assert len(key) == 16


class TestFormatToolResults:
    """Tests for _format_tool_results."""

    def test_formats_tool_messages(self) -> None:
        delta = [
            ChatMessageAssistant(
                content="calling",
                tool_calls=[ToolCall(id="tc_1", function="bash", arguments={"cmd": "ls"})],
            ),
            ChatMessageTool(content="file1.py", tool_call_id="tc_1", function="bash"),
        ]
        result = _format_tool_results(delta)
        assert "[Tool Result: bash]" in result
        assert "file1.py" in result
        # Assistant messages should be skipped
        assert "calling" not in result

    def test_empty_delta(self) -> None:
        assert _format_tool_results([]) == ""


class TestExtractSystemContent:
    """Tests for _extract_system_content."""

    def test_returns_system_content(self) -> None:
        msgs = [
            ChatMessageSystem(content="Be helpful"),
            ChatMessageUser(content="Hi"),
        ]
        assert _extract_system_content(msgs) == "Be helpful"

    def test_returns_none_when_no_system(self) -> None:
        msgs = [ChatMessageUser(content="Hi")]
        assert _extract_system_content(msgs) is None


class TestExtractFirstUserContent:
    """Tests for _extract_first_user_content."""

    def test_returns_first_user(self) -> None:
        msgs = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="First user"),
            ChatMessageUser(content="Second user"),
        ]
        assert _extract_first_user_content(msgs) == "First user"

    def test_returns_empty_when_no_user(self) -> None:
        msgs = [ChatMessageSystem(content="System")]
        assert _extract_first_user_content(msgs) == ""


# ---------------------------------------------------------------------------
# Multi-turn session reuse
# ---------------------------------------------------------------------------
class TestMultiTurnSessionReuse:
    """Tests that generate() reuses sessions across calls for the same conversation."""

    def _make_api(self, monkeypatch: pytest.MonkeyPatch) -> CopilotModelAPI:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        return CopilotModelAPI(model_name="gpt-4o")

    def _prepopulate_pool(self, mock_client: object) -> None:
        import asyncio

        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = [mock_client]

    def _make_session_with_response(
        self,
        session_id: str,
        content: str,
        tool_requests: list[object] | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        mock_session_event_type = MagicMock()
        session = self._make_session_with_event_type(
            session_id, content, mock_session_event_type, tool_requests
        )
        return session, mock_session_event_type

    def _make_session_with_event_type(
        self,
        session_id: str,
        content: str,
        mock_session_event_type: MagicMock,
        tool_requests: list[object] | None = None,
    ) -> MagicMock:
        mock_session = MagicMock()
        mock_session.session_id = session_id

        captured_callback = None

        def on_side_effect(callback: object) -> MagicMock:
            nonlocal captured_callback
            captured_callback = callback
            return MagicMock()

        mock_session.on = MagicMock(side_effect=on_side_effect)

        response_data = SimpleNamespace(
            content=content,
            tool_requests=tool_requests or [],
        )

        async def send_side_effect(prompt: str) -> None:
            event = SimpleNamespace(
                type=mock_session_event_type.ASSISTANT_MESSAGE,
                data=response_data,
            )
            if captured_callback is not None:
                captured_callback(event)

        mock_session.send = AsyncMock(side_effect=send_side_effect)
        mock_session.get_messages = AsyncMock(return_value=[])
        return mock_session

    async def test_second_call_reuses_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When turn 1 returns tool_calls the session is cached with pending
        futures.  Turn 2 resolves those futures and gets the next response
        without creating a new session."""
        import asyncio

        api = self._make_api(monkeypatch)

        # Turn 1: session fires EXTERNAL_TOOL_REQUESTED
        mock_event_type = MagicMock()
        turn1_callback = None

        def on_turn1(callback: object) -> MagicMock:
            nonlocal turn1_callback
            turn1_callback = callback
            return MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-mt-1"
        mock_session.on = MagicMock(side_effect=on_turn1)

        async def send_turn1(prompt: str) -> None:
            if turn1_callback is not None:
                turn1_callback(SimpleNamespace(
                    type=mock_event_type.EXTERNAL_TOOL_REQUESTED,
                    data=SimpleNamespace(
                        tool_name="bash",
                        tool_call_id="tc_1",
                        arguments={"cmd": "ls"},
                    ),
                ))

        mock_session.send = AsyncMock(side_effect=send_turn1)
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()
        self._prepopulate_pool(mock_client)

        turn1_msgs = [
            ChatMessageSystem(content="You are a security scanner."),
            ChatMessageUser(content="Scan /workspace"),
        ]

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model._generate_config import GenerateConfig

            # Turn 1 — tool_calls returned, session cached
            output1 = await api.generate(
                input=turn1_msgs,
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output1.stop_reason == "tool_calls"
        assert len(CopilotModelAPI._conversations) == 1
        conv_key = list(CopilotModelAPI._conversations.keys())[0]
        state = CopilotModelAPI._conversations[conv_key]
        assert "tc_1" in state.pending_futures

        # Reconfigure mock session for Turn 2:
        # When subscribed, fire ASSISTANT_MESSAGE after a tick
        def on_turn2(callback: object) -> MagicMock:
            async def fire_response() -> None:
                await asyncio.sleep(0)
                callback(SimpleNamespace(  # type: ignore[operator]
                    type=mock_event_type.ASSISTANT_MESSAGE,
                    data=SimpleNamespace(
                        content="Found vulnerability in file1.py",
                        tool_requests=[],
                    ),
                ))
            asyncio.ensure_future(fire_response())
            return MagicMock()

        mock_session.on = MagicMock(side_effect=on_turn2)

        turn2_msgs = turn1_msgs + [
            ChatMessageAssistant(
                content="",
                tool_calls=[ToolCall(id="tc_1", function="bash", arguments={"cmd": "ls"})],
            ),
            ChatMessageTool(content="file1.py\nfile2.py", tool_call_id="tc_1", function="bash"),
        ]

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            # Turn 2 — futures resolved, text response returned
            output2 = await api.generate(
                input=turn2_msgs,
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output2.choices[0].message.text == "Found vulnerability in file1.py"
        # Session created only ONCE (turn 1) — no eviction
        assert mock_client.create_session.await_count == 1
        # delete_session never called
        mock_client.delete_session.assert_not_awaited()
        # Conversation still cached
        assert len(CopilotModelAPI._conversations) == 1

    async def test_different_conversations_get_different_sessions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two different conversations should create separate sessions."""
        api = self._make_api(monkeypatch)

        session_a, event_type_a = self._make_session_with_response("sess-a", "Response A")
        # Reuse event_type_a so session_b's events match the patched SessionEventType
        session_b = self._make_session_with_event_type("sess-b", "Response B", event_type_a)

        mock_client_a = AsyncMock()
        mock_client_a.create_session = AsyncMock(return_value=session_a)
        mock_client_a.delete_session = AsyncMock()

        mock_client_b = AsyncMock()
        mock_client_b.create_session = AsyncMock(return_value=session_b)
        mock_client_b.delete_session = AsyncMock()

        import asyncio
        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client_a)
        pool.put_nowait(mock_client_b)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = [mock_client_a, mock_client_b]

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=event_type_a
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model._generate_config import GenerateConfig

            output_a = await api.generate(
                input=[
                    ChatMessageSystem(content="Task A"),
                    ChatMessageUser(content="Do A"),
                ],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )
            output_b = await api.generate(
                input=[
                    ChatMessageSystem(content="Task B"),
                    ChatMessageUser(content="Do B"),
                ],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output_a.choices[0].message.text == "Response A"
        assert output_b.choices[0].message.text == "Response B"
        assert len(CopilotModelAPI._conversations) == 2

    async def test_continuation_with_tool_calls_keeps_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a continuation (turn 2+) returns tool_calls, the session
        should be kept alive with pending_futures populated."""
        api = self._make_api(monkeypatch)

        # Turn 1: text-only → session cached
        mock_session1, mock_event_type = self._make_session_with_response(
            "sess-ct-1", "Scanning..."
        )

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session1)
        mock_client.delete_session = AsyncMock()
        self._prepopulate_pool(mock_client)

        turn1_msgs = [
            ChatMessageSystem(content="You are a scanner."),
            ChatMessageUser(content="Scan /workspace"),
        ]

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model._generate_config import GenerateConfig

            output1 = await api.generate(
                input=turn1_msgs,
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )
            assert output1.choices[0].message.text == "Scanning..."

        # Session should be cached (text-only response)
        assert len(CopilotModelAPI._conversations) == 1

        # Turn 2: continuation returns tool_calls (via ASSISTANT_MESSAGE)
        # Session should be KEPT with pending_futures populated
        turn2_msgs = turn1_msgs + [
            ChatMessageUser(content="Now run a deeper scan"),
        ]

        # Reconfigure session1 to return tool_calls on next send
        tool_requests_turn2 = [
            SimpleNamespace(
                tool_call_id="tc_deep",
                name="bash",
                arguments={"cmd": "find / -name '*.py'"},
            )
        ]
        captured_callback_2 = None

        def on_side_effect_2(callback: object) -> MagicMock:
            nonlocal captured_callback_2
            captured_callback_2 = callback
            return MagicMock()

        mock_session1.on = MagicMock(side_effect=on_side_effect_2)

        response_data_2 = SimpleNamespace(
            content="Running deep scan.",
            tool_requests=tool_requests_turn2,
        )

        async def send_side_effect_2(prompt: str) -> None:
            event = SimpleNamespace(
                type=mock_event_type.ASSISTANT_MESSAGE,
                data=response_data_2,
            )
            if captured_callback_2 is not None:
                captured_callback_2(event)

        mock_session1.send = AsyncMock(side_effect=send_side_effect_2)

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            output2 = await api.generate(
                input=turn2_msgs,
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output2.stop_reason == "tool_calls"
        # Session should be KEPT (not evicted) with pending_futures
        assert len(CopilotModelAPI._conversations) == 1
        conv_key = list(CopilotModelAPI._conversations.keys())[0]
        state = CopilotModelAPI._conversations[conv_key]
        assert "tc_deep" in state.pending_futures
        # delete_session NOT called
        mock_client.delete_session.assert_not_awaited()
        # Client NOT returned to pool (still held by conversation)
        assert CopilotModelAPI._pool is not None
        assert CopilotModelAPI._pool.qsize() == 0


# ---------------------------------------------------------------------------
# CopilotClientPool — pool lifecycle and acquire/release
# ---------------------------------------------------------------------------
class TestCopilotClientPool:
    """Tests for the asyncio.Queue client-pool mechanics."""

    def _make_api(self, monkeypatch: pytest.MonkeyPatch) -> CopilotModelAPI:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        return CopilotModelAPI(model_name="gpt-4o")

    def test_init_default_pool_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._pool_size == 8

    def test_init_custom_pool_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        CopilotModelAPI(model_name="gpt-4o", pool_size="8")
        assert CopilotModelAPI._pool_size == 8

    async def test_init_pool_creates_n_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._make_api(monkeypatch)
        CopilotModelAPI._pool_size = 3

        mock_clients = [AsyncMock() for _ in range(3)]
        call_idx = 0

        async def _mock_create() -> object:
            nonlocal call_idx
            client = mock_clients[call_idx]
            call_idx += 1
            return client

        with patch.object(
            CopilotModelAPI, "_create_single_client", side_effect=_mock_create
        ):
            await CopilotModelAPI._init_pool()

        assert CopilotModelAPI._pool is not None
        assert CopilotModelAPI._pool.qsize() == 3
        assert len(CopilotModelAPI._all_clients) == 3

    async def test_init_pool_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._make_api(monkeypatch)
        CopilotModelAPI._pool_size = 2

        create_count = 0

        async def _mock_create() -> object:
            nonlocal create_count
            create_count += 1
            return AsyncMock()

        with patch.object(
            CopilotModelAPI, "_create_single_client", side_effect=_mock_create
        ):
            await CopilotModelAPI._init_pool()
            await CopilotModelAPI._init_pool()  # Second call — no-op

        assert create_count == 2  # Only created during first init

    @pytest.mark.asyncio
    async def test_init_pool_partial_failure_stops_started_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If client creation fails mid-init, already-started clients are stopped."""
        self._make_api(monkeypatch)
        CopilotModelAPI._pool_size = 3

        started_clients: list[AsyncMock] = []

        async def _mock_create() -> object:
            if len(started_clients) == 2:
                raise RuntimeError("SDK install broken")
            client = AsyncMock()
            started_clients.append(client)
            return client

        with (
            patch.object(
                CopilotModelAPI, "_create_single_client", side_effect=_mock_create
            ),
            pytest.raises(RuntimeError, match="SDK install broken"),
        ):
            await CopilotModelAPI._init_pool()

        # Both started clients should have been stopped
        assert len(started_clients) == 2
        for c in started_clients:
            c.stop.assert_awaited_once()

        # Pool should NOT have been set (init failed)
        assert CopilotModelAPI._pool is None
        assert CopilotModelAPI._all_clients == []

    async def test_generate_acquires_and_returns_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After generate completes, client should be back in pool."""
        import asyncio

        api = self._make_api(monkeypatch)

        mock_session_event_type = MagicMock()

        captured_callback = None

        def on_side_effect(callback: object) -> MagicMock:
            nonlocal captured_callback
            captured_callback = callback
            return MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-pool-1"
        mock_session.on = MagicMock(side_effect=on_side_effect)

        async def send_side_effect(prompt: str) -> None:
            event = SimpleNamespace(
                type=mock_session_event_type.ASSISTANT_MESSAGE,
                data=SimpleNamespace(content="Hi", tool_requests=[]),
            )
            if captured_callback is not None:
                captured_callback(event)

        mock_session.send = AsyncMock(side_effect=send_side_effect)
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()

        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = [mock_client]

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_session_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            await api.generate(
                input=[ChatMessageUser(content="Hi")],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        # Client is now held by the cached conversation, NOT in the pool
        assert CopilotModelAPI._pool.qsize() == 0
        assert len(CopilotModelAPI._conversations) == 1

    async def test_pool_acquire_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty pool → generate returns empty output after timeout."""
        import asyncio

        api = self._make_api(monkeypatch)

        # Pool exists but is empty — no clients available
        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = []

        monkeypatch.setattr(
            "saber.inspect_ai.integration.copilot_model._POOL_ACQUIRE_TIMEOUT",
            0.05,
        )

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

    async def test_client_failure_replaces_in_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BrokenPipeError during generate replaces dead client with a new one."""
        import asyncio

        api = self._make_api(monkeypatch)

        mock_session_event_type = MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-fail"
        mock_session.on = MagicMock(return_value=MagicMock())
        mock_session.send = AsyncMock(side_effect=BrokenPipeError("pipe broken"))
        mock_session.get_messages = AsyncMock(return_value=[])

        dead_client = AsyncMock()
        dead_client.create_session = AsyncMock(return_value=mock_session)
        dead_client.delete_session = AsyncMock()

        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(dead_client)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = [dead_client]

        replacement_client = AsyncMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "copilot": MagicMock(PermissionHandler=MagicMock()),
                    "copilot.generated.session_events": MagicMock(
                        SessionEventType=mock_session_event_type
                    ),
                    "copilot.tools": MagicMock(),
                },
            ),
            patch.object(
                CopilotModelAPI,
                "_create_single_client",
                return_value=replacement_client,
            ),
        ):
            from inspect_ai.model import ChatMessageUser
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=[ChatMessageUser(content="Hi")],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        # Should return empty output
        assert output.stop_reason == "unknown"
        # Replacement client should be in pool
        assert CopilotModelAPI._pool.qsize() == 1
        assert replacement_client in CopilotModelAPI._all_clients
        assert dead_client not in CopilotModelAPI._all_clients

    async def test_aclose_stops_all_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Last instance aclose drains queue and stops all tracked clients."""
        import asyncio

        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api = CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._instance_count == 1

        clients = [AsyncMock() for _ in range(3)]
        for c in clients:
            c.stop = AsyncMock()

        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        for c in clients:
            pool.put_nowait(c)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = list(clients)

        await api.aclose()

        for c in clients:
            c.stop.assert_awaited_once()
        assert CopilotModelAPI._pool is None
        assert CopilotModelAPI._all_clients == []
        assert CopilotModelAPI._instance_count == 0

    async def test_aclose_non_last_decrements_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-last instance only decrements count, doesn't stop pool."""
        import asyncio

        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        api1 = CopilotModelAPI(model_name="gpt-4o")
        CopilotModelAPI(model_name="gpt-4o")
        assert CopilotModelAPI._instance_count == 2

        mock_client = AsyncMock()
        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = [mock_client]

        await api1.aclose()

        assert CopilotModelAPI._instance_count == 1
        assert CopilotModelAPI._pool is not None  # Pool still alive

    async def test_shutdown_flag_prevents_pool_return(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _shutting_down=True, client is stopped not returned to pool."""
        import asyncio

        CopilotModelAPI._shutting_down = True
        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        CopilotModelAPI._pool = pool

        mock_client = AsyncMock()
        mock_client.stop = AsyncMock()

        await CopilotModelAPI._return_or_replace_client(mock_client, failed=False)

        # Client stopped, NOT returned to pool
        mock_client.stop.assert_awaited_once()
        assert pool.qsize() == 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestCopilotRegistration:
    """Tests for the @modelapi('copilot') registration."""

    def test_modelapi_decorator_applied(self) -> None:
        """Verify that the copilot factory function has the modelapi registration."""
        from saber.inspect_ai.integration.copilot_model import copilot

        # The function should exist and be callable
        assert callable(copilot)

    def test_copilot_returns_model_class(self) -> None:
        """Verify the factory returns CopilotModelAPI when SDK is available."""
        from saber.inspect_ai.integration.copilot_model import copilot

        # Mock the copilot SDK import and call the decorated factory
        # The @modelapi wrapper calls our factory then instantiates the class,
        # so we need to pass model_name through.
        with patch.dict("sys.modules", {"copilot": MagicMock()}):
            with patch.object(CopilotModelAPI, "__init__", return_value=None):
                result = copilot(model_name="gpt-4o")
        assert isinstance(result, CopilotModelAPI)


# ---------------------------------------------------------------------------
# _make_tool_handler
# ---------------------------------------------------------------------------
class TestMakeToolHandler:
    """Tests for _make_tool_handler factory function."""

    async def test_handler_creates_future_in_dict(self) -> None:
        """Handler should create and store a future keyed by tool_call_id."""
        import asyncio

        pending_futures: dict[str, asyncio.Future[str]] = {}
        mock_tool_result_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {"copilot": MagicMock(), "copilot.tools": MagicMock(ToolResult=mock_tool_result_cls)},
        ):
            handler = _make_tool_handler(pending_futures)
            invocation = SimpleNamespace(tool_call_id="tc_123")

            # Run handler in background (it awaits the future)
            task = asyncio.create_task(handler(invocation))
            await asyncio.sleep(0)  # Let handler start

            assert "tc_123" in pending_futures
            assert not pending_futures["tc_123"].done()

            # Resolve and clean up
            pending_futures["tc_123"].set_result("tool output")
            await task

    async def test_handler_returns_tool_result_on_resolve(self) -> None:
        """Resolving the future should return ToolResult with success."""
        import asyncio

        pending_futures: dict[str, asyncio.Future[str]] = {}
        mock_tool_result_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {"copilot": MagicMock(), "copilot.tools": MagicMock(ToolResult=mock_tool_result_cls)},
        ):
            handler = _make_tool_handler(pending_futures)
            invocation = SimpleNamespace(tool_call_id="tc_456")

            task = asyncio.create_task(handler(invocation))
            await asyncio.sleep(0)

            pending_futures["tc_456"].set_result("file contents here")
            await task

        call_kwargs = mock_tool_result_cls.call_args[1]
        assert call_kwargs["text_result_for_llm"] == "file contents here"
        assert call_kwargs["result_type"] == "success"
        assert call_kwargs["error"] is None

    async def test_handler_returns_failure_on_exception(self) -> None:
        """Setting exception on future should return failure ToolResult."""
        import asyncio

        pending_futures: dict[str, asyncio.Future[str]] = {}
        mock_tool_result_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {"copilot": MagicMock(), "copilot.tools": MagicMock(ToolResult=mock_tool_result_cls)},
        ):
            handler = _make_tool_handler(pending_futures)
            invocation = SimpleNamespace(tool_call_id="tc_err")

            task = asyncio.create_task(handler(invocation))
            await asyncio.sleep(0)

            pending_futures["tc_err"].set_exception(RuntimeError("tool crashed"))
            await task

        call_kwargs = mock_tool_result_cls.call_args[1]
        assert call_kwargs["result_type"] == "failure"
        assert call_kwargs["text_result_for_llm"] == ""
        assert "tool crashed" in call_kwargs["error"]

    async def test_handler_empty_tool_call_id(self) -> None:
        """Handler with missing tool_call_id should use empty string."""
        import asyncio

        pending_futures: dict[str, asyncio.Future[str]] = {}
        mock_tool_result_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {"copilot": MagicMock(), "copilot.tools": MagicMock(ToolResult=mock_tool_result_cls)},
        ):
            handler = _make_tool_handler(pending_futures)
            invocation = SimpleNamespace()  # No tool_call_id attr

            task = asyncio.create_task(handler(invocation))
            await asyncio.sleep(0)

            assert "" in pending_futures
            pending_futures[""].set_result("ok")
            await task


# ---------------------------------------------------------------------------
# _convert_tools_for_sdk
# ---------------------------------------------------------------------------
class TestConvertToolsForSdk:
    """Tests for _convert_tools_for_sdk tool handler registration."""

    def test_tools_have_callable_handler(self) -> None:
        """SDK tools should have a callable handler (Future-based) so the SDK
        session stays alive while awaiting external tool execution."""
        from dataclasses import dataclass

        @dataclass
        class FakeTool:
            name: str
            description: str
            handler: object
            parameters: object = None
            overrides_built_in_tool: bool = False
            skip_permission: bool = False

        mock_tools_module = MagicMock()
        mock_tools_module.Tool = FakeTool

        tools = [
            ToolInfo(name="run_command", description="Run a command"),
            ToolInfo(name="read_file", description="Read a file"),
        ]
        with patch.dict("sys.modules", {"copilot": MagicMock(), "copilot.tools": mock_tools_module}):
            sdk_tools = _convert_tools_for_sdk(tools, {})
        assert len(sdk_tools) == 2
        for sdk_tool in sdk_tools:
            assert callable(sdk_tool.handler)

    def test_tools_share_pending_futures_dict(self) -> None:
        """All tools should share the same pending_futures dict via handler."""
        from dataclasses import dataclass

        @dataclass
        class FakeTool:
            name: str
            description: str
            handler: object
            parameters: object = None
            overrides_built_in_tool: bool = False
            skip_permission: bool = False

        mock_tools_module = MagicMock()
        mock_tools_module.Tool = FakeTool

        tools = [
            ToolInfo(name="tool_a", description="A"),
            ToolInfo(name="tool_b", description="B"),
        ]
        with patch.dict("sys.modules", {"copilot": MagicMock(), "copilot.tools": mock_tools_module}):
            sdk_tools = _convert_tools_for_sdk(tools, {})
        # All tools should have the same handler (same closure)
        assert sdk_tools[0].handler is sdk_tools[1].handler

    def test_tools_preserve_name_and_description(self) -> None:
        """SDK tools should have correct name and description."""
        from dataclasses import dataclass

        @dataclass
        class FakeTool:
            name: str
            description: str
            handler: object
            parameters: object = None
            overrides_built_in_tool: bool = False
            skip_permission: bool = False

        mock_tools_module = MagicMock()
        mock_tools_module.Tool = FakeTool

        tools = [
            ToolInfo(name="run_command", description="Execute a shell command"),
        ]
        with patch.dict("sys.modules", {"copilot": MagicMock(), "copilot.tools": mock_tools_module}):
            sdk_tools = _convert_tools_for_sdk(tools, {})
        assert sdk_tools[0].name == "run_command"
        assert sdk_tools[0].description == "Execute a shell command"

    def test_tools_have_overrides_built_in_tool(self) -> None:
        """SDK tools should set overrides_built_in_tool=True."""
        from dataclasses import dataclass

        @dataclass
        class FakeTool:
            name: str
            description: str
            handler: object
            parameters: object = None
            overrides_built_in_tool: bool = False
            skip_permission: bool = False

        mock_tools_module = MagicMock()
        mock_tools_module.Tool = FakeTool

        tools = [ToolInfo(name="bash", description="Run bash")]
        with patch.dict("sys.modules", {"copilot": MagicMock(), "copilot.tools": mock_tools_module}):
            sdk_tools = _convert_tools_for_sdk(tools, {})
        assert sdk_tools[0].overrides_built_in_tool is True

    def test_empty_tools_returns_empty(self) -> None:
        """Empty tool list returns empty SDK tool list."""
        from dataclasses import dataclass

        @dataclass
        class FakeTool:
            name: str
            description: str
            handler: object
            parameters: object = None
            overrides_built_in_tool: bool = False
            skip_permission: bool = False

        mock_tools_module = MagicMock()
        mock_tools_module.Tool = FakeTool

        with patch.dict("sys.modules", {"copilot": MagicMock(), "copilot.tools": mock_tools_module}):
            assert _convert_tools_for_sdk([], {}) == []


# ---------------------------------------------------------------------------
# Future-based tool call handling (full round-trip tests)
# ---------------------------------------------------------------------------
class TestFutureBasedToolCalls:
    """Tests for the Future-based tool call handling — no session eviction."""

    def _make_api(self, monkeypatch: pytest.MonkeyPatch) -> CopilotModelAPI:
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        return CopilotModelAPI(model_name="gpt-4o")

    def _prepopulate_pool(self, mock_client: object) -> None:
        import asyncio

        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        pool.put_nowait(mock_client)
        CopilotModelAPI._pool = pool
        CopilotModelAPI._all_clients = [mock_client]

    async def test_start_conversation_caches_on_tool_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When tool_calls are in the response, session is cached with
        pending_futures — NOT evicted."""
        api = self._make_api(monkeypatch)
        mock_event_type = MagicMock()

        turn1_callback = None

        def on_turn1(callback: object) -> MagicMock:
            nonlocal turn1_callback
            turn1_callback = callback
            return MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-cache-tc"
        mock_session.on = MagicMock(side_effect=on_turn1)

        async def send_side_effect(prompt: str) -> None:
            if turn1_callback is not None:
                turn1_callback(SimpleNamespace(
                    type=mock_event_type.EXTERNAL_TOOL_REQUESTED,
                    data=SimpleNamespace(
                        tool_name="read_file",
                        tool_call_id="tc_rf1",
                        arguments={"path": "/etc/passwd"},
                    ),
                ))

        mock_session.send = AsyncMock(side_effect=send_side_effect)
        mock_session.get_messages = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.delete_session = AsyncMock()
        self._prepopulate_pool(mock_client)

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=[
                    ChatMessageSystem(content="System"),
                    ChatMessageUser(content="Read file"),
                ],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output.stop_reason == "tool_calls"
        tc = output.choices[0].message.tool_calls[0]
        assert tc.function == "read_file"
        assert tc.id == "tc_rf1"

        # Session CACHED (not evicted)
        assert len(CopilotModelAPI._conversations) == 1
        conv_key = list(CopilotModelAPI._conversations.keys())[0]
        state = CopilotModelAPI._conversations[conv_key]
        assert "tc_rf1" in state.pending_futures

        # delete_session NOT called
        mock_client.delete_session.assert_not_awaited()
        # Client NOT returned to pool (held by conversation)
        assert CopilotModelAPI._pool is not None
        assert CopilotModelAPI._pool.qsize() == 0

    async def test_resolve_and_collect_returns_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_resolve_and_collect should resolve futures and return the
        next ASSISTANT_MESSAGE response."""
        import asyncio

        from saber.inspect_ai.integration.copilot_model import (
            _ConversationState,
        )

        api = self._make_api(monkeypatch)
        mock_event_type = MagicMock()

        # Create a conversation state with a pending future
        loop = asyncio.get_running_loop()
        pending_futures: dict[str, asyncio.Future[str]] = {
            "tc_1": loop.create_future(),
        }

        # Mock session that fires ASSISTANT_MESSAGE when subscribed
        mock_session = MagicMock()
        mock_session.session_id = "sess-resolve"

        def on_side_effect(callback: object) -> MagicMock:
            async def fire_response() -> None:
                await asyncio.sleep(0)
                callback(SimpleNamespace(  # type: ignore[operator]
                    type=mock_event_type.ASSISTANT_MESSAGE,
                    data=SimpleNamespace(
                        content="File contains sensitive data.",
                        tool_requests=[],
                    ),
                ))
            asyncio.ensure_future(fire_response())
            return MagicMock()

        mock_session.on = MagicMock(side_effect=on_side_effect)
        mock_session.get_messages = AsyncMock(return_value=[])

        state = _ConversationState(
            session=mock_session,
            client=MagicMock(),
            messages_sent=2,
            pending_futures=pending_futures,
        )

        with patch.dict(
            "sys.modules",
            {
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
            },
        ):
            import time
            result = await api._resolve_and_collect(
                state,
                {"tc_1": "root:x:0:0:root:/root:/bin/bash"},
                t_start=time.monotonic(),
                t_session=time.monotonic(),
                conv_key="test1234",
            )

        assert result.choices[0].message.text == "File contains sensitive data."
        assert result.stop_reason == "stop"
        # Future should have been popped from pending_futures
        assert "tc_1" not in pending_futures

    async def test_resolve_timeout_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Timeout during _resolve_and_collect returns empty output."""
        import asyncio

        from saber.inspect_ai.integration.copilot_model import (
            _ConversationState,
        )

        api = self._make_api(monkeypatch)
        api._timeout = 0.05  # Very short timeout
        mock_event_type = MagicMock()

        loop = asyncio.get_running_loop()
        pending_futures: dict[str, asyncio.Future[str]] = {
            "tc_timeout": loop.create_future(),
        }

        # Mock session that does NOT fire any events (causes timeout)
        mock_session = MagicMock()
        mock_session.session_id = "sess-timeout"
        mock_session.on = MagicMock(return_value=MagicMock())
        mock_session.get_messages = AsyncMock(return_value=[])

        state = _ConversationState(
            session=mock_session,
            client=MagicMock(),
            messages_sent=2,
            pending_futures=pending_futures,
        )

        with patch.dict(
            "sys.modules",
            {
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
            },
        ):
            import time
            result = await api._resolve_and_collect(
                state,
                {"tc_timeout": "some result"},
                t_start=time.monotonic(),
                t_session=time.monotonic(),
                conv_key="timeoutkey",
            )

        assert result.stop_reason == "unknown"
        assert result.choices[0].message.content == ""

    async def test_continue_without_pending_futures_falls_back_to_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If no pending_futures, _continue_conversation sends via session.send."""
        import asyncio

        from saber.inspect_ai.integration.copilot_model import (
            _ConversationState,
        )

        api = self._make_api(monkeypatch)
        mock_event_type = MagicMock()

        # Session fires ASSISTANT_MESSAGE on send
        captured_callback = None

        def on_side_effect(callback: object) -> MagicMock:
            nonlocal captured_callback
            captured_callback = callback
            return MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-fb"
        mock_session.on = MagicMock(side_effect=on_side_effect)

        async def send_side_effect(prompt: str) -> None:
            if captured_callback is not None:
                captured_callback(SimpleNamespace(
                    type=mock_event_type.ASSISTANT_MESSAGE,
                    data=SimpleNamespace(
                        content="Fallback response.",
                        tool_requests=[],
                    ),
                ))

        mock_session.send = AsyncMock(side_effect=send_side_effect)
        mock_session.get_messages = AsyncMock(return_value=[])

        # No pending_futures → should use session.send fallback
        state = _ConversationState(
            session=mock_session,
            client=MagicMock(),
            messages_sent=2,
            pending_futures={},
        )
        conv_key = _conversation_key([
            ChatMessageSystem(content="Sys"),
            ChatMessageUser(content="User"),
        ])
        CopilotModelAPI._conversations[conv_key] = state

        # Pre-populate pool dummy (not used since conversation found)
        pool: asyncio.Queue[object] = asyncio.Queue(maxsize=4)
        CopilotModelAPI._pool = pool

        msgs = [
            ChatMessageSystem(content="Sys"),
            ChatMessageUser(content="User"),
            ChatMessageAssistant(
                content="Calling tool.",
                tool_calls=[ToolCall(id="tc_fb", function="bash", arguments={})],
            ),
            ChatMessageTool(content="output", tool_call_id="tc_fb", function="bash"),
        ]

        with patch.dict(
            "sys.modules",
            {
                "copilot": MagicMock(PermissionHandler=MagicMock()),
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
                "copilot.tools": MagicMock(),
            },
        ):
            from inspect_ai.model._generate_config import GenerateConfig

            output = await api.generate(
                input=msgs,
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
            )

        assert output.choices[0].message.text == "Fallback response."
        # session.send was used (fallback path)
        mock_session.send.assert_awaited_once()

    async def test_multiple_tool_calls_all_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple pending futures should all be resolved."""
        import asyncio

        from saber.inspect_ai.integration.copilot_model import (
            _ConversationState,
        )

        api = self._make_api(monkeypatch)
        mock_event_type = MagicMock()

        loop = asyncio.get_running_loop()
        pending_futures: dict[str, asyncio.Future[str]] = {
            "tc_a": loop.create_future(),
            "tc_b": loop.create_future(),
        }

        def on_side_effect(callback: object) -> MagicMock:
            async def fire_response() -> None:
                await asyncio.sleep(0)
                callback(SimpleNamespace(  # type: ignore[operator]
                    type=mock_event_type.ASSISTANT_MESSAGE,
                    data=SimpleNamespace(
                        content="Both tools completed.",
                        tool_requests=[],
                    ),
                ))
            asyncio.ensure_future(fire_response())
            return MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-multi"
        mock_session.on = MagicMock(side_effect=on_side_effect)
        mock_session.get_messages = AsyncMock(return_value=[])

        state = _ConversationState(
            session=mock_session,
            client=MagicMock(),
            messages_sent=2,
            pending_futures=pending_futures,
        )

        with patch.dict(
            "sys.modules",
            {
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
            },
        ):
            import time
            result = await api._resolve_and_collect(
                state,
                {"tc_a": "result_a", "tc_b": "result_b"},
                t_start=time.monotonic(),
                t_session=time.monotonic(),
                conv_key="multikey1",
            )

        assert result.choices[0].message.text == "Both tools completed."
        # All futures should be popped
        assert "tc_a" not in pending_futures
        assert "tc_b" not in pending_futures


class TestEvictionCancelsFutures:
    """Test that _evict_conversation cancels pending Futures."""

    @pytest.mark.asyncio
    async def test_evict_cancels_pending_futures(self) -> None:
        """Pending futures must be cancelled when conversation is evicted."""
        import asyncio

        from saber.inspect_ai.integration.copilot_model import (
            CopilotModelAPI,
            _ConversationState,
        )

        loop = asyncio.get_running_loop()
        fut_a: asyncio.Future[str] = loop.create_future()
        fut_b: asyncio.Future[str] = loop.create_future()

        mock_client = MagicMock()
        mock_client.delete_session = AsyncMock(return_value=None)

        state = _ConversationState(
            session=MagicMock(session_id="sess_1"),
            client=mock_client,
            messages_sent=2,
            pending_futures={"tc_1": fut_a, "tc_2": fut_b},
        )

        # Manually inject into conversations dict
        CopilotModelAPI._conversations["test_evict_key"] = state
        CopilotModelAPI._pool = asyncio.Queue()
        CopilotModelAPI._shutting_down = False

        try:
            await CopilotModelAPI._evict_conversation(
                "test_evict_key", replace_client=False
            )
            # Futures should be cancelled
            assert fut_a.cancelled()
            assert fut_b.cancelled()
            assert len(state.pending_futures) == 0
        finally:
            # Cleanup: drain pool and reset
            while not CopilotModelAPI._pool.empty():
                CopilotModelAPI._pool.get_nowait()
            CopilotModelAPI._conversations.pop("test_evict_key", None)
            CopilotModelAPI._pool = None


class TestPartialToolResults:
    """Test that unresolved Futures get fallback results."""

    @pytest.mark.asyncio
    async def test_orphaned_futures_get_fallback_result(self) -> None:
        """When inspect_ai sends partial tool results, orphaned futures get fallback."""
        import asyncio

        from saber.inspect_ai.integration.copilot_model import (
            CopilotModelAPI,
            _ConversationState,
        )

        mock_event_type = MagicMock()
        loop = asyncio.get_running_loop()

        # Two futures but we'll only resolve one
        fut_a: asyncio.Future[str] = loop.create_future()
        fut_b: asyncio.Future[str] = loop.create_future()

        def on_side_effect(callback: object) -> MagicMock:
            async def fire_response() -> None:
                await asyncio.sleep(0)
                callback(SimpleNamespace(  # type: ignore[operator]
                    type=mock_event_type.ASSISTANT_MESSAGE,
                    data=SimpleNamespace(
                        content="Partial results handled.",
                        tool_requests=[],
                    ),
                ))
            asyncio.ensure_future(fire_response())
            return MagicMock()

        mock_session = MagicMock()
        mock_session.session_id = "sess-partial"
        mock_session.on = MagicMock(side_effect=on_side_effect)
        mock_session.get_messages = AsyncMock(return_value=[])

        state = _ConversationState(
            session=mock_session,
            client=MagicMock(),
            messages_sent=2,
            pending_futures={"tc_a": fut_a, "tc_b": fut_b},
        )

        api = CopilotModelAPI.__new__(CopilotModelAPI)
        api.model_name = "copilot/test"
        api._timeout = 5

        with patch.dict(
            "sys.modules",
            {
                "copilot.generated.session_events": MagicMock(
                    SessionEventType=mock_event_type
                ),
            },
        ):
            import time
            # Only send result for tc_a, tc_b is orphaned
            result = await api._resolve_and_collect(
                state,
                {"tc_a": "result for a"},
                t_start=time.monotonic(),
                t_session=time.monotonic(),
                conv_key="partialkey",
            )

        assert result.choices[0].message.text == "Partial results handled."
        # fut_a was resolved with the provided result
        assert fut_a.result() == "result for a"
        # fut_b was resolved with fallback
        assert fut_b.result() == "Tool result not available"
        # All futures cleared
        assert len(state.pending_futures) == 0
