"""Tests for shared bridge_utils helpers.

These utilities are used by both claude_code and copilot agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBuildSystemPrompt:
    """build_system_prompt shared helper."""

    def test_combines_all_parts(self) -> None:
        """Combines instruction and assistant prompts."""
        from saber.agents.bridge_utils import build_system_prompt

        result = build_system_prompt(
            instruction_prompt="Instructions here.",
            assistant_prompt="You are an agent.",
        )
        assert "Instructions here." in result
        assert "You are an agent." in result

    def test_skips_empty_parts(self) -> None:
        """Skips empty strings from the combined prompt."""
        from saber.agents.bridge_utils import build_system_prompt

        result = build_system_prompt(
            instruction_prompt="Only this.",
            assistant_prompt="",
        )
        assert result == "Only this."

    def test_all_empty_returns_empty(self) -> None:
        """Returns empty string when all parts are empty."""
        from saber.agents.bridge_utils import build_system_prompt

        assert build_system_prompt("", "") == ""


class TestBuildBridgedTools:
    """build_bridged_tools shared helper."""

    def test_creates_spec_from_tools(self) -> None:
        """Converts a list of Tools to a BridgedToolsSpec list."""
        from inspect_ai.agent import BridgedToolsSpec
        from inspect_ai.tool import bash

        from saber.agents.bridge_utils import build_bridged_tools

        tools = [bash()]
        result = build_bridged_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], BridgedToolsSpec)
        assert result[0].name == "saber_tools"

    def test_returns_none_for_empty(self) -> None:
        """Returns None when no tools provided."""
        from saber.agents.bridge_utils import build_bridged_tools

        assert build_bridged_tools(None) is None
        assert build_bridged_tools([]) is None


class TestBuildUserPrompt:
    """build_user_prompt extracts user messages from state.messages."""

    def test_single_user_message(self) -> None:
        """Returns user text when only one user message exists."""
        from inspect_ai.model import ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [ChatMessageUser(content="Do this task.")]
        prompt, has_assistant = build_user_prompt(messages)
        assert prompt == "Do this task."
        assert has_assistant is False

    def test_user_after_assistant(self) -> None:
        """Returns only user messages after the last assistant message."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [
            ChatMessageUser(content="First question"),
            ChatMessageAssistant(content="First answer"),
            ChatMessageUser(content="Follow-up question"),
        ]
        prompt, has_assistant = build_user_prompt(messages)
        assert prompt == "Follow-up question"
        assert has_assistant is True

    def test_multiple_user_after_assistant(self) -> None:
        """Joins multiple user messages after assistant with double newlines."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [
            ChatMessageUser(content="First"),
            ChatMessageAssistant(content="Reply"),
            ChatMessageUser(content="Second"),
            ChatMessageUser(content="Third"),
        ]
        prompt, has_assistant = build_user_prompt(messages)
        assert prompt == "Second\n\nThird"
        assert has_assistant is True

    def test_no_user_messages_returns_empty(self) -> None:
        """Returns empty string when no user messages exist."""
        from inspect_ai.model import ChatMessageSystem

        from saber.agents.bridge_utils import build_user_prompt

        messages = [ChatMessageSystem(content="System context")]
        prompt, has_assistant = build_user_prompt(messages)
        assert prompt == ""
        assert has_assistant is False

    def test_skips_system_messages(self) -> None:
        """System messages are not included in the prompt."""
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [
            ChatMessageSystem(content="System setup"),
            ChatMessageUser(content="User question"),
        ]
        prompt, has_assistant = build_user_prompt(messages)
        assert prompt == "User question"
        assert has_assistant is False

    def test_empty_messages_list(self) -> None:
        """Returns empty string and False for empty messages list."""
        from saber.agents.bridge_utils import build_user_prompt

        prompt, has_assistant = build_user_prompt([])
        assert prompt == ""
        assert has_assistant is False

    def test_raises_on_trailing_assistant(self) -> None:
        """Raises ValueError when last message is an assistant message."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [
            ChatMessageUser(content="Hi"),
            ChatMessageAssistant(content="Hello"),
        ]
        with pytest.raises(ValueError, match="ends with an assistant"):
            build_user_prompt(messages)

    def test_all_before_assistant_ignored(self) -> None:
        """User messages before last assistant are excluded."""
        from inspect_ai.model import (
            ChatMessageAssistant,
            ChatMessageUser,
        )

        from saber.agents.bridge_utils import build_user_prompt

        messages = [
            ChatMessageUser(content="Old question"),
            ChatMessageAssistant(content="Old answer"),
            ChatMessageUser(content="Continue"),
            ChatMessageAssistant(content="Answer 2"),
            ChatMessageUser(content="Latest"),
        ]
        prompt, has_assistant = build_user_prompt(messages)
        assert prompt == "Latest"
        assert has_assistant is True


class TestResolveMcpServers:
    """resolve_mcp_servers uses model_dump for proper serialization."""

    def test_single_server_all_tools(self) -> None:
        """Single HTTP server with tools='all' produces wildcard pattern."""
        from inspect_ai.tool._mcp._config import MCPServerConfigHTTP

        from saber.agents.bridge_utils import resolve_mcp_servers

        configs = [
            MCPServerConfigHTTP(
                type="http",
                name="saber_tools",
                url="http://localhost:3001/mcp",
                tools="all",
            ),
        ]
        mcp_json_str, allowed_tools = resolve_mcp_servers(configs)
        mcp_json = json.loads(mcp_json_str)

        assert "mcpServers" in mcp_json
        assert "saber_tools" in mcp_json["mcpServers"]
        server_cfg = mcp_json["mcpServers"]["saber_tools"]
        assert server_cfg["type"] == "http"
        assert server_cfg["url"] == "http://localhost:3001/mcp"
        # name and tools should be excluded
        assert "name" not in server_cfg
        assert "tools" not in server_cfg
        assert allowed_tools == ["mcp__saber_tools__*"]

    def test_server_with_specific_tools(self) -> None:
        """Server with specific tool list produces per-tool patterns."""
        from inspect_ai.tool._mcp._config import MCPServerConfigHTTP

        from saber.agents.bridge_utils import resolve_mcp_servers

        configs = [
            MCPServerConfigHTTP(
                type="http",
                name="my_server",
                url="http://localhost:5000/mcp",
                tools=["run_cmd", "read_file"],
            ),
        ]
        mcp_json_str, allowed_tools = resolve_mcp_servers(configs)
        mcp_json = json.loads(mcp_json_str)

        assert "my_server" in mcp_json["mcpServers"]
        assert allowed_tools == [
            "mcp__my_server__run_cmd",
            "mcp__my_server__read_file",
        ]

    def test_empty_configs(self) -> None:
        """Empty config list produces empty JSON object and no tools."""
        from saber.agents.bridge_utils import resolve_mcp_servers

        mcp_json_str, allowed_tools = resolve_mcp_servers([])
        mcp_json = json.loads(mcp_json_str)

        assert mcp_json == {"mcpServers": {}}
        assert allowed_tools == []

    def test_multiple_servers(self) -> None:
        """Multiple servers are all included in config."""
        from inspect_ai.tool._mcp._config import MCPServerConfigHTTP

        from saber.agents.bridge_utils import resolve_mcp_servers

        configs = [
            MCPServerConfigHTTP(
                type="http",
                name="server_a",
                url="http://localhost:3001/mcp",
                tools="all",
            ),
            MCPServerConfigHTTP(
                type="http",
                name="server_b",
                url="http://localhost:3002/mcp",
                tools=["tool_x"],
            ),
        ]
        mcp_json_str, allowed_tools = resolve_mcp_servers(configs)
        mcp_json = json.loads(mcp_json_str)

        assert "server_a" in mcp_json["mcpServers"]
        assert "server_b" in mcp_json["mcpServers"]
        assert "mcp__server_a__*" in allowed_tools
        assert "mcp__server_b__tool_x" in allowed_tools

    def test_excludes_none_headers(self) -> None:
        """None headers are excluded from serialization."""
        from inspect_ai.tool._mcp._config import MCPServerConfigHTTP

        from saber.agents.bridge_utils import resolve_mcp_servers

        configs = [
            MCPServerConfigHTTP(
                type="http",
                name="srv",
                url="http://localhost:3001/mcp",
                headers=None,
            ),
        ]
        mcp_json_str, _allowed = resolve_mcp_servers(configs)
        mcp_json = json.loads(mcp_json_str)

        assert "headers" not in mcp_json["mcpServers"]["srv"]


class TestValidateModelAvailability:
    """validate_model_availability pre-flight check."""

    @pytest.mark.asyncio
    async def test_raises_on_not_found_error(self) -> None:
        """Raises ModelValidationError when API returns not_found."""
        from saber.agents.bridge_utils import (
            ModelValidationError,
            validate_model_availability,
        )

        mock_api = MagicMock()
        mock_api.generate = AsyncMock(
            side_effect=Exception(
                "Error code: 404 - {'type': 'error', 'error': "
                "{'type': 'not_found_error', 'message': "
                "'model: claude-opus-4.6 was not found'}}"
            )
        )
        mock_model = MagicMock()
        mock_model.name = "anthropic/claude-opus-4.6"
        mock_model.api = mock_api

        with patch("inspect_ai.model._model.get_model", return_value=mock_model):
            with pytest.raises(ModelValidationError, match="Model validation failed"):
                await validate_model_availability()

    @pytest.mark.asyncio
    async def test_raises_on_authentication_error(self) -> None:
        """Raises ModelValidationError on authentication failures."""
        from saber.agents.bridge_utils import (
            ModelValidationError,
            validate_model_availability,
        )

        mock_api = MagicMock()
        mock_api.generate = AsyncMock(
            side_effect=Exception(
                "Error code: 401 - {'type': 'error', 'error': "
                "{'type': 'authentication_error', 'message': 'invalid x-api-key'}}"
            )
        )
        mock_model = MagicMock()
        mock_model.name = "anthropic/claude-sonnet-4-20250514"
        mock_model.api = mock_api

        with patch("inspect_ai.model._model.get_model", return_value=mock_model):
            with pytest.raises(ModelValidationError, match="Model validation failed"):
                await validate_model_availability()

    @pytest.mark.asyncio
    async def test_passes_on_success(self) -> None:
        """Does not raise when model responds successfully."""
        from saber.agents.bridge_utils import validate_model_availability

        mock_api = MagicMock()
        mock_api.generate = AsyncMock(return_value=MagicMock())
        mock_model = MagicMock()
        mock_model.name = "anthropic/claude-sonnet-4-20250514"
        mock_model.api = mock_api

        with patch("inspect_ai.model._model.get_model", return_value=mock_model):
            await validate_model_availability()  # Should not raise

    @pytest.mark.asyncio
    async def test_swallows_transient_errors(self) -> None:
        """Non-fatal errors (rate limits, network) are logged but not raised."""
        from saber.agents.bridge_utils import validate_model_availability

        mock_api = MagicMock()
        mock_api.generate = AsyncMock(side_effect=Exception("Connection timeout after 30s"))
        mock_model = MagicMock()
        mock_model.name = "anthropic/claude-sonnet-4-20250514"
        mock_model.api = mock_api

        with patch("inspect_ai.model._model.get_model", return_value=mock_model):
            # Should not raise — transient errors are swallowed
            await validate_model_availability()

    @pytest.mark.asyncio
    async def test_raises_on_permission_denied(self) -> None:
        """Raises ModelValidationError on permission_denied."""
        from saber.agents.bridge_utils import (
            ModelValidationError,
            validate_model_availability,
        )

        mock_api = MagicMock()
        mock_api.generate = AsyncMock(side_effect=Exception("permission_denied: insufficient quota"))
        mock_model = MagicMock()
        mock_model.name = "anthropic/claude-sonnet-4-20250514"
        mock_model.api = mock_api

        with patch("inspect_ai.model._model.get_model", return_value=mock_model):
            with pytest.raises(ModelValidationError):
                await validate_model_availability()

    def test_model_validation_error_is_runtime_error(self) -> None:
        """ModelValidationError is a subclass of RuntimeError."""
        from saber.agents.bridge_utils import ModelValidationError

        assert issubclass(ModelValidationError, RuntimeError)


class TestUploadSkillsToSandbox:
    """upload_skills_to_sandbox copies host files into sandbox."""

    @pytest.mark.asyncio
    async def test_upload_skills_writes_files(self, tmp_path: Path) -> None:
        """Creates 2 files and verifies write_file called with correct paths."""
        from saber.agents.bridge_utils import upload_skills_to_sandbox

        (tmp_path / "skill_a.md").write_bytes(b"# Skill A")
        (tmp_path / "skill_b.md").write_bytes(b"# Skill B")

        sbox = MagicMock()
        sbox.write_file = AsyncMock()

        await upload_skills_to_sandbox(sbox, tmp_path, ".claude/skills")

        assert sbox.write_file.call_count == 2
        calls = {call.args[0]: call.args[1] for call in sbox.write_file.call_args_list}
        assert calls[".claude/skills/skill_a.md"] == b"# Skill A"
        assert calls[".claude/skills/skill_b.md"] == b"# Skill B"

    @pytest.mark.asyncio
    async def test_upload_skills_nested_directories(self, tmp_path: Path) -> None:
        """Nested directory structure preserves relative paths."""
        from saber.agents.bridge_utils import upload_skills_to_sandbox

        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        (tmp_path / "root.md").write_bytes(b"root")
        (nested / "leaf.md").write_bytes(b"leaf")

        sbox = MagicMock()
        sbox.write_file = AsyncMock()

        await upload_skills_to_sandbox(sbox, tmp_path, ".github/skills")

        call_paths = {call.args[0] for call in sbox.write_file.call_args_list}
        assert ".github/skills/root.md" in call_paths
        assert ".github/skills/sub/deep/leaf.md" in call_paths

    @pytest.mark.asyncio
    async def test_upload_skills_returns_sandbox_base(self, tmp_path: Path) -> None:
        """Returns the sandbox_base string."""
        from saber.agents.bridge_utils import upload_skills_to_sandbox

        (tmp_path / "f.txt").write_bytes(b"x")

        sbox = MagicMock()
        sbox.write_file = AsyncMock()

        result = await upload_skills_to_sandbox(sbox, tmp_path, ".claude/skills")
        assert result == ".claude/skills"

    @pytest.mark.asyncio
    async def test_upload_skills_nonexistent_dir(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when directory does not exist."""
        from saber.agents.bridge_utils import upload_skills_to_sandbox

        sbox = MagicMock()
        sbox.write_file = AsyncMock()

        with pytest.raises(FileNotFoundError):
            await upload_skills_to_sandbox(sbox, tmp_path / "nope", ".claude/skills")

    @pytest.mark.asyncio
    async def test_upload_skills_not_a_directory(self, tmp_path: Path) -> None:
        """Raises ValueError when path is a file, not a directory."""
        from saber.agents.bridge_utils import upload_skills_to_sandbox

        file_path = tmp_path / "file.txt"
        file_path.write_bytes(b"data")

        sbox = MagicMock()
        sbox.write_file = AsyncMock()

        with pytest.raises(ValueError, match="not a directory"):
            await upload_skills_to_sandbox(sbox, file_path, ".claude/skills")

    @pytest.mark.asyncio
    async def test_upload_skills_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory results in no write_file calls."""
        from saber.agents.bridge_utils import upload_skills_to_sandbox

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        sbox = MagicMock()
        sbox.write_file = AsyncMock()

        result = await upload_skills_to_sandbox(sbox, empty_dir, ".github/skills")
        assert result == ".github/skills"
        sbox.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_skills_skips_symlinks(self, tmp_path: Path) -> None:
        """Symlinks are skipped for security."""
        from saber.agents.bridge_utils import upload_skills_to_sandbox

        real_file = tmp_path / "real.md"
        real_file.write_bytes(b"real content")

        link_file = tmp_path / "link.md"
        link_file.symlink_to(real_file)

        sbox = MagicMock()
        sbox.write_file = AsyncMock()

        await upload_skills_to_sandbox(sbox, tmp_path, ".claude/skills")

        assert sbox.write_file.call_count == 1
        call_paths = {call.args[0] for call in sbox.write_file.call_args_list}
        assert ".claude/skills/real.md" in call_paths
        assert ".claude/skills/link.md" not in call_paths


class TestCreateToolCallLimitFilter:
    """Tests for create_tool_call_limit_filter() — tool-call limiting logic."""

    def _make_assistant_msg(self, num_tool_calls: int = 1) -> object:
        """Create a ChatMessageAssistant with the given number of ToolCalls."""
        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.tool import ToolCall

        tool_calls = [
            ToolCall(
                id=f"tc_{i}",
                function=f"tool_{i}",
                arguments={"arg": "val"},
                type="function",
            )
            for i in range(num_tool_calls)
        ]
        return ChatMessageAssistant(content="ok", tool_calls=tool_calls)

    @pytest.mark.asyncio
    async def test_filter_returns_none_when_under_limit(self) -> None:
        """Normal case: filter returns None when tool call limit is not hit."""
        from inspect_ai.model import ChatMessageUser, GenerateConfig

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, _check = create_tool_call_limit_filter()

        messages: list[object] = [ChatMessageUser(content="hi")]
        tools: list[object] = [MagicMock()]
        config = GenerateConfig()

        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch("inspect_ai.util._limit.check_tool_call_limit"):
                result = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )
        assert result is None

    @pytest.mark.asyncio
    async def test_filter_returns_generate_input_on_limit(self) -> None:
        """When the tool call limit is reached, filter returns GenerateInput
        with empty tools and tool_choice='none'."""
        from inspect_ai.model import ChatMessageUser, GenerateConfig, GenerateInput
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, _check = create_tool_call_limit_filter()

        assistant_msg = self._make_assistant_msg(num_tool_calls=5)
        messages: list[object] = [
            ChatMessageUser(content="do stuff"),
            assistant_msg,
        ]
        tools: list[object] = [MagicMock(), MagicMock()]
        config = GenerateConfig()

        with patch("inspect_ai.util._limit.record_tool_call_usage") as mock_record:
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=5, limit=5),
            ):
                result = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )

        # Must return GenerateInput, NOT None
        assert isinstance(result, GenerateInput), f"Expected GenerateInput, got {type(result)}"
        # Empty tools
        assert result.tools == []
        # tool_choice should be "none"
        assert result.tool_choice == "none"
        # The limit message should have been appended to messages
        assert any("tool call limit" in str(m) for m in messages)
        mock_record.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_filter_grace_returns_generate_input(self) -> None:
        """During grace period, filter returns GenerateInput with empty tools."""
        from inspect_ai.model import ChatMessageUser, GenerateConfig, GenerateInput
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, _check = create_tool_call_limit_filter()

        # First call: trigger the limit
        assistant_msg = self._make_assistant_msg(num_tool_calls=3)
        messages: list[object] = [
            ChatMessageUser(content="do stuff"),
            assistant_msg,
        ]
        tools: list[object] = [MagicMock()]
        config = GenerateConfig()

        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=3, limit=3),
            ):
                result1 = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )

        assert isinstance(result1, GenerateInput)

        # Second call: grace period, same messages (no new tool calls)
        tools2: list[object] = [MagicMock()]
        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch("inspect_ai.util._limit.check_tool_call_limit"):
                result2 = await filt(
                    MagicMock(),
                    messages,
                    tools2,
                    "auto",
                    config,  # type: ignore[arg-type]
                )

        # Grace should return GenerateInput with empty tools
        assert isinstance(result2, GenerateInput), f"Grace period should return GenerateInput, got {type(result2)}"
        assert result2.tools == []
        assert result2.tool_choice == "none"

    @pytest.mark.asyncio
    async def test_filter_hard_stop_after_grace_exhausted(self) -> None:
        """After grace period is used up, filter returns ModelOutput."""
        from inspect_ai.model import ChatMessageUser, GenerateConfig, GenerateInput
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, _check = create_tool_call_limit_filter()

        assistant_msg = self._make_assistant_msg(num_tool_calls=3)
        messages: list[object] = [
            ChatMessageUser(content="do stuff"),
            assistant_msg,
        ]
        tools: list[object] = [MagicMock()]
        config = GenerateConfig()

        # Call 1: trigger limit (uses grace_remaining = 1)
        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=3, limit=3),
            ):
                result1 = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )
        assert isinstance(result1, GenerateInput)

        # Call 2: grace generation (decrements grace_remaining to 0)
        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch("inspect_ai.util._limit.check_tool_call_limit"):
                result2 = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )
        assert isinstance(result2, GenerateInput)

        # Call 3: grace exhausted — hard stop
        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch("inspect_ai.util._limit.check_tool_call_limit"):
                result3 = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )
        assert isinstance(result3, ModelOutput), f"Expected ModelOutput hard stop, got {type(result3)}"

    @pytest.mark.asyncio
    async def test_filter_no_grace_reset_on_subsequent_calls(self) -> None:
        """Once limit is exceeded, the _limit_exceeded check fires BEFORE
        the delta block, so grace does NOT reset — the key bug-3 test."""
        from inspect_ai.model import ChatMessageUser, GenerateConfig, GenerateInput
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, _check = create_tool_call_limit_filter()

        # Call 1: 3 tool calls, trigger limit
        msg1 = self._make_assistant_msg(num_tool_calls=3)
        messages: list[object] = [
            ChatMessageUser(content="go"),
            msg1,
        ]
        tools: list[object] = [MagicMock()]
        config = GenerateConfig()

        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=3, limit=3),
            ):
                await filt(MagicMock(), messages, tools, "auto", config)  # type: ignore[arg-type]

        # Call 2: model adds MORE tool calls during grace (simulates bug 3)
        msg2 = self._make_assistant_msg(num_tool_calls=2)
        messages.append(msg2)  # Now total = 5, delta = 2

        with patch("inspect_ai.util._limit.record_tool_call_usage") as mock_rec:
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=5, limit=3),
            ) as mock_check:
                result2 = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )

        # The _limit_exceeded check should fire BEFORE delta block,
        # so record_tool_call_usage and check_tool_call_limit should
        # NOT be called — grace should have decremented, not reset.
        mock_rec.assert_not_called()
        mock_check.assert_not_called()

        # Should be a grace generation (GenerateInput), not a reset
        assert isinstance(result2, GenerateInput)

        # Call 3: grace exhausted — hard stop
        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch("inspect_ai.util._limit.check_tool_call_limit"):
                result3 = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )
        assert isinstance(result3, ModelOutput), "Grace should be exhausted, expected ModelOutput hard stop"

    @pytest.mark.asyncio
    async def test_filter_injects_tool_results_for_orphaned_calls(self) -> None:
        """When limit is hit, dummy tool results are injected for any
        assistant tool_calls that lack a corresponding ChatMessageTool.

        This prevents OpenAI API 400 errors:
        'No tool output found for function call <id>'.
        """
        from inspect_ai.model import (
            ChatMessageTool,
            ChatMessageUser,
            GenerateConfig,
            GenerateInput,
        )
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, _check = create_tool_call_limit_filter()

        # Assistant made 3 tool calls, none have results yet
        assistant_msg = self._make_assistant_msg(num_tool_calls=3)
        messages: list[object] = [
            ChatMessageUser(content="do stuff"),
            assistant_msg,
        ]
        tools: list[object] = [MagicMock()]
        config = GenerateConfig()

        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=3, limit=3),
            ):
                result = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )

        assert isinstance(result, GenerateInput)

        # Verify dummy tool results were injected for all 3 orphaned calls
        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        assert len(tool_msgs) == 3
        orphaned_ids = {f"tc_{i}" for i in range(3)}
        injected_ids = {m.tool_call_id for m in tool_msgs}
        assert injected_ids == orphaned_ids

        # The limit user message should come AFTER the tool results
        user_msgs = [m for m in messages if isinstance(m, ChatMessageUser)]
        assert any("tool call limit" in str(m.content) for m in user_msgs)

        # Tool results should appear before the limit message in order
        last_user_idx = max(i for i, m in enumerate(messages) if isinstance(m, ChatMessageUser))
        for i, m in enumerate(messages):
            if isinstance(m, ChatMessageTool):
                assert i < last_user_idx, "Dummy tool results must appear before the limit user message"

    @pytest.mark.asyncio
    async def test_filter_skips_injection_for_already_resolved_calls(
        self,
    ) -> None:
        """When some tool calls already have results, only orphaned ones
        get dummy results injected."""
        from inspect_ai.model import (
            ChatMessageTool,
            ChatMessageUser,
            GenerateConfig,
            GenerateInput,
        )
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, _check = create_tool_call_limit_filter()

        # Assistant made 3 tool calls
        assistant_msg = self._make_assistant_msg(num_tool_calls=3)
        # But tool_0 already has a result
        existing_tool_result = ChatMessageTool(content="result for tool_0", tool_call_id="tc_0")
        messages: list[object] = [
            ChatMessageUser(content="do stuff"),
            assistant_msg,
            existing_tool_result,
        ]
        tools: list[object] = [MagicMock()]
        config = GenerateConfig()

        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=3, limit=3),
            ):
                result = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )

        assert isinstance(result, GenerateInput)

        # Should have the original tool result + 2 injected ones = 3 total
        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        assert len(tool_msgs) == 3

        # The original result should be unchanged
        original = [m for m in tool_msgs if m.tool_call_id == "tc_0"]
        assert len(original) == 1
        assert original[0].content == "result for tool_0"

        # The injected ones should be for tc_1 and tc_2
        injected = [m for m in tool_msgs if m.tool_call_id in ("tc_1", "tc_2")]
        assert len(injected) == 2

    @pytest.mark.asyncio
    async def test_filter_grace_also_patches_orphaned_calls(self) -> None:
        """During grace generations, orphaned tool calls in messages
        should also have dummy results injected."""
        from inspect_ai.model import (
            ChatMessageTool,
            ChatMessageUser,
            GenerateConfig,
            GenerateInput,
        )
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, _check = create_tool_call_limit_filter()

        # First call: trigger the limit (no orphans here)
        msg1 = self._make_assistant_msg(num_tool_calls=3)
        tool_results = [ChatMessageTool(content=f"r{i}", tool_call_id=f"tc_{i}") for i in range(3)]
        messages: list[object] = [
            ChatMessageUser(content="go"),
            msg1,
            *tool_results,
        ]
        tools: list[object] = [MagicMock()]
        config = GenerateConfig()

        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=3, limit=3),
            ):
                await filt(MagicMock(), messages, tools, "auto", config)  # type: ignore[arg-type]

        # Now simulate: the model made MORE tool calls during grace
        # (the agent CLI ignored the stop and made calls anyway)
        from inspect_ai.tool import ToolCall

        msg2_calls = [
            ToolCall(id="grace_tc_0", function="bash", arguments={}, type="function"),
            ToolCall(id="grace_tc_1", function="bash", arguments={}, type="function"),
        ]
        from inspect_ai.model import ChatMessageAssistant

        msg2 = ChatMessageAssistant(content="more work", tool_calls=msg2_calls)
        messages.append(msg2)

        # Second call: grace period — orphaned calls from msg2
        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch("inspect_ai.util._limit.check_tool_call_limit"):
                result2 = await filt(
                    MagicMock(),
                    messages,
                    tools,
                    "auto",
                    config,  # type: ignore[arg-type]
                )

        assert isinstance(result2, GenerateInput)

        # The orphaned grace calls should have dummy results
        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        grace_results = [m for m in tool_msgs if m.tool_call_id in ("grace_tc_0", "grace_tc_1")]
        assert len(grace_results) == 2

    @pytest.mark.asyncio
    async def test_check_after_exec_raises_when_limit_exceeded(self) -> None:
        """check_after_exec raises LimitExceededError after limit was hit."""
        from inspect_ai.model import ChatMessageUser, GenerateConfig
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        filt, check_after_exec = create_tool_call_limit_filter()

        # Trigger the limit
        assistant_msg = self._make_assistant_msg(num_tool_calls=2)
        messages: list[object] = [
            ChatMessageUser(content="go"),
            assistant_msg,
        ]
        tools: list[object] = [MagicMock()]
        config = GenerateConfig()

        with patch("inspect_ai.util._limit.record_tool_call_usage"):
            with patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=2, limit=2),
            ):
                await filt(MagicMock(), messages, tools, "auto", config)  # type: ignore[arg-type]

        # check_after_exec should raise
        with patch(
            "inspect_ai.util._limit.check_tool_call_limit",
            side_effect=LimitExceededError("tool_call", value=2, limit=2),
        ):
            with pytest.raises(LimitExceededError):
                check_after_exec()
