"""Tests for claude_code agent module — sandbox_agent_bridge architecture."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Factory interface tests
# ---------------------------------------------------------------------------


class TestClaudeCodeCreateAgent:
    """claude_code.solver.create_agent two-level factory."""

    def test_create_agent_is_callable(self) -> None:
        """create_agent function exists and is callable."""
        from saber.agents.registry.claude_code.solver import create_agent

        assert callable(create_agent)

    def test_create_agent_returns_callable(self) -> None:
        """create_agent() returns create_with_prompts callable."""
        from saber.agents.registry.claude_code.solver import create_agent

        create_with_prompts = create_agent()
        assert callable(create_with_prompts)

    def test_create_with_prompts_signature_has_standard_kwargs(self) -> None:
        """create_with_prompts accepts all standard prompt kwargs."""
        from saber.agents.registry.claude_code.solver import create_agent

        create_with_prompts = create_agent()
        sig = inspect.signature(create_with_prompts)
        param_names = set(sig.parameters.keys())

        for name in (
            "instruction_prompt",
            "assistant_prompt",
            "tools",
            "extra_kwargs",
        ):
            assert name in param_names, f"missing parameter: {name}"

    def test_create_with_prompts_accepts_tools(self) -> None:
        """create_with_prompts has a 'tools' parameter (bridge architecture)."""
        from saber.agents.registry.claude_code.solver import create_agent

        create_with_prompts = create_agent()
        sig = inspect.signature(create_with_prompts)
        assert "tools" in sig.parameters

    def test_create_with_prompts_returns_solver(self) -> None:
        """create_with_prompts() returns a Solver instance."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.claude_code.solver import create_agent

        create_with_prompts = create_agent()
        solver = create_with_prompts(instruction_prompt="Do the task.", max_steps=200)
        assert isinstance(solver, Solver)

    def test_create_with_prompts_returns_solver_with_tools(self) -> None:
        """create_with_prompts() returns a Solver when tools are provided."""
        from inspect_ai.solver import Solver
        from inspect_ai.tool import bash

        from saber.agents.registry.claude_code.solver import create_agent

        create_with_prompts = create_agent()
        solver = create_with_prompts(
            instruction_prompt="Do the task.",
            tools=[bash()],
            max_steps=200,
        )
        assert isinstance(solver, Solver)

    def test_module_exports(self) -> None:
        """Package __init__ exports create_agent."""
        from saber.agents.registry.claude_code import create_agent

        assert callable(create_agent)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """build_system_prompt helper (shared via bridge_utils)."""

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


class TestBuildClaudeCmd:
    """_build_claude_cmd helper."""

    def test_basic_command_with_session_id(self) -> None:
        """Builds basic claude CLI args with session-id for first run."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Begin.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="abc-123",
            has_assistant_response=False,
        )
        assert "--dangerously-skip-permissions" in cmd
        assert "--model" in cmd
        assert "inspect" in cmd
        assert "--print" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd
        # Session management
        assert "--session-id" in cmd
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == "abc-123"
        # -- separator before prompt
        assert "--" in cmd
        dash_idx = cmd.index("--")
        assert cmd[dash_idx + 1] == "Begin."

    def test_continue_mode_for_subsequent_runs(self) -> None:
        """Uses --continue instead of --session-id on subsequent runs."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Continue.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="abc-123",
            has_assistant_response=True,
        )
        assert "--continue" in cmd
        assert "--session-id" not in cmd

    def test_no_append_system_prompt_flag(self) -> None:
        """System prompt is no longer passed via --append-system-prompt."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
        )
        assert "--append-system-prompt" not in cmd

    def test_user_prompt_appears_after_separator(self) -> None:
        """User prompt (which may include system prompt) appears after -- separator."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="System context here\n\nActual prompt.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
        )
        dash_idx = cmd.index("--")
        assert cmd[dash_idx + 1] == "System context here\n\nActual prompt."

    def test_mcp_config_from_json(self) -> None:
        """Adds --mcp-config and --allowed-tools from resolved values."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        mcp_json = json.dumps({"mcpServers": {"srv": {"type": "http", "url": "http://localhost:3001"}}})
        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json=mcp_json,
            allowed_tools=["mcp__srv__*"],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
        )
        assert "--mcp-config" in cmd
        mcp_idx = cmd.index("--mcp-config")
        assert cmd[mcp_idx + 1] == mcp_json
        assert "--allowed-tools" in cmd
        at_idx = cmd.index("--allowed-tools")
        assert cmd[at_idx + 1] == "mcp__srv__*"

    def test_disallowed_tools(self) -> None:
        """Adds --disallowed-tools flags."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=["Edit", "WebFetch"],
            session_id="id-1",
            has_assistant_response=False,
        )
        dt_indices = [i for i, v in enumerate(cmd) if v == "--disallowed-tools"]
        assert len(dt_indices) == 2
        disallowed_values = [cmd[i + 1] for i in dt_indices]
        assert "Edit" in disallowed_values
        assert "WebFetch" in disallowed_values

    def test_separator_before_user_prompt(self) -> None:
        """-- separator appears directly before user prompt."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="My prompt here.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
        )
        # Last two elements should be ["--", "My prompt here."]
        assert cmd[-2] == "--"
        assert cmd[-1] == "My prompt here."

    def test_does_not_include_binary(self) -> None:
        """Command args do not include the binary — handled at call site."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
        )
        # First element should be a flag, not a binary path
        assert cmd[0].startswith("--")

    def test_build_claude_cmd_append_system_prompt(self) -> None:
        """When append_system_prompt is provided, cmd contains --append-system-prompt."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
            append_system_prompt="Custom persona",
        )
        assert "--append-system-prompt" in cmd
        idx = cmd.index("--append-system-prompt")
        assert cmd[idx + 1] == "Custom persona"

    def test_build_claude_cmd_no_append_system_prompt(self) -> None:
        """When append_system_prompt=None, no --append-system-prompt in cmd."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
            append_system_prompt=None,
        )
        assert "--append-system-prompt" not in cmd

    def test_build_claude_cmd_empty_append_system_prompt(self) -> None:
        """When append_system_prompt is empty string, no --append-system-prompt in cmd."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
            append_system_prompt="",
        )
        assert "--append-system-prompt" not in cmd

    def test_build_claude_cmd_add_dirs_single(self) -> None:
        """When add_dirs has one entry, cmd contains --add-dir with that path."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
            add_dirs=[".claude/skills"],
        )
        assert "--add-dir" in cmd
        idx = cmd.index("--add-dir")
        assert cmd[idx + 1] == ".claude/skills"

    def test_build_claude_cmd_add_dirs_multiple(self) -> None:
        """When add_dirs has two entries, cmd has two --add-dir flags."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
            add_dirs=[".claude/skills", "/extra/dir"],
        )
        add_dir_indices = [i for i, v in enumerate(cmd) if v == "--add-dir"]
        assert len(add_dir_indices) == 2
        add_dir_values = [cmd[i + 1] for i in add_dir_indices]
        assert ".claude/skills" in add_dir_values
        assert "/extra/dir" in add_dir_values

    def test_build_claude_cmd_no_add_dirs(self) -> None:
        """When add_dirs is empty, no --add-dir in cmd."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="Go.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
            add_dirs=(),
        )
        assert "--add-dir" not in cmd

    def test_build_claude_cmd_prompt_is_last(self) -> None:
        """Even with persona+dirs, ['--', user_prompt] is still last."""
        from saber.agents.registry.claude_code.solver import _build_claude_cmd

        cmd = _build_claude_cmd(
            user_prompt="My prompt.",
            mcp_config_json="",
            allowed_tools=[],
            disallowed_tools=[],
            session_id="id-1",
            has_assistant_response=False,
            append_system_prompt="Persona text",
            add_dirs=[".claude/skills", "/other"],
        )
        assert cmd[-2] == "--"
        assert cmd[-1] == "My prompt."


class TestClaudeCodePersonaSkills:
    """Persona file and skills dir support in create_agent."""

    def test_create_agent_passes_persona_file_to_inner(self) -> None:
        """create_agent(persona_file=...) produces a Solver via create_with_prompts."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.claude_code.solver import create_agent

        create_with_prompts = create_agent(persona_file="/tmp/persona.md")
        solver = create_with_prompts(instruction_prompt="Do the task.", max_steps=200)
        assert isinstance(solver, Solver)

    def test_create_agent_passes_skills_dir_to_inner(self) -> None:
        """create_agent(skills_dir=...) produces a Solver via create_with_prompts."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.claude_code.solver import create_agent

        create_with_prompts = create_agent(skills_dir="/tmp/skills")
        solver = create_with_prompts(instruction_prompt="Do the task.", max_steps=200)
        assert isinstance(solver, Solver)

    def test_create_agent_accepts_agent_bundle_options(self) -> None:
        """Runtime bundle kwargs produce a Solver without scenario YAML changes."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.claude_code.solver import create_agent

        create_with_prompts = create_agent(
            agent_bundle="/tmp/recon-agent",
            main_agent="Recon Agent",
            mcp_config="/tmp/recon-agent/.mcp.json",
        )
        solver = create_with_prompts(instruction_prompt="Do the task.", max_steps=200)
        assert isinstance(solver, Solver)

    def test_solver_stages_claude_agents_directory(self) -> None:
        """Claude Code solver contains bundle staging hooks for .claude/agents."""
        import saber.agents.registry.claude_code.solver as mod

        source = inspect.getsource(mod)
        assert "upload_agent_bundle_to_sandbox" in source
        assert '".claude/agents"' in source


class TestBuildAgentEnv:
    """_build_agent_env returns environment variables dict."""

    def test_contains_required_keys(self) -> None:
        """All required environment variables are present."""
        from saber.agents.registry.claude_code.solver import _build_agent_env

        env = _build_agent_env(bridge_port=13131, model="inspect")
        required_keys = {
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
            "IS_SANDBOX",
        }
        assert required_keys.issubset(set(env.keys()))

    def test_base_url_uses_bridge_port(self) -> None:
        """ANTHROPIC_BASE_URL points to bridge port."""
        from saber.agents.registry.claude_code.solver import _build_agent_env

        env = _build_agent_env(bridge_port=5555, model="inspect")
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:5555"

    def test_model_set_across_all_keys(self) -> None:
        """Model name propagated to all model-related keys."""
        from saber.agents.registry.claude_code.solver import _build_agent_env

        env = _build_agent_env(bridge_port=3000, model="inspect")
        model_keys = [
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
        ]
        for key in model_keys:
            assert env[key] == "inspect"

    def test_extra_env_overrides(self) -> None:
        """Extra env vars are merged and can override defaults."""
        from saber.agents.registry.claude_code.solver import _build_agent_env

        env = _build_agent_env(
            bridge_port=3000,
            model="inspect",
            extra_env={"CUSTOM_VAR": "custom_value", "IS_SANDBOX": "0"},
        )
        assert env["CUSTOM_VAR"] == "custom_value"
        assert env["IS_SANDBOX"] == "0"  # overridden

    def test_no_extra_env(self) -> None:
        """Works without extra_env parameter."""
        from saber.agents.registry.claude_code.solver import _build_agent_env

        env = _build_agent_env(bridge_port=3000, model="inspect")
        assert env["IS_SANDBOX"] == "1"


class TestSeedClaudeConfig:
    """_seed_claude_config writes settings.json to sandbox."""

    @pytest.mark.asyncio
    async def test_creates_settings_file(self) -> None:
        """Calls sbox.exec with bash command to create settings."""
        from unittest.mock import AsyncMock

        from saber.agents.registry.claude_code.solver import _seed_claude_config

        sbox = AsyncMock()
        sbox.exec.return_value = AsyncMock(returncode=0)

        await _seed_claude_config(sbox, "sk-ant-test-key")

        sbox.exec.assert_called_once()
        call_args = sbox.exec.call_args
        # cmd is passed as keyword arg
        cmd = call_args.kwargs.get("cmd") or call_args[0][0]
        assert cmd[0] == "bash"
        assert cmd[1] == "-c"
        # The bash script should create .claude dir and write settings.json
        script = cmd[2]
        assert ".claude" in script
        assert "settings.json" in script
        assert "sk-ant-test-key" in script
        assert "apiKeyHelper" in script

    @pytest.mark.asyncio
    async def test_forwards_user_param(self) -> None:
        """Passes user kwarg through to sbox.exec."""
        from unittest.mock import AsyncMock

        from saber.agents.registry.claude_code.solver import _seed_claude_config

        sbox = AsyncMock()
        sbox.exec.return_value = AsyncMock(returncode=0)

        await _seed_claude_config(sbox, "sk-test", user="root")

        call_kwargs = sbox.exec.call_args.kwargs
        assert call_kwargs.get("user") == "root"

    @pytest.mark.asyncio
    async def test_forwards_cwd_param(self) -> None:
        """Passes cwd kwarg through to sbox.exec."""
        from unittest.mock import AsyncMock

        from saber.agents.registry.claude_code.solver import _seed_claude_config

        sbox = AsyncMock()
        sbox.exec.return_value = AsyncMock(returncode=0)

        await _seed_claude_config(sbox, "sk-test", cwd="/home/user")

        call_kwargs = sbox.exec.call_args.kwargs
        assert call_kwargs.get("cwd") == "/home/user"

    @pytest.mark.asyncio
    async def test_no_extra_kwargs_when_none(self) -> None:
        """Does not pass user/cwd kwargs when they are None."""
        from unittest.mock import AsyncMock

        from saber.agents.registry.claude_code.solver import _seed_claude_config

        sbox = AsyncMock()
        sbox.exec.return_value = AsyncMock(returncode=0)

        await _seed_claude_config(sbox, "sk-test")

        call_kwargs = sbox.exec.call_args.kwargs
        assert "user" not in call_kwargs
        assert "cwd" not in call_kwargs


class TestResolveBinary:
    """_resolve_claude_binary helper."""

    @pytest.mark.asyncio
    async def test_auto_finds_existing_binary(self) -> None:
        """Auto version resolves to found binary path."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from saber.agents.registry.claude_code.solver import _resolve_claude_binary

        sbox = AsyncMock()
        sbox.exec.return_value = SimpleNamespace(returncode=0, stdout="/usr/local/bin/claude\n")

        result = await _resolve_claude_binary(sbox, "auto")
        assert result == "/usr/local/bin/claude"

    @pytest.mark.asyncio
    async def test_auto_fallback_when_not_found(self) -> None:
        """Auto version falls back to 'claude' when which fails."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from saber.agents.registry.claude_code.solver import _resolve_claude_binary

        sbox = AsyncMock()
        sbox.exec.return_value = SimpleNamespace(returncode=1, stdout="")

        result = await _resolve_claude_binary(sbox, "auto")
        assert result == "claude"

    @pytest.mark.asyncio
    async def test_explicit_binary_path(self) -> None:
        """Non-auto version returns the version string as binary name."""
        from unittest.mock import AsyncMock

        from saber.agents.registry.claude_code.solver import _resolve_claude_binary

        sbox = AsyncMock()
        result = await _resolve_claude_binary(sbox, "/opt/claude-v2")
        assert result == "/opt/claude-v2"
        sbox.exec.assert_not_called()


class TestSessionIdPerExecution:
    """session_id must be unique per execute() call, not per solver."""

    def test_session_id_not_in_create_with_prompts_scope(self) -> None:
        """session_id is generated inside execute(), not create_with_prompts()."""
        import ast
        import textwrap

        from saber.agents.registry.claude_code import solver as mod

        source = textwrap.dedent(inspect.getsource(mod.create_agent))
        tree = ast.parse(source)

        # Walk AST to find where uuid.uuid4() or uuid4() is called
        # It must be inside the execute() function, not in create_with_prompts
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "create_with_prompts":
                # Direct body assignments (not nested in execute) should NOT have uuid
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        src = ast.dump(stmt)
                        assert "uuid" not in src.lower(), (
                            "session_id = uuid.uuid4() must be inside execute(), not in create_with_prompts() body"
                        )

    def test_build_claude_cmd_receives_session_id_param(self) -> None:
        """_build_claude_cmd accepts session_id as a parameter."""
        sig = inspect.signature(
            __import__(
                "saber.agents.registry.claude_code.solver",
                fromlist=["_build_claude_cmd"],
            )._build_claude_cmd
        )
        assert "session_id" in sig.parameters


class TestSboxTyping:
    """_resolve_claude_binary and _seed_claude_config use SandboxEnvironment type."""

    def test_resolve_binary_uses_sandbox_type(self) -> None:
        """_resolve_claude_binary parameter annotation is not 'object'."""
        from saber.agents.registry.claude_code.solver import _resolve_claude_binary

        sig = inspect.signature(_resolve_claude_binary)
        annotation = sig.parameters["sbox"].annotation
        assert annotation is not object, "sbox should not be typed as 'object'"
        assert "object" not in str(annotation), "sbox annotation should not be 'object'"

    def test_seed_config_uses_sandbox_type(self) -> None:
        """_seed_claude_config parameter annotation is not 'object'."""
        from saber.agents.registry.claude_code.solver import _seed_claude_config

        sig = inspect.signature(_seed_claude_config)
        annotation = sig.parameters["sbox"].annotation
        assert annotation is not object, "sbox should not be typed as 'object'"
        assert "object" not in str(annotation), "sbox annotation should not be 'object'"


class TestBuildBridgedTools:
    """build_bridged_tools helper (shared via bridge_utils)."""

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


class TestClaudeCodeTrackingIntegration:
    """Verify tracking filter is wired into Claude Code solver."""

    def test_imports_tracking_filter(self) -> None:
        """Claude Code solver imports create_tracking_filter."""
        from saber.agents.registry.claude_code import solver

        source = Path(solver.__file__).read_text()
        assert "create_tracking_filter" in source

    def test_imports_compose_filters(self) -> None:
        """Claude Code solver imports compose_filters."""
        from saber.agents.registry.claude_code import solver

        source = Path(solver.__file__).read_text()
        assert "compose_filters" in source


class TestClaudeCodeModelAliases:
    """Verify model_aliases support is wired into Claude Code solver."""

    def test_imports_resolve_model_aliases(self) -> None:
        """Claude Code solver imports resolve_model_aliases."""
        from saber.agents.registry.claude_code import solver

        source = Path(solver.__file__).read_text()
        assert "resolve_model_aliases" in source

class TestClaudeCodeCLIParserIntegration:
    """Verify CLI output parser is wired into Claude Code solver."""

    def test_imports_cli_parser(self) -> None:
        """Claude Code solver imports parse_claude_code_stream_json."""
        from saber.agents.registry.claude_code import solver

        source = Path(solver.__file__).read_text()
        assert "parse_claude_code_stream_json" in source

    def test_uses_record_bridge_summary(self) -> None:
        """Claude Code solver uses the shared record_bridge_summary helper."""
        from saber.agents.registry.claude_code import solver

        source = Path(solver.__file__).read_text()
        assert "record_bridge_summary" in source
