"""Claude Code agent solver — sandbox_agent_bridge integration.

Uses inspect_ai's ``sandbox_agent_bridge()`` to run the Claude Code CLI
binary INSIDE the Docker sandbox.  The bridge proxies model calls back
to inspect_ai's configured model.

Two-level factory pattern:
    ``create_agent(**kwargs)`` -> ``create_with_prompts(**prompt_kwargs)`` -> ``Solver``
"""

import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from saber.agents.bridge_utils import (
    build_bridged_tools_for_claude_code,
    build_system_prompt,
    build_user_prompt,
    create_tool_call_limit_filter,
    parse_bridge_stderr,
    resolve_mcp_servers,
    upload_skills_to_sandbox,
    validate_model_availability,
)
from saber.logging import get_logger

if TYPE_CHECKING:
    from inspect_ai.solver import Solver
    from inspect_ai.tool import Tool
    from inspect_ai.util import SandboxEnvironment

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_STEPS = 50
_STORE_PORT_KEY = "claude_code_model_port"
_DEFAULT_PORT_BASE = 3000
_AUTH_TOKEN = "sk-ant-api03-DOq5tyLPrk9M4hPE"


# ---------------------------------------------------------------------------
# Helper: resolve claude binary inside sandbox
# ---------------------------------------------------------------------------


async def _resolve_claude_binary(sbox: "SandboxEnvironment", version: str) -> str:
    """Locate the Claude Code CLI binary inside the sandbox.

    Args:
        sbox: inspect_ai SandboxEnvironment instance.
        version: ``"auto"`` to search, or an explicit binary path.

    Returns:
        Path to the claude binary.
    """
    if version != "auto":
        return version

    try:
        result = await sbox.exec(["which", "claude"])
        if result.returncode == 0 and result.stdout.strip():
            return str(result.stdout.strip())
    except Exception:
        logger.debug("'which claude' failed in sandbox", exc_info=True)

    return "claude"


# ---------------------------------------------------------------------------
# Helper: seed claude config in sandbox
# ---------------------------------------------------------------------------


async def _seed_claude_config(
    sbox: "SandboxEnvironment",
    api_key: str,
    user: str | None = None,
    cwd: str | None = None,
) -> None:
    """Write ``~/.claude/settings.json`` with an ``apiKeyHelper``.

    This is the authentication workaround used by claude code — the CLI
    reads the helper command to obtain the API key at runtime.

    Args:
        sbox: inspect_ai SandboxEnvironment instance.
        api_key: API key value to embed in the helper.
        user: Optional user to run the command as.
        cwd: Optional working directory for the command.
    """
    kwargs: dict[str, str] = {}
    if user is not None:
        kwargs["user"] = user
    if cwd is not None:
        kwargs["cwd"] = cwd
    await sbox.exec(
        cmd=[
            "bash",
            "-c",
            'mkdir -p "$HOME/.claude"'
            " && echo '"
            '{"apiKeyHelper": "echo ' + api_key + '"}'
            '\' > "$HOME/.claude/settings.json"',
        ],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Helper: build agent environment variables
# ---------------------------------------------------------------------------


def _build_agent_env(
    *,
    bridge_port: int,
    model: str,
    extra_env: "dict[str, str] | None" = None,
) -> dict[str, str]:
    """Build environment variables for the Claude Code CLI process.

    Follows the inspect_swe pattern of setting all model-related env vars
    to route through the bridge proxy.

    Args:
        bridge_port: Port where the bridge proxy is listening.
        model: Model identifier (typically ``"inspect"``).
        extra_env: Additional env vars to merge (overrides defaults).

    Returns:
        Dict of environment variables for ``sbox.exec(env=...)``.
    """
    env: dict[str, str] = {
        "ANTHROPIC_BASE_URL": f"http://localhost:{bridge_port}",
        "ANTHROPIC_AUTH_TOKEN": _AUTH_TOKEN,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "IS_SANDBOX": "1",
    }
    if extra_env:
        env.update(extra_env)
    return env


# ---------------------------------------------------------------------------
# Helper: build claude CLI command args (without binary)
# ---------------------------------------------------------------------------


def _build_claude_cmd(
    *,
    user_prompt: str,
    mcp_config_json: str,
    allowed_tools: Sequence[str],
    disallowed_tools: Sequence[str],
    session_id: str,
    has_assistant_response: bool,
    append_system_prompt: str | None = None,
    add_dirs: Sequence[str] = (),
) -> list[str]:
    """Build the Claude Code CLI argument list (binary handled at call site).

    Args:
        user_prompt: User-facing prompt (after ``--`` separator).
            The SABER system prompt is prepended here by the caller.
        mcp_config_json: Serialized MCP config JSON, or empty string.
        allowed_tools: Tool patterns for ``--allowed-tools``.
        disallowed_tools: Tool names to disallow.
        session_id: UUID for session management.
        has_assistant_response: If True, use ``--continue`` instead of
            ``--session-id`` (subsequent run in same session).
        append_system_prompt: Optional text to append to Claude Code's
            default system prompt via ``--append-system-prompt``.
        add_dirs: Paths to additional directories to pass via ``--add-dir``.

    Returns:
        Argument list suitable for prepending the binary path.
    """
    cmd: list[str] = [
        "--dangerously-skip-permissions",
        "--model",
        "inspect",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
    ]

    # Persona / append-system-prompt
    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    # Additional directories
    for dir_path in add_dirs:
        cmd.extend(["--add-dir", dir_path])

    # Session management
    if has_assistant_response:
        cmd.append("--continue")
    else:
        cmd.extend(["--session-id", session_id])

    # MCP server config
    if mcp_config_json:
        cmd.extend(["--mcp-config", mcp_config_json])
        for tool_pattern in allowed_tools:
            cmd.extend(["--allowed-tools", tool_pattern])

    # Disallowed tools
    for dt in disallowed_tools:
        cmd.extend(["--disallowed-tools", dt])

    # User prompt (after -- separator)
    cmd.extend(["--", user_prompt])

    return cmd


# ---------------------------------------------------------------------------
# Two-level factory
# ---------------------------------------------------------------------------


def create_agent(**kwargs: object) -> "Callable[..., Solver]":
    """Create a Claude Code agent factory with SABER integration.

    Uses ``sandbox_agent_bridge()`` to run the Claude Code CLI inside
    the Docker sandbox, proxying model calls back to inspect_ai.

    Returns:
        A callable that accepts prompt kwargs and returns a ``Solver``.
    """
    outer_kwargs = kwargs

    def create_with_prompts(
        instruction_prompt: str = "",
        assistant_prompt: str = "",
        tools: "Sequence[Tool] | None" = None,
        **extra_kwargs: object,
    ) -> "Solver":
        """Build a Solver that drives Claude Code CLI via sandbox_agent_bridge.

        Args:
            instruction_prompt: Main task instructions for the agent.
            assistant_prompt: Initial assistant message / persona.
            tools: Sequence of inspect_ai Tools to expose via MCP bridge.
            **extra_kwargs: Absorbed for forward compatibility.

        Returns:
            An inspect_ai ``Solver`` instance.
        """
        from inspect_ai.agent import Agent, AgentState, agent, as_solver, sandbox_agent_bridge
        from inspect_ai.util import sandbox as sandbox_env
        from inspect_ai.util import store, tool_call_limit

        sandbox_name: str = str(outer_kwargs.get("sandbox_name", "default"))
        version: str = str(outer_kwargs.get("version", "auto"))
        disallowed_tools: list[str] = list(  # type: ignore[call-overload]
            outer_kwargs.get("disallowed_tools", [])
        )
        max_steps: int = int(outer_kwargs.get("max_steps", _DEFAULT_MAX_STEPS))  # type: ignore[call-overload]
        _pf = outer_kwargs.get("persona_file")
        persona_file: str | None = str(_pf) if _pf else None
        _sd = outer_kwargs.get("skills_dir")
        skills_dir: str | None = str(_sd) if _sd else None

        our_system_prompt = build_system_prompt(
            instruction_prompt=instruction_prompt,
            assistant_prompt=assistant_prompt,
        )

        bridged = build_bridged_tools_for_claude_code(tools)

        @agent  # type: ignore[misc]
        def _claude_code_agent() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                # Fail fast on model misconfiguration (e.g. wrong name)
                await validate_model_availability()

                session_id = str(uuid.uuid4())
                port = store().get(_STORE_PORT_KEY, _DEFAULT_PORT_BASE) + 1
                store().set(_STORE_PORT_KEY, port)

                bridge_filter, check_tool_limit = create_tool_call_limit_filter()

                async with sandbox_agent_bridge(
                    state,
                    model="inspect",
                    port=port,
                    sandbox=sandbox_name,
                    bridged_tools=bridged,
                    filter=bridge_filter,
                ) as bridge:
                    sbox = sandbox_env(sandbox_name)
                    claude_binary = await _resolve_claude_binary(sbox, version)

                    # Build user prompt from message history
                    user_prompt, has_assistant_response = build_user_prompt(
                        state.messages,
                    )
                    if not user_prompt:
                        user_prompt = "Begin the task."

                    # Prepend the SABER system prompt to the user message
                    # instead of using --append-system-prompt, which would
                    # append to Claude Code's own default system prompt.
                    if our_system_prompt:
                        user_prompt = f"{our_system_prompt}\n\n{user_prompt}"

                    # Resolve MCP servers
                    mcp_config_json = ""
                    allowed_tools: list[str] = []
                    if bridge.mcp_server_configs:
                        mcp_config_json, allowed_tools = resolve_mcp_servers(
                            bridge.mcp_server_configs,
                        )

                    # Read persona file content from host
                    persona_content: str | None = None
                    if persona_file:
                        persona_path = Path(persona_file)
                        if not persona_path.is_file():
                            msg = f"Persona file not found: {persona_path}"
                            raise FileNotFoundError(msg)
                        persona_content = persona_path.read_text(encoding="utf-8")
                        logger.info(
                            "Loaded persona file: %s (%d chars)",
                            persona_path,
                            len(persona_content),
                        )

                    # Upload skills directory to sandbox
                    add_dirs: list[str] = []
                    if skills_dir:
                        sandbox_skills = await upload_skills_to_sandbox(sbox, skills_dir, ".claude/skills")
                        add_dirs.append(sandbox_skills)

                    # Build CLI args
                    cmd_args = _build_claude_cmd(
                        user_prompt=user_prompt,
                        mcp_config_json=mcp_config_json,
                        allowed_tools=allowed_tools,
                        disallowed_tools=disallowed_tools,
                        session_id=session_id,
                        has_assistant_response=has_assistant_response,
                        append_system_prompt=persona_content,
                        add_dirs=add_dirs,
                    )

                    # Build environment and seed config
                    agent_env = _build_agent_env(
                        bridge_port=bridge.port,
                        model="inspect",
                    )
                    await _seed_claude_config(sbox, _AUTH_TOKEN)

                    # Execute with stdin protection.
                    # No timeout — inspect_ai's --time-limit governs the
                    # overall sample wall-clock budget.  A redundant
                    # sbox.exec timeout can orphan the CLI process inside
                    # the container (docker compose exec without a TTY
                    # does not propagate signals) and trigger spurious
                    # "session already in use" errors on retry.
                    agent_cmd = [claude_binary] + cmd_args
                    result = await sbox.exec(
                        [
                            "bash",
                            "-c",
                            'exec 0</dev/null; "$@"',
                            "bash",
                        ]
                        + agent_cmd,
                        env=agent_env,
                    )
                    if result.returncode != 0:
                        stderr = result.stderr or ""
                        stdout = result.stdout or ""

                        # Parse bridge proxy errors for structured diagnostics
                        diagnostics = parse_bridge_stderr(stderr)
                        if diagnostics:
                            logger.error(
                                "Bridge proxy error detected in Claude Code CLI:\n%s",
                                diagnostics,
                            )

                        # Log full stderr at debug level for forensics
                        if stderr:
                            logger.debug(
                                "Claude Code CLI full stderr (%d chars):\n%s",
                                len(stderr),
                                stderr[:2000],
                            )

                        # If the bridge has accumulated conversation state
                        # (at least one successful model call), salvage it
                        # instead of crashing.  This handles transient
                        # failures like ECONNREFUSED on a follow-up call
                        # after the main conversation already completed.
                        if bridge.state.messages:
                            detail = stderr[:300] if stderr else stdout[:300] if stdout else "(no output)"
                            logger.warning(
                                "Claude Code CLI exited with code %d but bridge "
                                "has %d messages — returning salvaged state. "
                                "Detail: %s",
                                result.returncode,
                                len(bridge.state.messages),
                                detail,
                            )
                            return bridge.state

                        # Truncated detail for the exception message
                        detail = stderr[:500] if stderr else stdout[:500] if stdout else "(no output)"
                        msg = f"Claude Code CLI exited with code {result.returncode}: {detail}"
                        logger.error(msg)
                        raise RuntimeError(msg)

                    # Log bridge warnings even on success
                    if result.returncode == 0 and result.stderr:
                        warnings = parse_bridge_stderr(result.stderr)
                        if warnings:
                            logger.warning(
                                "Bridge proxy warnings in Claude Code CLI (exit 0):\n%s",
                                warnings,
                            )

                    # Raise LimitExceededError in the main flow if the
                    # tool-call limit was hit so that apply_limits handles
                    # it cleanly (rather than via cancel-scope propagation
                    # which can stall waiting for the subprocess).
                    check_tool_limit()

                    return bridge.state

            return execute

        return as_solver(
            _claude_code_agent(),
            limits=[tool_call_limit(max_steps)],
        )

    return create_with_prompts
