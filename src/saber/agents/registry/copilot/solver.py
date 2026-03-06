"""Copilot agent solver — sandbox_agent_bridge integration.

Uses inspect_ai's ``sandbox_agent_bridge()`` to run a self-contained
Copilot SDK runner script INSIDE the Docker sandbox.  The bridge proxies
model API calls back to inspect_ai's configured model, so the Copilot SDK
never needs direct access to the real model endpoint.

Two-level factory pattern:
    ``create_agent(**kwargs)`` -> ``create_with_prompts(**prompt_kwargs)`` -> ``Solver``

Architecture:
    1. solver writes RUNNER_SCRIPT to sandbox filesystem
    2. sandbox_agent_bridge() opens a local proxy on a free port
    3. runner script starts CopilotClient with BYOK provider pointed at proxy
    4. all model traffic flows: runner -> bridge proxy -> inspect_ai model
    5. bridge.state captures the full message transcript
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from saber.agents.bridge_utils import (
    build_bridged_tools_for_copilot,
    build_system_prompt,
    build_user_prompt,
    create_tool_call_limit_filter,
    parse_bridge_stderr,
    upload_skills_to_sandbox,
    validate_model_availability,
)
from saber.logging import get_logger

if TYPE_CHECKING:
    from inspect_ai.solver import Solver
    from inspect_ai.tool import Tool

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STORE_PORT_KEY = "copilot_model_port"
_DEFAULT_PORT_BASE = 3000
_RUNNER_PATH = "/tmp/_copilot_runner.py"

# ---------------------------------------------------------------------------
# Embedded runner script (executed inside the Docker sandbox)
# ---------------------------------------------------------------------------

RUNNER_SCRIPT = r'''#!/usr/bin/env python3
"""Copilot runner — executes inside the Docker sandbox.

Reads configuration from environment variables:
- OPENAI_BASE_URL: Bridge proxy URL (e.g., http://localhost:13131/v1)
- OPENAI_API_KEY: Placeholder API key for the bridge
- COPILOT_MODEL: Model name to request (routed through bridge)
- COPILOT_PROMPT: User prompt text (SABER system prompt prepended)
- COPILOT_MCP_CONFIG: JSON string of MCP server configs (optional)
- COPILOT_PERSONA_PROMPT: Persona prompt text to append to system message (optional)
- COPILOT_TIMEOUT: Timeout in seconds for session.send_and_wait (default: 3600)
"""

import asyncio
import json
import os
import sys


def _approve_all(
    _request: dict,
    _context: dict,
) -> dict:
    """Auto-approve every permission request.

    The Copilot SDK's PermissionHandler signature is
    ``(PermissionRequest, Dict[str, str]) -> PermissionRequestResult``.
    Returning ``{"kind": "approved"}`` grants the request.
    This is safe because the runner executes inside an isolated Docker
    sandbox used exclusively for benchmarking.
    """
    return {"kind": "approved"}


async def main() -> int:
    """Run Copilot session with bridge-routed model."""
    from copilot import CopilotClient

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:13131/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    model = os.environ.get("COPILOT_MODEL", "inspect")
    prompt = os.environ.get("COPILOT_PROMPT", "")
    mcp_config_str = os.environ.get("COPILOT_MCP_CONFIG", "")
    timeout = int(os.environ.get("COPILOT_TIMEOUT", "3600"))

    if not prompt:
        print("ERROR: COPILOT_PROMPT is required", file=sys.stderr)
        return 1

    # Build session config
    # wire_api="responses" tells the Copilot CLI to use the OpenAI
    # Responses API (POST /v1/responses) instead of Chat Completions.
    # The bridge's Chat Completions handler serialises
    # ChatCompletion.tool_calls as ``null`` for text-only model turns
    # (Pydantic model_dump default).  The CLI's JS code accesses
    # ``tool_calls.length`` without a null-guard, crashing with
    # "TypeError: Cannot read properties of null (reading 'length')".
    # The Responses API schema has no ``tool_calls`` field at all —
    # tool calls are separate output items in a list, so the null
    # issue never arises.
    session_config: dict = {
        "model": model,
        "provider": {
            "type": "openai",
            "base_url": base_url,
            "api_key": api_key,
            "wire_api": "responses",
        },
        "on_permission_request": _approve_all,
    }

    # MCP servers from bridge
    if mcp_config_str:
        try:
            mcp_servers = json.loads(mcp_config_str)
            if mcp_servers:
                session_config["mcp_servers"] = mcp_servers
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid COPILOT_MCP_CONFIG: {mcp_config_str} — {exc}", file=sys.stderr)
            return 1

    # Persona prompt — appended to the default Copilot system message
    persona_prompt = os.environ.get("COPILOT_PERSONA_PROMPT", "")
    if persona_prompt:
        session_config["system_message"] = {
            "mode": "append",
            "content": persona_prompt,
        }

    # Skill directories (paths inside sandbox, uploaded from host)
    skill_dirs_str = os.environ.get("COPILOT_SKILL_DIRECTORIES", "")
    if skill_dirs_str:
        try:
            skill_dirs = json.loads(skill_dirs_str)
            if skill_dirs:
                session_config["skill_directories"] = skill_dirs
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid COPILOT_SKILL_DIRECTORIES: {skill_dirs_str} — {exc}", file=sys.stderr)
            return 1

    client = CopilotClient()
    try:
        await client.start()
        session = await client.create_session(session_config)

        try:
            response = await session.send_and_wait(
                {"prompt": prompt},
                timeout=timeout,
            )
            if response:
                content = getattr(getattr(response, "data", None), "content", None)
                if content:
                    print(f"COPILOT_RESPONSE: {content[:500]}", file=sys.stderr)
        except Exception as exc:
            print(f"ERROR: Session failed: {exc}", file=sys.stderr)
            return 1
        finally:
            await session.destroy()
    finally:
        await client.stop()

    print("COPILOT_RUNNER_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
'''


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class CopilotBridgeConfig(BaseModel):
    """Bridge-relevant configuration for the copilot agent.

    Minimal config — the bridge handles model routing, BridgedToolsSpec
    handles tools.  Only sandbox/port configuration remains.
    """

    model_config = ConfigDict(frozen=True)

    sandbox_name: str = "default"
    max_steps: int = 50
    port_base: int = _DEFAULT_PORT_BASE
    model: str = "inspect"
    persona_file: str | None = None
    skills_dir: str | None = None

    @classmethod
    def from_kwargs(cls, kwargs: "dict[str, object]") -> "CopilotBridgeConfig":
        """Extract typed config from untyped ``**kwargs``.

        Args:
            kwargs: Raw keyword arguments from ``create_agent()``.

        Returns:
            Validated config with defaults for missing fields.
        """
        return cls.model_validate({k: v for k, v in kwargs.items() if k in cls.model_fields})


# ---------------------------------------------------------------------------
# Helper: build runner environment variables
# ---------------------------------------------------------------------------


def _build_runner_env(
    *,
    bridge_port: int,
    model: str,
    prompt: str,
    mcp_configs: Sequence[object],
    persona_prompt: str = "",
    skill_directories_json: str = "[]",
) -> dict[str, str]:
    """Build environment variables for the runner script.

    Args:
        bridge_port: Port where the bridge proxy is listening.
        model: Model identifier to pass to the runner.
        prompt: User prompt text (SABER system prompt already prepended).
        mcp_configs: MCP server config objects from bridge
            (each with ``.name``, ``.url``, ``.type`` attributes).
        persona_prompt: Persona prompt text to append to the Copilot
            system message via ``system_message`` mode=append.
        skill_directories_json: JSON-serialized list of sandbox skill
            directory paths.

    Returns:
        Dict of env vars to pass to ``sbox.exec()``.
    """
    mcp_list = [
        {
            "name": getattr(c, "name", ""),
            "url": getattr(c, "url", ""),
            "type": getattr(c, "type", "http"),
        }
        for c in mcp_configs
    ]
    return {
        "OPENAI_BASE_URL": f"http://localhost:{bridge_port}/v1",
        "OPENAI_API_KEY": "sk-placeholder-for-bridge",
        "COPILOT_MODEL": model,
        "COPILOT_PROMPT": prompt,
        "COPILOT_MCP_CONFIG": json.dumps(mcp_list),
        "COPILOT_PERSONA_PROMPT": persona_prompt,
        "COPILOT_SKILL_DIRECTORIES": skill_directories_json,
    }


# ---------------------------------------------------------------------------
# Two-level factory
# ---------------------------------------------------------------------------


def create_agent(**kwargs: object) -> "Callable[..., Solver]":
    """Create a Copilot agent factory using sandbox_agent_bridge.

    Uses ``sandbox_agent_bridge()`` to run the Copilot SDK runner script
    inside the Docker sandbox, proxying model calls back to inspect_ai.

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
        """Build a Solver that drives Copilot via sandbox_agent_bridge.

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

        config = CopilotBridgeConfig.from_kwargs(dict(outer_kwargs))

        system_prompt = build_system_prompt(
            instruction_prompt=instruction_prompt,
            assistant_prompt=assistant_prompt,
        )

        bridged = build_bridged_tools_for_copilot(tools)

        @agent  # type: ignore[misc]
        def _copilot_agent() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                # Fail fast on model misconfiguration (e.g. wrong name)
                await validate_model_availability()

                port = store().get(_STORE_PORT_KEY, config.port_base) + 1
                store().set(_STORE_PORT_KEY, port)

                bridge_filter, check_tool_limit = create_tool_call_limit_filter()

                async with sandbox_agent_bridge(
                    state,
                    model="inspect",
                    port=port,
                    sandbox=config.sandbox_name,
                    bridged_tools=bridged,
                    filter=bridge_filter,
                ) as bridge:
                    sbox = sandbox_env(config.sandbox_name)

                    # Write runner script to sandbox
                    await sbox.write_file(
                        _RUNNER_PATH,
                        RUNNER_SCRIPT,
                    )

                    # Build user prompt from message history
                    user_prompt, _has_assistant = build_user_prompt(
                        state.messages,
                    )
                    if not user_prompt:
                        user_prompt = "Begin the task."

                    # Prepend the SABER system prompt to the user message
                    # so that Copilot's own default system prompt is preserved.
                    if system_prompt:
                        user_prompt = f"{system_prompt}\n\n{user_prompt}"

                    # --- Persona & Skills handling ---
                    persona_prompt = ""
                    skill_directories_json = "[]"

                    if config.persona_file:
                        from saber.agents.persona import parse_agent_file

                        persona_path = Path(config.persona_file)
                        metadata, prompt_body = parse_agent_file(persona_path)

                        # Copy the raw persona file into sandbox
                        sandbox_persona_path = f".github/agents/{persona_path.name}"
                        await sbox.write_file(
                            sandbox_persona_path,
                            persona_path.read_text(encoding="utf-8"),
                        )
                        logger.info(
                            "Copied persona file to sandbox: %s",
                            sandbox_persona_path,
                        )

                        # Use the prompt body as the persona prompt
                        # The SDK injects this via system_message mode=append
                        persona_prompt = prompt_body

                    if config.skills_dir:
                        sandbox_skills = await upload_skills_to_sandbox(sbox, config.skills_dir, ".github/skills")
                        skill_directories_json = json.dumps([sandbox_skills])

                    # Build environment
                    runner_env = _build_runner_env(
                        bridge_port=bridge.port,
                        model=config.model,
                        prompt=user_prompt,
                        mcp_configs=bridge.mcp_server_configs,
                        persona_prompt=persona_prompt,
                        skill_directories_json=skill_directories_json,
                    )

                    # Execute runner in sandbox (stdin closed to prevent hangs).
                    # No timeout — inspect_ai's --time-limit governs the
                    # overall sample wall-clock budget.
                    result = await sbox.exec(
                        [
                            "bash",
                            "-c",
                            'exec 0</dev/null; "$@"',
                            "bash",
                            "python3",
                            _RUNNER_PATH,
                        ],
                        env=runner_env,
                    )

                    if result.returncode != 0:
                        stderr = result.stderr or ""
                        stdout = result.stdout or ""

                        # Parse bridge proxy errors for structured diagnostics
                        diagnostics = parse_bridge_stderr(stderr)
                        if diagnostics:
                            logger.error(
                                "Bridge proxy error detected in Copilot runner:\n%s",
                                diagnostics,
                            )

                        # Log full stderr at debug level for forensics
                        if stderr:
                            logger.debug(
                                "Copilot runner full stderr (%d chars):\n%s",
                                len(stderr),
                                stderr[:2000],
                            )

                        # Truncated detail for the exception message
                        detail = stderr[:500] if stderr else stdout[:500] if stdout else "(no output)"
                        msg = f"Copilot runner exited with code {result.returncode}: {detail}"
                        logger.error(msg)
                        raise RuntimeError(msg)

                    # Log bridge warnings even on success
                    if result.returncode == 0 and result.stderr:
                        warnings = parse_bridge_stderr(result.stderr)
                        if warnings:
                            logger.warning(
                                "Bridge proxy warnings in Copilot runner (exit 0):\n%s",
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
            _copilot_agent(),
            limits=[tool_call_limit(config.max_steps)],
        )

    return create_with_prompts
