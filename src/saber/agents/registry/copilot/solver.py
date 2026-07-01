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
import os
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from saber.agents.bridge_tracking import create_tracking_filter
from saber.agents.bridge_utils import (
    build_bridged_tools_for_copilot,
    build_system_prompt,
    build_user_prompt,
    compose_filters,
    create_reasoning_effort_filter,
    create_responses_store_filter,
    create_tool_call_limit_filter,
    merge_mcp_configs,
    parse_bridge_stderr,
    parse_idle_decision,
    parse_runner_metrics,
    read_mcp_config,
    record_bridge_summary,
    resolve_model_aliases,
    upload_agent_bundle_to_sandbox,
    upload_skills_to_sandbox,
    validate_model_availability,
)
from saber.agents.cli_output_parser import parse_copilot_stderr as parse_copilot_cli
from saber.agents.persona import load_agent_bundle
from saber.logging import get_logger

if TYPE_CHECKING:
    from inspect_ai.solver import Solver
    from inspect_ai.tool import Tool
    from inspect_ai.tool._mcp._config import MCPServerConfigHTTP

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STORE_PORT_KEY = "copilot_model_port"
_DEFAULT_PORT_BASE = 3000
_RUNNER_PATH = "/tmp/_copilot_runner.py"
_DEFAULT_TIMEOUT = 3600
_MIN_TIMEOUT = 30  # Floor to allow graceful completion

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
- COPILOT_CUSTOM_AGENTS: JSON list of Copilot SDK custom agent configs (optional)
- COPILOT_AGENT: Selected custom agent name (optional)
- COPILOT_NESTED_AGENTS: Nested-agent mode: native, bridge, disabled, github_auth
- COPILOT_TIMEOUT: Max wall-clock timeout in seconds (default: 3600)
- COPILOT_IDLE_TIMEOUT: Max idle time with no activity events (default: 300)
"""

import asyncio
import json
import os
import signal
import sys
import time

try:
    from copilot.types import PermissionRequestResult
except ModuleNotFoundError:
    try:
        from copilot.session import PermissionRequestResult
    except ImportError:
        PermissionRequestResult = None

try:
    from copilot.session import PermissionDecisionApproveOnce
except (ModuleNotFoundError, ImportError):
    PermissionDecisionApproveOnce = None

try:
    from copilot.tools import Tool, ToolResult
except ModuleNotFoundError:
    Tool = None
    ToolResult = None

from copilot.generated.session_events import SessionEventType

# ---------------------------------------------------------------------------
# Constants for process health classification
# ---------------------------------------------------------------------------

INTERACTIVE_COMMANDS = frozenset({
    "less", "more", "vi", "vim", "nano", "man", "pager", "view", "ed",
})
STDIN_WAIT_FNS = frozenset({
    "wait_woken", "n_tty_read", "read_chan", "unix_stream_read_generic", "pipe_read",
})
NETWORK_WAIT_FNS = frozenset({
    "poll_schedule_timeout", "ep_poll", "inet_csk_accept",
    "tcp_recvmsg", "sk_wait_data", "do_select",
})
MAX_IDLE_EXTENSIONS = 3


# ---------------------------------------------------------------------------
# /proc filesystem readers
# ---------------------------------------------------------------------------


def read_proc_stat(pid, proc_root="/proc"):
    """Read /proc/PID/stat and return parsed dict or None."""
    try:
        with open(f"{proc_root}/{pid}/stat") as f:
            data = f.read().strip()
    except (OSError, IOError):
        return None
    # Parse: pid (comm) state ppid ... utime(14) stime(15)
    # comm can contain spaces/parens, so find the last ')'
    start = data.index("(") + 1
    end = data.rindex(")")
    comm = data[start:end]
    rest = data[end + 2:].split()
    # rest[0]=state, rest[1]=ppid, ..., rest[11]=utime, rest[12]=stime
    try:
        return {
            "pid": pid,
            "comm": comm,
            "state": rest[0],
            "ppid": int(rest[1]),
            "utime": int(rest[11]),
            "stime": int(rest[12]),
        }
    except (IndexError, ValueError):
        return None


def read_proc_wchan(pid, proc_root="/proc"):
    """Read /proc/PID/wchan and return string (or "")."""
    try:
        with open(f"{proc_root}/{pid}/wchan") as f:
            return f.read().strip()
    except (OSError, IOError):
        return ""


def read_proc_io(pid, proc_root="/proc"):
    """Read /proc/PID/io and return dict with read_bytes, write_bytes (or None)."""
    try:
        with open(f"{proc_root}/{pid}/io") as f:
            lines = f.readlines()
    except (OSError, IOError):
        return None
    result = {}
    for line in lines:
        parts = line.strip().split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            if key in ("read_bytes", "write_bytes"):
                try:
                    result[key] = int(parts[1].strip())
                except ValueError:
                    pass
    if "read_bytes" in result and "write_bytes" in result:
        return result
    return None


def read_proc_fd0(pid, proc_root="/proc"):
    """Read symlink target of /proc/PID/fd/0 (or "")."""
    try:
        return os.readlink(f"{proc_root}/{pid}/fd/0")
    except (OSError, IOError):
        return ""


def read_proc_cmdline(pid, proc_root="/proc"):
    """Read /proc/PID/cmdline and return list of strings (or [])."""
    try:
        with open(f"{proc_root}/{pid}/cmdline", "rb") as f:
            data = f.read()
    except (OSError, IOError):
        return []
    if not data:
        return []
    # cmdline is null-separated; strip trailing null
    if data.endswith(b"\x00"):
        data = data[:-1]
    return data.decode("utf-8", errors="replace").split("\x00")


def get_descendant_pids(ancestor_pid, proc_root="/proc"):
    """Return set of all descendant PIDs of ancestor_pid."""
    # Build parent->children map from all /proc entries
    children_map = {}  # ppid -> set of pids
    try:
        entries = os.listdir(proc_root)
    except (OSError, IOError):
        return set()
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == ancestor_pid:
            continue
        stat = read_proc_stat(pid, proc_root)
        if stat is None:
            continue
        ppid = stat["ppid"]
        if ppid not in children_map:
            children_map[ppid] = set()
        children_map[ppid].add(pid)
    # BFS from ancestor
    descendants = set()
    queue = list(children_map.get(ancestor_pid, set()))
    while queue:
        pid = queue.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        queue.extend(children_map.get(pid, set()))
    return descendants


# ---------------------------------------------------------------------------
# Process health classifier
# ---------------------------------------------------------------------------


def classify_process(pid, prev_snapshot, curr_snapshot, proc_root="/proc"):
    """Classify a process as stuck, working, or indeterminate.

    prev_snapshot/curr_snapshot: dicts with read_bytes, write_bytes, utime, stime

    Decision tree (checked in order):
    1. cmdline basename in INTERACTIVE_COMMANDS -> stuck
    2. state == "T" -> stuck
    3. wchan in STDIN_WAIT_FNS AND fd0 is pipe/pty -> stuck
    4. delta(read_bytes + write_bytes) > 0 -> working
    5. delta(utime + stime) > 0 -> working
    6. fd0 is socket -> working
    7. wchan in NETWORK_WAIT_FNS -> working
    8. else -> indeterminate
    """
    cmdline = read_proc_cmdline(pid, proc_root)
    cmdline_str = " ".join(cmdline) if cmdline else ""
    stat = read_proc_stat(pid, proc_root)
    state = stat["state"] if stat else "?"
    wchan = read_proc_wchan(pid, proc_root)
    fd0 = read_proc_fd0(pid, proc_root)

    basename = os.path.basename(cmdline[0]) if cmdline else ""

    def _result(health, reason):
        return {
            "pid": pid,
            "cmdline": cmdline_str,
            "state": state,
            "health": health,
            "reason": reason,
            "wchan": wchan,
            "fd0_target": fd0,
        }

    # 1. Interactive command
    if basename in INTERACTIVE_COMMANDS:
        return _result("stuck", f"interactive command: {basename}")

    # 2. Stopped process
    if state == "T":
        return _result("stuck", "stopped (SIGSTOP/traced)")

    # 3. Stdin-blocked
    if wchan in STDIN_WAIT_FNS and ("pipe:" in fd0 or "pts" in fd0 or "pty" in fd0):
        return _result("stuck", f"stdin-blocked on {wchan}, fd0={fd0}")

    # 4. Active I/O
    prev_io = (prev_snapshot.get("read_bytes", 0) + prev_snapshot.get("write_bytes", 0))
    curr_io = (curr_snapshot.get("read_bytes", 0) + curr_snapshot.get("write_bytes", 0))
    if curr_io - prev_io > 0:
        return _result("working", f"active I/O (delta={curr_io - prev_io})")

    # 5. Active CPU
    prev_cpu = (prev_snapshot.get("utime", 0) + prev_snapshot.get("stime", 0))
    curr_cpu = (curr_snapshot.get("utime", 0) + curr_snapshot.get("stime", 0))
    if curr_cpu - prev_cpu > 0:
        return _result("working", f"active CPU (delta={curr_cpu - prev_cpu})")

    # 6. Socket on fd0
    if "socket:" in fd0:
        return _result("working", f"fd0 is socket: {fd0}")

    # 7. Network wait
    if wchan in NETWORK_WAIT_FNS:
        return _result("working", f"network wait: {wchan}")

    # 8. Indeterminate
    return _result("indeterminate", f"no definitive signal (wchan={wchan})")


def make_idle_decision(ancestor_pid, prev_snapshots, in_flight_tools, idle_seconds, total_seconds, proc_root="/proc"):
    """Decide what to do when idle timeout fires.

    1. Get descendant PIDs
    2. Take current I/O+CPU snapshot
    3. Classify each leaf process
    4. Aggregate decision:
       - in_flight_tools non-empty -> "extend"
       - ALL leaf processes stuck -> "kill_stuck"
       - ANY working or indeterminate -> "extend"
       - No descendants -> "timeout"

    Returns (decision_dict, new_snapshots).
    """
    descendants = get_descendant_pids(ancestor_pid, proc_root)

    if not descendants:
        decision = {
            "action": "timeout",
            "classifications": [],
            "in_flight_tools": (
                list(in_flight_tools.keys())
                if isinstance(in_flight_tools, dict)
                else list(in_flight_tools)
            ),
            "idle_seconds": idle_seconds,
            "total_seconds": total_seconds,
        }
        return decision, {}

    # Take current snapshot
    curr_snapshots = {}
    for pid in descendants:
        io_data = read_proc_io(pid, proc_root)
        stat_data = read_proc_stat(pid, proc_root)
        snap = {
            "read_bytes": io_data["read_bytes"] if io_data else 0,
            "write_bytes": io_data["write_bytes"] if io_data else 0,
            "utime": stat_data["utime"] if stat_data else 0,
            "stime": stat_data["stime"] if stat_data else 0,
        }
        curr_snapshots[pid] = snap

    # Classify each descendant
    classifications = []
    for pid in descendants:
        prev = prev_snapshots.get(pid, {"read_bytes": 0, "write_bytes": 0, "utime": 0, "stime": 0})
        curr = curr_snapshots.get(pid, {"read_bytes": 0, "write_bytes": 0, "utime": 0, "stime": 0})
        c = classify_process(pid, prev, curr, proc_root)
        classifications.append(c)

    tool_names = list(in_flight_tools.keys()) if isinstance(in_flight_tools, dict) else list(in_flight_tools)

    # Aggregate
    if tool_names:
        action = "extend"
    elif all(c["health"] == "stuck" for c in classifications):
        action = "kill_stuck"
    elif any(c["health"] in ("working", "indeterminate") for c in classifications):
        action = "extend"
    else:
        # Defensive: unreachable with current classifier outputs
        action = "timeout"

    decision = {
        "action": action,
        "classifications": classifications,
        "in_flight_tools": tool_names,
        "idle_seconds": idle_seconds,
        "total_seconds": total_seconds,
    }
    return decision, curr_snapshots


def _approve_all(
    _request: dict,
    _context: dict,
) -> object:
    """Auto-approve every permission request.

    Newer Copilot SDKs return typed permission decisions. Older SDKs used a
    constructible ``PermissionRequestResult`` with ``kind="approve-once"``.
    Prefer the typed decision and fall back only when running against an older
    SDK.

    This is safe because the runner executes inside an isolated Docker
    sandbox used exclusively for benchmarking.
    """
    if PermissionDecisionApproveOnce is not None:
        return PermissionDecisionApproveOnce()
    if PermissionRequestResult is None:
        raise RuntimeError("Copilot SDK does not expose a permission approval result")
    return PermissionRequestResult(kind="approve-once")


def _normalize_agent_selector(value):
    """Normalize agent names/filenames for delegation lookup."""
    return "-".join(part for part in "".join(ch.lower() if ch.isalnum() else "-" for ch in value).split("-") if part)


def _agent_aliases(custom_agents):
    """Build normalized alias -> SDK agent name mapping."""
    aliases = {}
    for agent_config in custom_agents:
        if not isinstance(agent_config, dict):
            continue
        name = agent_config.get("name")
        display_name = agent_config.get("display_name")
        for value in (name, display_name):
            if isinstance(value, str) and value.strip():
                aliases[value] = name
                aliases[_normalize_agent_selector(value)] = name
                aliases[_normalize_agent_selector(value).removesuffix("-agent")] = name
    return {k: v for k, v in aliases.items() if k and v}


def _coerce_agent_tool_args(arguments):
    """Accept common argument shapes for the bridge-backed agent tool."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        return {"prompt": arguments}
    return {}


def _first_arg(args, names):
    """Return the first string argument from a set of aliases."""
    for name in names:
        value = args.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _event_content(event):
    """Extract assistant content from a Copilot SDK event."""
    data = getattr(event, "data", None)
    content = getattr(data, "content", None)
    if content is None and isinstance(data, dict):
        content = data.get("content")
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _tool_failure(message):
    """Return a Copilot ToolResult failure."""
    return ToolResult(
        text_result_for_llm=f"Nested agent failed: {message}",
        result_type="failure",
        error=message,
    )


def _build_bridge_agent_tool(
    *,
    client,
    base_session_config,
    custom_agents,
    timeout_seconds,
    quiet_timeout_seconds,
    max_calls,
    output_limit,
    tool_name="agent",
    allow_ad_hoc=False,
):
    """Create a bridge-backed `agent` tool for delegated custom agents.

    The Copilot SDK's native delegation path may require GitHub auth in the
    sandbox. This tool is SABER-owned: it creates a nested SDK session with the
    same custom OpenAI provider, MCP config, skills, and custom-agent registry
    as the parent, but without re-exposing the delegation tool recursively.
    """
    if Tool is None or ToolResult is None:
        raise RuntimeError("github-copilot-sdk does not expose custom Tool support; install >=0.2.2.")

    aliases = _agent_aliases(custom_agents)
    agents_by_name = {
        agent_config["name"]: agent_config
        for agent_config in custom_agents
        if isinstance(agent_config, dict) and isinstance(agent_config.get("name"), str)
    }
    calls = {"count": 0}

    async def _run_nested_session(agent_name, prompt, call_id, agent_prompt_override=None):
        agent_config = agents_by_name.get(agent_name, {})
        agent_prompt = agent_prompt_override or agent_config.get("prompt")
        if not isinstance(agent_prompt, str) or not agent_prompt.strip():
            return _tool_failure(f"agent '{agent_name}' has no prompt")

        nested_config = {
            key: value
            for key, value in base_session_config.items()
            if key not in (
                "agent",
                "config_dir",
                "custom_agents",
                "enable_config_discovery",
                "system_message",
                "tools",
            )
        }
        nested_config["system_message"] = {
            "mode": "append",
            "content": agent_prompt,
        }
        provider = nested_config.get("provider")
        if isinstance(provider, dict) and provider.get("type") == "openai":
            nested_provider = dict(provider)
            nested_provider["wire_api"] = os.environ.get("COPILOT_WIRE_API", "responses")
            nested_config["provider"] = nested_provider

        idle_event = asyncio.Event()
        session_error = ""
        last_content = ""
        assistant_messages = 0
        tool_calls_started = 0
        tool_calls_completed = 0
        in_flight_tools = {}
        start_time = time.monotonic()
        last_activity_time = start_time
        poll_error_reported = False

        nested_client = type(client)()
        try:
            await nested_client.start()
            session = await nested_client.create_session(**nested_config)

            async def _poll_nested_messages():
                nonlocal last_content, tool_calls_started, tool_calls_completed
                nonlocal last_activity_time
                nonlocal poll_error_reported

                try:
                    events = await session.get_messages()
                except Exception as exc:
                    if not poll_error_reported:
                        print(
                            "COPILOT_NESTED_AGENT_POLL_WARNING: "
                            + json.dumps(
                                {
                                    "call_id": call_id,
                                    "agent_name": agent_name,
                                    "message": str(exc)[:300],
                                }
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        poll_error_reported = True
                    return

                polled_content = ""
                polled_tool_starts = 0
                polled_tool_completes = 0
                for event in events:
                    if event.type == SessionEventType.ASSISTANT_MESSAGE:
                        content = _event_content(event)
                        if content:
                            polled_content = content
                    elif event.type == SessionEventType.TOOL_EXECUTION_START:
                        polled_tool_starts += 1
                    elif event.type == SessionEventType.TOOL_EXECUTION_COMPLETE:
                        polled_tool_completes += 1

                if polled_content and polled_content != last_content:
                    last_content = polled_content
                    last_activity_time = time.monotonic()

                if polled_tool_starts or polled_tool_completes:
                    tool_calls_started = max(tool_calls_started, polled_tool_starts)
                    tool_calls_completed = max(tool_calls_completed, polled_tool_completes)
                    if polled_tool_starts > polled_tool_completes:
                        in_flight_tools["__polled__"] = polled_tool_starts - polled_tool_completes
                    else:
                        in_flight_tools.pop("__polled__", None)

            def _on_nested_event(event):
                nonlocal session_error, last_content, assistant_messages
                nonlocal tool_calls_started, tool_calls_completed
                nonlocal last_activity_time

                last_activity_time = time.monotonic()
                if event.type == SessionEventType.ASSISTANT_MESSAGE:
                    assistant_messages += 1
                    content = _event_content(event)
                    if content:
                        last_content = content
                elif event.type == SessionEventType.TOOL_EXECUTION_START:
                    tool_calls_started += 1
                    data = getattr(event, "data", None)
                    tool_name = getattr(data, "tool_name", None) or getattr(data, "name", None)
                    if tool_name:
                        in_flight_tools[tool_name] = in_flight_tools.get(tool_name, 0) + 1
                elif event.type == SessionEventType.TOOL_EXECUTION_COMPLETE:
                    tool_calls_completed += 1
                    data = getattr(event, "data", None)
                    tool_name = getattr(data, "tool_name", None) or getattr(data, "name", None)
                    if tool_name:
                        count = in_flight_tools.get(tool_name, 1) - 1
                        if count <= 0:
                            in_flight_tools.pop(tool_name, None)
                        else:
                            in_flight_tools[tool_name] = count
                elif event.type == SessionEventType.SESSION_IDLE:
                    idle_event.set()
                elif event.type == SessionEventType.SESSION_ERROR:
                    session_error = str(getattr(getattr(event, "data", None), "message", event.data))
                    idle_event.set()

            unsubscribe = session.on(_on_nested_event)
            try:
                await session.send(prompt)
                completion_reason = "idle"
                while True:
                    await _poll_nested_messages()
                    now = time.monotonic()
                    if idle_event.is_set():
                        completion_reason = "idle"
                        break
                    if last_content and not in_flight_tools and (now - last_activity_time) >= quiet_timeout_seconds:
                        completion_reason = "quiet"
                        break
                    if (now - start_time) >= timeout_seconds:
                        completion_reason = "timeout"
                        break
                    await asyncio.sleep(1.0)

                if completion_reason == "timeout":
                    elapsed = round(time.monotonic() - start_time, 1)
                    print(
                        "COPILOT_NESTED_AGENT_ERROR: "
                        + json.dumps(
                            {
                                "call_id": call_id,
                                "agent_name": agent_name,
                                "error_kind": "timeout",
                                "elapsed_seconds": elapsed,
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    return _tool_failure(f"{agent_name} timed out after {timeout_seconds}s")

                elapsed = round(time.monotonic() - start_time, 1)
                if session_error:
                    print(
                        "COPILOT_NESTED_AGENT_ERROR: "
                        + json.dumps(
                            {
                                "call_id": call_id,
                                "agent_name": agent_name,
                                "error_kind": "session_error",
                                "elapsed_seconds": elapsed,
                                "message": session_error[:300],
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    return _tool_failure(session_error)

                print(
                    "COPILOT_NESTED_AGENT_END: "
                    + json.dumps(
                        {
                            "call_id": call_id,
                            "agent_name": agent_name,
                            "status": "success",
                            "elapsed_seconds": elapsed,
                            "assistant_messages": assistant_messages,
                            "tool_calls_started": tool_calls_started,
                            "tool_calls_completed": tool_calls_completed,
                            "completion_reason": completion_reason,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if len(last_content) > output_limit:
                    last_content = last_content[:output_limit] + "\n\n[truncated]"
                return ToolResult(text_result_for_llm=last_content or f"{agent_name} completed without text output.")
            finally:
                unsubscribe()
        finally:
            if "session" in locals():
                if hasattr(session, "disconnect"):
                    await session.disconnect()
                else:
                    await session.destroy()
            await nested_client.stop()

    async def _agent_handler(invocation):
        args = _coerce_agent_tool_args(getattr(invocation, "arguments", None))
        requested_agent = _first_arg(args, ("agent", "agent_name", "name", "subagent", "recipient"))
        prompt = _first_arg(args, ("prompt", "task", "instructions", "input", "message"))

        if not requested_agent and not allow_ad_hoc:
            return _tool_failure("missing required argument: agent")
        if not requested_agent:
            requested_agent = "task"
        if not prompt:
            return _tool_failure("missing required argument: prompt")

        selected_agent = aliases.get(requested_agent) or aliases.get(_normalize_agent_selector(requested_agent))
        if not selected_agent:
            if allow_ad_hoc:
                selected_agent = f"ad-hoc-{_normalize_agent_selector(requested_agent) or 'task'}"
                ad_hoc_prompt = (
                    "You are a focused delegated assistant running inside SABER. "
                    "Complete only the task provided by the parent agent, use available tools when needed, "
                    "return concise evidence and results, and do not modify cloud resources."
                )
                if calls["count"] >= max_calls:
                    return _tool_failure(f"nested agent call limit exceeded ({max_calls})")
                calls["count"] += 1
                call_id = f"nested-{calls['count']}"
                print(
                    "COPILOT_NESTED_AGENT_START: "
                    + json.dumps(
                        {
                            "call_id": call_id,
                            "agent_name": selected_agent,
                            "depth": 1,
                            "model": base_session_config.get("model"),
                            "timeout": timeout_seconds,
                            "tool": tool_name,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                return await _run_nested_session(selected_agent, prompt, call_id, ad_hoc_prompt)
            known = ", ".join(sorted({v for v in aliases.values()}))
            return _tool_failure(f"unknown agent '{requested_agent}'. Known agents: {known}")

        if calls["count"] >= max_calls:
            return _tool_failure(f"nested agent call limit exceeded ({max_calls})")

        calls["count"] += 1
        call_id = f"nested-{calls['count']}"
        print(
            "COPILOT_NESTED_AGENT_START: "
            + json.dumps(
                {
                    "call_id": call_id,
                    "agent_name": selected_agent,
                    "depth": 1,
                        "model": base_session_config.get("model"),
                        "timeout": timeout_seconds,
                        "tool": tool_name,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        return await _run_nested_session(selected_agent, prompt, call_id)

    return Tool(
        name=tool_name,
        description=(
            "Delegate a focused task to a named custom agent from the current agent bundle. "
            "Use this for subagents such as fingerprint, attack-path, choke-point, or report."
            if tool_name == "agent"
            else "Run a focused delegated task through SABER's bridge-routed nested session."
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the subagent to invoke.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Task instructions and all context the subagent needs.",
                },
            },
            "required": ["agent", "prompt"],
            "additionalProperties": True,
        },
        handler=_agent_handler,
        overrides_built_in_tool=True,
        skip_permission=True,
    )


async def main() -> int:
    """Run Copilot session with bridge-routed model."""
    from copilot import CopilotClient
    from copilot.generated.session_events import SessionEventType

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:13131/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    model = os.environ.get("COPILOT_MODEL", "inspect")
    prompt = os.environ.get("COPILOT_PROMPT", "")
    mcp_config_str = os.environ.get("COPILOT_MCP_CONFIG", "")
    custom_agents_str = os.environ.get("COPILOT_CUSTOM_AGENTS", "")
    active_agent = os.environ.get("COPILOT_AGENT", "")
    config_dir = os.environ.get("COPILOT_CONFIG_DIR", "")
    nested_agents_mode = os.environ.get("COPILOT_NESTED_AGENTS", "native")
    nested_agent_timeout = int(os.environ.get("COPILOT_NESTED_AGENT_TIMEOUT", "600"))
    nested_agent_quiet_timeout = int(os.environ.get("COPILOT_NESTED_AGENT_QUIET_TIMEOUT", "5"))
    nested_agent_max_calls = int(os.environ.get("COPILOT_NESTED_AGENT_MAX_CALLS", "16"))
    nested_agent_output_limit = int(os.environ.get("COPILOT_NESTED_AGENT_OUTPUT_LIMIT", "20000"))
    max_timeout = int(os.environ.get("COPILOT_TIMEOUT", "3600"))
    idle_timeout = int(os.environ.get("COPILOT_IDLE_TIMEOUT", "300"))

    if not prompt:
        print("ERROR: COPILOT_PROMPT is required", file=sys.stderr)
        return 1

    # Build session config
    # Wire protocol for the bridge-facing custom OpenAI provider.
    #
    # Default is Chat Completions: in this environment the Copilot CLI has
    # historically stalled when routed through the Responses API bridge,
    # never emitting an assistant turn or tool call, so non-streaming Chat
    # Completions is the compatible default path.
    #
    # However, Chat Completions cannot represent Responses-API reasoning
    # items, so they are stripped between turns. Reasoning models served
    # over the Responses API (e.g. MAI flash-code) reject the next turn
    # because a replayed `function_call` is missing its required `reasoning`
    # item. For those models set COPILOT_WIRE_API=responses so reasoning
    # items round-trip through the runner's conversation history.
    wire_api = os.environ.get("COPILOT_WIRE_API", "completions")
    session_config: dict = {
        "model": model,
        # The Inspect bridge only implements non-streaming OpenAI proxy calls.
        # Copilot's SDK may default to streaming for custom providers, which can
        # leave the session hanging without ever producing an assistant message.
        "streaming": False,
        "provider": {
            "type": "openai",
            "base_url": base_url,
            "api_key": api_key,
            "wire_api": wire_api,
        },
        "on_permission_request": _approve_all,
    }

    # MCP servers from bridge
    if mcp_config_str:
        try:
            mcp_servers = json.loads(mcp_config_str)
            if not isinstance(mcp_servers, dict):
                print("ERROR: COPILOT_MCP_CONFIG must decode to an object", file=sys.stderr)
                return 1
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

    # Custom agent bundle definitions, passed directly to the SDK. These are
    # snake_case dictionaries; the SDK converts them to wire format.
    custom_agents = []
    if custom_agents_str:
        try:
            custom_agents = json.loads(custom_agents_str)
            if not isinstance(custom_agents, list):
                print("ERROR: COPILOT_CUSTOM_AGENTS must decode to a list", file=sys.stderr)
                return 1
            if custom_agents and nested_agents_mode != "bridge":
                session_config["custom_agents"] = custom_agents
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid COPILOT_CUSTOM_AGENTS: {custom_agents_str} — {exc}", file=sys.stderr)
            return 1

    if active_agent and nested_agents_mode != "bridge":
        session_config["agent"] = active_agent

    if config_dir and nested_agents_mode != "bridge":
        session_config["config_dir"] = config_dir
        session_config["enable_config_discovery"] = True

    # Activity tracking
    start_time = time.monotonic()
    last_activity_time = start_time
    total_events = 0
    assistant_messages = 0
    tool_calls_started = 0
    tool_calls_completed = 0
    turn_count = 0
    last_event_type = ""
    idle_event = asyncio.Event()
    session_error_msg = ""
    last_response = None
    in_flight_tools = {}
    idle_extensions = 0
    proc_snapshots = {}
    last_heartbeat_time = start_time

    # Event types that count as "activity" (agent is doing work)
    ACTIVITY_EVENTS = {
        SessionEventType.ASSISTANT_MESSAGE,
        SessionEventType.ASSISTANT_MESSAGE_DELTA,
        SessionEventType.TOOL_EXECUTION_START,
        SessionEventType.TOOL_EXECUTION_COMPLETE,
        SessionEventType.TOOL_EXECUTION_PROGRESS,
        SessionEventType.ASSISTANT_TURN_START,
        SessionEventType.ASSISTANT_TURN_END,
        SessionEventType.ASSISTANT_REASONING,
        SessionEventType.ASSISTANT_REASONING_DELTA,
        SessionEventType.ASSISTANT_INTENT,
        SessionEventType.SESSION_COMPACTION_START,
        SessionEventType.SESSION_COMPACTION_COMPLETE,
        SessionEventType.ASSISTANT_USAGE,
    }
    # Expand with optional event types (may not exist in all SDK versions)
    for _evt_name in ("TOOL_EXECUTION_PARTIAL_RESULT", "COMMAND_QUEUED", "COMMAND_COMPLETED"):
        if hasattr(SessionEventType, _evt_name):
            ACTIVITY_EVENTS.add(getattr(SessionEventType, _evt_name))

    def _on_event(event):
        nonlocal last_activity_time, total_events, assistant_messages
        nonlocal tool_calls_started, tool_calls_completed, turn_count
        nonlocal last_event_type, session_error_msg, last_response

        total_events += 1
        last_event_type = str(event.type.value) if hasattr(event.type, 'value') else str(event.type)

        if event.type in ACTIVITY_EVENTS:
            last_activity_time = time.monotonic()

        if event.type == SessionEventType.ASSISTANT_MESSAGE:
            assistant_messages += 1
            last_response = event
        elif event.type == SessionEventType.TOOL_EXECUTION_START:
            tool_calls_started += 1
            # Track in-flight tool (ref-counted)
            data = getattr(event, "data", None)
            tool_name = getattr(data, "tool_name", None) or getattr(data, "name", None)
            if tool_name:
                in_flight_tools[tool_name] = in_flight_tools.get(tool_name, 0) + 1
        elif event.type == SessionEventType.TOOL_EXECUTION_COMPLETE:
            tool_calls_completed += 1
            # Decrement ref-count for in-flight tool
            data = getattr(event, "data", None)
            tool_name = getattr(data, "tool_name", None) or getattr(data, "name", None)
            if tool_name:
                count = in_flight_tools.get(tool_name, 1) - 1
                if count <= 0:
                    in_flight_tools.pop(tool_name, None)
                else:
                    in_flight_tools[tool_name] = count
        elif event.type == SessionEventType.ASSISTANT_TURN_END:
            turn_count += 1
        elif event.type == SessionEventType.SESSION_IDLE:
            idle_event.set()
        elif event.type == SessionEventType.SESSION_ERROR:
            session_error_msg = str(getattr(getattr(event, 'data', None), 'message', event.data))
            idle_event.set()

    def _build_metrics(exit_reason):
        now = time.monotonic()
        return {
            "total_events": total_events,
            "assistant_messages": assistant_messages,
            "tool_calls_started": tool_calls_started,
            "tool_calls_completed": tool_calls_completed,
            "turns": turn_count,
            "elapsed_seconds": round(now - start_time, 1),
            "idle_seconds": round(now - last_activity_time, 1),
            "exit_reason": exit_reason,
            "last_event_type": last_event_type,
            "idle_extensions": idle_extensions,
            "in_flight_tools_at_exit": list(in_flight_tools.keys()),
        }

    def _print_metrics(exit_reason):
        metrics = _build_metrics(exit_reason)
        print(f"COPILOT_METRICS: {json.dumps(metrics)}", file=sys.stderr)

    client = CopilotClient()
    try:
        await client.start()
        if nested_agents_mode == "bridge" and custom_agents:
            session_config["tools"] = [
                _build_bridge_agent_tool(
                    client=client,
                    base_session_config=session_config,
                    custom_agents=custom_agents,
                    timeout_seconds=nested_agent_timeout,
                    quiet_timeout_seconds=nested_agent_quiet_timeout,
                    max_calls=nested_agent_max_calls,
                    output_limit=nested_agent_output_limit,
                    tool_name="agent",
                ),
                _build_bridge_agent_tool(
                    client=client,
                    base_session_config=session_config,
                    custom_agents=custom_agents,
                    timeout_seconds=nested_agent_timeout,
                    quiet_timeout_seconds=nested_agent_quiet_timeout,
                    max_calls=nested_agent_max_calls,
                    output_limit=nested_agent_output_limit,
                    tool_name="task",
                    allow_ad_hoc=True,
                )
            ]
        session = await client.create_session(**session_config)

        try:
            unsubscribe = session.on(_on_event)
            try:
                await session.send(prompt)
                last_activity_time = time.monotonic()  # Reset after send

                # Poll loop with activity-aware timeout
                while not idle_event.is_set():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(idle_event.wait()),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        pass  # Check timeouts below

                    if idle_event.is_set():
                        break

                    now = time.monotonic()
                    idle_elapsed = now - last_activity_time
                    total_elapsed = now - start_time

                    # Heartbeat logging every 30s
                    if now - last_heartbeat_time > 30.0:
                        descendants = get_descendant_pids(os.getpid())
                        heartbeat = {
                            "elapsed_s": round(total_elapsed, 1),
                            "idle_s": round(idle_elapsed, 1),
                            "total_events": total_events,
                            "in_flight_tools": list(in_flight_tools.keys()),
                            "descendant_count": len(descendants),
                            "idle_extensions": idle_extensions,
                        }
                        print(f"COPILOT_HEARTBEAT: {json.dumps(heartbeat)}", file=sys.stderr)
                        last_heartbeat_time = now

                    if idle_elapsed > idle_timeout:
                        decision, proc_snapshots = make_idle_decision(
                            os.getpid(), proc_snapshots, in_flight_tools,
                            idle_elapsed, total_elapsed
                        )
                        print(f"COPILOT_IDLE_DECISION: {json.dumps(decision)}", file=sys.stderr)

                        if decision["action"] == "kill_stuck":
                            my_pid = os.getpid()
                            for c in decision["classifications"]:
                                if c["health"] == "stuck" and c["pid"] not in (1, my_pid):
                                    try:
                                        os.kill(c["pid"], signal.SIGKILL)
                                    except (ProcessLookupError, OSError):
                                        pass
                            _print_metrics("stuck_processes")
                            break
                        elif decision["action"] == "extend":
                            last_activity_time = time.monotonic()
                            idle_extensions += 1
                            if idle_extensions > MAX_IDLE_EXTENSIONS:
                                _print_metrics("max_idle_extensions")
                                break
                            continue
                        else:  # "timeout"
                            print(
                                f"COPILOT_RUNNER_IDLE_TIMEOUT: No activity for "
                                f"{idle_elapsed:.0f}s (limit: {idle_timeout}s)",
                                file=sys.stderr,
                            )
                            _print_metrics("idle_timeout")
                            break

                    if total_elapsed > max_timeout:
                        print(
                            f"COPILOT_RUNNER_MAX_TIMEOUT: Session active for "
                            f"{total_elapsed:.0f}s (limit: {max_timeout}s)",
                            file=sys.stderr,
                        )
                        _print_metrics("max_timeout")
                        break

                else:
                    # idle_event was set — normal completion or session error
                    if session_error_msg:
                        print(f"COPILOT_SESSION_ERROR: {session_error_msg}", file=sys.stderr)
                        _print_metrics("session_error")
                    else:
                        if last_response:
                            content = getattr(getattr(last_response, "data", None), "content", None)
                            if content:
                                print(f"COPILOT_RESPONSE: {content[:500]}", file=sys.stderr)
                        _print_metrics("completed")

            finally:
                unsubscribe()
        finally:
            if hasattr(session, "disconnect"):
                await session.disconnect()
            else:
                await session.destroy()
    except Exception as exc:
        # Non-timeout exceptions (session creation failure, etc.)
        print(f"ERROR: Session failed: {exc}", file=sys.stderr)
        return 1
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
    port_base: int = _DEFAULT_PORT_BASE
    model: str = "inspect"
    agent_bundle: str | None = None
    agents_dir: str | None = None
    main_agent: str | None = None
    persona_file: str | None = None
    skills_dir: str | None = None
    mcp_config: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    nested_agents: Literal["native", "bridge", "disabled", "github_auth"] = "bridge"
    nested_agent_timeout: int = 600
    nested_agent_quiet_timeout: int = 5
    nested_agent_max_calls: int = 16
    nested_agent_output_limit: int = 20000
    idle_timeout: int = 300

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
# Process health classifier models (host-side)
# ---------------------------------------------------------------------------


class ProcessHealth(StrEnum):
    """Health classification for a running process."""

    STUCK = "stuck"
    WORKING = "working"
    INDETERMINATE = "indeterminate"


class ProcessClassification(BaseModel):
    """Classification result for a single process."""

    model_config = ConfigDict(frozen=True)

    pid: int
    cmdline: str
    state: str
    health: ProcessHealth
    reason: str
    wchan: str
    fd0_target: str


class IdleDecision(BaseModel):
    """Decision made when idle timeout fires."""

    model_config = ConfigDict(frozen=True)

    action: Literal["kill_stuck", "extend", "timeout"]
    classifications: list[ProcessClassification]
    in_flight_tools: list[str]
    idle_seconds: float
    total_seconds: float


# ---------------------------------------------------------------------------
# Helper: build runner environment variables
# ---------------------------------------------------------------------------


def _build_runner_env(
    *,
    bridge_port: int,
    model: str,
    prompt: str,
    mcp_configs: "Sequence[MCPServerConfigHTTP]",
    timeout: int = _DEFAULT_TIMEOUT,
    persona_prompt: str = "",
    skill_directories_json: str = "[]",
    extra_mcp_config: dict[str, object] | None = None,
    custom_agents_json: str = "[]",
    active_agent: str = "",
    config_dir: str = "",
    nested_agents_mode: str = "bridge",
    nested_agent_timeout: int = 600,
    nested_agent_quiet_timeout: int = 5,
    nested_agent_max_calls: int = 16,
    nested_agent_output_limit: int = 20000,
    idle_timeout: int = 300,
) -> dict[str, str]:
    """Build environment variables for the runner script.

    Args:
        bridge_port: Port where the bridge proxy is listening.
        model: Model identifier to pass to the runner.
        prompt: User prompt text (SABER system prompt already prepended).
        mcp_configs: MCP server config objects from the bridge.
        timeout: Timeout in seconds for ``session.send_and_wait()``.
            Derived from inspect_ai's ``sample_limits().time.remaining``
            so the runner respects the overall sample time budget.
        persona_prompt: Persona prompt text to append to the Copilot
            system message via ``system_message`` mode=append.
        skill_directories_json: JSON-serialized list of sandbox skill
            directory paths.
        extra_mcp_config: Operator-provided MCP config object, typically
            read from an agent bundle ``.mcp.json``.
        custom_agents_json: JSON-serialized Copilot SDK custom-agent configs.
        active_agent: Name of the custom agent to select for the session.
        config_dir: Sandbox config directory for optional SDK discovery.
        nested_agents_mode: Copilot nested-agent mode.
        nested_agent_timeout: Per-delegation timeout in seconds for bridge
            mode.
        nested_agent_quiet_timeout: Seconds of quiet after a nested assistant
            response with no in-flight tools before treating the call complete.
        nested_agent_max_calls: Maximum number of bridge-backed nested calls
            per sample.
        nested_agent_output_limit: Maximum nested output characters returned
            to the parent.
        idle_timeout: Maximum idle time (no activity events) in seconds
            before the runner exits gracefully.

    Returns:
        Dict of env vars to pass to ``sbox.exec()``.
    """
    # The Copilot SDK expects mcp_servers as dict[str, MCPRemoteServerConfig]
    # (server name → config), NOT a list.  See:
    # copilot.types.MCPRemoteServerConfig — type, url, tools, headers.
    mcp_dict: dict[str, dict[str, str | list[str]]] = {}
    for c in mcp_configs:
        mcp_dict[c.name] = {
            "type": c.type,
            "url": c.url,
            "tools": ["*"],
        }
    merged_mcp = merge_mcp_configs({"mcpServers": mcp_dict} if mcp_dict else {}, extra_mcp_config)
    mcp_servers = merged_mcp.get("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        raise ValueError("Merged MCP config must contain object-valued mcpServers.")
    return {
        "OPENAI_BASE_URL": f"http://localhost:{bridge_port}/v1",
        "OPENAI_API_KEY": "sk-placeholder-for-bridge",
        "COPILOT_MODEL": model,
        "COPILOT_PROMPT": prompt,
        "COPILOT_MCP_CONFIG": json.dumps(mcp_servers),
        "COPILOT_TIMEOUT": str(timeout),
        "COPILOT_PERSONA_PROMPT": persona_prompt,
        "COPILOT_SKILL_DIRECTORIES": skill_directories_json,
        "COPILOT_CUSTOM_AGENTS": custom_agents_json,
        "COPILOT_AGENT": active_agent,
        "COPILOT_CONFIG_DIR": config_dir,
        "COPILOT_NESTED_AGENTS": nested_agents_mode,
        "COPILOT_WIRE_API": os.environ.get("COPILOT_WIRE_API", "completions"),
        "COPILOT_NESTED_AGENT_TIMEOUT": str(nested_agent_timeout),
        "COPILOT_NESTED_AGENT_QUIET_TIMEOUT": str(nested_agent_quiet_timeout),
        "COPILOT_NESTED_AGENT_MAX_CALLS": str(nested_agent_max_calls),
        "COPILOT_NESTED_AGENT_OUTPUT_LIMIT": str(nested_agent_output_limit),
        "COPILOT_IDLE_TIMEOUT": str(idle_timeout),
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
        *,
        max_steps: int,
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
                tracking_filter, get_tracking_summary = create_tracking_filter()
                store_filter = create_responses_store_filter()
                # Reasoning effort: prefer the task param (-T reasoning_effort=...,
                # recorded in task_args/CONFIG), falling back to the env var.
                _reasoning_effort = config.reasoning_effort or (os.environ.get("COPILOT_REASONING_EFFORT") or None)
                reasoning_filter = create_reasoning_effort_filter(_reasoning_effort)
                if _reasoning_effort:
                    logger.info("copilot bridge reasoning effort override: %s", _reasoning_effort)
                    try:
                        from inspect_ai.log._transcript import transcript

                        transcript().info(
                            {"reasoning_effort": _reasoning_effort},
                            source="saber.reasoning_effort",
                        )
                    except Exception:
                        logger.debug("Could not record reasoning_effort InfoEvent")
                composed = compose_filters(tracking_filter, reasoning_filter, store_filter, bridge_filter)

                _raw_aliases = outer_kwargs.get("model_aliases")
                model_aliases = resolve_model_aliases(_raw_aliases if isinstance(_raw_aliases, dict) else None)

                # Propagate the eval's primary model (with its -M model args,
                # e.g. responses_api / responses_store) across the sandbox bridge
                # boundary. The bridge otherwise re-resolves the requested model
                # name via get_model() in a context where the active model
                # contextvar is not propagated, yielding a fresh model that drops
                # the -M args. For Responses-API reasoning models this means
                # store=false is sent without reasoning.encrypted_content, so the
                # next turn's reasoning-item reference fails ("Item ... not found.
                # Items are not persisted when store is set to false"). Injecting
                # the primary Model as an alias for the requested name makes the
                # bridge use the correctly-configured instance.
                #
                # The Copilot SDK forwards an OpenAI-style model field that drops
                # the inspect provider prefix (e.g. "openai/azure/foo-bar" becomes
                # "foo-bar"), so alias both the full provider-qualified name and
                # the bare service-model segment to cover whichever the runner
                # actually sends across the bridge.
                #
                # Use active_model() (the Model instance inspect constructed for
                # this eval, carrying its -M args) rather than get_model() with no
                # args: the latter falls back to building a fresh model from the
                # INSPECT_EVAL_MODEL name string, dropping responses_store /
                # responses_api and re-introducing the store=false failure.
                from inspect_ai.model._model import active_model as _active_model
                from inspect_ai.model._model import get_model as _get_primary_model

                primary_model = _active_model() or _get_primary_model()
                _store_flag = getattr(getattr(primary_model, "api", None), "responses_store", None)
                _api_flag = getattr(getattr(primary_model, "api", None), "responses_api", None)
                logger.info(
                    "copilot bridge primary model resolved: %s (responses_api=%s responses_store=%s active=%s)",
                    getattr(primary_model, "name", primary_model),
                    _api_flag,
                    _store_flag,
                    _active_model() is not None,
                )
                _alias_keys = {config.model, config.model.rsplit("/", 1)[-1]}
                model_aliases = {
                    **(model_aliases or {}),
                    **dict.fromkeys(_alias_keys, primary_model),
                }

                async with sandbox_agent_bridge(
                    state,
                    model="inspect",
                    port=port,
                    sandbox=config.sandbox_name,
                    bridged_tools=bridged,
                    filter=composed,
                    model_aliases=model_aliases,
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
                    custom_agents_json = "[]"
                    active_agent = ""
                    config_dir = ""

                    bundle = load_agent_bundle(
                        agent_bundle=config.agent_bundle,
                        agents_dir=config.agents_dir,
                        main_agent=config.main_agent,
                        persona_file=config.persona_file,
                        skills_dir=config.skills_dir,
                        mcp_config=config.mcp_config,
                    )
                    if bundle.main_agent is not None:
                        persona_prompt = bundle.main_agent.prompt

                    should_register_custom_agents = (
                        bundle.has_agents
                        and config.nested_agents != "disabled"
                        and bool(config.agent_bundle or config.agents_dir or config.main_agent)
                    )
                    if should_register_custom_agents:
                        sandbox_agents_dir = await upload_agent_bundle_to_sandbox(sbox, bundle, ".github/agents")
                        if sandbox_agents_dir:
                            config_dir = "."
                        custom_agents_json = json.dumps(
                            [agent_config.to_sdk_dict() for agent_config in bundle.custom_agent_configs()]
                        )
                        if bundle.main_agent is not None:
                            active_agent = bundle.main_agent.name
                        logger.info(
                            "Registered %d Copilot custom agent(s); active_agent=%s",
                            len(bundle.agents),
                            active_agent or "(default)",
                        )

                    if bundle.skills_dir:
                        sandbox_skills = await upload_skills_to_sandbox(sbox, bundle.skills_dir, ".github/skills")
                        skill_directories_json = json.dumps([sandbox_skills])

                    extra_mcp_config = read_mcp_config(bundle.mcp_config)

                    # Derive timeout from inspect_ai's sample time limit.
                    # sample_limits().time.remaining gives the seconds left
                    # in the current sample's time budget (set by Task(
                    # time_limit=...) or --time-limit CLI flag).
                    # Import deferred to avoid import-time context errors.
                    from inspect_ai.util import sample_limits

                    try:
                        remaining = sample_limits().time.remaining
                        runner_timeout = (
                            max(int(remaining), _MIN_TIMEOUT) if remaining is not None else _DEFAULT_TIMEOUT
                        )
                    except RuntimeError:
                        # No active sample context (e.g. during testing)
                        runner_timeout = _DEFAULT_TIMEOUT

                    # Build environment
                    runner_env = _build_runner_env(
                        bridge_port=bridge.port,
                        model=config.model,
                        prompt=user_prompt,
                        mcp_configs=bridge.mcp_server_configs,
                        timeout=runner_timeout,
                        persona_prompt=persona_prompt,
                        skill_directories_json=skill_directories_json,
                        extra_mcp_config=extra_mcp_config,
                        custom_agents_json=custom_agents_json,
                        active_agent=active_agent,
                        config_dir=config_dir,
                        nested_agents_mode=config.nested_agents,
                        nested_agent_timeout=config.nested_agent_timeout,
                        nested_agent_quiet_timeout=config.nested_agent_quiet_timeout,
                        nested_agent_max_calls=config.nested_agent_max_calls,
                        nested_agent_output_limit=config.nested_agent_output_limit,
                        idle_timeout=config.idle_timeout,
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

                        # Check if this is a timeout from the old-style
                        # runner ("Timeout after ...s waiting for session.idle")
                        # — treat as non-fatal so partial scoring can proceed.
                        if "Timeout after" in stderr and "session.idle" in stderr:
                            logger.warning(
                                "Copilot runner timed out (non-fatal, returning partial transcript): %s",
                                stderr[:500],
                            )
                            return bridge.state

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

                    # Parse and log runner metrics on success
                    if result.returncode == 0 and result.stderr:
                        metrics = parse_runner_metrics(result.stderr)
                        if metrics:
                            logger.info(
                                "Copilot runner metrics: %s",
                                json.dumps(metrics),
                            )

                        # Parse and log idle decision if present
                        idle_decision = parse_idle_decision(result.stderr)
                        if idle_decision:
                            logger.info(
                                "Copilot idle decision: action=%s classifications=%d",
                                idle_decision.action,
                                len(idle_decision.classifications),
                            )

                        if "COPILOT_RUNNER_IDLE_TIMEOUT" in result.stderr:
                            logger.warning(
                                "Copilot session appeared orphaned (idle timeout). "
                                "Returning partial transcript for scoring.",
                            )
                        elif "COPILOT_RUNNER_MAX_TIMEOUT" in result.stderr:
                            logger.info(
                                "Copilot session hit max time limit but was actively working. "
                                "Returning partial transcript for scoring.",
                            )

                        # Log bridge warnings even on success
                        warnings = parse_bridge_stderr(result.stderr)
                        if warnings:
                            logger.warning(
                                "Bridge proxy warnings in Copilot runner (exit 0):\n%s",
                                warnings,
                            )

                    # Parse CLI stderr for enriched observability
                    parsed_output = parse_copilot_cli(result.stderr or "")
                    if parsed_output.error_messages or parsed_output.final_response:
                        try:
                            from inspect_ai.log._transcript import transcript

                            transcript().info(
                                parsed_output.model_dump(mode="json"),
                                source="saber.cli_output",
                            )
                        except Exception:
                            logger.debug("Could not record CLI output InfoEvent")

                # Record bridge session summary
                record_bridge_summary(get_tracking_summary, logger)

                # Raise LimitExceededError in the main flow if the
                # tool-call limit was hit so that apply_limits handles
                # it cleanly (rather than via cancel-scope propagation
                # which can stall waiting for the subprocess).
                check_tool_limit()

                return bridge.state

            return execute

        return as_solver(
            _copilot_agent(),
            limits=[tool_call_limit(max_steps)],
        )

    return create_with_prompts
