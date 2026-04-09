"""Copilot agent solver — sandbox_agent_bridge integration.

Uses inspect_ai's ``sandbox_agent_bridge()`` to run a self-contained
Copilot SDK runner script INSIDE the Docker sandbox.  The bridge proxies
model API calls back to inspect_ai's configured model, so the Copilot SDK
never needs direct access to the real model endpoint.

Two-level factory pattern:
    ``create_agent(**kwargs)`` -> ``create_with_prompts(**prompt_kwargs)`` -> ``Solver``

Architecture:
    1. solver writes RUNNER_SCRIPT to sandbox filesystem
    2. sandbox_agent_bridge() opens a local proxy on the configured port
    3. runner script starts CopilotClient with BYOK provider pointed at proxy
    4. all model traffic flows: runner -> bridge proxy -> inspect_ai model
    5. bridge.state captures the full message transcript
"""

import json
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from saber.agents.bridge_tracking import create_tracking_filter
from saber.agents.bridge_utils import (
    build_bridged_tools_for_copilot,
    build_system_prompt,
    build_user_prompt,
    compose_filters,
    create_tool_call_limit_filter,
    parse_bridge_stderr,
    parse_idle_decision,
    parse_runner_metrics,
    record_bridge_summary,
    resolve_model_aliases,
    tool_call_limit,
    upload_skills_to_sandbox,
    validate_model_availability,
)
from saber.agents.cli_output_parser import parse_copilot_stderr as parse_copilot_cli
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
- COPILOT_TIMEOUT: Max wall-clock timeout in seconds (default: 3600)
- COPILOT_IDLE_TIMEOUT: Max idle time with no activity events (default: 300)
"""

import asyncio
import json
import os
import signal
import sys
import time

from copilot.types import PermissionRequestResult

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
) -> PermissionRequestResult:
    """Auto-approve every permission request.

    The Copilot SDK's PermissionHandler signature is
    ``(PermissionRequest, Dict[str, str]) -> PermissionRequestResult``.
    Returning a ``PermissionRequestResult(kind="approved")`` grants the
    request.  The SDK accesses result attributes (``result.kind``), so a
    plain dict would raise ``AttributeError`` and be silently converted to
    a denial.

    This is safe because the runner executes inside an isolated Docker
    sandbox used exclusively for benchmarking.
    """
    return PermissionRequestResult(kind="approved")


async def main() -> int:
    """Run Copilot session with bridge-routed model."""
    from copilot import CopilotClient
    from copilot.generated.session_events import SessionEventType

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:13131/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    model = os.environ.get("COPILOT_MODEL", "inspect")
    prompt = os.environ.get("COPILOT_PROMPT", "")
    mcp_config_str = os.environ.get("COPILOT_MCP_CONFIG", "")
    max_timeout = int(os.environ.get("COPILOT_TIMEOUT", "3600"))
    idle_timeout = int(os.environ.get("COPILOT_IDLE_TIMEOUT", "300"))

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
    exit_reason = "started"

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
        session = await client.create_session(session_config)

        try:
            unsubscribe = session.on(_on_event)
            try:
                await session.send({"prompt": prompt})
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
                            exit_reason = "stuck_processes"
                            break
                        elif decision["action"] == "extend":
                            last_activity_time = time.monotonic()
                            idle_extensions += 1
                            if idle_extensions > MAX_IDLE_EXTENSIONS:
                                _print_metrics("max_idle_extensions")
                                exit_reason = "max_idle_extensions"
                                break
                            continue
                        else:  # "timeout"
                            print(
                                f"COPILOT_RUNNER_IDLE_TIMEOUT: No activity for "
                                f"{idle_elapsed:.0f}s (limit: {idle_timeout}s)",
                                file=sys.stderr,
                            )
                            _print_metrics("idle_timeout")
                            exit_reason = "idle_timeout"
                            break

                    if total_elapsed > max_timeout:
                        print(
                            f"COPILOT_RUNNER_MAX_TIMEOUT: Session active for "
                            f"{total_elapsed:.0f}s (limit: {max_timeout}s)",
                            file=sys.stderr,
                        )
                        _print_metrics("max_timeout")
                        exit_reason = "max_timeout"
                        break

                else:
                    # idle_event was set — normal completion or session error
                    if session_error_msg:
                        print(f"COPILOT_SESSION_ERROR: {session_error_msg}", file=sys.stderr)
                        _print_metrics("session_error")
                        exit_reason = "session_error"
                    else:
                        if last_response:
                            content = getattr(getattr(last_response, "data", None), "content", None)
                            if content:
                                print(f"COPILOT_RESPONSE: {content[:500]}", file=sys.stderr)
                        _print_metrics("completed")
                        exit_reason = "completed"

            finally:
                unsubscribe()
        finally:
            await session.destroy()
    except Exception as exc:
        # Non-timeout exceptions (session creation failure, etc.)
        print(f"ERROR: Session failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.stop()

    if exit_reason != "completed":
        print(f"ERROR: Copilot runner incomplete exit: {exit_reason}", file=sys.stderr)
        return 1

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
    persona_file: str | None = None
    skills_dir: str | None = None
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
    return {
        "OPENAI_BASE_URL": f"http://localhost:{bridge_port}/v1",
        "OPENAI_API_KEY": "sk-placeholder-for-bridge",
        "COPILOT_MODEL": model,
        "COPILOT_PROMPT": prompt,
        "COPILOT_MCP_CONFIG": json.dumps(mcp_dict),
        "COPILOT_TIMEOUT": str(timeout),
        "COPILOT_PERSONA_PROMPT": persona_prompt,
        "COPILOT_SKILL_DIRECTORIES": skill_directories_json,
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
        from inspect_ai.util import store

        config = CopilotBridgeConfig.from_kwargs(dict(outer_kwargs))

        system_prompt = build_system_prompt(
            instruction_prompt=instruction_prompt,
            assistant_prompt=assistant_prompt,
        )

        bridged = build_bridged_tools_for_copilot(tools)

        @agent
        def _copilot_agent() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                # Fail fast on model misconfiguration (e.g. wrong name)
                await validate_model_availability()

                port = store().get(_STORE_PORT_KEY, config.port_base) + 1
                store().set(_STORE_PORT_KEY, port)

                bridge_filter, check_tool_limit = create_tool_call_limit_filter()
                tracking_filter, get_tracking_summary = create_tracking_filter()
                composed = compose_filters(tracking_filter, bridge_filter)

                _raw_aliases = outer_kwargs.get("model_aliases")
                model_aliases = resolve_model_aliases(_raw_aliases if isinstance(_raw_aliases, dict) else None)

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
