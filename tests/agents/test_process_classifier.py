"""Tests for the deterministic process health classifier.

Tests cover:
1. Pydantic models (ProcessHealth, ProcessClassification, IdleDecision)
2. /proc reader functions (extracted from RUNNER_SCRIPT via exec)
3. classify_process decision tree
4. make_idle_decision aggregation logic
5. parse_idle_decision host-side parsing
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helper: extract testable functions from RUNNER_SCRIPT
# ---------------------------------------------------------------------------


def _load_runner_functions() -> dict[str, object]:
    """Extract testable functions from RUNNER_SCRIPT.

    Executes everything before ``async def main`` so constants and
    module-level functions are available in the returned namespace.

    A stub ``copilot`` package is injected into ``sys.modules`` so the
    script's ``from copilot.types import PermissionRequestResult`` succeeds
    even when the real SDK is not installed.
    """
    import sys
    import types

    # Stub the copilot SDK so the import inside RUNNER_SCRIPT succeeds.
    _copilot_stub = types.ModuleType("copilot")
    _copilot_types_stub = types.ModuleType("copilot.types")
    _copilot_types_stub.PermissionRequestResult = type(  # type: ignore[attr-defined]
        "PermissionRequestResult", (), {}
    )
    _copilot_stub.types = _copilot_types_stub  # type: ignore[attr-defined]

    prev_copilot = sys.modules.get("copilot")
    prev_copilot_types = sys.modules.get("copilot.types")
    sys.modules["copilot"] = _copilot_stub
    sys.modules["copilot.types"] = _copilot_types_stub

    try:
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        main_idx = RUNNER_SCRIPT.index("async def main")
        preamble = RUNNER_SCRIPT[:main_idx]
        namespace: dict[str, object] = {"__builtins__": __builtins__}
        exec(compile(preamble, "<runner>", "exec"), namespace)  # noqa: S102
        return namespace
    finally:
        # Restore original module state to avoid leaking stubs.
        if prev_copilot is None:
            sys.modules.pop("copilot", None)
        else:
            sys.modules["copilot"] = prev_copilot
        if prev_copilot_types is None:
            sys.modules.pop("copilot.types", None)
        else:
            sys.modules["copilot.types"] = prev_copilot_types


# Cache the namespace so we don't recompile for every test
_NS: dict[str, object] | None = None


def _ns() -> dict[str, object]:
    global _NS  # noqa: PLW0603
    if _NS is None:
        _NS = _load_runner_functions()
    return _NS


# ---------------------------------------------------------------------------
# Shared test helper
# ---------------------------------------------------------------------------


def _make_snapshot(
    read_bytes: int = 0,
    write_bytes: int = 0,
    utime: int = 0,
    stime: int = 0,
) -> dict[str, int]:
    """Build a minimal I/O + CPU snapshot dict for testing."""
    return {
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "utime": utime,
        "stime": stime,
    }


# ---------------------------------------------------------------------------
# 1. Pydantic model tests
# ---------------------------------------------------------------------------


class TestPydanticModels:
    """Test host-side Pydantic models for classifier results."""

    def test_process_health_enum_values(self) -> None:
        from saber.agents.registry.copilot.solver import ProcessHealth

        assert ProcessHealth.STUCK == "stuck"
        assert ProcessHealth.WORKING == "working"
        assert ProcessHealth.INDETERMINATE == "indeterminate"

    def test_process_classification_creation(self) -> None:
        from saber.agents.registry.copilot.solver import ProcessClassification, ProcessHealth

        pc = ProcessClassification(
            pid=42,
            cmdline="bash -c sleep 10",
            state="S",
            health=ProcessHealth.WORKING,
            reason="active I/O",
            wchan="poll_schedule_timeout",
            fd0_target="pipe:[12345]",
        )
        assert pc.pid == 42
        assert pc.health == ProcessHealth.WORKING
        assert pc.state == "S"

    def test_process_classification_frozen(self) -> None:
        from saber.agents.registry.copilot.solver import ProcessClassification, ProcessHealth

        pc = ProcessClassification(
            pid=1,
            cmdline="cat",
            state="S",
            health=ProcessHealth.STUCK,
            reason="interactive",
            wchan="",
            fd0_target="",
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            pc.pid = 99  # type: ignore[misc]

    def test_idle_decision_creation(self) -> None:
        from saber.agents.registry.copilot.solver import IdleDecision, ProcessClassification, ProcessHealth

        classifications = [
            ProcessClassification(
                pid=10,
                cmdline="less foo",
                state="S",
                health=ProcessHealth.STUCK,
                reason="interactive command",
                wchan="n_tty_read",
                fd0_target="/dev/pts/0",
            ),
        ]
        decision = IdleDecision(
            action="kill_stuck",
            classifications=classifications,
            in_flight_tools=["bash"],
            idle_seconds=120.5,
            total_seconds=600.0,
        )
        assert decision.action == "kill_stuck"
        assert len(decision.classifications) == 1
        assert decision.idle_seconds == 120.5

    def test_idle_decision_frozen(self) -> None:
        from saber.agents.registry.copilot.solver import IdleDecision

        decision = IdleDecision(
            action="timeout",
            classifications=[],
            in_flight_tools=[],
            idle_seconds=300.0,
            total_seconds=900.0,
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            decision.action = "extend"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. /proc reader tests (using tmp_path as mock /proc)
# ---------------------------------------------------------------------------


class TestProcReaders:
    """Test /proc filesystem reader functions."""

    def test_read_proc_stat_happy(self, tmp_path: Path) -> None:
        read_proc_stat = _ns()["read_proc_stat"]
        pid_dir = tmp_path / "42"
        pid_dir.mkdir()
        # Minimal /proc/PID/stat line:
        # pid (comm) state ppid ... (fields 1-52)
        stat_line = "42 (bash) S 1 42 42 0 -1 4194304 100 0 0 0 50 25 0 0 20 0 1 0 100 0 0"
        (pid_dir / "stat").write_text(stat_line)

        result = read_proc_stat(42, str(tmp_path))
        assert result is not None
        assert result["pid"] == 42
        assert result["comm"] == "bash"
        assert result["state"] == "S"
        assert result["ppid"] == 1
        assert result["utime"] == 50
        assert result["stime"] == 25

    def test_read_proc_stat_missing(self, tmp_path: Path) -> None:
        read_proc_stat = _ns()["read_proc_stat"]
        result = read_proc_stat(9999, str(tmp_path))
        assert result is None

    def test_read_proc_wchan_happy(self, tmp_path: Path) -> None:
        read_proc_wchan = _ns()["read_proc_wchan"]
        pid_dir = tmp_path / "42"
        pid_dir.mkdir()
        (pid_dir / "wchan").write_text("poll_schedule_timeout")

        result = read_proc_wchan(42, str(tmp_path))
        assert result == "poll_schedule_timeout"

    def test_read_proc_wchan_missing(self, tmp_path: Path) -> None:
        read_proc_wchan = _ns()["read_proc_wchan"]
        result = read_proc_wchan(9999, str(tmp_path))
        assert result == ""

    def test_read_proc_io_happy(self, tmp_path: Path) -> None:
        read_proc_io = _ns()["read_proc_io"]
        pid_dir = tmp_path / "42"
        pid_dir.mkdir()
        io_content = (
            "rchar: 1000\n"
            "wchar: 500\n"
            "syscr: 10\n"
            "syscw: 5\n"
            "read_bytes: 4096\n"
            "write_bytes: 2048\n"
            "cancelled_write_bytes: 0\n"
        )
        (pid_dir / "io").write_text(io_content)

        result = read_proc_io(42, str(tmp_path))
        assert result is not None
        assert result["read_bytes"] == 4096
        assert result["write_bytes"] == 2048

    def test_read_proc_io_missing(self, tmp_path: Path) -> None:
        read_proc_io = _ns()["read_proc_io"]
        result = read_proc_io(9999, str(tmp_path))
        assert result is None

    def test_read_proc_fd0_happy(self, tmp_path: Path) -> None:
        read_proc_fd0 = _ns()["read_proc_fd0"]
        pid_dir = tmp_path / "42"
        fd_dir = pid_dir / "fd"
        fd_dir.mkdir(parents=True)
        # Create a symlink for fd 0
        target = tmp_path / "some_pipe"
        target.write_text("")
        (fd_dir / "0").symlink_to(target)

        result = read_proc_fd0(42, str(tmp_path))
        assert str(target) in result

    def test_read_proc_fd0_missing(self, tmp_path: Path) -> None:
        read_proc_fd0 = _ns()["read_proc_fd0"]
        result = read_proc_fd0(9999, str(tmp_path))
        assert result == ""

    def test_read_proc_cmdline_happy(self, tmp_path: Path) -> None:
        read_proc_cmdline = _ns()["read_proc_cmdline"]
        pid_dir = tmp_path / "42"
        pid_dir.mkdir()
        # /proc/PID/cmdline uses null bytes as separators
        (pid_dir / "cmdline").write_bytes(b"python3\x00-c\x00print('hi')\x00")

        result = read_proc_cmdline(42, str(tmp_path))
        assert result == ["python3", "-c", "print('hi')"]

    def test_read_proc_cmdline_missing(self, tmp_path: Path) -> None:
        read_proc_cmdline = _ns()["read_proc_cmdline"]
        result = read_proc_cmdline(9999, str(tmp_path))
        assert result == []

    def test_get_descendant_pids(self, tmp_path: Path) -> None:
        get_descendant_pids = _ns()["get_descendant_pids"]
        # Create a process tree: 1 -> 10 -> 100, 1 -> 20
        for pid, ppid in [(1, 0), (10, 1), (20, 1), (100, 10), (200, 99)]:
            d = tmp_path / str(pid)
            d.mkdir()
            stat_line = f"{pid} (proc) S {ppid} {pid} {pid} 0 -1 0 0 0 0 0 10 5 0 0 20 0 1 0 0 0 0"
            (d / "stat").write_text(stat_line)

        result = get_descendant_pids(1, str(tmp_path))
        assert result == {10, 20, 100}

    def test_get_descendant_pids_no_children(self, tmp_path: Path) -> None:
        get_descendant_pids = _ns()["get_descendant_pids"]
        d = tmp_path / "1"
        d.mkdir()
        stat_line = "1 (init) S 0 1 1 0 -1 0 0 0 0 0 10 5 0 0 20 0 1 0 0 0 0"
        (d / "stat").write_text(stat_line)

        result = get_descendant_pids(1, str(tmp_path))
        assert result == set()


# ---------------------------------------------------------------------------
# 3. classify_process tests (decision tree branches)
# ---------------------------------------------------------------------------


class TestClassifyProcess:
    """Test each branch of the classify_process decision tree."""

    def _setup_proc(
        self,
        tmp_path: Path,
        pid: int,
        *,
        cmdline: str = "bash",
        state: str = "S",
        wchan: str = "0",
        fd0_target: str = "",
    ) -> None:
        """Create mock /proc entries for a process."""
        pid_dir = tmp_path / str(pid)
        pid_dir.mkdir(exist_ok=True)

        # cmdline
        parts = cmdline.split()
        (pid_dir / "cmdline").write_bytes(b"\x00".join(p.encode() for p in parts) + b"\x00")

        # stat
        comm = os.path.basename(parts[0]) if parts else "unknown"
        stat_line = f"{pid} ({comm}) {state} 1 {pid} {pid} 0 -1 0 0 0 0 0 10 5 0 0 20 0 1 0 0 0 0"
        (pid_dir / "stat").write_text(stat_line)

        # wchan
        (pid_dir / "wchan").write_text(wchan)

        # fd/0
        fd_dir = pid_dir / "fd"
        fd_dir.mkdir(exist_ok=True)
        if fd0_target:
            # Create symlink to a path that looks like the target
            target_file = tmp_path / f"fd0_target_{pid}"
            target_file.write_text("")
            link = fd_dir / "0"
            if link.exists() or link.is_symlink():
                link.unlink()
            # For pipe/pty/socket testing, we use the target name
            # The function reads the symlink, so we make it point to a named target
            link.symlink_to(fd0_target)

    def test_interactive_command_stuck(self, tmp_path: Path) -> None:
        classify_process = _ns()["classify_process"]
        self._setup_proc(tmp_path, 42, cmdline="less /var/log/syslog", state="S")
        prev = _make_snapshot()
        curr = _make_snapshot()

        result = classify_process(42, prev, curr, str(tmp_path))
        assert result["health"] == "stuck"
        assert "interactive" in result["reason"].lower()

    def test_stopped_process_stuck(self, tmp_path: Path) -> None:
        classify_process = _ns()["classify_process"]
        self._setup_proc(tmp_path, 42, cmdline="python3 script.py", state="T")
        prev = _make_snapshot()
        curr = _make_snapshot()

        result = classify_process(42, prev, curr, str(tmp_path))
        assert result["health"] == "stuck"
        assert "stopped" in result["reason"].lower()

    def test_stdin_blocked_stuck(self, tmp_path: Path) -> None:
        classify_process = _ns()["classify_process"]
        self._setup_proc(
            tmp_path,
            42,
            cmdline="cat",
            state="S",
            wchan="n_tty_read",
            fd0_target="pipe:[12345]",
        )
        prev = _make_snapshot()
        curr = _make_snapshot()

        result = classify_process(42, prev, curr, str(tmp_path))
        assert result["health"] == "stuck"
        assert "stdin" in result["reason"].lower() or "blocked" in result["reason"].lower()

    def test_active_io_working(self, tmp_path: Path) -> None:
        classify_process = _ns()["classify_process"]
        self._setup_proc(tmp_path, 42, cmdline="python3 script.py", state="S")
        prev = _make_snapshot(read_bytes=100)
        curr = _make_snapshot(read_bytes=200)

        result = classify_process(42, prev, curr, str(tmp_path))
        assert result["health"] == "working"

    def test_active_cpu_working(self, tmp_path: Path) -> None:
        classify_process = _ns()["classify_process"]
        self._setup_proc(tmp_path, 42, cmdline="python3 script.py", state="S")
        prev = _make_snapshot(utime=100)
        curr = _make_snapshot(utime=150)

        result = classify_process(42, prev, curr, str(tmp_path))
        assert result["health"] == "working"

    def test_network_socket_working(self, tmp_path: Path) -> None:
        classify_process = _ns()["classify_process"]
        self._setup_proc(
            tmp_path,
            42,
            cmdline="curl http://example.com",
            state="S",
            fd0_target="socket:[67890]",
        )
        prev = _make_snapshot()
        curr = _make_snapshot()

        result = classify_process(42, prev, curr, str(tmp_path))
        assert result["health"] == "working"

    def test_network_wchan_working(self, tmp_path: Path) -> None:
        classify_process = _ns()["classify_process"]
        self._setup_proc(
            tmp_path,
            42,
            cmdline="python3 server.py",
            state="S",
            wchan="tcp_recvmsg",
        )
        prev = _make_snapshot()
        curr = _make_snapshot()

        result = classify_process(42, prev, curr, str(tmp_path))
        assert result["health"] == "working"

    def test_indeterminate(self, tmp_path: Path) -> None:
        classify_process = _ns()["classify_process"]
        self._setup_proc(
            tmp_path,
            42,
            cmdline="python3 mystery.py",
            state="S",
            wchan="some_unknown_function",
        )
        prev = _make_snapshot()
        curr = _make_snapshot()

        result = classify_process(42, prev, curr, str(tmp_path))
        assert result["health"] == "indeterminate"

    def test_vanished_process_indeterminate(self, tmp_path: Path) -> None:
        """SF-1: A PID with no /proc entry should return indeterminate, not crash."""
        classify_process = _ns()["classify_process"]
        prev = _make_snapshot()
        curr = _make_snapshot()

        # PID 9999 has NO /proc entry at all in tmp_path
        result = classify_process(9999, prev, curr, str(tmp_path))
        assert result["health"] == "indeterminate"


# ---------------------------------------------------------------------------
# 4. make_idle_decision tests
# ---------------------------------------------------------------------------


class TestMakeIdleDecision:
    """Test the make_idle_decision aggregation logic."""

    def _setup_tree(
        self,
        tmp_path: Path,
        ancestor_pid: int,
        children: list[dict[str, object]],
    ) -> None:
        """Set up a mock process tree.

        children is a list of dicts with keys:
            pid, cmdline, state, wchan, fd0_target, ppid
        """
        # Create ancestor
        d = tmp_path / str(ancestor_pid)
        d.mkdir(exist_ok=True)
        stat_line = f"{ancestor_pid} (main) S 0 {ancestor_pid} {ancestor_pid} 0 -1 0 0 0 0 0 10 5 0 0 20 0 1 0 0 0 0"
        (d / "stat").write_text(stat_line)

        for child in children:
            pid = child["pid"]
            ppid = child.get("ppid", ancestor_pid)
            cmdline = child.get("cmdline", "bash")
            state = child.get("state", "S")
            wchan = child.get("wchan", "0")
            fd0_target = child.get("fd0_target", "")

            cd = tmp_path / str(pid)
            cd.mkdir(exist_ok=True)

            # cmdline
            parts = str(cmdline).split()
            (cd / "cmdline").write_bytes(b"\x00".join(p.encode() for p in parts) + b"\x00")

            # stat
            comm = os.path.basename(parts[0]) if parts else "unknown"
            stat_line = f"{pid} ({comm}) {state} {ppid} {pid} {pid} 0 -1 0 0 0 0 0 10 5 0 0 20 0 1 0 0 0 0"
            (cd / "stat").write_text(stat_line)

            # wchan
            (cd / "wchan").write_text(str(wchan))

            # io
            io_content = (
                "rchar: 0\nwchar: 0\nsyscr: 0\nsyscw: 0\n"
                "read_bytes: 0\nwrite_bytes: 0\ncancelled_write_bytes: 0\n"
            )
            (cd / "io").write_text(io_content)

            # fd/0
            fd_dir = cd / "fd"
            fd_dir.mkdir(exist_ok=True)
            if fd0_target:
                (fd_dir / "0").symlink_to(str(fd0_target))

    def test_all_stuck_kills(self, tmp_path: Path) -> None:
        make_idle_decision = _ns()["make_idle_decision"]
        self._setup_tree(
            tmp_path,
            1,
            [
                {"pid": 10, "cmdline": "less /var/log/messages", "state": "S", "ppid": 1},
                {"pid": 20, "cmdline": "vi /etc/hosts", "state": "S", "ppid": 1},
            ],
        )
        prev_snapshots: dict[int, dict[str, int]] = {
            10: _make_snapshot(),
            20: _make_snapshot(),
        }

        decision, new_snapshots = make_idle_decision(
            1, prev_snapshots, {}, 120.0, 600.0, str(tmp_path)
        )
        assert decision["action"] == "kill_stuck"
        assert len(decision["classifications"]) == 2

    def test_any_working_extends(self, tmp_path: Path) -> None:
        make_idle_decision = _ns()["make_idle_decision"]
        self._setup_tree(
            tmp_path,
            1,
            [
                {"pid": 10, "cmdline": "less /var/log/messages", "state": "S", "ppid": 1},
                {"pid": 20, "cmdline": "python3 server.py", "state": "S", "wchan": "tcp_recvmsg", "ppid": 1},
            ],
        )
        prev_snapshots: dict[int, dict[str, int]] = {
            10: _make_snapshot(),
            20: _make_snapshot(),
        }

        decision, _ = make_idle_decision(
            1, prev_snapshots, {}, 120.0, 600.0, str(tmp_path)
        )
        assert decision["action"] == "extend"

    def test_indeterminate_extends(self, tmp_path: Path) -> None:
        make_idle_decision = _ns()["make_idle_decision"]
        self._setup_tree(
            tmp_path,
            1,
            [
                {"pid": 10, "cmdline": "python3 mystery.py", "state": "S", "wchan": "unknown_fn", "ppid": 1},
            ],
        )
        prev_snapshots: dict[int, dict[str, int]] = {}

        decision, _ = make_idle_decision(
            1, prev_snapshots, {}, 120.0, 600.0, str(tmp_path)
        )
        assert decision["action"] == "extend"

    def test_mixed_stuck_and_working_extends(self, tmp_path: Path) -> None:
        make_idle_decision = _ns()["make_idle_decision"]
        self._setup_tree(
            tmp_path,
            1,
            [
                {"pid": 10, "cmdline": "less foo", "state": "S", "ppid": 1},
                {"pid": 20, "cmdline": "python3 worker.py", "state": "S", "wchan": "poll_schedule_timeout", "ppid": 1},
            ],
        )
        prev_snapshots: dict[int, dict[str, int]] = {
            10: _make_snapshot(),
            20: _make_snapshot(),
        }

        decision, _ = make_idle_decision(
            1, prev_snapshots, {}, 120.0, 600.0, str(tmp_path)
        )
        assert decision["action"] == "extend"

    def test_no_descendants_timeout(self, tmp_path: Path) -> None:
        make_idle_decision = _ns()["make_idle_decision"]
        # Only ancestor, no children
        self._setup_tree(tmp_path, 1, [])

        decision, _ = make_idle_decision(
            1, {}, {}, 120.0, 600.0, str(tmp_path)
        )
        assert decision["action"] == "timeout"

    def test_in_flight_tools_forces_extend(self, tmp_path: Path) -> None:
        make_idle_decision = _ns()["make_idle_decision"]
        # All stuck, but in_flight_tools is non-empty
        self._setup_tree(
            tmp_path,
            1,
            [
                {"pid": 10, "cmdline": "less foo", "state": "S", "ppid": 1},
            ],
        )
        prev_snapshots: dict[int, dict[str, int]] = {
            10: _make_snapshot(),
        }

        decision, _ = make_idle_decision(
            1, prev_snapshots, {"bash": 1234.0}, 120.0, 600.0, str(tmp_path)
        )
        assert decision["action"] == "extend"


# ---------------------------------------------------------------------------
# 5. parse_idle_decision tests
# ---------------------------------------------------------------------------


class TestParseIdleDecision:
    """Test host-side parse_idle_decision from bridge_utils."""

    def test_valid_line(self) -> None:
        from saber.agents.bridge_utils import parse_idle_decision

        decision_data = {
            "action": "kill_stuck",
            "classifications": [
                {
                    "pid": 42,
                    "cmdline": "less foo",
                    "state": "S",
                    "health": "stuck",
                    "reason": "interactive command",
                    "wchan": "n_tty_read",
                    "fd0_target": "/dev/pts/0",
                },
            ],
            "in_flight_tools": [],
            "idle_seconds": 120.5,
            "total_seconds": 600.0,
        }
        stderr = f"some output\nCOPILOT_IDLE_DECISION: {json.dumps(decision_data)}\nmore output"
        result = parse_idle_decision(stderr)
        assert result is not None
        assert result.action == "kill_stuck"
        assert len(result.classifications) == 1
        assert result.classifications[0].pid == 42

    def test_missing_line(self) -> None:
        from saber.agents.bridge_utils import parse_idle_decision

        result = parse_idle_decision("no decision here\njust some output")
        assert result is None

    def test_empty_stderr(self) -> None:
        from saber.agents.bridge_utils import parse_idle_decision

        result = parse_idle_decision("")
        assert result is None

    def test_invalid_json(self) -> None:
        from saber.agents.bridge_utils import parse_idle_decision

        result = parse_idle_decision("COPILOT_IDLE_DECISION: {invalid json!!}")
        assert result is None

    def test_multiple_decisions_returns_last(self) -> None:
        """SF-2: When multiple COPILOT_IDLE_DECISION lines appear, the last wins."""
        from saber.agents.bridge_utils import parse_idle_decision

        extend_data = {
            "action": "extend",
            "classifications": [],
            "in_flight_tools": ["bash"],
            "idle_seconds": 60.0,
            "total_seconds": 300.0,
        }
        kill_data = {
            "action": "kill_stuck",
            "classifications": [
                {
                    "pid": 42,
                    "cmdline": "less foo",
                    "state": "S",
                    "health": "stuck",
                    "reason": "interactive command",
                    "wchan": "n_tty_read",
                    "fd0_target": "/dev/pts/0",
                },
            ],
            "in_flight_tools": [],
            "idle_seconds": 120.0,
            "total_seconds": 600.0,
        }
        stderr = (
            f"COPILOT_IDLE_DECISION: {json.dumps(extend_data)}\n"
            f"COPILOT_IDLE_DECISION: {json.dumps(extend_data)}\n"
            f"COPILOT_IDLE_DECISION: {json.dumps(extend_data)}\n"
            f"COPILOT_IDLE_DECISION: {json.dumps(kill_data)}\n"
        )
        result = parse_idle_decision(stderr)
        assert result is not None
        assert result.action == "kill_stuck"

    def test_extra_fields_ignored(self) -> None:
        """SF-3: Extra unknown keys in JSON should be ignored by Pydantic."""
        from saber.agents.bridge_utils import parse_idle_decision

        decision_data = {
            "action": "extend",
            "classifications": [],
            "in_flight_tools": [],
            "idle_seconds": 30.0,
            "total_seconds": 200.0,
            "extra_field": "should_be_ignored",
            "another_unknown": 42,
        }
        stderr = f"COPILOT_IDLE_DECISION: {json.dumps(decision_data)}"
        result = parse_idle_decision(stderr)
        assert result is not None
        assert result.action == "extend"
        assert not hasattr(result, "extra_field")
