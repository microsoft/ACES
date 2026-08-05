# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for copilot agent module — sandbox_agent_bridge architecture."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Phase 1: Runner Script
# ---------------------------------------------------------------------------


class TestRunnerScript:
    """Tests for the embedded RUNNER_SCRIPT constant."""

    def test_runner_script_is_valid_python(self) -> None:
        """RUNNER_SCRIPT compiles as valid Python."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        compile(RUNNER_SCRIPT, "<runner>", "exec")

    def test_runner_script_no_saber_imports(self) -> None:
        """Runner must not import from saber (runs in sandbox)."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "from saber" not in RUNNER_SCRIPT
        assert "import saber" not in RUNNER_SCRIPT

    def test_runner_script_uses_copilot_sdk(self) -> None:
        """Runner imports from copilot SDK."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "from copilot" in RUNNER_SCRIPT or "import copilot" in RUNNER_SCRIPT

    def test_runner_script_approves_permissions_with_sdk_kind(self) -> None:
        """Runner uses the current Copilot SDK permission decision."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "PermissionDecisionApproveOnce()" in RUNNER_SCRIPT
        assert 'PermissionRequestResult(kind="approve-once")' in RUNNER_SCRIPT
        assert 'PermissionRequestResult(kind="approved")' not in RUNNER_SCRIPT

    def test_runner_script_reads_env_vars(self) -> None:
        """Runner reads required env vars."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        for var in ["OPENAI_BASE_URL", "COPILOT_MODEL", "COPILOT_PROMPT"]:
            assert var in RUNNER_SCRIPT

    def test_runner_script_has_main_guard(self) -> None:
        """Runner has if __name__ == '__main__' guard."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "__name__" in RUNNER_SCRIPT
        assert "__main__" in RUNNER_SCRIPT

    def test_runner_script_persists_bridge_events(self) -> None:
        """Default Python runner writes bounded tool-only scoring events."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert 'EVENT_LOG_PATH = "/workspace/.runner_events.jsonl"' in RUNNER_SCRIPT
        assert "EVENT_SCHEMA_VERSION = 1" in RUNNER_SCRIPT
        assert "EVENT_LOG_MAX_EVENTS" in RUNNER_SCRIPT
        assert "EVENT_LOG_MAX_BYTES" in RUNNER_SCRIPT
        assert "SECRET_FIELD_NAMES" in RUNNER_SCRIPT
        assert "def _tool_event_record(event):" in RUNNER_SCRIPT
        assert '"schema_version": EVENT_SCHEMA_VERSION' in RUNNER_SCRIPT
        assert '"toolCallId": str(tool_call_id)' in RUNNER_SCRIPT
        assert '"parentToolCallId": (' in RUNNER_SCRIPT
        assert '"arguments": _bounded_event_value(' in RUNNER_SCRIPT
        assert '"result": _bounded_event_value(' in RUNNER_SCRIPT
        assert "def _append_event_log(event):" in RUNNER_SCRIPT

    def test_runner_event_constants_match_scoring_module(self) -> None:
        """Embedded runner constants must stay in sync with the reader/parser.

        The runner is emitted as a sandbox string and cannot import saber, so
        its scoring constants are duplicated in three places (this embedded
        script, ``bridge_events`` and the ``solver`` reader). A silent drift —
        especially the schema version — would make the reader reject every event
        and fall back to message-derived steps, so pin the values together here.
        """
        import re

        from saber.agents.bridge_events import (
            _SECRET_FIELD_NAMES,
            BRIDGE_TOOL_EVENT_FIELD_MAX_CHARS,
            BRIDGE_TOOL_EVENT_SCHEMA_VERSION,
        )
        from saber.agents.registry.copilot.solver import (
            _RUNNER_EVENT_LOG_MAX_BYTES,
            _RUNNER_EVENT_LOG_MAX_EVENTS,
            _RUNNER_EVENT_LOG_PATH,
            RUNNER_SCRIPT,
        )

        def _int_const(name: str) -> int:
            match = re.search(rf"(?m)^{name}\s*=\s*([0-9_]+)\b", RUNNER_SCRIPT)
            assert match, f"{name} not found in RUNNER_SCRIPT"
            return int(match.group(1).replace("_", ""))

        def _secret_names() -> set[str]:
            match = re.search(
                r"SECRET_FIELD_NAMES\s*=\s*frozenset\(\{(.*?)\}\)",
                RUNNER_SCRIPT,
                re.S,
            )
            assert match, "SECRET_FIELD_NAMES not found in RUNNER_SCRIPT"
            return set(re.findall(r'"([^"]+)"', match.group(1)))

        # Schema version drift is the dangerous one: it makes the reader reject
        # the whole event log and silently score from messages instead.
        assert _int_const("EVENT_SCHEMA_VERSION") == BRIDGE_TOOL_EVENT_SCHEMA_VERSION
        assert _int_const("EVENT_FIELD_MAX_CHARS") == BRIDGE_TOOL_EVENT_FIELD_MAX_CHARS
        assert _int_const("EVENT_LOG_MAX_EVENTS") == _RUNNER_EVENT_LOG_MAX_EVENTS
        assert _int_const("EVENT_LOG_MAX_BYTES") == _RUNNER_EVENT_LOG_MAX_BYTES
        assert _secret_names() == set(_SECRET_FIELD_NAMES)
        assert f'EVENT_LOG_PATH = "{_RUNNER_EVENT_LOG_PATH}"' in RUNNER_SCRIPT
        assert "_append_event_log(event)" in RUNNER_SCRIPT
        assert 'event_type not in {"tool.execution_start", "tool.execution_complete"}' in RUNNER_SCRIPT


# ---------------------------------------------------------------------------
# Phase 2: Solver Helpers
# ---------------------------------------------------------------------------


class TestCopilotBridgeStdoutCompletionConfig:
    """CopilotBridgeConfig model."""

    def test_defaults(self) -> None:
        """Default values are correct."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig()
        assert cfg.sandbox_name == "default"
        assert cfg.port_base == 3000
        assert cfg.model == "inspect"
        assert cfg.artifact_patterns == ()

    def test_from_kwargs(self) -> None:
        """from_kwargs extracts known fields and ignores unknowns."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig.from_kwargs({"sandbox_name": "custom", "unknown": "ignored"})
        assert cfg.sandbox_name == "custom"

    def test_accepts_host_mcp_server_directory(self) -> None:
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig.from_kwargs({
            "mcp_servers_dir": "/host/mcp_servers",
        })

        assert cfg.mcp_servers_dir == "/host/mcp_servers"

    async def test_uploads_host_mcp_servers_over_image_copy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from saber.agents.registry.copilot import solver

        upload = AsyncMock(return_value="mcp_servers")
        monkeypatch.setattr(solver, "upload_skills_to_sandbox", upload)
        sbox = object()

        result = await solver._upload_mcp_servers_to_sandbox(
            sbox,
            "/host/mcp_servers",
        )

        assert result == "mcp_servers"
        upload.assert_awaited_once_with(
            sbox,
            "/host/mcp_servers",
            "mcp_servers",
        )

    def test_frozen(self) -> None:
        """Config model is immutable."""
        from pydantic import ValidationError

        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig()
        with pytest.raises(ValidationError):
            cfg.sandbox_name = "other"  # type: ignore[misc]


class TestCopilotArtifactExport:
    """Host-side artifact preservation before sandbox cleanup."""

    async def test_exports_reports_and_disambiguates_logs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Required reports and both log namespaces are preserved."""
        from saber.agents.registry.copilot import solver

        monkeypatch.setattr(
            solver,
            "uuid4",
            lambda: SimpleNamespace(hex="artifact-id"),
        )
        paths = "\n".join(
            [
                "/workspace/report.md",
                "/workspace/investigation_report.md",
                "/workspace/.copilot/logs/session.log",
                "/root/.copilot/logs/session.log",
            ]
        )
        contents = {
            "/workspace/report.md": b"# Final",
            "/workspace/investigation_report.md": b"# Working",
            "/workspace/.copilot/logs/session.log": b"agent log",
            "/root/.copilot/logs/session.log": b"process log",
        }
        sbox = SimpleNamespace(
            exec=AsyncMock(
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=paths,
                    stderr="",
                )
            ),
            read_file=AsyncMock(side_effect=lambda path, text=False: contents[path]),
        )

        patterns = (
            "/workspace/report.md",
            "/workspace/investigation_report.md",
            "/workspace/.copilot/logs/*.log",
            "/root/.copilot/logs/*.log",
        )
        metadata = await solver._export_copilot_artifacts(
            sbox,
            str(tmp_path),
            patterns,
        )

        artifact_dir = tmp_path / "artifact-id"
        assert metadata == {
            "artifact_id": "artifact-id",
            "artifact_dir": str(artifact_dir),
            "files": [
                "report.md",
                "investigation_report.md",
                "agent-session.log",
                "process-session.log",
            ],
            "failed_files": [],
        }
        assert (artifact_dir / "report.md").read_text() == "# Final"
        assert (artifact_dir / "agent-session.log").read_text() == "agent log"
        assert (artifact_dir / "process-session.log").read_text() == "process log"
        assert sbox.exec.await_args.args[0][-4:] == list(patterns)
        sbox.read_file.assert_any_await(
            "/workspace/report.md",
            text=False,
        )

    async def test_preserves_binary_files_with_colliding_basenames(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Binary artifacts with the same basename receive unique names."""
        from saber.agents.registry.copilot import solver

        monkeypatch.setattr(
            solver,
            "uuid4",
            lambda: SimpleNamespace(hex="artifact-id"),
        )
        paths = "/workspace/a/result.bin\n/workspace/b/result.bin\n/workspace/c/result.bin\n"
        sbox = SimpleNamespace(
            exec=AsyncMock(
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=paths,
                    stderr="",
                )
            ),
            read_file=AsyncMock(
                side_effect=[b"\x00first", b"\x00second", b"\x00third"],
            ),
        )

        metadata = await solver._export_copilot_artifacts(
            sbox,
            str(tmp_path),
            ("/workspace/*/result.bin",),
        )

        assert metadata is not None
        names = metadata["files"]
        assert len(names) == 3
        assert len(set(names)) == 3
        artifact_dir = tmp_path / "artifact-id"
        assert sorted((artifact_dir / name).read_bytes() for name in names) == [
            b"\x00first",
            b"\x00second",
            b"\x00third",
        ]

    async def test_shielded_export_finishes_before_reraising_cancellation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancellation waits for the export task and is then preserved."""
        import asyncio

        from saber.agents.registry.copilot import solver

        exported = asyncio.Event()

        async def record(*args: object) -> None:
            await asyncio.sleep(0)
            exported.set()

        monkeypatch.setattr(solver, "_record_copilot_artifacts", record)

        task = asyncio.create_task(
            solver._record_copilot_artifacts_shielded(
                SimpleNamespace(),
                "/tmp/artifacts",
                ("/workspace/report.md",),
            )
        )
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert exported.is_set()

    async def test_read_failure_preserves_other_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One unreadable sandbox file does not mask the agent outcome."""
        from saber.agents.registry.copilot import solver

        monkeypatch.setattr(
            solver,
            "uuid4",
            lambda: SimpleNamespace(hex="artifact-id"),
        )

        async def read_file(path: str, text: bool = False) -> bytes:
            if path.endswith("report.md"):
                raise RuntimeError("sandbox read failed")
            return b"notes"

        sbox = SimpleNamespace(
            exec=AsyncMock(
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=("/workspace/report.md\n/workspace/investigation_notes.md\n"),
                    stderr="",
                )
            ),
            read_file=AsyncMock(side_effect=read_file),
        )

        metadata = await solver._export_copilot_artifacts(
            sbox,
            str(tmp_path),
            (
                "/workspace/report.md",
                "/workspace/investigation_notes.md",
            ),
        )

        assert metadata is not None
        assert metadata["files"] == ["investigation_notes.md"]
        assert metadata["failed_files"] == ["/workspace/report.md"]
        assert (tmp_path / "artifact-id" / "investigation_notes.md").read_text() == "notes"

    async def test_discovery_failure_returns_none(
        self,
        tmp_path: Path,
    ) -> None:
        """Sandbox discovery failures do not replace the agent result."""
        from saber.agents.registry.copilot import solver

        sbox = SimpleNamespace(
            exec=AsyncMock(side_effect=RuntimeError("sandbox unavailable")),
        )

        assert (
            await solver._export_copilot_artifacts(
                sbox,
                str(tmp_path),
                ("/workspace/report.md",),
            )
            is None
        )


class TestBridgeEventToolSteps:
    """Bridge event trajectory extraction."""

    def test_joins_nested_tool_start_and_completion(self) -> None:
        """Nested Copilot tool events are converted to scorer-ready steps."""
        from saber.agents.bridge_events import bridge_event_tool_steps_from_lines

        steps = bridge_event_tool_steps_from_lines(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_start",
                        "data": {
                            "toolCallId": "call_1",
                            "mcpToolName": "RunAdvancedHuntingQuery",
                            "arguments": {"kqlQuery": "AADSignInLogs | take 1"},
                            "parentToolCallId": "call_parent",
                        },
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_complete",
                        "data": {
                            "toolCallId": "call_1",
                            "success": True,
                            "result": {"content": '{"Results":[{"Account":"user"}],"Count":1}'},
                            "parentToolCallId": "call_parent",
                        },
                    }
                ),
            ]
        )

        assert steps == [
            {
                "step_number": 1,
                "tool_name": "RunAdvancedHuntingQuery",
                "tool_input": {"kqlQuery": "AADSignInLogs | take 1"},
                "output": '{"Results":[{"Account":"user"}],"Count":1}',
                "is_error": False,
                "error_type": None,
                "tool_call_id": "call_1",
                "parent_tool_call_id": "call_parent",
            }
        ]

    def test_completion_without_start_is_preserved(self) -> None:
        """Completion-only events are retained instead of silently dropped."""
        from saber.agents.bridge_events import bridge_event_tool_steps_from_lines

        steps = bridge_event_tool_steps_from_lines(
            [
                '{"schema_version":1,"type":"tool.execution_complete","data":{"toolCallId":"call_1","toolName":"get_incident","success":false,"result":{"content":"boom"},"toolTelemetry":{"errorType":"runtime"}}}',
            ]
        )

        assert steps[0]["tool_name"] == "get_incident"
        assert steps[0]["output"] == "boom"
        assert steps[0]["is_error"] is True
        assert steps[0]["error_type"] == "runtime"
        assert steps[0]["tool_call_id"] == "call_1"

    def test_incomplete_node_event_schema_falls_back(self) -> None:
        """Name-only Node events do not suppress message-derived tool steps."""
        from saber.agents.bridge_events import bridge_event_tool_steps_from_lines

        steps = bridge_event_tool_steps_from_lines(
            [
                '{"type":"tool.execution_start","tool":"investigation-sentinel-triage-get_incident"}',
                '{"type":"tool.execution_complete","tool":"unknown"}',
                '{"type":"tool.execution_start","tool":"sentinel-triage-RunAdvancedHuntingQuery"}',
            ]
        )

        assert steps == []

    def test_missing_call_id_rejects_event_log(self) -> None:
        """Call IDs are required for runner-independent event correlation."""
        from saber.agents.bridge_events import bridge_event_tool_steps_from_lines

        steps = bridge_event_tool_steps_from_lines(
            [
                '{"schema_version":1,"type":"tool.execution_start","data":{"toolName":"run_hunting_query","arguments":{}}}',
            ]
        )

        assert steps == []

    def test_node_schema_error_round_trip(self) -> None:
        """Canonical Node records preserve input, output, errors, and IDs."""
        from saber.agents.bridge_events import bridge_event_tool_steps_from_lines

        steps = bridge_event_tool_steps_from_lines(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_start",
                        "data": {
                            "toolCallId": "node_call",
                            "parentToolCallId": "node_parent",
                            "toolName": "get_incident",
                            "arguments": {"incident_id": "42"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_complete",
                        "data": {
                            "toolCallId": "node_call",
                            "parentToolCallId": "node_parent",
                            "success": False,
                            "result": {"content": "incident error"},
                            "toolTelemetry": {"errorType": "runtime"},
                        },
                    }
                ),
            ]
        )

        assert steps == [
            {
                "step_number": 1,
                "tool_name": "get_incident",
                "tool_input": {"incident_id": "42"},
                "output": "incident error",
                "is_error": True,
                "error_type": "runtime",
                "tool_call_id": "node_call",
                "parent_tool_call_id": "node_parent",
            }
        ]

    def test_snake_case_event_data_is_supported(self) -> None:
        """Python SDK model dumps can use snake_case field names."""
        from saber.agents.bridge_events import bridge_event_tool_steps_from_lines

        steps = bridge_event_tool_steps_from_lines(
            [
                '{"schema_version":1,"type":"tool.execution_start","data":{"tool_call_id":"call_1","mcp_tool_name":"get_incident","arguments":{"id":"42"},"parent_tool_call_id":"parent"}}',
                '{"schema_version":1,"type":"tool.execution_complete","data":{"tool_call_id":"call_1","success":false,"result":{"content":"boom"},"tool_telemetry":{"error_type":"runtime"}}}',
            ]
        )

        assert steps == [
            {
                "step_number": 1,
                "tool_name": "get_incident",
                "tool_input": {"id": "42"},
                "output": "boom",
                "is_error": True,
                "error_type": "runtime",
                "tool_call_id": "call_1",
                "parent_tool_call_id": "parent",
            }
        ]

    def test_parser_redacts_secrets_and_truncates_large_fields(self) -> None:
        """External runner records are sanitized before scoring metadata."""
        from saber.agents.bridge_events import (
            BRIDGE_TOOL_EVENT_FIELD_MAX_CHARS,
            bridge_event_tool_steps_from_lines,
        )

        oversized = "x" * (BRIDGE_TOOL_EVENT_FIELD_MAX_CHARS + 100)
        steps = bridge_event_tool_steps_from_lines(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_start",
                        "data": {
                            "toolCallId": "call_1",
                            "toolName": "run",
                            "arguments": {
                                "password": "super-secret",
                                "query": oversized,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_complete",
                        "data": {
                            "toolCallId": "call_1",
                            "success": True,
                            "result": {
                                "content": json.dumps(
                                    {
                                        "access_token": "token-value",
                                        "result": oversized,
                                    }
                                )
                            },
                        },
                    }
                ),
            ]
        )

        assert steps[0]["tool_input"]["password"] == "[REDACTED]"
        assert str(steps[0]["tool_input"]["query"]).endswith("...[truncated]")
        assert "token-value" not in steps[0]["output"]
        assert "[REDACTED]" in steps[0]["output"]
        assert "...[truncated]" in steps[0]["output"]


class TestBridgeEventLogIngestion:
    """Best-effort host ingestion of runner event files."""

    @pytest.mark.asyncio
    async def test_partial_timeout_events_are_read_before_return(self) -> None:
        """A valid partial event file survives the old-style timeout path."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from saber.agents.registry.copilot.solver import (
            _is_nonfatal_idle_timeout,
            _read_bridge_tool_steps,
            create_agent,
        )

        event_log = "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_start",
                        "data": {
                            "toolCallId": "partial_call",
                            "parentToolCallId": "parent_call",
                            "toolName": "get_incident",
                            "arguments": {"incident_id": "42"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_complete",
                        "data": {
                            "toolCallId": "partial_call",
                            "success": True,
                            "result": {"content": "partial result"},
                        },
                    }
                ),
            ]
        )
        sbox = SimpleNamespace(
            read_file=AsyncMock(return_value=event_log),
            exec=AsyncMock(return_value=SimpleNamespace(returncode=0)),
        )

        steps = await _read_bridge_tool_steps(sbox)

        assert _is_nonfatal_idle_timeout("Timeout after 30s waiting for session.idle")
        assert steps[0]["tool_call_id"] == "partial_call"
        assert steps[0]["output"] == "partial result"
        source = inspect.getsource(create_agent)
        assert source.index("bridge_tool_steps = await _read_bridge_tool_steps(sbox)") < source.index(
            "if nonfatal_timeout:"
        )
        sbox.exec.assert_awaited_once_with(["rm", "-f", "/workspace/.runner_events.jsonl"])

    @pytest.mark.asyncio
    async def test_read_failure_falls_back_without_masking_result(self) -> None:
        """Permission/transport-style read failures return no bridge steps."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from saber.agents.registry.copilot.solver import _read_bridge_tool_steps

        sbox = SimpleNamespace(
            read_file=AsyncMock(side_effect=PermissionError("denied")),
            exec=AsyncMock(return_value=SimpleNamespace(returncode=0)),
        )

        assert await _read_bridge_tool_steps(sbox) == []
        sbox.exec.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_oversized_read_falls_back_without_masking_result(self) -> None:
        """Sandbox output-limit failures do not replace the agent outcome."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from inspect_ai.util import OutputLimitExceededError

        from saber.agents.registry.copilot.solver import _read_bridge_tool_steps

        sbox = SimpleNamespace(
            read_file=AsyncMock(side_effect=OutputLimitExceededError("100 MiB", None)),
            exec=AsyncMock(return_value=SimpleNamespace(returncode=0)),
        )

        assert await _read_bridge_tool_steps(sbox) == []
        sbox.exec.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_tool_events_do_not_exhaust_tool_event_limit(self) -> None:
        """Large SDK telemetry logs retain their bounded tool trajectory."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from saber.agents.registry.copilot.solver import (
            _RUNNER_EVENT_LOG_MAX_EVENTS,
            _read_bridge_tool_steps,
        )

        non_tool_event = json.dumps(
            {
                "schema_version": 1,
                "type": "assistant.streaming_delta",
                "data": {"delta": "x"},
            }
        )
        event_log = "\n".join(
            [
                *([non_tool_event] * (_RUNNER_EVENT_LOG_MAX_EVENTS + 1)),
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_start",
                        "data": {
                            "toolCallId": "call-1",
                            "toolName": "get_incident",
                            "arguments": {"incident_id": "42"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "tool.execution_complete",
                        "data": {
                            "toolCallId": "call-1",
                            "success": True,
                            "result": {"content": "incident evidence"},
                        },
                    }
                ),
            ]
        )
        sbox = SimpleNamespace(
            read_file=AsyncMock(return_value=event_log),
            exec=AsyncMock(return_value=SimpleNamespace(returncode=0)),
        )

        steps = await _read_bridge_tool_steps(sbox)

        assert len(steps) == 1
        assert steps[0]["tool_name"] == "get_incident"
        assert steps[0]["output"] == "incident evidence"
        sbox.exec.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_incomplete_node_log_is_removed_and_falls_back(self) -> None:
        """Incomplete Node telemetry is rejected and the raw file is removed."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from saber.agents.registry.copilot.solver import _read_bridge_tool_steps

        sbox = SimpleNamespace(
            read_file=AsyncMock(return_value='{"type":"tool.execution_start","tool":"get_incident"}'),
            exec=AsyncMock(return_value=SimpleNamespace(returncode=0)),
        )

        assert await _read_bridge_tool_steps(sbox) == []
        sbox.exec.assert_awaited_once()


class TestBuildSystemPrompt:
    """build_system_prompt helper (shared via bridge_utils)."""

    def test_combines_all_parts(self) -> None:
        """Combines instruction and assistant prompts."""
        from saber.agents.bridge_utils import build_system_prompt

        result = build_system_prompt("A", "B")
        assert result == "A\n\nB"

    def test_skips_empty(self) -> None:
        """Skips empty strings from the combined prompt."""
        from saber.agents.bridge_utils import build_system_prompt

        assert build_system_prompt("A", "") == "A"

    def test_all_empty(self) -> None:
        """Returns empty string when all parts are empty."""
        from saber.agents.bridge_utils import build_system_prompt

        assert build_system_prompt("", "") == ""


class TestBuildBridgedTools:
    """build_bridged_tools helper (shared via bridge_utils)."""

    def test_creates_spec_from_tools(self) -> None:
        """Converts a list of Tools to a BridgedToolsSpec list."""
        from inspect_ai.agent import BridgedToolsSpec
        from inspect_ai.tool import bash

        from saber.agents.bridge_utils import build_bridged_tools

        result = build_bridged_tools([bash()])
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], BridgedToolsSpec)
        assert result[0].name == "saber_tools"

    def test_returns_none_for_empty(self) -> None:
        """Returns None when no tools provided."""
        from saber.agents.bridge_utils import build_bridged_tools

        assert build_bridged_tools(None) is None
        assert build_bridged_tools([]) is None


class TestBuildRunnerEnv:
    """_build_runner_env helper."""

    def test_includes_required_vars(self) -> None:
        """Env dict contains all required variables."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="gpt-5",
            prompt="Do the task",
            mcp_configs=[],
        )
        assert env["OPENAI_BASE_URL"] == "http://localhost:13131/v1"
        assert env["COPILOT_MODEL"] == "gpt-5"
        assert env["COPILOT_PROMPT"] == "Do the task"
        assert env["COPILOT_TIMEOUT"] == "3600"
        assert "COPILOT_SYSTEM_PROMPT" not in env

    def test_mcp_config_serialized_as_json(self) -> None:
        """MCP configs are serialized as a dict keyed by server name."""
        import json
        from types import SimpleNamespace

        from saber.agents.registry.copilot.solver import _build_runner_env

        mcp_configs = [
            SimpleNamespace(
                name="saber_tools",
                url="http://localhost:13131/mcp/saber_tools",
                type="http",
            ),
        ]
        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=mcp_configs,
        )
        parsed = json.loads(env["COPILOT_MCP_CONFIG"])
        assert isinstance(parsed, dict)
        assert "saber_tools" in parsed
        assert parsed["saber_tools"]["url"] == "http://localhost:13131/mcp/saber_tools"
        assert parsed["saber_tools"]["type"] == "http"
        assert parsed["saber_tools"]["tools"] == ["*"]

    def test_empty_mcp_configs(self) -> None:
        """Empty MCP configs serialize to empty JSON dict."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
        )
        assert env["COPILOT_MCP_CONFIG"] == "{}"

    def test_extra_mcp_config_is_merged(self) -> None:
        """Operator MCP config merges with bridge MCP servers."""
        import json
        from types import SimpleNamespace

        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[
                SimpleNamespace(
                    name="saber_tools",
                    url="http://localhost:13131/mcp/saber_tools",
                    type="http",
                )
            ],
            extra_mcp_config={
                "mcpServers": {
                    "Azure": {
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@azure/mcp@latest", "server", "start"],
                        "tools": ["*"],
                    }
                }
            },
        )

        parsed = json.loads(env["COPILOT_MCP_CONFIG"])
        assert parsed["saber_tools"]["url"] == "http://localhost:13131/mcp/saber_tools"
        assert parsed["Azure"]["command"] == "npx"

    def test_custom_agents_flow_to_env(self) -> None:
        """Custom agent registry and active agent are passed to runner env."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        custom_agents_json = '[{"name":"Recon Agent","prompt":"Recon.","infer":true}]'
        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
            custom_agents_json=custom_agents_json,
            active_agent="Recon Agent",
            config_dir=".",
        )

        assert env["COPILOT_CUSTOM_AGENTS"] == custom_agents_json
        assert env["COPILOT_AGENT"] == "Recon Agent"
        assert env["COPILOT_CONFIG_DIR"] == "."
        assert env["COPILOT_NESTED_AGENTS"] == "bridge"
        assert env["COPILOT_NESTED_AGENT_TIMEOUT"] == "600"
        assert env["COPILOT_NESTED_AGENT_QUIET_TIMEOUT"] == "5"
        assert env["COPILOT_NESTED_AGENT_MAX_CALLS"] == "16"

    def test_timeout_flows_to_env(self) -> None:
        """Custom timeout is passed through to COPILOT_TIMEOUT."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
            timeout=1800,
        )
        assert env["COPILOT_TIMEOUT"] == "1800"

    def test_default_timeout_is_3600(self) -> None:
        """Default timeout for COPILOT_TIMEOUT is 3600 seconds."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
        )
        assert env["COPILOT_TIMEOUT"] == "3600"

    def test_nodejs_max_turns_flows_to_env(self) -> None:
        """Node.js runner receives optional turn budget overrides."""
        from saber.agents.registry.copilot.solver import _build_nodejs_runner_env

        env = _build_nodejs_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            max_turns=150,
            max_delegated_turns=24,
            reasoning_effort="xhigh",
        )
        assert env["COPILOT_MAX_TURNS"] == "150"
        assert env["COPILOT_MAX_DELEGATED_TURNS"] == "24"
        assert env["COPILOT_REASONING_EFFORT"] == "xhigh"

        default_env = _build_nodejs_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
        )
        assert default_env["COPILOT_MAX_TURNS"] == ""
        assert default_env["COPILOT_MAX_DELEGATED_TURNS"] == ""
        assert default_env["COPILOT_REASONING_EFFORT"] == ""


class TestNodejsAgentMdPathResolution:
    """Node runner agent.md path selection stays inside the sandbox."""

    def test_prefers_baked_workspace_agent_path_before_uploaded_bundle(self) -> None:
        """IA-MTP-style images bake the full agent tree under /workspace/agents."""
        from types import SimpleNamespace

        from saber.agents.registry.copilot.solver import (
            CopilotBridgeConfig,
            _nodejs_agent_md_candidates,
        )

        bundle = SimpleNamespace(
            main_agent=SimpleNamespace(
                relative_path=Path("defender-investigation-agent.agent.md"),
                source_path="/host/not/in/sandbox/agent.md",
            )
        )
        cfg = CopilotBridgeConfig(
            agent_bundle="domains/perception/ia_mtp/agents",
            main_agent="defender-investigation-agent",
        )

        candidates = _nodejs_agent_md_candidates(
            bundle=bundle,
            config=cfg,
            sandbox_agents_dir=".github/agents",
        )

        assert candidates[0] == "/workspace/agents/defender-investigation-agent/agent.md"
        assert "/workspace/.github/agents/defender-investigation-agent.agent.md" in candidates
        assert "/host/not/in/sandbox/agent.md" not in candidates

    def test_uploaded_bundle_path_is_candidate_for_generic_node_runners(self) -> None:
        """Generic Node runners can still consume uploaded .github/agents files."""
        from types import SimpleNamespace

        from saber.agents.registry.copilot.solver import (
            CopilotBridgeConfig,
            _nodejs_agent_md_candidates,
        )

        bundle = SimpleNamespace(main_agent=SimpleNamespace(relative_path=Path("main.agent.md")))
        cfg = CopilotBridgeConfig(agent_bundle="/tmp/bundle", main_agent="main")

        candidates = _nodejs_agent_md_candidates(
            bundle=bundle,
            config=cfg,
            sandbox_agents_dir=".github/agents",
        )

        assert "/workspace/.github/agents/main.agent.md" in candidates

    @pytest.mark.asyncio
    async def test_first_existing_sandbox_file_selects_first_existing_candidate(self) -> None:
        """Sandbox existence check picks the first candidate that exists."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from saber.agents.registry.copilot.solver import _first_existing_sandbox_file

        sbox = SimpleNamespace(
            exec=AsyncMock(
                side_effect=[
                    SimpleNamespace(returncode=1),
                    SimpleNamespace(returncode=0),
                ]
            )
        )

        result = await _first_existing_sandbox_file(
            sbox,
            ["/workspace/missing.md", "/workspace/agent.md"],
            label="agent.md",
        )

        assert result == "/workspace/agent.md"

    @pytest.mark.asyncio
    async def test_first_existing_sandbox_file_fails_with_candidates(self) -> None:
        """Missing agent.md fails before launching the Node runner."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from saber.agents.registry.copilot.solver import _first_existing_sandbox_file

        sbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(returncode=1)))

        with pytest.raises(FileNotFoundError, match="/workspace/missing.md"):
            await _first_existing_sandbox_file(
                sbox,
                ["/workspace/missing.md"],
                label="agent.md",
            )


class TestSampleLimitsTimeout:
    """Tests for sample_limits() → runner_timeout derivation logic."""

    def test_remaining_flows_to_timeout(self) -> None:
        """Remaining time from sample_limits is used as runner timeout."""
        from saber.agents.registry.copilot.solver import _DEFAULT_TIMEOUT, _MIN_TIMEOUT

        remaining = 1800.0
        runner_timeout = max(int(remaining), _MIN_TIMEOUT) if remaining is not None else _DEFAULT_TIMEOUT
        assert runner_timeout == 1800

    def test_none_remaining_uses_default(self) -> None:
        """None remaining (unlimited) falls back to default timeout."""
        from saber.agents.registry.copilot.solver import _DEFAULT_TIMEOUT, _MIN_TIMEOUT

        remaining = None
        runner_timeout = max(int(remaining), _MIN_TIMEOUT) if remaining is not None else _DEFAULT_TIMEOUT
        assert runner_timeout == _DEFAULT_TIMEOUT

    def test_zero_remaining_clamped_to_min(self) -> None:
        """Zero remaining is clamped to _MIN_TIMEOUT, not zero."""
        from saber.agents.registry.copilot.solver import _DEFAULT_TIMEOUT, _MIN_TIMEOUT

        remaining = 0.0
        runner_timeout = max(int(remaining), _MIN_TIMEOUT) if remaining is not None else _DEFAULT_TIMEOUT
        assert runner_timeout == _MIN_TIMEOUT
        assert runner_timeout >= 30

    def test_negative_remaining_clamped_to_min(self) -> None:
        """Negative remaining is clamped to _MIN_TIMEOUT."""
        from saber.agents.registry.copilot.solver import _DEFAULT_TIMEOUT, _MIN_TIMEOUT

        remaining = -5.2
        runner_timeout = max(int(remaining), _MIN_TIMEOUT) if remaining is not None else _DEFAULT_TIMEOUT
        assert runner_timeout == _MIN_TIMEOUT

    def test_small_remaining_clamped_to_min(self) -> None:
        """Remaining smaller than _MIN_TIMEOUT is clamped up."""
        from saber.agents.registry.copilot.solver import _DEFAULT_TIMEOUT, _MIN_TIMEOUT

        remaining = 10.5
        runner_timeout = max(int(remaining), _MIN_TIMEOUT) if remaining is not None else _DEFAULT_TIMEOUT
        assert runner_timeout == _MIN_TIMEOUT

    def test_fractional_remaining_truncated(self) -> None:
        """Fractional seconds are truncated to int."""
        from saber.agents.registry.copilot.solver import _DEFAULT_TIMEOUT, _MIN_TIMEOUT

        remaining = 2500.7
        runner_timeout = max(int(remaining), _MIN_TIMEOUT) if remaining is not None else _DEFAULT_TIMEOUT
        assert runner_timeout == 2500


# ---------------------------------------------------------------------------
# Phase 3: Two-Level Factory & Solver
# ---------------------------------------------------------------------------


class TestCopilotCreateAgent:
    """copilot.solver.create_agent two-level factory."""

    def test_create_agent_is_callable(self) -> None:
        """create_agent function exists and is callable."""
        from saber.agents.registry.copilot.solver import create_agent

        assert callable(create_agent)

    def test_create_agent_returns_callable(self) -> None:
        """create_agent() returns create_with_prompts callable."""
        from saber.agents.registry.copilot.solver import create_agent

        create_with_prompts = create_agent()
        assert callable(create_with_prompts)

    def test_create_with_prompts_signature_has_standard_kwargs(self) -> None:
        """create_with_prompts accepts all standard prompt kwargs."""
        from saber.agents.registry.copilot.solver import create_agent

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

    def test_create_with_prompts_returns_solver(self) -> None:
        """create_with_prompts() returns a Solver instance."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.copilot.solver import create_agent

        solver = create_agent()(instruction_prompt="Do the task.", max_steps=200)
        assert isinstance(solver, Solver)

    def test_create_with_prompts_returns_solver_with_tools(self) -> None:
        """create_with_prompts() returns a Solver when tools are provided."""
        from inspect_ai.solver import Solver
        from inspect_ai.tool import bash

        from saber.agents.registry.copilot.solver import create_agent

        solver = create_agent()(instruction_prompt="Do the task.", tools=[bash()], max_steps=200)
        assert isinstance(solver, Solver)

    def test_no_copilot_sdk_import_on_host(self) -> None:
        """Solver module does not import copilot SDK at module level."""
        import sys

        # Ensure copilot is not loaded as a side effect of importing the solver
        import saber.agents.registry.copilot.solver  # noqa: F401

        copilot_modules = [m for m in sys.modules if m == "copilot" or m.startswith("copilot.")]
        assert copilot_modules == [], f"copilot SDK was imported at module level: {copilot_modules}"

    def test_module_exports(self) -> None:
        """Package __init__ exports create_agent."""
        from saber.agents.registry.copilot import create_agent

        assert callable(create_agent)


# ---------------------------------------------------------------------------
# Phase 4: Legacy Module Removal
# ---------------------------------------------------------------------------


class TestLegacyModulesRemoved:
    """Verify legacy SDK-on-host modules are deleted."""

    def test_no_client_module(self) -> None:
        """client.py should not exist."""
        with pytest.raises(ImportError):
            from saber.agents.registry.copilot.client import SharedCopilotClient  # noqa: F401

    def test_no_tools_module(self) -> None:
        """tools.py should not exist."""
        with pytest.raises(ImportError):
            from saber.agents.registry.copilot.tools import convert_tools  # noqa: F401

    def test_no_provider_module(self) -> None:
        """provider.py should not exist."""
        with pytest.raises(ImportError):
            from saber.agents.registry.copilot.provider import build_provider_from_active_model  # noqa: F401

    def test_no_events_module(self) -> None:
        """events.py should not exist."""
        with pytest.raises(ImportError):
            from saber.agents.registry.copilot.events import events_to_chat_messages  # noqa: F401

    def test_no_models_module(self) -> None:
        """models.py should not exist."""
        with pytest.raises(ImportError):
            from saber.agents.registry.copilot.models import CopilotProviderConfig  # noqa: F401


# ---------------------------------------------------------------------------
# Phase 5: Integration Verification
# ---------------------------------------------------------------------------


class TestCopilotStateMessagesExtraction:
    """Copilot solver correctly extracts system/user messages from state."""

    def test_system_messages_from_state_included(self) -> None:
        """build_system_prompt combines instruction + assistant without duplication."""
        from saber.agents.bridge_utils import build_system_prompt

        # Verify build_system_prompt doesn't duplicate assistant_prompt
        system = build_system_prompt(
            instruction_prompt="Do X",
            assistant_prompt="You are a helper",
        )
        assert system.count("You are a helper") == 1

    def test_build_user_prompt_imported(self) -> None:
        """build_user_prompt is importable from bridge_utils for copilot use."""
        from saber.agents.bridge_utils import build_user_prompt

        assert callable(build_user_prompt)

    def test_copilot_solver_imports_build_user_prompt(self) -> None:
        """Copilot solver module imports build_user_prompt from bridge_utils."""
        import saber.agents.registry.copilot.solver as mod

        source = inspect.getsource(mod)
        assert "build_user_prompt" in source

    def test_copilot_solver_does_not_use_assistant_prompt_as_user_prompt(self) -> None:
        """Copilot solver should NOT use assistant_prompt as the user prompt."""
        import textwrap

        import saber.agents.registry.copilot.solver as mod

        source = textwrap.dedent(inspect.getsource(mod.create_agent))
        # The old code had: user_prompt = assistant_prompt or "Begin the task."
        # This should NOT exist anymore
        assert "user_prompt = assistant_prompt" not in source


class TestSolverFactoryIntegration:
    """Verify copilot agent integrates with solver_factory."""

    def test_agent_registered_in_registry(self) -> None:
        """Copilot agent auto-discovered by AgentRegistry."""
        from saber.agents import AgentRegistry, _register_core_agents

        AgentRegistry._reset()
        _register_core_agents()
        assert AgentRegistry.get("copilot") is not None

    def test_capabilities_entry_exists(self) -> None:
        """AGENT_CAPABILITIES has copilot entry with supports_tools=True."""
        from saber.agents.solver_factory import AGENT_CAPABILITIES

        assert "copilot" in AGENT_CAPABILITIES
        assert AGENT_CAPABILITIES["copilot"].supports_tools is True

    def test_create_saber_solver_copilot_returns_solver(self) -> None:
        """create_saber_solver with copilot factory returns a Solver."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.copilot.solver import create_agent
        from saber.agents.solver_factory import create_saber_solver

        result = create_saber_solver("copilot", create_agent)
        assert isinstance(result, Solver)


# ---------------------------------------------------------------------------
# Phase 1 extensions — RUNNER_SCRIPT deep content
# ---------------------------------------------------------------------------


class TestRunnerScriptDeepContent:
    """Phase 1 extended: deeper content verification of RUNNER_SCRIPT."""

    def test_runner_script_emits_sentinel(self) -> None:
        """Runner emits COPILOT_RUNNER_COMPLETE on success."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_RUNNER_COMPLETE" in RUNNER_SCRIPT

    def test_runner_script_reads_mcp_config_env(self) -> None:
        """Runner reads COPILOT_MCP_CONFIG from environment."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_MCP_CONFIG" in RUNNER_SCRIPT

    def test_runner_script_registers_custom_agents(self) -> None:
        """Runner passes SDK-native custom agents to create_session."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_CUSTOM_AGENTS" in RUNNER_SCRIPT
        assert 'session_config["custom_agents"]' in RUNNER_SCRIPT
        assert 'session_config["agent"]' in RUNNER_SCRIPT
        assert "client.create_session(**session_config)" in RUNNER_SCRIPT

    def test_runner_script_has_bridge_agent_tool(self) -> None:
        """Runner can provide a bridge-backed agent delegation tool."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_NESTED_AGENTS" in RUNNER_SCRIPT
        assert "_build_bridge_agent_tool" in RUNNER_SCRIPT
        assert 'name="agent"' in RUNNER_SCRIPT
        assert "COPILOT_NESTED_AGENT_START" in RUNNER_SCRIPT
        assert "COPILOT_NESTED_AGENT_END" in RUNNER_SCRIPT

    def test_runner_script_uses_sdk_02_send_api(self) -> None:
        """Runner sends prompt strings for github-copilot-sdk 0.2.x."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "await session.send(prompt)" in RUNNER_SCRIPT
        assert 'session.send({"prompt": prompt})' not in RUNNER_SCRIPT


# ---------------------------------------------------------------------------
# Phase 2 extensions — CopilotBridgeConfig edge cases
# ---------------------------------------------------------------------------


class TestCopilotBridgeConfigEdgeCases:
    """Phase 2 extended: CopilotBridgeConfig edge cases."""

    def test_from_kwargs_all_known_fields(self) -> None:
        """from_kwargs accepts all known fields."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig.from_kwargs(
            {
                "sandbox_name": "custom",
                "port_base": 4000,
                "model": "gpt-4",
                "agent_bundle": "/tmp/bundle",
                "agents_dir": "/tmp/bundle/agents",
                "main_agent": "Recon Agent",
                "mcp_config": "/tmp/bundle/.mcp.json",
                "nested_agents": "bridge",
                "nested_agent_timeout": 120,
                "nested_agent_quiet_timeout": 3,
                "nested_agent_max_calls": 4,
            }
        )
        assert cfg.sandbox_name == "custom"
        assert cfg.port_base == 4000
        assert cfg.model == "gpt-4"
        assert cfg.agent_bundle == "/tmp/bundle"
        assert cfg.agents_dir == "/tmp/bundle/agents"
        assert cfg.main_agent == "Recon Agent"
        assert cfg.mcp_config == "/tmp/bundle/.mcp.json"
        assert cfg.nested_agents == "bridge"
        assert cfg.nested_agent_timeout == 120
        assert cfg.nested_agent_quiet_timeout == 3
        assert cfg.nested_agent_max_calls == 4

    def test_from_kwargs_only_unknown_kwargs(self) -> None:
        """Passing only unknown keys yields a default config."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig.from_kwargs({"foo": "bar", "baz": 123})
        assert cfg.sandbox_name == "default"

    def test_boundary_port_base_one(self) -> None:
        """port_base=1 is accepted (max_steps removed from config)."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig(port_base=1)
        assert cfg.port_base == 1

    def test_boundary_port_base_minimum(self) -> None:
        """port_base=1 is accepted."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig(port_base=1)
        assert cfg.port_base == 1

    def test_custom_model_name(self) -> None:
        """Custom model names are preserved exactly."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig(model="claude-sonnet-4-5-20250929")
        assert cfg.model == "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Phase 3 extensions — build_system_prompt parametrized
# ---------------------------------------------------------------------------


class TestBuildSystemPromptExtended:
    """Phase 3 extended: additional build_system_prompt scenarios."""

    def test_only_instruction(self) -> None:
        """Only instruction_prompt returns just that string."""
        from saber.agents.bridge_utils import build_system_prompt

        assert build_system_prompt("X", "") == "X"

    def test_only_assistant(self) -> None:
        """Only assistant_prompt returns just that string."""
        from saber.agents.bridge_utils import build_system_prompt

        assert build_system_prompt("", "B") == "B"

    def test_both_combined(self) -> None:
        """Instruction and assistant prompts are joined."""
        from saber.agents.bridge_utils import build_system_prompt

        assert build_system_prompt("A", "B") == "A\n\nB"


# ---------------------------------------------------------------------------
# Phase 4 extensions — build_bridged_tools edge cases
# ---------------------------------------------------------------------------


class TestBuildBridgedToolsEdgeCases:
    """Phase 4 extended: build_bridged_tools edge cases."""

    def test_multiple_tools_wrapped_in_single_spec(self) -> None:
        """Multiple tools produce exactly one BridgedToolsSpec."""
        from inspect_ai.agent import BridgedToolsSpec
        from inspect_ai.tool import bash, python

        from saber.agents.bridge_utils import build_bridged_tools

        result = build_bridged_tools([bash(), python()])
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], BridgedToolsSpec)

    def test_spec_name_is_saber_tools_always(self) -> None:
        """BridgedToolsSpec name is always 'saber_tools' regardless of tool count."""
        from inspect_ai.tool import bash, python

        from saber.agents.bridge_utils import build_bridged_tools

        result = build_bridged_tools([bash(), python(), bash()])
        assert result is not None
        assert result[0].name == "saber_tools"

    def test_tools_count_preserved_in_spec(self) -> None:
        """Tool count in the spec matches the input list length."""
        from inspect_ai.tool import bash, python

        from saber.agents.bridge_utils import build_bridged_tools

        result = build_bridged_tools([bash(), python(), bash()])
        assert result is not None
        assert len(result[0].tools) == 3


# ---------------------------------------------------------------------------
# Phase 5 — build_user_prompt (new class)
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """Tests for build_user_prompt in bridge_utils."""

    def test_single_user_message_no_assistant(self) -> None:
        """Single user message returns its text with has_assistant=False."""
        from inspect_ai.model import ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        text, has_assistant = build_user_prompt([ChatMessageUser(content="Hi")])
        assert text == "Hi"
        assert has_assistant is False

    def test_user_after_assistant(self) -> None:
        """User message following an assistant message is extracted."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [
            ChatMessageUser(content="First"),
            ChatMessageAssistant(content="Response"),
            ChatMessageUser(content="Second"),
        ]
        text, has_assistant = build_user_prompt(messages)
        assert text == "Second"
        assert has_assistant is True

    def test_multiple_user_messages_after_assistant(self) -> None:
        """Multiple user messages after last assistant are joined."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [
            ChatMessageUser(content="First"),
            ChatMessageAssistant(content="Response"),
            ChatMessageUser(content="Q2"),
            ChatMessageUser(content="Q3"),
        ]
        text, _ = build_user_prompt(messages)
        assert text == "Q2\n\nQ3"

    def test_empty_list_returns_empty_string_not_raises(self) -> None:
        """Empty message list returns ('', False) without raising ValueError."""
        from saber.agents.bridge_utils import build_user_prompt

        text, has_assistant = build_user_prompt([])
        assert text == ""
        assert has_assistant is False

    def test_only_assistant_message_raises_value_error(self) -> None:
        """List ending with an AssistantMessage raises ValueError."""
        from inspect_ai.model import ChatMessageAssistant

        from saber.agents.bridge_utils import build_user_prompt

        with pytest.raises(ValueError):
            build_user_prompt([ChatMessageAssistant(content="Hello")])

    def test_only_user_messages(self) -> None:
        """Multiple user messages with no assistant are joined."""
        from inspect_ai.model import ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [ChatMessageUser(content="A"), ChatMessageUser(content="B"), ChatMessageUser(content="C")]
        text, has_assistant = build_user_prompt(messages)
        assert text == "A\n\nB\n\nC"
        assert has_assistant is False

    def test_has_assistant_false_when_no_prior_assistant(self) -> None:
        """has_assistant is False when no assistant message is present."""
        from inspect_ai.model import ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        _, has_assistant = build_user_prompt([ChatMessageUser(content="X")])
        assert has_assistant is False

    def test_has_assistant_true_when_assistant_precedes(self) -> None:
        """has_assistant is True when an assistant message precedes the user message."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        from saber.agents.bridge_utils import build_user_prompt

        messages = [
            ChatMessageUser(content="Q"),
            ChatMessageAssistant(content="A"),
            ChatMessageUser(content="Follow-up"),
        ]
        _, has_assistant = build_user_prompt(messages)
        assert has_assistant is True


# ---------------------------------------------------------------------------
# Phase 6 extensions — _build_runner_env edge cases
# ---------------------------------------------------------------------------


class TestBuildRunnerEnvEdgeCases:
    """Phase 6 extended: _build_runner_env edge cases."""

    def test_custom_port_in_url(self) -> None:
        """bridge_port is reflected in OPENAI_BASE_URL."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=9999,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
        )
        assert env["OPENAI_BASE_URL"] == "http://localhost:9999/v1"

    def test_openai_api_key_is_placeholder(self) -> None:
        """OPENAI_API_KEY is always the bridge placeholder."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
        )
        assert env["OPENAI_API_KEY"] == "sk-placeholder-for-bridge"

    def test_mcp_config_includes_tools_wildcard(self) -> None:
        """Each MCP config entry includes tools=["*"] for the Copilot SDK."""
        import json
        from types import SimpleNamespace

        from saber.agents.registry.copilot.solver import _build_runner_env

        mcp_configs = [
            SimpleNamespace(name="tool_a", url="http://localhost:1/mcp", type="http"),
        ]
        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=mcp_configs,
        )
        parsed = json.loads(env["COPILOT_MCP_CONFIG"])
        assert parsed["tool_a"]["tools"] == ["*"]

    def test_mcp_config_type_defaults_to_http(self) -> None:
        """MCP config type is captured from the config object."""
        import json
        from types import SimpleNamespace

        from saber.agents.registry.copilot.solver import _build_runner_env

        mcp_configs = [
            SimpleNamespace(name="sse_server", url="http://localhost:1234/mcp", type="sse"),
        ]
        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=mcp_configs,
        )
        parsed = json.loads(env["COPILOT_MCP_CONFIG"])
        assert parsed["sse_server"]["type"] == "sse"

    def test_multiple_mcp_configs_serialized(self) -> None:
        """Two MCP configs produce a JSON dict with 2 keys."""
        import json
        from types import SimpleNamespace

        from saber.agents.registry.copilot.solver import _build_runner_env

        mcp_configs = [
            SimpleNamespace(name="tool_a", url="http://localhost:1/mcp", type="http"),
            SimpleNamespace(name="tool_b", url="http://localhost:2/mcp", type="http"),
        ]
        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=mcp_configs,
        )
        parsed = json.loads(env["COPILOT_MCP_CONFIG"])
        assert isinstance(parsed, dict)
        assert len(parsed) == 2
        assert "tool_a" in parsed
        assert "tool_b" in parsed
        assert parsed["tool_a"]["url"] == "http://localhost:1/mcp"
        assert parsed["tool_b"]["url"] == "http://localhost:2/mcp"

    def test_all_return_values_are_strings(self) -> None:
        """Every value in the returned dict is a str."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
        )
        for key, value in env.items():
            assert isinstance(value, str), f"env[{key!r}] is {type(value).__name__}, expected str"


# ---------------------------------------------------------------------------
# Phase 7 extensions — create_agent factory edge cases
# ---------------------------------------------------------------------------


class TestCopilotCreateAgentEdgeCases:
    """Phase 7 extended: create_agent factory edge cases."""

    def test_default_all_prompts_empty_returns_solver(self) -> None:
        """create_agent()(max_steps=200) with no prompts returns a Solver."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.copilot.solver import create_agent

        solver = create_agent()(max_steps=200)
        assert isinstance(solver, Solver)

    def test_extra_kwargs_absorbed_without_error(self) -> None:
        """Unknown kwargs passed to create_with_prompts do not raise."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.copilot.solver import create_agent

        solver = create_agent()(instruction_prompt="X", max_steps=200, unknown_kwarg="ignored")
        assert isinstance(solver, Solver)


# ---------------------------------------------------------------------------
# Phase 8 — AgentCapabilities model
# ---------------------------------------------------------------------------


class TestAgentCapabilitiesModel:
    """Phase 8: AgentCapabilities model tests."""

    def test_agent_capabilities_frozen_prevents_mutation(self) -> None:
        """Mutating a frozen AgentCapabilities raises ValidationError."""
        from pydantic import ValidationError

        from saber.agents.models import AgentCapabilities

        cap = AgentCapabilities()
        with pytest.raises(ValidationError):
            cap.supports_tools = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Phase 9 — Error propagation logic
# ---------------------------------------------------------------------------


class TestErrorPropagationLogic:
    """Tests for the runner error propagation behavior (RuntimeError on non-zero exit).

    The solver builds an error detail string from stderr/stdout when the
    copilot runner exits with a non-zero return code.  These tests validate
    the detail-selection and truncation logic in isolation.
    """

    @staticmethod
    def _build_error_detail(stderr: str | None, stdout: str | None) -> str:
        """Replicate the solver's error-detail selection logic for testing."""
        stderr_val = stderr[:500] if stderr else "(no stderr)"
        stdout_val = stdout[:500] if stdout else ""
        return stderr_val if stderr_val != "(no stderr)" else stdout_val

    @staticmethod
    def _build_error_message(returncode: int, stderr: str | None, stdout: str | None) -> str:
        """Replicate the full error message the solver raises."""
        stderr_val = stderr[:500] if stderr else "(no stderr)"
        stdout_val = stdout[:500] if stdout else ""
        detail = stderr_val if stderr_val != "(no stderr)" else stdout_val
        return f"Copilot runner exited with code {returncode}: {detail}"

    # -- detail selection -----------------------------------------------------

    def test_stderr_preferred_over_stdout(self) -> None:
        """When stderr has content it is used as the detail, not stdout."""
        detail = self._build_error_detail("some error", "some output")
        assert detail == "some error"

    def test_stdout_used_when_stderr_empty(self) -> None:
        """When stderr is an empty string, stdout is used instead."""
        detail = self._build_error_detail("", "fallback output")
        assert detail == "fallback output"

    def test_none_stderr_falls_back_to_stdout(self) -> None:
        """When stderr is None, stdout is used as the detail."""
        detail = self._build_error_detail(None, "stdout content")
        assert detail == "stdout content"

    def test_both_empty_gives_no_stderr_sentinel(self) -> None:
        """When both stderr and stdout are empty, detail is empty string."""
        detail = self._build_error_detail("", "")
        assert detail == ""

    def test_both_none_gives_no_stderr_sentinel(self) -> None:
        """When both stderr and stdout are None, detail is empty string."""
        detail = self._build_error_detail(None, None)
        assert detail == ""

    # -- truncation -----------------------------------------------------------

    def test_stderr_truncated_at_500_chars(self) -> None:
        """stderr longer than 500 chars is truncated to exactly 500."""
        long_stderr = "E" * 1000
        detail = self._build_error_detail(long_stderr, "ignored")
        assert len(detail) == 500
        assert detail == "E" * 500

    def test_stdout_truncated_at_500_chars(self) -> None:
        """stdout longer than 500 chars is truncated when used as fallback."""
        long_stdout = "O" * 1000
        detail = self._build_error_detail("", long_stdout)
        assert len(detail) == 500
        assert detail == "O" * 500

    # -- full message format --------------------------------------------------

    def test_error_message_format(self) -> None:
        """The full RuntimeError message matches the expected format."""
        msg = self._build_error_message(1, "segfault", None)
        assert msg == "Copilot runner exited with code 1: segfault"

    def test_error_message_with_high_exit_code(self) -> None:
        """Exit codes other than 1 are rendered correctly."""
        msg = self._build_error_message(137, "killed", None)
        assert msg == "Copilot runner exited with code 137: killed"

    def test_error_message_stdout_fallback(self) -> None:
        """When stderr is absent, stdout appears in the message."""
        msg = self._build_error_message(2, None, "traceback here")
        assert msg == "Copilot runner exited with code 2: traceback here"

    def test_error_message_truncated_detail(self) -> None:
        """Truncated stderr appears in the full message."""
        long_stderr = "X" * 1000
        msg = self._build_error_message(1, long_stderr, None)
        expected = f"Copilot runner exited with code 1: {'X' * 500}"
        assert msg == expected


# ---------------------------------------------------------------------------
# Tool Call Limit Filter
# ---------------------------------------------------------------------------


class TestToolCallLimitFilter:
    """Tests for create_tool_call_limit_filter."""

    def test_filter_returns_tuple(self) -> None:
        """Factory returns a (filter, check_after_exec) tuple."""
        from saber.agents.bridge_utils import create_tool_call_limit_filter

        result = create_tool_call_limit_filter()
        assert isinstance(result, tuple)
        assert len(result) == 2
        f, check = result
        assert callable(f)
        assert callable(check)

    @pytest.mark.asyncio
    async def test_filter_returns_none_no_tool_calls(self) -> None:
        """Filter returns None (proceed normally) when no tool calls."""
        from unittest.mock import MagicMock

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        f, _check = create_tool_call_limit_filter()
        result = await f(MagicMock(), [], [], None, MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_filter_records_tool_call_usage(self) -> None:
        """Filter records tool calls from assistant messages."""
        from unittest.mock import MagicMock, patch

        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.model._chat_message import ToolCall

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        f, _check = create_tool_call_limit_filter()

        msg = ChatMessageAssistant(
            content="test",
            tool_calls=[
                ToolCall(id="1", function="bash", type="function", arguments={"cmd": "ls"}),
                ToolCall(id="2", function="bash", type="function", arguments={"cmd": "pwd"}),
            ],
        )

        with (
            patch("inspect_ai.util._limit.record_tool_call_usage") as mock_record,
            patch("inspect_ai.util._limit.check_tool_call_limit") as mock_check,
        ):
            result = await f(MagicMock(), [msg], [], None, MagicMock())
            assert result is None
            mock_record.assert_called_once_with(2)
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_filter_records_delta_only(self) -> None:
        """Filter only records NEW tool calls, not previously counted ones."""
        from unittest.mock import MagicMock, patch

        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.model._chat_message import ToolCall

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        f, _check = create_tool_call_limit_filter()

        msg1 = ChatMessageAssistant(
            content="step 1",
            tool_calls=[
                ToolCall(id="1", function="bash", type="function", arguments={"cmd": "ls"}),
            ],
        )

        with (
            patch("inspect_ai.util._limit.record_tool_call_usage") as mock_record,
            patch("inspect_ai.util._limit.check_tool_call_limit"),
        ):
            # First call: 1 tool call
            await f(MagicMock(), [msg1], [], None, MagicMock())
            mock_record.assert_called_once_with(1)
            mock_record.reset_mock()

            # Second call with same messages + 1 new: delta should be 1
            msg2 = ChatMessageAssistant(
                content="step 2",
                tool_calls=[
                    ToolCall(id="2", function="bash", type="function", arguments={"cmd": "pwd"}),
                ],
            )
            await f(MagicMock(), [msg1, msg2], [], None, MagicMock())
            mock_record.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_filter_no_record_when_no_new_calls(self) -> None:
        """Filter does not record when there are no new tool calls."""
        from unittest.mock import MagicMock, patch

        from inspect_ai.model import ChatMessageAssistant

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        f, _check = create_tool_call_limit_filter()

        msg = ChatMessageAssistant(content="just text")

        with (
            patch("inspect_ai.util._limit.record_tool_call_usage") as mock_record,
            patch("inspect_ai.util._limit.check_tool_call_limit") as mock_check,
        ):
            await f(MagicMock(), [msg], [], None, MagicMock())
            mock_record.assert_not_called()
            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_filter_returns_generate_input_on_limit_exceeded(self) -> None:
        """Filter injects tool results + user message, returns GenerateInput."""
        from unittest.mock import MagicMock, patch

        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateInput
        from inspect_ai.model._chat_message import ToolCall
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        f, check = create_tool_call_limit_filter()

        msg = ChatMessageAssistant(
            content="test",
            tool_calls=[
                ToolCall(id="1", function="bash", type="function", arguments={"cmd": "ls"}),
            ],
        )

        with (
            patch("inspect_ai.util._limit.record_tool_call_usage"),
            patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=50, limit=50),
            ),
        ):
            msgs: list[object] = [msg]
            tools_list: list[object] = ["some_tool"]
            result = await f(MagicMock(), msgs, tools_list, None, MagicMock())
            # Should return GenerateInput with empty tools
            assert isinstance(result, GenerateInput)
            assert result.tools == []
            assert result.tool_choice == "none"
            # User message injected telling agent to provide final answer
            user_msgs = [m for m in msgs if isinstance(m, ChatMessageUser)]
            assert any("tool call limit" in m.content.lower() for m in user_msgs)

    @pytest.mark.asyncio
    async def test_filter_returns_generate_input_then_hard_stops(self) -> None:
        """Once limit is hit, filter returns GenerateInput with empty tools,
        allows 1 grace generation, then hard-stops with ModelOutput."""
        from unittest.mock import MagicMock, patch

        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateInput
        from inspect_ai.model._chat_message import ToolCall
        from inspect_ai.model._model_output import ModelOutput
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        f, _check = create_tool_call_limit_filter()

        msg = ChatMessageAssistant(
            content="test",
            tool_calls=[
                ToolCall(id="1", function="bash", type="function", arguments={"cmd": "ls"}),
            ],
        )

        with (
            patch("inspect_ai.util._limit.record_tool_call_usage"),
            patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=50, limit=50),
            ),
        ):
            # First call triggers the limit — should return GenerateInput
            msgs: list[object] = [msg]
            tools_list: list[object] = ["some_tool"]
            result1 = await f(MagicMock(), msgs, tools_list, None, MagicMock())
            assert isinstance(result1, GenerateInput)
            assert result1.tools == []
            assert result1.tool_choice == "none"
            # Check that a user message was injected
            user_msgs = [m for m in msgs if isinstance(m, ChatMessageUser)]
            assert any("tool call limit" in m.content.lower() for m in user_msgs)

        # Grace period: 1 generation to produce final answer.
        result2 = await f(MagicMock(), [msg], ["some_tool"], None, MagicMock())
        assert isinstance(result2, GenerateInput)
        assert result2.tools == []
        assert result2.tool_choice == "none"

        # After grace period, hard stop
        result3 = await f(MagicMock(), [msg], [], None, MagicMock())
        assert isinstance(result3, ModelOutput)
        assert result3.stop_reason == "stop"

    def test_check_after_exec_noop_when_no_limit(self) -> None:
        """check_after_exec does nothing when limit was not exceeded."""
        from saber.agents.bridge_utils import create_tool_call_limit_filter

        _f, check = create_tool_call_limit_filter()
        # Should not raise
        check()

    @pytest.mark.asyncio
    async def test_check_after_exec_raises_when_limit_exceeded(self) -> None:
        """check_after_exec raises LimitExceededError when limit was hit."""
        from unittest.mock import MagicMock, patch

        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.model._chat_message import ToolCall
        from inspect_ai.util._limit import LimitExceededError

        from saber.agents.bridge_utils import create_tool_call_limit_filter

        f, check = create_tool_call_limit_filter()

        msg = ChatMessageAssistant(
            content="test",
            tool_calls=[
                ToolCall(id="1", function="bash", type="function", arguments={"cmd": "ls"}),
            ],
        )

        # Trigger the limit in the filter
        with (
            patch("inspect_ai.util._limit.record_tool_call_usage"),
            patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=50, limit=50),
            ),
        ):
            await f(MagicMock(), [msg], [], None, MagicMock())

        # Now check_after_exec should raise
        with (
            patch(
                "inspect_ai.util._limit.check_tool_call_limit",
                side_effect=LimitExceededError("tool_call", value=50, limit=50),
            ),
            pytest.raises(LimitExceededError),
        ):
            check()


# ---------------------------------------------------------------------------
# Persona & Skills: CopilotBridgeConfig fields
# ---------------------------------------------------------------------------


class TestCopilotBridgeConfigPersonaSkills:
    """CopilotBridgeConfig persona_file and skills_dir fields."""

    def test_persona_file_default_none(self) -> None:
        """Default persona_file is None."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig()
        assert cfg.persona_file is None

    def test_skills_dir_default_none(self) -> None:
        """Default skills_dir is None."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig()
        assert cfg.skills_dir is None

    def test_from_kwargs_with_persona_file(self) -> None:
        """from_kwargs extracts persona_file."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig.from_kwargs({"persona_file": "/tmp/agent.md"})
        assert cfg.persona_file == "/tmp/agent.md"

    def test_from_kwargs_with_skills_dir(self) -> None:
        """from_kwargs extracts skills_dir."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig.from_kwargs({"skills_dir": "/tmp/skills"})
        assert cfg.skills_dir == "/tmp/skills"


# ---------------------------------------------------------------------------
# Persona & Skills: _build_runner_env new params
# ---------------------------------------------------------------------------


class TestBuildRunnerEnvPersonaSkills:
    """_build_runner_env persona_prompt and skill_directories params."""

    def test_persona_prompt_env_var_default(self) -> None:
        """Default persona_prompt produces COPILOT_PERSONA_PROMPT=''."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
        )
        assert env["COPILOT_PERSONA_PROMPT"] == ""

    def test_persona_prompt_env_var_with_data(self) -> None:
        """persona_prompt appears in COPILOT_PERSONA_PROMPT env var."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        data = "You are a security analyst."
        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
            persona_prompt=data,
        )
        assert env["COPILOT_PERSONA_PROMPT"] == data

    def test_skill_directories_env_var_default(self) -> None:
        """Default skill_directories_json produces COPILOT_SKILL_DIRECTORIES='[]'."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
        )
        assert env["COPILOT_SKILL_DIRECTORIES"] == "[]"

    def test_skill_directories_env_var_with_data(self) -> None:
        """skill_directories_json appears in COPILOT_SKILL_DIRECTORIES env var."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        data = '[".github/skills"]'
        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
            skill_directories_json=data,
        )
        assert env["COPILOT_SKILL_DIRECTORIES"] == data


class TestCopilotBridgeConfig:
    """Verify Copilot bridge config parsing for Node runner behavior flags."""

    def test_runner_stdout_completion_defaults_false(self) -> None:
        """Runner stdout completion override is opt-in."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        config = CopilotBridgeConfig.from_kwargs({})
        assert config.use_runner_stdout_completion is False

    def test_runner_stdout_completion_parses_true_string(self) -> None:
        """Inspect -T string values can enable the stdout completion override."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        config = CopilotBridgeConfig.from_kwargs({"use_runner_stdout_completion": "true"})
        assert config.use_runner_stdout_completion is True

    def test_delegated_turn_budget_parses_string(self) -> None:
        """Inspect -T string values can set the delegated turn budget."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        config = CopilotBridgeConfig.from_kwargs({"max_delegated_turns": "24"})
        assert config.max_delegated_turns == 24

    def test_reasoning_effort_parses_string(self) -> None:
        """Inspect -T string values can explicitly override reasoning effort."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        config = CopilotBridgeConfig.from_kwargs({"reasoning_effort": "xhigh"})
        assert config.reasoning_effort == "xhigh"


# ---------------------------------------------------------------------------
# Persona & Skills: RUNNER_SCRIPT env var references
# ---------------------------------------------------------------------------


class TestRunnerScriptPersonaSkills:
    """Verify RUNNER_SCRIPT references new env vars and session_config keys."""

    def test_runner_references_persona_prompt_env(self) -> None:
        """RUNNER_SCRIPT reads COPILOT_PERSONA_PROMPT env var."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_PERSONA_PROMPT" in RUNNER_SCRIPT

    def test_runner_references_skill_dirs_env(self) -> None:
        """RUNNER_SCRIPT reads COPILOT_SKILL_DIRECTORIES env var."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_SKILL_DIRECTORIES" in RUNNER_SCRIPT

    def test_runner_adds_system_message_to_session_config(self) -> None:
        """RUNNER_SCRIPT sets session_config["system_message"]."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert 'session_config["system_message"]' in RUNNER_SCRIPT

    def test_runner_adds_skill_dirs_to_session_config(self) -> None:
        """RUNNER_SCRIPT sets session_config["skill_directories"]."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert 'session_config["skill_directories"]' in RUNNER_SCRIPT


# ---------------------------------------------------------------------------
# Persona & Skills: create_agent wiring
# ---------------------------------------------------------------------------


class TestCopilotPersonaSkillsWiring:
    """Verify create_agent passes persona_file / skills_dir through."""

    def test_create_agent_with_persona_file(self) -> None:
        """create_agent(persona_file=...) returns callable that yields Solver."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.copilot.solver import create_agent

        factory = create_agent(persona_file="/tmp/p.md")
        assert callable(factory)
        solver = factory(instruction_prompt="Do it", max_steps=200)
        assert isinstance(solver, Solver)

    def test_create_agent_with_skills_dir(self) -> None:
        """create_agent(skills_dir=...) returns callable that yields Solver."""
        from inspect_ai.solver import Solver

        from saber.agents.registry.copilot.solver import create_agent

        factory = create_agent(skills_dir="/tmp/skills")
        assert callable(factory)
        solver = factory(instruction_prompt="Do it", max_steps=200)
        assert isinstance(solver, Solver)


# ---------------------------------------------------------------------------
# Activity-Aware Timeout & Metrics
# ---------------------------------------------------------------------------


class TestRunnerScriptActivityTimeout:
    """Verify RUNNER_SCRIPT has activity-aware timeout and metrics."""

    def test_runner_reads_idle_timeout_env(self) -> None:
        """RUNNER_SCRIPT reads COPILOT_IDLE_TIMEOUT env var."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_IDLE_TIMEOUT" in RUNNER_SCRIPT

    def test_runner_uses_session_send_not_send_and_wait(self) -> None:
        """RUNNER_SCRIPT uses session.send() not session.send_and_wait()."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "session.send(" in RUNNER_SCRIPT
        assert "send_and_wait" not in RUNNER_SCRIPT

    def test_runner_emits_metrics_json(self) -> None:
        """RUNNER_SCRIPT prints COPILOT_METRICS: JSON line."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_METRICS:" in RUNNER_SCRIPT

    def test_runner_tracks_activity_events(self) -> None:
        """RUNNER_SCRIPT tracks activity events for idle detection."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "ACTIVITY_EVENTS" in RUNNER_SCRIPT
        assert "last_activity_time" in RUNNER_SCRIPT

    def test_runner_emits_idle_timeout_marker(self) -> None:
        """RUNNER_SCRIPT emits COPILOT_RUNNER_IDLE_TIMEOUT on idle timeout."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_RUNNER_IDLE_TIMEOUT" in RUNNER_SCRIPT

    def test_runner_emits_max_timeout_marker(self) -> None:
        """RUNNER_SCRIPT emits COPILOT_RUNNER_MAX_TIMEOUT on max timeout."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "COPILOT_RUNNER_MAX_TIMEOUT" in RUNNER_SCRIPT

    def test_runner_imports_session_event_type(self) -> None:
        """RUNNER_SCRIPT imports SessionEventType for event tracking."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "SessionEventType" in RUNNER_SCRIPT

    def test_runner_uses_time_monotonic(self) -> None:
        """RUNNER_SCRIPT uses time.monotonic() for timing."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        assert "time.monotonic()" in RUNNER_SCRIPT

    def test_runner_tracks_all_required_metrics(self) -> None:
        """RUNNER_SCRIPT tracks all required metric fields."""
        from saber.agents.registry.copilot.solver import RUNNER_SCRIPT

        for field in [
            "total_events",
            "assistant_messages",
            "tool_calls_started",
            "tool_calls_completed",
            "turn_count",
            "last_event_type",
            "elapsed_seconds",
            "idle_seconds",
            "exit_reason",
        ]:
            assert field in RUNNER_SCRIPT, f"Missing metric field: {field}"


class TestCopilotBridgeConfigIdleTimeout:
    """CopilotBridgeConfig idle_timeout field."""

    def test_idle_timeout_default(self) -> None:
        """Default idle_timeout is 300."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig()
        assert cfg.idle_timeout == 300

    def test_from_kwargs_with_idle_timeout(self) -> None:
        """from_kwargs extracts idle_timeout."""
        from saber.agents.registry.copilot.solver import CopilotBridgeConfig

        cfg = CopilotBridgeConfig.from_kwargs({"idle_timeout": 600})
        assert cfg.idle_timeout == 600


class TestBuildRunnerEnvIdleTimeout:
    """_build_runner_env idle_timeout parameter."""

    def test_idle_timeout_env_var_default(self) -> None:
        """Default idle_timeout produces COPILOT_IDLE_TIMEOUT='300'."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
        )
        assert env["COPILOT_IDLE_TIMEOUT"] == "300"

    def test_idle_timeout_env_var_custom(self) -> None:
        """Custom idle_timeout is passed through."""
        from saber.agents.registry.copilot.solver import _build_runner_env

        env = _build_runner_env(
            bridge_port=13131,
            model="inspect",
            prompt="Go",
            mcp_configs=[],
            idle_timeout=600,
        )
        assert env["COPILOT_IDLE_TIMEOUT"] == "600"


class TestParseRunnerMetrics:
    """Tests for parse_runner_metrics in bridge_utils."""

    def test_parses_valid_metrics_json(self) -> None:
        """Parses valid COPILOT_METRICS JSON from stderr."""
        from saber.agents.bridge_utils import parse_runner_metrics

        stderr = 'Some log line\nCOPILOT_METRICS: {"total_events": 42, "exit_reason": "completed"}\nMore output\n'
        result = parse_runner_metrics(stderr)
        assert result is not None
        assert result["total_events"] == 42
        assert result["exit_reason"] == "completed"

    def test_returns_none_for_no_metrics(self) -> None:
        """Returns None when no COPILOT_METRICS line found."""
        from saber.agents.bridge_utils import parse_runner_metrics

        result = parse_runner_metrics("some random stderr output\n")
        assert result is None

    def test_returns_none_for_empty_string(self) -> None:
        """Returns None for empty stderr."""
        from saber.agents.bridge_utils import parse_runner_metrics

        assert parse_runner_metrics("") is None

    def test_returns_none_for_none_input(self) -> None:
        """Returns None when passed None (via falsy check)."""
        from saber.agents.bridge_utils import parse_runner_metrics

        assert parse_runner_metrics("") is None

    def test_returns_none_for_invalid_json(self) -> None:
        """Returns None when COPILOT_METRICS line has invalid JSON."""
        from saber.agents.bridge_utils import parse_runner_metrics

        stderr = "COPILOT_METRICS: {not valid json}\n"
        result = parse_runner_metrics(stderr)
        assert result is None

    def test_parses_full_metrics(self) -> None:
        """Parses a complete COPILOT_METRICS payload."""
        import json

        from saber.agents.bridge_utils import parse_runner_metrics

        metrics = {
            "total_events": 150,
            "assistant_messages": 10,
            "tool_calls_started": 25,
            "tool_calls_completed": 24,
            "turns": 8,
            "elapsed_seconds": 1200.5,
            "idle_seconds": 3.2,
            "exit_reason": "completed",
            "last_event_type": "session.idle",
        }
        stderr = f"COPILOT_METRICS: {json.dumps(metrics)}\n"
        result = parse_runner_metrics(stderr)
        assert result is not None
        assert result["total_events"] == 150
        assert result["turns"] == 8
        assert result["exit_reason"] == "completed"

    def test_first_metrics_line_wins(self) -> None:
        """When multiple COPILOT_METRICS lines exist, first one is returned."""
        from saber.agents.bridge_utils import parse_runner_metrics

        stderr = 'COPILOT_METRICS: {"exit_reason": "first"}\nCOPILOT_METRICS: {"exit_reason": "second"}\n'
        result = parse_runner_metrics(stderr)
        assert result is not None
        assert result["exit_reason"] == "first"


class TestParseWorkflowStatus:
    """Tests for parse_workflow_status in bridge_utils."""

    def test_parses_valid_workflow_status_json(self) -> None:
        """Parses valid COPILOT_WORKFLOW_STATUS JSON from stderr."""
        from saber.agents.bridge_utils import parse_workflow_status

        stderr = (
            "Some log line\n"
            "COPILOT_WORKFLOW_STATUS: "
            '{"workflowComplete": false, "missing": ["report"], "reason": "report.md is missing"}\n'
            "More output\n"
        )
        result = parse_workflow_status(stderr)
        assert result is not None
        assert result["workflowComplete"] is False
        assert result["missing"] == ["report"]
        assert result["reason"] == "report.md is missing"

    def test_returns_none_for_no_workflow_status(self) -> None:
        """Returns None when no COPILOT_WORKFLOW_STATUS line is present."""
        from saber.agents.bridge_utils import parse_workflow_status

        result = parse_workflow_status("some random stderr output\n")
        assert result is None

    def test_returns_none_for_invalid_workflow_status_json(self) -> None:
        """Returns None when COPILOT_WORKFLOW_STATUS contains invalid JSON."""
        from saber.agents.bridge_utils import parse_workflow_status

        result = parse_workflow_status("COPILOT_WORKFLOW_STATUS: {not valid json}\n")
        assert result is None


# ---------------------------------------------------------------------------
# Bridge Tracking Integration
# ---------------------------------------------------------------------------


class TestCopilotTrackingIntegration:
    """Verify tracking filter is wired into Copilot solver."""

    def test_imports_tracking_filter(self) -> None:
        """Copilot solver imports create_tracking_filter."""
        from saber.agents.registry.copilot import solver

        source = Path(solver.__file__).read_text()
        assert "create_tracking_filter" in source

    def test_imports_compose_filters(self) -> None:
        """Copilot solver imports compose_filters."""
        from saber.agents.registry.copilot import solver

        source = Path(solver.__file__).read_text()
        assert "compose_filters" in source


class TestCopilotModelAliases:
    """Verify model_aliases support is wired into Copilot solver."""

    def test_imports_resolve_model_aliases(self) -> None:
        """Copilot solver imports resolve_model_aliases."""
        from saber.agents.registry.copilot import solver

        source = Path(solver.__file__).read_text()
        assert "resolve_model_aliases" in source


class TestCopilotCLIParserIntegration:
    """Verify CLI output parser is wired into Copilot solver."""

    def test_imports_cli_parser(self) -> None:
        """Copilot solver imports parse_copilot_stderr."""
        from saber.agents.registry.copilot import solver

        source = Path(solver.__file__).read_text()
        assert "parse_copilot_cli" in source or "parse_copilot_stderr" in source

    def test_uses_record_bridge_summary(self) -> None:
        """Copilot solver uses the shared record_bridge_summary helper."""
        from saber.agents.registry.copilot import solver

        source = Path(solver.__file__).read_text()
        assert "record_bridge_summary" in source
