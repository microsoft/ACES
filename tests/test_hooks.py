# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for saber.hooks — setup hook protocol and executor."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from saber.hooks import (
    SetupHookResult,
    SetupHooksResult,
    SetupHookStatus,
    SimpleSetupHook,
    run_setup_hooks,
)

# ── Result model tests ──────────────────────────────────────────────


class TestSetupHookResult:
    def test_immutable(self) -> None:
        r = SetupHookResult(name="test", status=SetupHookStatus.COMPLETED)
        with pytest.raises(ValidationError):
            r.name = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        r = SetupHookResult(name="test", status=SetupHookStatus.SKIPPED)
        assert r.duration_seconds == 0.0
        assert r.message == ""


class TestSetupHooksResult:
    def test_all_succeeded_empty(self) -> None:
        assert SetupHooksResult().all_succeeded is True

    def test_all_succeeded_with_skipped(self) -> None:
        r = SetupHooksResult(
            results=(
                SetupHookResult(name="a", status=SetupHookStatus.SKIPPED),
                SetupHookResult(name="b", status=SetupHookStatus.COMPLETED),
            )
        )
        assert r.all_succeeded is True

    def test_all_succeeded_with_failure(self) -> None:
        r = SetupHooksResult(
            results=(
                SetupHookResult(name="a", status=SetupHookStatus.COMPLETED),
                SetupHookResult(name="b", status=SetupHookStatus.FAILED, message="boom"),
            )
        )
        assert r.all_succeeded is False
        assert len(r.failed_hooks) == 1
        assert r.failed_hooks[0].name == "b"

    def test_counts(self) -> None:
        r = SetupHooksResult(
            results=(
                SetupHookResult(name="a", status=SetupHookStatus.COMPLETED),
                SetupHookResult(name="b", status=SetupHookStatus.SKIPPED),
                SetupHookResult(name="c", status=SetupHookStatus.FAILED),
            )
        )
        assert r.completed_count == 1
        assert r.skipped_count == 1


# ── SimpleSetupHook tests ───────────────────────────────────────────


class TestSimpleSetupHook:
    def test_name(self) -> None:
        h = SimpleSetupHook(name="test", run_fn=lambda _: None)
        assert h.name == "test"

    def test_should_run_default_true(self, tmp_path: Path) -> None:
        h = SimpleSetupHook(name="test", run_fn=lambda _: None)
        assert h.should_run(tmp_path) is True

    def test_should_run_guard_false(self, tmp_path: Path) -> None:
        h = SimpleSetupHook(name="test", run_fn=lambda _: None, guard=lambda _: False)
        assert h.should_run(tmp_path) is False

    def test_run_delegates(self, tmp_path: Path) -> None:
        marker = tmp_path / "ran"
        h = SimpleSetupHook(name="test", run_fn=lambda root: marker.write_text("yes"))
        h.run(tmp_path)
        assert marker.read_text() == "yes"

    def test_satisfies_protocol(self) -> None:
        from saber.hooks import SetupHook

        h = SimpleSetupHook(name="test", run_fn=lambda _: None)
        assert isinstance(h, SetupHook)


# ── run_setup_hooks tests ───────────────────────────────────────────


class TestRunSetupHooks:
    def test_empty_hooks(self, tmp_path: Path) -> None:
        result = run_setup_hooks([], tmp_path)
        assert result.all_succeeded is True
        assert len(result.results) == 0

    def test_hook_runs_and_completes(self, tmp_path: Path) -> None:
        marker = tmp_path / "ran"
        hook = SimpleSetupHook(
            name="writer",
            run_fn=lambda root: (root / "ran").write_text("done"),
        )
        result = run_setup_hooks([hook], tmp_path)
        assert result.all_succeeded is True
        assert result.completed_count == 1
        assert marker.read_text() == "done"

    def test_hook_skipped_when_guard_false(self, tmp_path: Path) -> None:
        hook = SimpleSetupHook(
            name="skippable",
            run_fn=lambda _: None,
            guard=lambda _: False,
        )
        result = run_setup_hooks([hook], tmp_path)
        assert result.all_succeeded is True
        assert result.skipped_count == 1
        assert result.completed_count == 0

    def test_hook_failure_captured(self, tmp_path: Path) -> None:
        def failing_fn(root: Path) -> None:
            raise RuntimeError("download failed")

        hook = SimpleSetupHook(name="failing", run_fn=failing_fn)
        result = run_setup_hooks([hook], tmp_path)
        assert result.all_succeeded is False
        assert result.results[0].status == SetupHookStatus.FAILED
        assert "download failed" in result.results[0].message

    def test_all_hooks_run_even_after_failure(self, tmp_path: Path) -> None:
        """No short-circuit: all hooks run, caller decides on abort."""
        marker = tmp_path / "second_ran"

        def fail_fn(root: Path) -> None:
            raise RuntimeError("oops")

        hooks = [
            SimpleSetupHook(name="fail_first", run_fn=fail_fn),
            SimpleSetupHook(
                name="runs_second",
                run_fn=lambda root: (root / "second_ran").write_text("yes"),
            ),
        ]
        result = run_setup_hooks(hooks, tmp_path)
        assert not result.all_succeeded
        assert marker.read_text() == "yes"
        assert result.completed_count == 1

    def test_hooks_run_in_order(self, tmp_path: Path) -> None:
        log: list[str] = []
        hooks = [
            SimpleSetupHook(name="first", run_fn=lambda _: log.append("first")),
            SimpleSetupHook(name="second", run_fn=lambda _: log.append("second")),
        ]
        run_setup_hooks(hooks, tmp_path)
        assert log == ["first", "second"]

    def test_duration_is_positive(self, tmp_path: Path) -> None:
        import time

        hook = SimpleSetupHook(
            name="slow",
            run_fn=lambda _: time.sleep(0.05),
        )
        result = run_setup_hooks([hook], tmp_path)
        assert result.results[0].duration_seconds >= 0.04

    def test_should_run_exception_treated_as_failed(self, tmp_path: Path) -> None:
        """If should_run() raises, the hook is treated as FAILED."""

        def bad_guard(root: Path) -> bool:
            raise ValueError("guard exploded")

        hook = SimpleSetupHook(name="bad_guard", run_fn=lambda _: None, guard=bad_guard)
        result = run_setup_hooks([hook], tmp_path)
        assert not result.all_succeeded
        assert result.results[0].status == SetupHookStatus.FAILED
        assert "guard exploded" in result.results[0].message
