"""Domain setup hooks — pre-task setup actions.

Provides the ``SetupHook`` protocol and ``run_setup_hooks()`` executor for
domain-specific pre-evaluation setup (data downloads, validation, etc.).

Hooks run synchronously in ``create_task()`` Phase 0, before config loading.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from saber.logging import get_logger

logger = get_logger(__name__)


class SetupHookStatus(StrEnum):
    """Outcome of a setup hook execution."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class SetupHookResult(BaseModel):
    """Immutable result of a single setup hook execution."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: SetupHookStatus
    duration_seconds: float = 0.0
    message: str = ""


class SetupHooksResult(BaseModel):
    """Immutable aggregate result of all setup hooks."""

    model_config = ConfigDict(frozen=True)

    results: tuple[SetupHookResult, ...] = ()

    @property
    def all_succeeded(self) -> bool:
        """True if no hook failed."""
        return all(r.status != SetupHookStatus.FAILED for r in self.results)

    @property
    def failed_hooks(self) -> list[SetupHookResult]:
        """List of hooks that failed."""
        return [r for r in self.results if r.status == SetupHookStatus.FAILED]

    @property
    def completed_count(self) -> int:
        """Count of hooks that completed successfully."""
        return sum(1 for r in self.results if r.status == SetupHookStatus.COMPLETED)

    @property
    def skipped_count(self) -> int:
        """Count of hooks that were skipped."""
        return sum(1 for r in self.results if r.status == SetupHookStatus.SKIPPED)


@runtime_checkable
class SetupHook(Protocol):
    """Protocol for domain setup hooks.

    Each hook has a name, a guard to check if it needs to run, and the
    actual setup logic. Hooks are synchronous — they run before the
    Task object is constructed.
    """

    @property
    def name(self) -> str:
        """Human-readable hook name for logging."""
        ...

    def should_run(self, domain_root: Path) -> bool:
        """Return True if this hook needs to execute.

        Called before ``run()``. Return False to skip (e.g., data already
        downloaded). This method must be fast and side-effect-free.
        """
        ...

    def run(self, domain_root: Path) -> None:
        """Execute the setup action.

        Raise any exception to signal failure. The exception message
        will be captured in ``SetupHookResult.message``.
        """
        ...


class SimpleSetupHook:
    """Convenience implementation for straightforward setup hooks.

    Use when a simple callable + guard is sufficient. For complex hooks,
    implement the ``SetupHook`` protocol directly.

    Args:
        name: Human-readable hook name.
        run_fn: Callable that performs the setup. Receives ``domain_root``.
        guard: Optional callable that returns True if the hook should run.
            Receives ``domain_root``. Defaults to always-run.
    """

    def __init__(
        self,
        name: str,
        run_fn: Callable[[Path], None],
        guard: Callable[[Path], bool] | None = None,
    ) -> None:
        self._name = name
        self._run_fn = run_fn
        self._guard = guard

    @property
    def name(self) -> str:
        """Human-readable hook name."""
        return self._name

    def should_run(self, domain_root: Path) -> bool:
        """Check guard, defaulting to True if no guard was provided."""
        if self._guard is None:
            return True
        return self._guard(domain_root)

    def run(self, domain_root: Path) -> None:
        """Delegate to the run_fn callable."""
        self._run_fn(domain_root)


def run_setup_hooks(
    hooks: list[SetupHook],
    domain_root: Path,
) -> SetupHooksResult:
    """Execute setup hooks sequentially, collecting results.

    Each hook's ``should_run()`` is checked first. If False, the hook
    is skipped. If True, ``run()`` is called. Exceptions from ``run()``
    produce a FAILED result. The function does NOT short-circuit on
    failure — all hooks run, and the caller decides whether to abort.

    Args:
        hooks: Ordered list of setup hooks to execute.
        domain_root: Domain root directory passed to each hook.

    Returns:
        Aggregate result with per-hook outcomes.
    """
    results: list[SetupHookResult] = []

    for hook in hooks:
        try:
            should_run = hook.should_run(domain_root)
        except Exception as exc:
            logger.error("Setup hook '%s': should_run() failed — %s", hook.name, exc)
            results.append(
                SetupHookResult(
                    name=hook.name,
                    status=SetupHookStatus.FAILED,
                    message=f"should_run() raised: {exc}",
                )
            )
            continue

        if not should_run:
            logger.info("Setup hook '%s': skipped (already done)", hook.name)
            results.append(
                SetupHookResult(
                    name=hook.name,
                    status=SetupHookStatus.SKIPPED,
                )
            )
            continue

        logger.info("Setup hook '%s': running...", hook.name)
        start = time.monotonic()
        try:
            hook.run(domain_root)
            elapsed = time.monotonic() - start
            logger.info("Setup hook '%s': completed in %.1fs", hook.name, elapsed)
            results.append(
                SetupHookResult(
                    name=hook.name,
                    status=SetupHookStatus.COMPLETED,
                    duration_seconds=elapsed,
                )
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("Setup hook '%s': failed after %.1fs — %s", hook.name, elapsed, exc)
            results.append(
                SetupHookResult(
                    name=hook.name,
                    status=SetupHookStatus.FAILED,
                    duration_seconds=elapsed,
                    message=str(exc),
                )
            )

    return SetupHooksResult(results=tuple(results))
