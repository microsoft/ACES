"""Thin sandbox subclass adding permanent environment lifecycle.

All per-sample behavior (exec, read_file, write_file, sample_init,
sample_cleanup) is inherited from DockerSandboxEnvironment unchanged.
This subclass adds only task-level start/stop of shared services and
``SABER_PROJECT`` env-var injection so that per-sample compose
files can reference the permanent network/project.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment
from inspect_ai.util._sandbox.environment import (
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
)
from inspect_ai.util._sandbox.registry import sandboxenv
from typing_extensions import override

from saber.environments.images import PreflightBuildError, RebuildMode, build_domain_images
from saber.logging import display_progress, get_logger

__all__ = ["SaberSandboxEnvironment"]

_MONITOR_POLL_INTERVAL: float = 15.0
_STALE_CONTAINER_MIN_AGE_SECONDS: int = 180  # 3 minutes
_CONTAINER_CONFLICT_PATTERN: re.Pattern[str] = re.compile(
    r'container "([0-9a-f]+)"\. You have to remove',
)
_MAX_SAMPLE_INIT_RETRIES: int = 2

logger = get_logger(__name__)


@sandboxenv(name="saber")
class SaberSandboxEnvironment(DockerSandboxEnvironment):
    """Docker sandbox + permanent environment lifecycle.

    Manages task-level start/stop of shared services (databases, caches)
    that persist across samples.

    Usage in @task::

        SaberSandboxEnvironment.set_permanent_compose(
            domain_root / "compose" / "databases.compose.yml",
            project="excytin-databases",
        )
        await SaberSandboxEnvironment.task_init("my_task", None)
        # ... run evaluation ...
        await SaberSandboxEnvironment.task_cleanup("my_task", None, True)
    """

    _permanent_compose: ClassVar[Path | None] = None
    _permanent_project: ClassVar[str] = "saber-permanent"
    _permanent_domain_root: ClassVar[Path | None] = None
    _keep_permanent: ClassVar[bool] = False
    _preflight_domain_root: ClassVar[Path | None] = None
    _preflight_rebuild: ClassVar[RebuildMode | None] = None
    _preflight_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _preflight_done: ClassVar[bool] = False
    _preflight_error: ClassVar[RuntimeError | None] = None

    @classmethod
    def set_permanent_compose(
        cls,
        path: Path,
        project: str = "saber-permanent",
        domain_root: Path | None = None,
    ) -> None:
        """Configure the permanent services compose file.

        Args:
            path: Path to a Docker Compose file for permanent services.
            project: Docker Compose project name.
            domain_root: Explicit domain root directory for volume paths.
                Falls back to ``path.parent.parent`` if not provided.
        """
        cls._permanent_compose = path
        cls._permanent_project = project
        cls._permanent_domain_root = domain_root

    @classmethod
    def set_keep_permanent(cls, keep: bool) -> None:
        """Configure whether permanent services should survive task cleanup.

        When True, ``task_cleanup()`` will skip stopping permanent services
        and removing volumes, allowing the next run to reuse them.

        Args:
            keep: If True, permanent services are not stopped on cleanup.
        """
        cls._keep_permanent = keep

    @classmethod
    def set_preflight_config(cls, domain_root: Path, rebuild: RebuildMode | None = None) -> None:
        """Configure preflight image building for the next task_init cycle.

        Args:
            domain_root: Domain directory path.
            rebuild: Which images to rebuild, or ``None`` to skip.
        """
        cls._preflight_domain_root = domain_root
        cls._preflight_rebuild = rebuild
        cls._preflight_done = False

    @classmethod
    async def task_init(cls, task_name: str, config: SandboxEnvironmentConfigType | None) -> None:
        """Start permanent services before any samples run.

        Args:
            task_name: Name of the task being evaluated.
            config: Sandbox configuration forwarded to the parent class.
        """
        # Remove stale containers from previous failed runs
        try:
            await _cleanup_stale_containers(task_name)
        except Exception:
            logger.warning("Stale container cleanup failed, continuing with task_init")

        # Preflight image build (lock ensures only one concurrent run)
        async with cls._preflight_lock:
            if cls._preflight_error is not None:
                raise cls._preflight_error
            if cls._preflight_domain_root and not cls._preflight_done:
                display_progress("Building domain Docker images (preflight)...")
                result = await build_domain_images(
                    cls._preflight_domain_root,
                    rebuild=cls._preflight_rebuild,
                )
                cls._preflight_done = True
                if not result.all_succeeded:
                    failed = [r for r in result.results if r.action == "failed"]
                    cls._preflight_error = PreflightBuildError(failed)
                    raise cls._preflight_error
                display_progress("Docker image preflight complete.")

        if cls._permanent_compose and cls._permanent_compose.exists():
            domain_root = cls._permanent_domain_root or cls._permanent_compose.parent.parent
            await _start_permanent_services(
                cls._permanent_compose,
                cls._permanent_project,
                project_directory=domain_root,
            )
        if cls._permanent_compose and cls._permanent_compose.exists() and cls._permanent_project:
            os.environ["SABER_PROJECT"] = cls._permanent_project
        try:
            await super().task_init(task_name, config)
        except BaseException:
            os.environ.pop("SABER_PROJECT", None)
            if cls._permanent_compose:
                try:
                    await _stop_permanent_services(cls._permanent_project)
                except Exception:
                    logger.warning("Failed to stop permanent services during init recovery")
            raise

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        """Start per-sample sandbox with conflict-aware retry.

        Wraps the parent ``sample_init`` to handle Docker container name
        collisions caused by stale containers from previous crashed runs.
        When a "Conflict" error is detected, the offending container is
        force-removed and the operation is retried.

        Args:
            task_name: Name of the task being evaluated.
            config: Sandbox configuration forwarded to the parent class.
            metadata: Sample metadata dict.

        Returns:
            Mapping of service name to :class:`SandboxEnvironment`.
        """
        last_error: BaseException | None = None
        for attempt in range(_MAX_SAMPLE_INIT_RETRIES + 1):
            try:
                return await super().sample_init(task_name, config, metadata)
            except RuntimeError as exc:
                err_msg = str(exc)
                if "already in use" not in err_msg:
                    raise
                last_error = exc
                removed = await _remove_conflicting_containers(err_msg)
                if not removed:
                    raise
                logger.warning(
                    "Removed conflicting container(s) on attempt %d/%d, retrying sample_init",
                    attempt + 1,
                    _MAX_SAMPLE_INIT_RETRIES + 1,
                )
        # Should not reach here, but satisfy the type checker
        raise last_error  # type: ignore[misc]

    @classmethod
    async def task_cleanup(cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool) -> None:
        """Stop permanent services after all samples complete.

        Args:
            task_name: Name of the task being evaluated.
            config: Sandbox configuration forwarded to the parent class.
            cleanup: Whether to actually tear down resources.
        """
        try:
            await super().task_cleanup(task_name, config, cleanup)
        finally:
            try:
                if cleanup and cls._permanent_compose and not cls._keep_permanent:
                    await _stop_permanent_services(cls._permanent_project)
                elif cls._keep_permanent and cls._permanent_compose:
                    display_progress(
                        f"Keeping permanent services alive (project={cls._permanent_project})"
                    )
            except Exception:
                logger.warning("Failed to stop permanent services during cleanup")
            finally:
                os.environ.pop("SABER_PROJECT", None)

    @classmethod
    def _reset(cls) -> None:
        """Reset class state. For testing only."""
        cls._permanent_compose = None
        cls._permanent_project = "saber-permanent"
        cls._permanent_domain_root = None
        cls._keep_permanent = False
        cls._preflight_domain_root = None
        cls._preflight_rebuild = None
        cls._preflight_done = False
        cls._preflight_error = None
        cls._preflight_lock = asyncio.Lock()
        os.environ.pop("SABER_PROJECT", None)


def _sanitize_task_name(task_name: str) -> str:
    """Sanitize a task name the same way inspect_ai's ``task_project_name`` does.

    Args:
        task_name: Raw task name string.

    Returns:
        Sanitized name suitable for Docker project name matching.
    """
    sanitized = task_name.lower()
    sanitized = re.sub(r"[^a-z\d\-_]", "-", sanitized)
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized[:12].rstrip("_")


def _parse_docker_timestamp(timestamp: str) -> datetime:
    """Parse a Docker ``CreatedAt`` timestamp into a timezone-aware datetime.

    Docker outputs timestamps like ``2026-03-06 05:58:30 +0000 UTC``.
    The trailing ``UTC`` is stripped before parsing.

    Args:
        timestamp: Raw timestamp string from ``docker ps --format``.

    Returns:
        Parsed UTC datetime.
    """
    cleaned = timestamp.strip().removesuffix(" UTC").strip()
    return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S %z")


async def _cleanup_stale_containers(
    task_name: str,
    min_age_seconds: int = _STALE_CONTAINER_MIN_AGE_SECONDS,
) -> None:
    """Remove stale Docker containers left by killed compose-up processes.

    Finds containers in "Created" or "Dead" state whose names match the
    inspect_ai naming pattern for the given task, filters to only those
    older than ``min_age_seconds``, and force-removes them.  This avoids
    accidentally killing containers that are legitimately starting up
    from concurrent evaluation runs.

    Args:
        task_name: The task name used to derive the container name filter.
        min_age_seconds: Minimum container age in seconds before it is
            considered stale.  Defaults to ``_STALE_CONTAINER_MIN_AGE_SECONDS``
            (180 s / 3 minutes).

    Note:
        Errors are caught and logged so cleanup never blocks evaluation.
    """
    try:
        name_filter = f"inspect-{_sanitize_task_name(task_name)}-"

        proc = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={name_filter}",
            "--filter",
            "status=created",
            "--filter",
            "status=dead",
            "--format",
            "{{.ID}} {{.CreatedAt}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await proc.communicate()
        lines = [line for line in stdout.decode().strip().splitlines() if line.strip()]

        if not lines:
            return

        now = datetime.now(tz=UTC)
        stale_ids: list[str] = []
        for line in lines:
            # Format: "<id> <timestamp...>"  — split on first space only
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            container_id, created_at_str = parts
            try:
                created_at = _parse_docker_timestamp(created_at_str)
                age = (now - created_at).total_seconds()
                if age >= min_age_seconds:
                    stale_ids.append(container_id)
            except (ValueError, TypeError):
                # Cannot parse timestamp — treat as stale to be safe
                stale_ids.append(container_id)

        if not stale_ids:
            return

        logger.warning(
            "Found %d stale container(s) matching '%s' (older than %ds), removing: %s",
            len(stale_ids),
            name_filter,
            min_age_seconds,
            ", ".join(stale_ids),
        )

        rm_proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            *stale_ids,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await rm_proc.communicate()
    except Exception:
        logger.warning("Failed to cleanup stale containers for task '%s'", task_name, exc_info=True)


async def _remove_conflicting_containers(
    error_message: str,
    min_age_seconds: int = _STALE_CONTAINER_MIN_AGE_SECONDS,
) -> bool:
    """Extract and force-remove conflicting container(s) from a Docker error.

    Parses the Docker daemon "Conflict" error message to find the
    container ID(s) blocking creation.  Before removing, each container
    is inspected to verify it is genuinely stale — i.e. in a non-running
    state (``created`` or ``dead``) **and** older than
    ``min_age_seconds``.  Containers that appear healthy or too young
    are left untouched.

    Args:
        error_message: The full error message from a failed ``compose up``.
        min_age_seconds: Minimum container age in seconds before it is
            considered safe to remove.

    Returns:
        ``True`` if at least one stale container was found and removal
        was attempted; ``False`` if no container IDs could be parsed or
        none qualified as stale.
    """
    container_ids = _CONTAINER_CONFLICT_PATTERN.findall(error_message)
    if not container_ids:
        return False

    removable: list[str] = []
    now = datetime.now(tz=UTC)

    for cid in container_ids:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}} {{.Created}}",
                cid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                # Container may already be gone — safe to skip
                continue
            output = stdout.decode().strip()
            parts = output.split(" ", 1)
            if len(parts) != 2:
                continue
            status, created_str = parts

            # Only remove non-running containers
            if status not in ("created", "dead", "exited"):
                logger.info(
                    "Conflicting container %s is in '%s' state — skipping removal",
                    cid[:12],
                    status,
                )
                continue

            # Verify age threshold
            try:
                # Docker inspect uses ISO-8601: 2026-03-06T05:58:30.123456789Z
                created_at = datetime.fromisoformat(
                    created_str.replace("Z", "+00:00").split(".")[0] + "+00:00"
                )
                age = (now - created_at).total_seconds()
                if age < min_age_seconds:
                    logger.info(
                        "Conflicting container %s is only %ds old (threshold %ds) — skipping removal",
                        cid[:12],
                        int(age),
                        min_age_seconds,
                    )
                    continue
            except (ValueError, TypeError):
                # Cannot parse timestamp — treat as stale to be safe
                pass

            removable.append(cid)
        except Exception:
            logger.warning("Failed to inspect conflicting container %s", cid[:12], exc_info=True)

    if not removable:
        return False

    logger.warning(
        "Removing %d stale conflicting container(s): %s",
        len(removable),
        ", ".join(c[:12] for c in removable),
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            *removable,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    except Exception:
        logger.warning("Failed to remove conflicting containers", exc_info=True)
    return True


async def _start_permanent_services(
    compose_file: Path,
    project: str,
    *,
    project_directory: Path | None = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> None:
    """Start shared services with exponential backoff retry.

    Args:
        compose_file: Path to the permanent services compose file.
        project: Docker Compose project name.
        project_directory: Working directory for relative volume paths.
        max_retries: Maximum number of startup attempts.
        base_delay: Base delay in seconds for exponential backoff.

    Raises:
        RuntimeError: After max_retries failures.
    """
    cmd: tuple[str, ...] = (
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "-p",
        project,
    )
    if project_directory is not None:
        cmd = (*cmd, "--project-directory", str(project_directory))
    cmd = (*cmd, "up", "-d", "--wait", "--build")

    display_progress(f"Starting permanent services (project={project})...")

    for attempt in range(max_retries):
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        monitor = asyncio.create_task(_monitor_startup_progress(project))
        try:
            _stdout, stderr = await proc.communicate()
        finally:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
        if proc.returncode == 0:
            display_progress(f"Permanent services started (project={project})")
            return
        if attempt < max_retries - 1:
            delay = base_delay * (2**attempt)
            logger.warning(
                "Permanent service startup failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                stderr.decode().strip(),
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed to start permanent services after {max_retries} attempts")


async def _stop_permanent_services(project: str) -> None:
    """Stop permanent services and remove volumes.

    Args:
        project: Docker Compose project name to stop.
    """
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        "-p",
        project,
        "down",
        "--volumes",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode == 0:
        display_progress(f"Permanent services stopped (project={project})")
    else:
        logger.warning("Permanent service shutdown failed: %s", stderr.decode().strip())


async def _poll_service_health(project: str) -> list[tuple[str, str, str]]:
    """Poll Docker Compose for container health status.

    Args:
        project: Docker Compose project name.

    Returns:
        List of ``(service, state, health)`` tuples.
        Empty list on any error (non-fatal).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "compose",
            "-p",
            project,
            "ps",
            "--format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return []
        results: list[tuple[str, str, str]] = []
        for line in stdout.decode().strip().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            service = obj.get("Service", obj.get("Name", "unknown"))
            state = obj.get("State", "unknown")
            health = obj.get("Health", "")
            results.append((service, state, health))
        return results
    except Exception:
        return []


async def _monitor_startup_progress(project: str, interval: float = _MONITOR_POLL_INTERVAL) -> None:
    """Log service health periodically until cancelled.

    Designed to run as an ``asyncio.Task`` alongside
    ``docker compose up --wait``.  The caller must cancel this task
    when startup completes.

    Args:
        project: Docker Compose project name.
        interval: Seconds between health polls.
    """
    while True:
        await asyncio.sleep(interval)
        statuses = await _poll_service_health(project)
        waiting = [
            f"{svc} (health: {health or state})"
            for svc, state, health in statuses
            if health.lower() not in ("healthy", "") or state.lower() != "running"
        ]
        if waiting:
            display_progress(f"Waiting for services: {', '.join(waiting)}")
