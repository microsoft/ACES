# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Docker image rebuild mode parsing and build primitives for SABER environments.

Provides ``RebuildMode`` (which images to rebuild),
``parse_rebuild_param`` (convert a raw CLI/task-param string into a
``RebuildMode``), and async helpers for checking / building Docker
images.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import subprocess
import sys
import time
import types
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from saber.logging import display_progress, get_logger

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.text import Text as RichText

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False

#: Callback signature for build progress: (image_name, image_tag, event).
#: Events: "checking", "building", "built", "rebuilt", "skipped", "failed".
BuildProgressCallback = Callable[[str, str, str], None]

logger = get_logger(__name__)

__all__ = [
    "AGENTS_DOCKERFILE_NAME",
    "AGENTS_IMAGE_TAG",
    "BASE_DOCKERFILE_NAME",
    "BASE_IMAGE_TAG",
    "BuildProgressCallback",
    "ImageBuildError",
    "ImageBuildResult",
    "PreflightResult",
    "RebuildMode",
    "RebuildScope",
    "base_variant_for_agent",
    "build_domain_images",
    "build_image",
    "find_base_dockerfile",
    "image_exists",
    "parse_rebuild_param",
]

# ── Constants ────────────────────────────────────────────────────────

BASE_IMAGE_TAG: str = "saber/sandbox:latest"
BASE_DOCKERFILE_NAME: str = "Dockerfile.saber_sandbox"
#: Variant of the base image that adds the Copilot / Claude Code CLIs. Built on
#: top of BASE_IMAGE_TAG and selected automatically for the CLI harnesses.
AGENTS_IMAGE_TAG: str = "saber/sandbox:agents"
AGENTS_DOCKERFILE_NAME: str = "Dockerfile.saber_sandbox_agents"
#: Label stamped on each base variant; derived images inherit it, which lets
#: saber detect a domain image built from the wrong variant and rebuild it.
BASE_VARIANT_LABEL: str = "saber.base_variant"
#: Agents that run entirely in-process and so need no CLI tooling in the sandbox.
#: Everything else drives a CLI over the sandbox bridge and needs AGENTS_IMAGE_TAG.
_REACT_ONLY_AGENTS: frozenset[str] = frozenset({"react"})


def base_variant_for_agent(agent: str | None) -> str:
    """Return the base image variant an agent needs.

    Args:
        agent: Selected agent name, or ``None`` for the default.

    Returns:
        ``"base"`` for the react-only sandbox, ``"agents"`` for harnesses that
        run the Copilot / Claude Code CLIs inside the sandbox.
    """
    if agent is None or agent in _REACT_ONLY_AGENTS:
        return "base"
    return "agents"


class RebuildScope(Enum):
    """Scope of a Docker image rebuild request."""

    NONE = auto()
    ALL = auto()
    SPECIFIC = auto()


class RebuildMode(BaseModel):
    """Immutable specification of which Docker images should be rebuilt."""

    model_config = ConfigDict(frozen=True)

    scope: RebuildScope
    names: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def _validate_specific_has_names(self) -> RebuildMode:
        if self.scope is RebuildScope.SPECIFIC and not self.names:
            msg = "SPECIFIC scope requires non-empty names"
            raise ValueError(msg)
        return self

    @classmethod
    def none(cls) -> RebuildMode:
        """Create a mode that rebuilds nothing."""
        return cls(scope=RebuildScope.NONE)

    @classmethod
    def all(cls) -> RebuildMode:
        """Create a mode that rebuilds all images."""
        return cls(scope=RebuildScope.ALL)

    @classmethod
    def specific(cls, names: frozenset[str]) -> RebuildMode:
        """Create a mode that rebuilds only the named images.

        Args:
            names: Non-empty frozenset of image names to rebuild.

        Raises:
            ValueError: If *names* is empty.
        """
        return cls(scope=RebuildScope.SPECIFIC, names=names)

    def should_rebuild(self, image_name: str) -> bool:
        """Return whether *image_name* should be rebuilt under this mode.

        Args:
            image_name: The image name to check.

        Returns:
            True if the image should be rebuilt, False otherwise.
        """
        if self.scope is RebuildScope.NONE:
            return False
        if self.scope is RebuildScope.ALL:
            return True
        return image_name in self.names


def parse_rebuild_param(raw: str | bool | None) -> RebuildMode:
    """Parse a raw rebuild parameter into a ``RebuildMode``.

    Args:
        raw: The raw parameter value. ``None`` means no rebuild.
            Accepts ``bool`` (from inspect_ai ``-T`` flag parsing)
            or a string.

    Returns:
        A ``RebuildMode`` instance.

    Raises:
        ValueError: If *raw* is empty, whitespace-only, or contains
            only commas.
    """
    if raw is None:
        return RebuildMode.none()

    # inspect_ai -T flag parsing may pass a Python bool directly
    if isinstance(raw, bool):
        return RebuildMode.all() if raw else RebuildMode.none()

    stripped = raw.strip()

    if stripped == "":
        msg = "Rebuild parameter must not be empty"
        raise ValueError(msg)

    lower = stripped.lower()
    if lower == "true":
        return RebuildMode.all()
    if lower == "false":
        return RebuildMode.none()

    # Parse comma-separated list of image names
    names = frozenset(name for part in stripped.split(",") if (name := part.strip()))

    if not names:
        msg = "Rebuild parameter must not be empty"
        raise ValueError(msg)

    return RebuildMode.specific(names)


# ── ImageBuildResult ─────────────────────────────────────────────────


class ImageBuildResult(BaseModel):
    """Result of a single image build operation."""

    model_config = ConfigDict(frozen=True)

    name: str
    tag: str
    action: Literal["built", "skipped", "rebuilt", "failed"]
    error: str | None = None


# ── ImageBuildError ──────────────────────────────────────────────────


class ImageBuildError(Exception):
    """Raised when a Docker image build fails."""

    def __init__(self, tag: str, stderr: str, returncode: int, stdout: str = "") -> None:
        self.tag = tag
        self.stderr = stderr
        self.stdout = stdout
        self.returncode = returncode
        # Combine stdout and stderr for a complete picture — Docker
        # (especially BuildKit) sends build output to stdout.
        combined = ""
        if stdout and stdout.strip():
            combined += stdout.strip()
        if stderr and stderr.strip():
            if combined:
                combined += "\n"
            combined += stderr.strip()
        if combined:
            msg = f"Failed to build {tag} (exit {returncode}):\n{combined}"
        else:
            msg = f"Failed to build {tag} (exit {returncode}): (no output captured)"
        super().__init__(msg)


# ── Async helpers ────────────────────────────────────────────────────


def _detect_build_phase(line: str) -> str | None:
    """Detect the current build phase from a Docker build output line.

    Returns a short human-readable status string when a recognisable
    build step is detected, or ``None`` otherwise.
    """
    step_match = re.match(r"Step\s+(\d+/\d+)\s*:\s*(.*)", line)
    if step_match:
        return f"Step {step_match.group(1)}: {step_match.group(2)[:60]}"

    buildkit_match = re.match(r"#\d+\s+\[.*?\]\s*(.*)", line)
    if buildkit_match:
        return buildkit_match.group(1)[:60]

    lower = line.lower()
    if "apt-get" in lower:
        return "installing packages\u2026"
    if "pip install" in lower:
        return "installing Python packages\u2026"
    if "npm install" in lower or "yarn install" in lower:
        return "installing Node packages\u2026"
    if "successfully built" in lower or "exporting to image" in lower:
        return "finalizing\u2026"
    return None


async def image_exists(tag: str) -> bool:
    """Check whether a Docker image with *tag* exists locally.

    Args:
        tag: Full image tag, e.g. ``"saber/sandbox:latest"``.

    Returns:
        ``True`` if the image is present, ``False`` otherwise.
    """
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "image",
        "inspect",
        tag,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0


async def build_image(
    tag: str,
    dockerfile: Path,
    context: Path | None = None,
    build_args: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    no_cache: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    """Build a Docker image.

    Args:
        tag: Image tag to assign, e.g. ``"saber/sandbox:latest"``.
        dockerfile: Path to the Dockerfile.
        context: Docker build context directory.  Defaults to
            ``dockerfile.parent`` when *None*.
        build_args: Optional ``--build-arg`` key/value pairs.
        labels: Optional ``--label`` key/value pairs.
        no_cache: If ``True``, pass ``--no-cache`` to ``docker build``
            so that all layers are rebuilt from scratch.
        status_callback: Optional callback invoked with a short phase
            description whenever a recognisable build step is detected.

    Raises:
        ImageBuildError: If the build exits with a non-zero code.
    """
    if context is None:
        context = dockerfile.parent

    cmd: list[str] = ["docker", "build", "-f", str(dockerfile), "-t", tag, "--progress=plain"]

    if no_cache:
        cmd.append("--no-cache")

    for key, value in (build_args or {}).items():
        cmd.extend(["--build-arg", f"{key}={value}"])

    for key, value in (labels or {}).items():
        cmd.extend(["--label", f"{key}={value}"])

    cmd.append(str(context))

    logger.info("Running: %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    async def _read_stream(
        stream: asyncio.StreamReader | None,
        *,
        is_stderr: bool = False,
    ) -> None:
        if stream is None:
            return
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                # readline() raises ValueError when a single line exceeds
                # the StreamReader buffer limit (default 64 KiB).  Docker
                # --progress=plain can emit very long compiler commands.
                chunk = await stream.read(65536)
                if not chunk:
                    break
                line = chunk
            if not line:
                break
            if is_stderr:
                stderr_chunks.append(line)
            else:
                stdout_chunks.append(line)
            decoded = line.decode(errors="replace").rstrip()
            if decoded and status_callback is not None:
                phase = _detect_build_phase(decoded)
                if phase is not None:
                    status_callback(phase)

    await asyncio.gather(
        _read_stream(proc.stdout),
        _read_stream(proc.stderr, is_stderr=True),
    )
    await proc.wait()

    if proc.returncode != 0:
        stdout_text = b"".join(stdout_chunks).decode(errors="replace")
        stderr_text = b"".join(stderr_chunks).decode(errors="replace")
        # Log full build output so it's always available in logs
        logger.error(
            "Docker build failed for %s (exit %d)\n--- stdout ---\n%s\n--- stderr ---\n%s",
            tag,
            proc.returncode,
            stdout_text or "(empty)",
            stderr_text or "(empty)",
        )
        raise ImageBuildError(
            tag=tag,
            stderr=stderr_text,
            stdout=stdout_text,
            returncode=proc.returncode if proc.returncode is not None else -1,
        )


def find_base_dockerfile(saber_root: Path | None = None) -> Path:
    """Locate the base SABER sandbox Dockerfile.

    Searches in order:
    1. ``saber_root / docker / Dockerfile.saber_sandbox`` (development / source checkout)
    2. Bundled inside the installed package at ``saber/environments/_dockerfiles/``

    Args:
        saber_root: Explicit project root.  When *None*, auto-discovered
            relative to this source file.

    Returns:
        Resolved ``Path`` to the Dockerfile.

    Raises:
        FileNotFoundError: If the Dockerfile does not exist at any
            expected location.
    """
    return _find_dockerfile(BASE_DOCKERFILE_NAME, saber_root)


def find_agents_dockerfile(saber_root: Path | None = None) -> Path:
    """Locate the Dockerfile for the agent-harness base image variant.

    Args:
        saber_root: Explicit project root.  When *None*, auto-discovered
            relative to this source file.

    Returns:
        Resolved ``Path`` to the Dockerfile.

    Raises:
        FileNotFoundError: If the Dockerfile does not exist at any
            expected location.
    """
    return _find_dockerfile(AGENTS_DOCKERFILE_NAME, saber_root)


def _find_dockerfile(name: str, saber_root: Path | None = None) -> Path:
    """Resolve *name* in the source checkout, falling back to the packaged copy."""
    # 1. Check repo / source tree location
    if saber_root is None:
        saber_root = Path(__file__).parent.parent.parent.parent

    repo_path = saber_root / "docker" / name
    if repo_path.exists():
        return repo_path.resolve()

    # 2. Check package-bundled location (installed from wheel)
    package_path = Path(__file__).parent / "_dockerfiles" / name
    if package_path.exists():
        return package_path.resolve()

    msg = f"Dockerfile {name} not found at {repo_path} or bundled location {package_path}"
    raise FileNotFoundError(msg)


async def image_base_variant(tag: str) -> str | None:
    """Return the base variant an image was built from, or ``None`` if unknown.

    Derived images inherit the label from their base, so this reports whether a
    domain image was built on ``saber/sandbox:latest`` or ``saber/sandbox:agents``.

    Args:
        tag: Image tag to inspect.

    Returns:
        ``"base"``, ``"agents"``, or ``None`` when the image or label is absent.
    """
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "image",
        "inspect",
        tag,
        "--format",
        f'{{{{index .Config.Labels "{BASE_VARIANT_LABEL}"}}}}',
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    variant = stdout.decode().strip()
    return variant or None


# ── Progress helpers ─────────────────────────────────────────────────


def _create_build_progress(total: int) -> _BuildProgressDisplay | None:
    """Create a Rich live progress display for image builds, or None if unavailable.

    Returns None when rich is not installed or stderr is not a TTY.

    Args:
        total: Total number of images to build.

    Returns:
        A configured _BuildProgressDisplay instance, or None.
    """
    if not _HAS_RICH or not sys.stderr.isatty():
        return None
    return _BuildProgressDisplay(total)


class _BuildProgressDisplay:
    """Rich Live display showing progress bar + per-image status lines."""

    def __init__(self, total: int) -> None:
        self._console = Console(stderr=True)
        self._bar = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        self._task_id = self._bar.add_task("Building images", total=total)
        self._statuses: dict[str, str] = {}
        self._live = Live(self._render(), console=self._console, refresh_per_second=4)

    def _render(self) -> Group:
        """Render the progress bar and per-image status lines."""
        parts: list[Progress | RichText] = [self._bar]
        for name, status in self._statuses.items():
            parts.append(RichText.from_markup(f"  [dim]{name:<16}[/dim] {status}"))
        return Group(*parts)

    def set_status(self, name: str, status: str) -> None:
        """Update the status line for a specific image."""
        self._statuses[name] = status
        self._live.update(self._render())

    def complete_image(self, name: str, status: str) -> None:
        """Mark an image as complete and advance the progress bar."""
        self._statuses[name] = status
        self._bar.update(self._task_id, advance=1)
        self._live.update(self._render())

    def set_description(self, desc: str) -> None:
        """Update the progress bar description."""
        self._bar.update(self._task_id, description=desc)
        self._live.update(self._render())

    def __enter__(self) -> _BuildProgressDisplay:
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self._live.__exit__(exc_type, exc_val, exc_tb)


def _print_build_summary(result: PreflightResult, elapsed: float) -> None:
    """Print a final build summary line to stderr via display_progress.

    Args:
        result: The preflight result to summarize.
        elapsed: Total elapsed time in seconds.
    """
    parts: list[str] = []
    if result.built_count:
        parts.append(f"{result.built_count} built")
    if result.skipped_count:
        parts.append(f"{result.skipped_count} skipped")
    if result.failed_count:
        parts.append(f"{result.failed_count} failed")
    counts = ", ".join(parts) if parts else "0 images"
    display_progress(f"Image preflight complete: {counts} ({elapsed:.1f}s)")


# ── PreflightResult ──────────────────────────────────────────────────


class PreflightResult(BaseModel):
    """Summary of all preflight image operations."""

    model_config = ConfigDict(frozen=True)

    results: tuple[ImageBuildResult, ...] = ()
    domain_slug: str = ""

    @property
    def built_count(self) -> int:
        """Number of images that were built or rebuilt."""
        return sum(1 for r in self.results if r.action in ("built", "rebuilt"))

    @property
    def skipped_count(self) -> int:
        """Number of images that were skipped."""
        return sum(1 for r in self.results if r.action == "skipped")

    @property
    def failed_count(self) -> int:
        """Number of images that failed to build."""
        return sum(1 for r in self.results if r.action == "failed")

    @property
    def all_succeeded(self) -> bool:
        """Whether all image operations succeeded (no failures)."""
        return self.failed_count == 0


class PreflightBuildError(RuntimeError):
    """Raised when one or more preflight image builds fail.

    Provides a human-readable summary with per-image error details so
    the root cause is immediately visible in the terminal output.
    """

    def __init__(self, failed_results: list[ImageBuildResult]) -> None:
        self.failed_results = failed_results
        names = [r.name for r in failed_results]
        lines = [
            "",
            "=" * 70,
            "PREFLIGHT IMAGE BUILD FAILED",
            "=" * 70,
            f"{len(failed_results)} image(s) failed to build: {', '.join(names)}",
            "",
        ]
        for r in failed_results:
            lines.append(f"--- {r.name} ({r.tag}) ---")
            if r.error:
                # Show last ~40 lines of Docker build output to keep it readable
                error_lines = r.error.strip().splitlines()
                if len(error_lines) > 40:
                    lines.append(f"  ... ({len(error_lines) - 40} lines truncated)")
                    error_lines = error_lines[-40:]
                for el in error_lines:
                    lines.append(f"  {el}")
            else:
                lines.append("  (no error details captured)")
            lines.append("")
        lines.append("=" * 70)
        msg = "\n".join(lines)
        super().__init__(msg)


# ── build_domain_images orchestrator ─────────────────────────────────


async def build_domain_images(
    domain_root: Path,
    rebuild: RebuildMode | None = None,
    saber_root: Path | None = None,
    on_progress: BuildProgressCallback | None = None,
    base_variant: str = "base",
) -> PreflightResult:
    """Build or verify all Docker images required by a domain.

    Args:
        domain_root: Path to the domain root directory containing ``eval.yaml``.
        rebuild: Which images to rebuild. ``None`` means rebuild nothing.
        saber_root: Explicit SABER project root for locating the base Dockerfile.
        on_progress: Optional callback invoked before and after each image
            build with ``(name, tag, event)``.  Events are ``"checking"``,
            ``"building"``, ``"built"``, ``"rebuilt"``, ``"skipped"``,
            or ``"failed"``.
        base_variant: ``"base"`` for the react-only sandbox, or ``"agents"`` to
            also build the CLI image that the copilot / claude_code harnesses
            need and derive domain images from it.

    Returns:
        A :class:`PreflightResult` summarising the outcome of each image.
    """

    def _notify(name: str, tag: str, event: str) -> None:
        if on_progress is not None:
            on_progress(name, tag, event)

    from saber.config.loader import load_domain_config

    try:
        config = load_domain_config(domain_root)
    except FileNotFoundError:
        logger.warning("No eval.yaml found in %s — skipping image preflight", domain_root)
        return PreflightResult()

    if not config.images:
        return PreflightResult(domain_slug=config.slug)

    if rebuild is None:
        rebuild = RebuildMode.none()

    # Warn about unknown names in SPECIFIC mode
    if rebuild.scope is RebuildScope.SPECIFIC:
        known_names = set(config.images.keys()) | {"base", "saber_sandbox"}
        for name in rebuild.names:
            if name not in known_names:
                logger.warning(
                    "Rebuild name %r not found in domain images or reserved names",
                    name,
                )

    all_results: list[ImageBuildResult] = []

    wants_agents = base_variant == "agents"
    #: Tag domain images are built FROM, and the variant they should carry.
    domain_base_tag = AGENTS_IMAGE_TAG if wants_agents else BASE_IMAGE_TAG

    total_images = 1 + int(wants_agents) + len(config.images)
    display = _create_build_progress(total_images)
    start_time = time.monotonic()

    ctx: contextlib.AbstractContextManager[object] = display if display is not None else contextlib.nullcontext()
    with ctx:
        # ── Handle base image ────────────────────────────────────────────
        rebuild_base = rebuild.should_rebuild("base") or rebuild.should_rebuild("saber_sandbox")

        if rebuild_base or not await image_exists(BASE_IMAGE_TAG):
            action_label: Literal["built", "rebuilt"] = "rebuilt" if rebuild_base else "built"
            base_df = find_base_dockerfile(saber_root)
            _notify("base", BASE_IMAGE_TAG, "building")

            def _base_status(status: str) -> None:
                if display is not None:
                    display.set_status("base", status)

            try:
                await build_image(
                    tag=BASE_IMAGE_TAG,
                    dockerfile=base_df,
                    context=base_df.parent,
                    no_cache=rebuild_base,
                    status_callback=_base_status if display else None,
                )
                base_result = ImageBuildResult(name="base", tag=BASE_IMAGE_TAG, action=action_label)
                logger.info("Image %s (%s): %s", "base", BASE_IMAGE_TAG, action_label)
                _notify("base", BASE_IMAGE_TAG, action_label)
            except ImageBuildError as exc:
                base_result = ImageBuildResult(name="base", tag=BASE_IMAGE_TAG, action="failed", error=str(exc))
                logger.warning("Image %s (%s): failed \u2014 %s", "base", BASE_IMAGE_TAG, exc)
                _notify("base", BASE_IMAGE_TAG, "failed")
        else:
            base_result = ImageBuildResult(name="base", tag=BASE_IMAGE_TAG, action="skipped")
            logger.info("Image %s (%s): skipped (already exists)", "base", BASE_IMAGE_TAG)
            _notify("base", BASE_IMAGE_TAG, "skipped")

        all_results.append(base_result)
        if display is not None:
            status_label = (
                f"[green]\u2713[/green] {base_result.action}"
                if base_result.action != "failed"
                else "[red]\u2717 failed[/red]"
            )
            display.complete_image("base", status_label)

        # Short-circuit if base image failed — domain images depend on it
        if base_result.action == "failed":
            logger.warning("Skipping domain images \u2014 base image build failed")
            result = PreflightResult(results=tuple(all_results), domain_slug=config.slug)
            elapsed = time.monotonic() - start_time
            _print_build_summary(result, elapsed)
            return result

        # ── Handle the agent-CLI base variant ────────────────────────────
        if wants_agents:
            rebuild_agents = rebuild_base or rebuild.should_rebuild("agents")
            if rebuild_agents or not await image_exists(AGENTS_IMAGE_TAG):
                agents_action: Literal["built", "rebuilt"] = "rebuilt" if rebuild_agents else "built"
                agents_df = find_agents_dockerfile(saber_root)
                _notify("agents", AGENTS_IMAGE_TAG, "building")

                def _agents_status(status: str) -> None:
                    if display is not None:
                        display.set_status("agents", status)

                try:
                    await build_image(
                        tag=AGENTS_IMAGE_TAG,
                        dockerfile=agents_df,
                        context=agents_df.parent,
                        build_args={"SABER_BASE_IMAGE": BASE_IMAGE_TAG},
                        no_cache=rebuild_agents,
                        status_callback=_agents_status if display else None,
                    )
                    agents_result = ImageBuildResult(name="agents", tag=AGENTS_IMAGE_TAG, action=agents_action)
                    logger.info("Image %s (%s): %s", "agents", AGENTS_IMAGE_TAG, agents_action)
                    _notify("agents", AGENTS_IMAGE_TAG, agents_action)
                except ImageBuildError as exc:
                    agents_result = ImageBuildResult(
                        name="agents", tag=AGENTS_IMAGE_TAG, action="failed", error=str(exc)
                    )
                    logger.warning("Image %s (%s): failed \u2014 %s", "agents", AGENTS_IMAGE_TAG, exc)
                    _notify("agents", AGENTS_IMAGE_TAG, "failed")
            else:
                agents_result = ImageBuildResult(name="agents", tag=AGENTS_IMAGE_TAG, action="skipped")
                logger.info("Image %s (%s): skipped (already exists)", "agents", AGENTS_IMAGE_TAG)
                _notify("agents", AGENTS_IMAGE_TAG, "skipped")

            all_results.append(agents_result)
            if display is not None:
                status_label = (
                    f"[green]\u2713[/green] {agents_result.action}"
                    if agents_result.action != "failed"
                    else "[red]\u2717 failed[/red]"
                )
                display.complete_image("agents", status_label)

            if agents_result.action == "failed":
                logger.warning("Skipping domain images \u2014 agent CLI image build failed")
                result = PreflightResult(results=tuple(all_results), domain_slug=config.slug)
                elapsed = time.monotonic() - start_time
                _print_build_summary(result, elapsed)
                return result

        # ── Handle domain images ─────────────────────────────────────────
        for name, image_config in config.images.items():
            dockerfile = domain_root / image_config.dockerfile
            context = domain_root / image_config.context if image_config.context is not None else domain_root

            if display is not None:
                display.set_description(f"Checking {name}")

            _notify(name, image_config.tag, "checking")
            force_rebuild = rebuild.should_rebuild(name)
            if force_rebuild:
                action: Literal["built", "rebuilt"] = "rebuilt"
            elif await image_exists(image_config.tag):
                # Only images that derive from the saber base can be the wrong
                # variant; services built from their own base (a database image,
                # say) are unrelated and must not be rebuilt on every run.
                derives_from_base = False
                with contextlib.suppress(OSError):
                    derives_from_base = "SABER_BASE_IMAGE" in dockerfile.read_text(errors="ignore")
                # An unlabelled image predates the variants, so its CLI tooling is
                # unknown - treat it as unusable when a CLI harness is requested.
                stale_variant = (
                    wants_agents
                    and derives_from_base
                    and await image_base_variant(image_config.tag) != "agents"
                )
                if not stale_variant:
                    all_results.append(ImageBuildResult(name=name, tag=image_config.tag, action="skipped"))
                    logger.info("Image %s (%s): skipped (already exists)", name, image_config.tag)
                    _notify(name, image_config.tag, "skipped")
                    if display is not None:
                        display.complete_image(name, "[dim]skipped[/dim]")
                    continue
                logger.info(
                    "Image %s (%s): rebuilding \u2014 %s needs the agent CLIs and this image lacks them",
                    name,
                    image_config.tag,
                    base_variant,
                )
                action = "rebuilt"
            else:
                action = "built"

            if display is not None:
                display.set_description(f"Building {name}")
            _notify(name, image_config.tag, "building")

            def _image_status(status: str, _name: str = name) -> None:
                if display is not None:
                    display.set_status(_name, status)

            try:
                await build_image(
                    tag=image_config.tag,
                    dockerfile=dockerfile,
                    context=context,
                    build_args={"SABER_BASE_IMAGE": domain_base_tag, **(image_config.build_args or {})},
                    labels=image_config.labels or None,
                    no_cache=force_rebuild,
                    status_callback=_image_status if display else None,
                )
                all_results.append(ImageBuildResult(name=name, tag=image_config.tag, action=action))
                logger.info("Image %s (%s): %s", name, image_config.tag, action)
                _notify(name, image_config.tag, action)
                if display is not None:
                    display.complete_image(name, f"[green]\u2713[/green] {action}")
            except ImageBuildError as exc:
                all_results.append(ImageBuildResult(name=name, tag=image_config.tag, action="failed", error=str(exc)))
                logger.warning("Image %s (%s): failed \u2014 %s", name, image_config.tag, exc)
                _notify(name, image_config.tag, "failed")
                if display is not None:
                    display.complete_image(name, "[red]\u2717 failed[/red]")

    result = PreflightResult(results=tuple(all_results), domain_slug=config.slug)
    elapsed = time.monotonic() - start_time
    _print_build_summary(result, elapsed)
    return result
