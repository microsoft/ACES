"""Docker image rebuild mode parsing and build primitives for SABER environments.

Provides ``RebuildMode`` (which images to rebuild),
``parse_rebuild_param`` (convert a raw CLI/task-param string into a
``RebuildMode``), and async helpers for checking / building Docker
images.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from saber.logging import get_logger

#: Callback signature for build progress: (image_name, image_tag, event).
#: Events: "checking", "building", "built", "rebuilt", "skipped", "failed".
BuildProgressCallback = Callable[[str, str, str], None]

logger = get_logger(__name__)

__all__ = [
    "BASE_DOCKERFILE_NAME",
    "BASE_IMAGE_TAG",
    "BuildProgressCallback",
    "ImageBuildError",
    "ImageBuildResult",
    "PreflightResult",
    "RebuildMode",
    "RebuildScope",
    "build_domain_images",
    "build_image",
    "find_base_dockerfile",
    "image_exists",
    "parse_rebuild_param",
]

# ── Constants ────────────────────────────────────────────────────────

BASE_IMAGE_TAG: str = "saber/sandbox:latest"
BASE_DOCKERFILE_NAME: str = "Dockerfile.saber_sandbox"


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


def parse_rebuild_param(raw: str | None) -> RebuildMode:
    """Parse a raw rebuild parameter into a ``RebuildMode``.

    Args:
        raw: The raw parameter value. ``None`` means no rebuild.

    Returns:
        A ``RebuildMode`` instance.

    Raises:
        ValueError: If *raw* is empty, whitespace-only, or contains
            only commas.
    """
    if raw is None:
        return RebuildMode.none()

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

    def __init__(self, tag: str, stderr: str, returncode: int) -> None:
        self.tag = tag
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(f"Failed to build {tag} (exit {returncode}): {stderr}")


# ── Async helpers ────────────────────────────────────────────────────


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

    Raises:
        ImageBuildError: If the build exits with a non-zero code.
    """
    if context is None:
        context = dockerfile.parent

    cmd: list[str] = ["docker", "build", "-f", str(dockerfile), "-t", tag]

    if no_cache:
        cmd.append("--no-cache")

    for key, value in (build_args or {}).items():
        cmd.extend(["--build-arg", f"{key}={value}"])

    for key, value in (labels or {}).items():
        cmd.extend(["--label", f"{key}={value}"])

    cmd.append(str(context))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr_bytes = await proc.communicate()

    if proc.returncode != 0:
        raise ImageBuildError(
            tag=tag,
            stderr=stderr_bytes.decode(errors="replace"),
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
    # 1. Check repo / source tree location
    if saber_root is None:
        saber_root = Path(__file__).parent.parent.parent.parent

    repo_path = saber_root / "docker" / BASE_DOCKERFILE_NAME
    if repo_path.exists():
        return repo_path.resolve()

    # 2. Check package-bundled location (installed from wheel / ADO)
    package_path = Path(__file__).parent / "_dockerfiles" / BASE_DOCKERFILE_NAME
    if package_path.exists():
        return package_path.resolve()

    msg = f"Base Dockerfile not found at {repo_path} or bundled location {package_path}"
    raise FileNotFoundError(msg)


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
                # Show last ~20 lines of Docker build output to keep it readable
                error_lines = r.error.strip().splitlines()
                if len(error_lines) > 20:
                    lines.append(f"  ... ({len(error_lines) - 20} lines truncated)")
                    error_lines = error_lines[-20:]
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

    # ── Handle base image ────────────────────────────────────────────
    rebuild_base = rebuild.should_rebuild("base") or rebuild.should_rebuild("saber_sandbox")

    if rebuild_base or not await image_exists(BASE_IMAGE_TAG):
        action_label: Literal["built", "rebuilt"] = "rebuilt" if rebuild_base else "built"
        base_df = find_base_dockerfile(saber_root)
        _notify("base", BASE_IMAGE_TAG, "building")
        try:
            await build_image(
                tag=BASE_IMAGE_TAG,
                dockerfile=base_df,
                context=base_df.parent,
                no_cache=rebuild_base,
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

    # Short-circuit if base image failed — domain images depend on it
    if base_result.action == "failed":
        logger.warning("Skipping domain images \u2014 base image build failed")
        return PreflightResult(results=tuple(all_results), domain_slug=config.slug)

    # ── Handle domain images ─────────────────────────────────────────
    for name, image_config in config.images.items():
        dockerfile = domain_root / image_config.dockerfile
        context = domain_root / image_config.context if image_config.context is not None else domain_root

        _notify(name, image_config.tag, "checking")
        force_rebuild = rebuild.should_rebuild(name)
        if force_rebuild:
            action: Literal["built", "rebuilt"] = "rebuilt"
        elif await image_exists(image_config.tag):
            all_results.append(ImageBuildResult(name=name, tag=image_config.tag, action="skipped"))
            logger.info("Image %s (%s): skipped (already exists)", name, image_config.tag)
            _notify(name, image_config.tag, "skipped")
            continue
        else:
            action = "built"

        _notify(name, image_config.tag, "building")
        try:
            await build_image(
                tag=image_config.tag,
                dockerfile=dockerfile,
                context=context,
                build_args=image_config.build_args or None,
                labels=image_config.labels or None,
                no_cache=force_rebuild,
            )
            all_results.append(ImageBuildResult(name=name, tag=image_config.tag, action=action))
            logger.info("Image %s (%s): %s", name, image_config.tag, action)
            _notify(name, image_config.tag, action)
        except ImageBuildError as exc:
            all_results.append(ImageBuildResult(name=name, tag=image_config.tag, action="failed", error=str(exc)))
            logger.warning("Image %s (%s): failed \u2014 %s", name, image_config.tag, exc)
            _notify(name, image_config.tag, "failed")

    return PreflightResult(results=tuple(all_results), domain_slug=config.slug)
