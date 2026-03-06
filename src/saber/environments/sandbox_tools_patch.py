"""Auto-patch inspect-sandbox-tools binary in site-packages.

When inspect-ai is installed as a git VCS reference (not editable), the
sandbox-tools binary in site-packages may be outdated.  This module
compares a locally-built binary from ``external/inspect_ai`` against the
installed copy and overwrites it when they differ.

The entry point :func:`patch_sandbox_tools_binary` is called once from
:meth:`SaberSandboxEnvironment.task_init` so every evaluation
automatically gets the correct binary without a manual copy step.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from saber.logging import get_logger

__all__ = ["SandboxToolsPatchResult", "patch_sandbox_tools_binary"]

logger = get_logger(__name__)

PatchAction = Literal[
    "patched",
    "already_current",
    "no_source",
    "no_destination",
    "error",
]


class SandboxToolsPatchResult(BaseModel):
    """Immutable result of a sandbox-tools binary patch attempt."""

    model_config = ConfigDict(frozen=True)

    action: PatchAction
    source_path: Path | None = None
    dest_path: Path | None = None
    message: str = ""


def _detect_arch() -> str:
    """Detect current CPU architecture as a sandbox-tools arch string."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    return machine


def _find_source_binary(repo_root: Path) -> tuple[Path | None, str]:
    """Locate the locally-built sandbox-tools binary.

    Args:
        repo_root: Root of the oss_saber (or similar) repository.

    Returns:
        ``(path_or_None, expected_filename)``.
    """
    version_file = (
        repo_root
        / "external"
        / "inspect_ai"
        / "src"
        / "inspect_ai"
        / "tool"
        / "_sandbox_tools_utils"
        / "sandbox_tools_version.txt"
    )
    if not version_file.exists():
        return None, ""

    version = version_file.read_text().strip()
    arch = _detect_arch()
    binary_name = f"inspect-sandbox-tools-{arch}-v{version}"

    binaries_dir = repo_root / "external" / "inspect_ai" / "src" / "inspect_ai" / "binaries"

    # Prefer the production name, fall back to -dev
    source = binaries_dir / binary_name
    if source.exists():
        return source, binary_name

    dev_source = binaries_dir / f"{binary_name}-dev"
    if dev_source.exists():
        return dev_source, binary_name

    return None, binary_name


def _find_installed_binary(binary_name: str) -> Path | None:
    """Locate the sandbox-tools binary inside site-packages.

    Args:
        binary_name: Expected filename, e.g. ``inspect-sandbox-tools-amd64-v8``.

    Returns:
        Path to the binary (may not yet exist), or ``None`` when the
        ``inspect_ai/binaries`` directory cannot be found.
    """
    try:
        import inspect_ai  # noqa: WPS433 — runtime import to find site-packages path

        binaries_dir = Path(inspect_ai.__file__).parent / "binaries"
        if binaries_dir.is_dir():
            return binaries_dir / binary_name
    except ImportError:
        pass
    return None


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_sandbox_tools_binary(
    repo_root: Path | None = None,
) -> SandboxToolsPatchResult:
    """Compare and patch the sandbox-tools binary if needed.

    Idempotent: returns ``already_current`` when the installed binary
    already matches the source.

    Args:
        repo_root: Root of the repository that contains
            ``external/inspect_ai/``.  Auto-detected from this file's
            location if ``None``.

    Returns:
        A :class:`SandboxToolsPatchResult` describing what happened.
    """
    if repo_root is None:
        # .../external/saber/src/saber/environments/sandbox_tools_patch.py
        # parents: [environments, saber, src, saber(ext), external, REPO_ROOT]
        repo_root = Path(__file__).resolve().parents[5]

    source, binary_name = _find_source_binary(repo_root)

    if source is None:
        msg = (
            f"No local sandbox-tools binary found ({binary_name!r}). "
            "The S3 binary will be used. If Claude Code agents crash, "
            "build the binary with: cd external/inspect_ai/src/inspect_ai/"
            "tool/_sandbox_tools_utils && "
            "uv run python build_within_container.py --arch amd64"
        )
        logger.warning(msg)
        return SandboxToolsPatchResult(action="no_source", message=msg)

    dest = _find_installed_binary(binary_name)
    if dest is None:
        msg = "Cannot locate inspect_ai/binaries/ in site-packages."
        logger.warning(msg)
        return SandboxToolsPatchResult(action="no_destination", source_path=source, message=msg)

    # Fast path: same size → compare hashes
    if dest.exists() and dest.stat().st_size == source.stat().st_size:
        if _file_sha256(dest) == _file_sha256(source):
            logger.debug("Sandbox-tools binary already current: %s", dest)
            return SandboxToolsPatchResult(
                action="already_current",
                source_path=source,
                dest_path=dest,
                message="Binary already matches source.",
            )

    # Patch
    try:
        shutil.copy2(source, dest)
        dest.chmod(0o755)
        msg = f"Patched sandbox-tools binary: {dest} <- {source}"
        logger.info(msg)
        return SandboxToolsPatchResult(
            action="patched",
            source_path=source,
            dest_path=dest,
            message=msg,
        )
    except OSError as exc:
        msg = f"Failed to patch sandbox-tools binary: {exc}"
        logger.error(msg)
        return SandboxToolsPatchResult(
            action="error",
            source_path=source,
            dest_path=dest,
            message=msg,
        )
