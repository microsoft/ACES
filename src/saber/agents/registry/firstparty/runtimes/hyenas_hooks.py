"""Sandbox hook functions for Hyenas: repo injection and findings extraction."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING

from saber.agents.registry.firstparty.runtimes.hyenas_models import HyenasFinding
from saber.logging import get_logger

if TYPE_CHECKING:
    from inspect_ai.util import SandboxEnvironment

logger = get_logger(__name__)

_SANDBOX_TAR_PATH = "/tmp/repo.tar.gz"
_WORKSPACE_DIR = "/workspace"
_FINDINGS_PATH = "/output/.hyenas/scan/findings.jsonl"


async def inject_target_repo(
    sandbox: SandboxEnvironment,
    metadata: dict[str, str],
) -> None:
    """Inject a target repository into the sandbox at /workspace.

    Reads repository data from metadata. Supports two modes:
    - ``repo_tarball``: Base64-encoded tarball (priority if both present)
    - ``repo_path``: Path to a tarball file accessible from host

    Args:
        sandbox: The sandbox environment to write into.
        metadata: Sample metadata with repo_tarball or repo_path key.

    Raises:
        ValueError: If neither repo_tarball nor repo_path is in metadata.
        RuntimeError: If tar extraction fails inside the sandbox.
    """
    tar_bytes = _resolve_tar_bytes(metadata)

    await sandbox.write_file(_SANDBOX_TAR_PATH, tar_bytes)

    result = await sandbox.exec(
        ["tar", "xzf", _SANDBOX_TAR_PATH, "-C", _WORKSPACE_DIR]
    )
    if result.returncode != 0:
        msg = f"tar extraction failed (exit {result.returncode}): {result.stderr}"
        raise RuntimeError(msg)


def _resolve_tar_bytes(metadata: dict[str, str]) -> bytes:
    """Resolve tarball bytes from metadata (tarball takes priority over path).

    Args:
        metadata: Sample metadata dict.

    Returns:
        Raw bytes of the tarball.

    Raises:
        ValueError: If neither repo_tarball nor repo_path is present.
    """
    if "repo_tarball" in metadata:
        return base64.b64decode(metadata["repo_tarball"])

    if "repo_path" in metadata:
        return Path(metadata["repo_path"]).read_bytes()

    msg = "metadata must contain 'repo_tarball' or 'repo_path'"
    raise ValueError(msg)


async def extract_hyenas_findings(
    sandbox: SandboxEnvironment,
) -> list[HyenasFinding]:
    """Extract findings from Hyenas scan output.

    Reads ``/output/.hyenas/scan/findings.jsonl`` from the sandbox.
    Each line is a JSON object parsed into :class:`HyenasFinding`.
    Invalid lines are skipped with a warning.

    Args:
        sandbox: The sandbox environment to read from.

    Returns:
        List of parsed HyenasFinding objects. Empty if file missing or empty.
    """
    try:
        content = await sandbox.read_file(_FINDINGS_PATH, text=True)
    except FileNotFoundError:
        logger.warning("Findings file not found at %s", _FINDINGS_PATH)
        return []
    except Exception:
        logger.exception("Unexpected error reading findings file at %s", _FINDINGS_PATH)
        return []

    if not isinstance(content, str):
        logger.warning("Unexpected binary content from %s", _FINDINGS_PATH)
        return []

    findings: list[HyenasFinding] = []
    for line_num, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
            findings.append(HyenasFinding.model_validate(parsed))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Skipping invalid finding on line %d: %s", line_num, exc
            )
    return findings
