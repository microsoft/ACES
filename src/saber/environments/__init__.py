# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Sandbox environment resolution.

Maps domain configuration to the correct inspect_ai sandbox specification.
Per DEC-009, always returns ``("saber", ...)`` when the sandbox compose
file exists.
"""

from __future__ import annotations

from pathlib import Path

from saber.environments.images import RebuildMode
from saber.logging import get_logger

__all__ = ["resolve_sandbox_spec"]

logger = get_logger(__name__)


def resolve_sandbox_spec(
    domain_root: Path,
    sandbox_compose: str,
    permanent_compose: str | None,
    permanent_project: str,
    rebuild: RebuildMode | None = None,
    keep_permanent: bool = False,
) -> tuple[str, str] | None:
    """Resolve sandbox specification for Task(sandbox=...).

    Per DEC-009, always returns ``("saber", compose_path)`` when the
    sandbox compose file exists.

    Args:
        domain_root: Domain directory path.
        sandbox_compose: Relative path to sandbox compose file
            (e.g. ``"compose/sandbox.compose.yml"``).
        permanent_compose: Relative path to permanent compose, or ``None``.
        permanent_project: Docker Compose project name for permanent services.
        rebuild: Which images to rebuild, or ``None`` to skip.
        keep_permanent: When True, permanent services are not stopped on
            task cleanup, allowing the next run to reuse them.

    Returns:
        ``("saber", compose_path)`` when sandbox compose exists,
        ``None`` if sandbox compose file doesn't exist.
    """
    sandbox_path = domain_root / sandbox_compose
    if not sandbox_path.exists():
        return None

    # Deferred import to break circular dependency:
    # saber.sandbox -> saber.environments.images -> (this __init__)
    from saber.sandbox import SaberSandboxEnvironment

    # Configure permanent services if compose file exists
    if permanent_compose:
        perm_path = domain_root / permanent_compose
        if perm_path.exists():
            SaberSandboxEnvironment.set_permanent_compose(
                perm_path,
                project=permanent_project,
                domain_root=domain_root,
            )
        else:
            logger.warning(
                "permanent_compose %s not found, ignoring",
                perm_path,
            )

    # Configure keep_permanent
    SaberSandboxEnvironment.set_keep_permanent(keep_permanent)

    # Configure preflight
    SaberSandboxEnvironment.set_preflight_config(domain_root, rebuild)

    # Always return "saber" — DEC-009
    return ("saber", str(sandbox_path))
