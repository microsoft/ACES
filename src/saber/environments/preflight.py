# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Compose file preflight validation.

Validates Docker Compose YAML files for structural correctness and
security resource limits before evaluation starts.  All validation
is synchronous and read-only — no Docker daemon calls.

See also :mod:`saber.environments.images` for image-build preflight
(a separate, complementary system).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from saber.logging import get_logger

__all__ = [
    "ComposePreflightResult",
    "PreflightFinding",
    "validate_compose_file",
    "validate_domain",
    "validate_network_references",
]

logger = get_logger(__name__)

_SABER_PROJECT_PATTERN = re.compile(r"\$\{SABER_PROJECT\}_(.+)")


class PreflightFinding(BaseModel):
    """A single compose file validation finding."""

    model_config = ConfigDict(frozen=True)

    path: str
    message: str
    severity: Literal["error", "warning"]


class ComposePreflightResult(BaseModel):
    """Aggregated result of compose file preflight validation."""

    model_config = ConfigDict(frozen=True)

    findings: tuple[PreflightFinding, ...] = ()

    @property
    def passed(self) -> bool:
        """True when there are no error-severity findings."""
        return not any(f.severity == "error" for f in self.findings)

    @property
    def errors(self) -> tuple[PreflightFinding, ...]:
        """All error-severity findings."""
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[PreflightFinding, ...]:
        """All warning-severity findings."""
        return tuple(f for f in self.findings if f.severity == "warning")


def validate_compose_file(
    path: Path,
    domain_root: Path,
    *,
    is_permanent: bool = False,
) -> list[PreflightFinding]:
    """Validate a single compose file for syntax and resource limits.

    Checks:
    1. YAML parses without errors
    2. Top-level ``services`` key exists
    3. Each service has ``deploy.resources.limits.pids`` (warning if missing,
       skipped when *is_permanent* is True)

    Args:
        path: Absolute path to the compose YAML file.
        domain_root: Domain root for computing relative paths.
        is_permanent: When True, skip pids-limit checks.

    Returns:
        List of findings (may be empty).
    """
    rel_path = str(path.relative_to(domain_root))
    findings: list[PreflightFinding] = []

    # 1. Parse YAML
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        findings.append(
            PreflightFinding(
                path=rel_path,
                message=f"YAML parse error: {exc}",
                severity="error",
            )
        )
        return findings

    # 2. Check for empty file
    if data is None:
        findings.append(
            PreflightFinding(
                path=rel_path,
                message="File is empty or contains only comments",
                severity="error",
            )
        )
        return findings

    # 3. Check for 'services' key
    if not isinstance(data, dict) or "services" not in data:
        findings.append(
            PreflightFinding(
                path=rel_path,
                message="Missing required 'services' key",
                severity="error",
            )
        )
        return findings

    # 4. Check pids limits (skip for permanent services)
    if not is_permanent:
        services = data.get("services") or {}
        for svc_name, svc_config in services.items():
            if not isinstance(svc_config, dict):
                continue
            deploy = svc_config.get("deploy") or {}
            resources = deploy.get("resources") if isinstance(deploy, dict) else None
            limits = resources.get("limits") if isinstance(resources, dict) else None
            pids = limits.get("pids") if isinstance(limits, dict) else None
            if pids is None:
                findings.append(
                    PreflightFinding(
                        path=rel_path,
                        message=f"Service '{svc_name}' missing deploy.resources.limits.pids",
                        severity="warning",
                    )
                )

    return findings


def validate_network_references(
    sandbox_path: Path,
    permanent_path: Path,
    domain_root: Path,
) -> list[PreflightFinding]:
    """Cross-validate sandbox network references against permanent compose.

    Checks that every ``external: true`` network in the sandbox compose
    whose name matches ``${SABER_PROJECT}_<net>`` has a corresponding
    ``<net>`` defined in the permanent compose's top-level networks.

    Args:
        sandbox_path: Absolute path to sandbox compose file.
        permanent_path: Absolute path to permanent compose file.
        domain_root: Domain root for computing relative paths.

    Returns:
        List of findings (may be empty).
    """
    sandbox_rel = str(sandbox_path.relative_to(domain_root))
    findings: list[PreflightFinding] = []

    # Parse sandbox compose
    try:
        with open(sandbox_path) as f:
            sandbox_data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        # Syntax errors are caught by validate_compose_file; skip here
        return findings

    # Extract SABER_PROJECT-prefixed external network names
    networks = sandbox_data.get("networks") or {}
    if not isinstance(networks, dict):
        return findings
    saber_nets: list[str] = []
    for _net_key, net_config in networks.items():
        if not isinstance(net_config, dict):
            continue
        if not net_config.get("external"):
            continue
        name = net_config.get("name", "")
        match = _SABER_PROJECT_PATTERN.match(str(name))
        if match:
            saber_nets.append(match.group(1))

    if not saber_nets:
        return findings

    # Check permanent compose exists
    if not permanent_path.exists():
        findings.append(
            PreflightFinding(
                path=sandbox_rel,
                message=f"Permanent compose file not found: {permanent_path.name}",
                severity="error",
            )
        )
        return findings

    # Parse permanent compose
    try:
        with open(permanent_path) as f:
            permanent_data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        findings.append(
            PreflightFinding(
                path=str(permanent_path.relative_to(domain_root)),
                message="Failed to parse permanent compose file",
                severity="error",
            )
        )
        return findings

    # Get permanent network names
    perm_networks = permanent_data.get("networks") or {}
    permanent_networks = set(perm_networks.keys()) if isinstance(perm_networks, dict) else set()

    # Cross-reference
    for net_name in saber_nets:
        if net_name not in permanent_networks:
            findings.append(
                PreflightFinding(
                    path=sandbox_rel,
                    message=(
                        f"Network '${{SABER_PROJECT}}_{net_name}' references "
                        f"'{net_name}' which is not defined in permanent compose"
                    ),
                    severity="error",
                )
            )

    return findings


def validate_domain(
    domain_root: Path,
    sandbox_compose: str = "compose/sandbox.compose.yml",
    permanent_compose: str | None = None,
) -> ComposePreflightResult:
    """Run all preflight validation for a domain.

    Validates:
    1. Sandbox compose file syntax + pids limits
    2. Permanent compose file syntax (pids exempt) if provided
    3. Network cross-references between sandbox and permanent

    Args:
        domain_root: Path to the domain root directory.
        sandbox_compose: Relative path to sandbox compose file within domain.
        permanent_compose: Relative path to permanent compose file, or None.

    Returns:
        Aggregated preflight result with all findings.
    """
    findings: list[PreflightFinding] = []
    sandbox_path = domain_root / sandbox_compose

    if not sandbox_path.exists():
        findings.append(
            PreflightFinding(
                path=sandbox_compose,
                message=f"Sandbox compose file not found: {sandbox_compose}",
                severity="error",
            )
        )
        return ComposePreflightResult(findings=tuple(findings))

    # 1. Validate sandbox compose
    findings.extend(validate_compose_file(sandbox_path, domain_root))

    # 2. Validate permanent compose (if provided)
    if permanent_compose is not None:
        permanent_path = domain_root / permanent_compose
        if permanent_path.exists():
            findings.extend(validate_compose_file(permanent_path, domain_root, is_permanent=True))
        # 3. Cross-validate network references
        findings.extend(validate_network_references(sandbox_path, permanent_path, domain_root))

    result = ComposePreflightResult(findings=tuple(findings))

    # Log summary
    if result.errors:
        logger.error(
            "Compose preflight failed with %d error(s): %s",
            len(result.errors),
            "; ".join(e.message for e in result.errors),
        )
    if result.warnings:
        logger.warning(
            "Compose preflight: %d warning(s): %s",
            len(result.warnings),
            "; ".join(w.message for w in result.warnings),
        )
    if result.passed and not result.warnings:
        logger.info("Compose preflight passed with no issues")

    return result
