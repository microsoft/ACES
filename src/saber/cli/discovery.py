"""Domain discovery utilities for the SABER CLI."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from saber.config.loader import load_domain_config
from saber.logging import get_logger

logger = get_logger(__name__)


class ComposeProject(BaseModel):
    """A running Docker Compose project."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: str


class DiscoveredDomain(BaseModel):
    """A domain directory discovered on disk."""

    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    root: Path
    permanent_compose: Path | None = None
    permanent_project: str | None = None


def find_domains_root(start: Path | None = None) -> Path:
    """Walk up from *start* (default: CWD) to find a ``domains/`` directory.

    The ``domains/`` directory must contain at least one subdirectory
    with an ``eval.yaml`` file.

    Args:
        start: Starting directory. Defaults to ``Path.cwd()``.

    Returns:
        Absolute path to the ``domains/`` directory.

    Raises:
        FileNotFoundError: If no qualifying ``domains/`` directory is found.
    """
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / "domains"
        if candidate.is_dir():
            # Check that at least one subdirectory has eval.yaml
            for child in candidate.iterdir():
                if child.is_dir() and (child / "eval.yaml").is_file():
                    return candidate
        # Also check if current IS the domains dir
        if current.name == "domains":
            for child in current.iterdir():
                if child.is_dir() and (child / "eval.yaml").is_file():
                    return current
    raise FileNotFoundError("No domains/ directory found. Run from within the SABER workspace.")


def discover_domains(domains_root: Path) -> list[DiscoveredDomain]:
    """Scan *domains_root* for subdirectories with ``eval.yaml``.

    For each domain, resolve permanent compose from
    ``tasks/global.yaml``'s ``permanent_environment`` key.

    Args:
        domains_root: Path to the ``domains/`` directory.

    Returns:
        List of discovered domains, sorted by slug.
    """
    domains: list[DiscoveredDomain] = []
    for child in sorted(domains_root.iterdir()):
        if not child.is_dir():
            continue
        eval_path = child / "eval.yaml"
        if not eval_path.is_file():
            continue
        try:
            config = load_domain_config(child)
        except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"Domain {child.name!r} has invalid eval.yaml: {exc}") from exc

        perm_compose, perm_project = _resolve_permanent(child, config.slug)
        domains.append(
            DiscoveredDomain(
                slug=config.slug,
                name=config.name,
                root=child,
                permanent_compose=perm_compose,
                permanent_project=perm_project,
            )
        )
    return domains


def resolve_domain(domains_root: Path, domain_name: str) -> DiscoveredDomain:
    """Find a single domain by slug or directory name.

    Args:
        domains_root: Path to the ``domains/`` directory.
        domain_name: Domain slug or directory name to find.

    Returns:
        The matching :class:`DiscoveredDomain`.

    Raises:
        typer.BadParameter: If no matching domain is found.
    """
    import typer

    all_domains = discover_domains(domains_root)
    for domain in all_domains:
        if domain.slug == domain_name or domain.root.name == domain_name:
            return domain
    available = ", ".join(d.slug for d in all_domains) or "(none)"
    raise typer.BadParameter(f"Domain {domain_name!r} not found. Available: {available}")


def _resolve_permanent(domain_root: Path, slug: str) -> tuple[Path | None, str | None]:
    """Resolve permanent compose file from global.yaml.

    Args:
        domain_root: Path to the domain directory.
        slug: Domain slug.

    Returns:
        Tuple of (compose_path, project_name) or (None, None).
    """
    global_yaml = domain_root / "tasks" / "global.yaml"
    if not global_yaml.is_file():
        return None, None
    try:
        with global_yaml.open() as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {global_yaml}: {exc}") from exc
    # New nested format: permanent_environment: {name: ..., compose: ...}
    # Also supports reading from global_defaults section
    perm_env = data.get("permanent_environment")
    if not perm_env:
        gd = data.get("global_defaults", {})
        perm_env = gd.get("permanent_environment") if isinstance(gd, dict) else None
    if not perm_env or not isinstance(perm_env, dict):
        return None, None
    compose_rel = perm_env.get("compose")
    if not compose_rel:
        return None, None
    compose_path = domain_root / compose_rel
    if not compose_path.is_file():
        return None, None
    project_name = perm_env.get("name", f"{slug}-permanent")
    return compose_path, project_name
