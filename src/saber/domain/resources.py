"""Resource resolution for SABER domain orchestration.

This module handles loading packaged resources using importlib.resources
with fail-fast principles - no silent fallbacks that mask configuration issues.
"""

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator

from .exceptions import ResourceNotFoundError


@contextmanager
def resolve_compose_file(explicit_path: Path | None = None) -> Iterator[Path]:
    """Resolve docker-compose.yml file path that definitely exists on disk.

    Args:
        explicit_path: Optional explicit path override for testing/development

    Yields:
        Path: Absolute path to docker-compose.yml file on disk

    Raises:
        ResourceNotFoundError: If compose file cannot be found
    """
    if explicit_path:
        resolved = explicit_path.expanduser().resolve()
        if not resolved.exists():
            raise ResourceNotFoundError("docker-compose.yml", f"Explicit override not found: {resolved}")
        yield resolved
        return

    # Try packaged resource first (primary path for installed SABER)
    try:
        resource = files("saber.domain.package_resources").joinpath("docker-compose.yml")
        with as_file(resource) as packaged_path:
            if packaged_path.exists():
                yield packaged_path
                return
    except (ImportError, FileNotFoundError, AttributeError):
        # Package resource not available, try development fallback
        pass

    # Development fallback - only for editable installs
    dev_compose = _find_development_compose()
    if dev_compose and dev_compose.exists():
        yield dev_compose
        return

    # Fail fast - no silent failures
    raise ResourceNotFoundError(
        "docker-compose.yml",
        "Not found in package resources or development environment. "
        "Ensure SABER is properly installed or provide --compose-file explicitly.",
    )


@contextmanager
def resolve_schema_file(schema_name: str = "domain-manifest.schema.json") -> Iterator[Path]:
    """Resolve schema file path that definitely exists on disk.

    Args:
        schema_name: Name of the schema file to resolve

    Yields:
        Path: Absolute path to schema file on disk

    Raises:
        ResourceNotFoundError: If schema file cannot be found
    """
    # Try packaged resource first
    try:
        resource = files("saber.domain.package_resources.schemas").joinpath(schema_name)
        with as_file(resource) as packaged_path:
            if packaged_path.exists():
                yield packaged_path
                return
    except (ImportError, FileNotFoundError, AttributeError):
        # Package resource not available, try development fallback
        pass

    # Development fallback
    dev_schema = _find_development_schema(schema_name)
    if dev_schema and dev_schema.exists():
        yield dev_schema
        return

    # Fail fast
    raise ResourceNotFoundError(
        f"schema file '{schema_name}'",
        "Not found in package resources or development environment. " "Ensure SABER is properly installed.",
    )


def get_env_example_content() -> str:
    """Get the content of env.example template.

    Returns:
        str: Content of env.example file

    Raises:
        ResourceNotFoundError: If env.example cannot be found
    """
    # Try packaged resource first
    try:
        resource = files("saber.domain.package_resources").joinpath("env.example")
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    except (ImportError, FileNotFoundError, AttributeError):
        pass

    # Development fallback
    dev_env = _find_development_env_example()
    if dev_env and dev_env.exists():
        return dev_env.read_text(encoding="utf-8")

    # Fail fast
    raise ResourceNotFoundError(
        "env.example",
        "Not found in package resources or development environment. " "Ensure SABER is properly installed.",
    )


def _find_development_compose() -> Path | None:
    """Find docker-compose.yml in development environment.

    Only searches in well-known development locations to avoid
    silent configuration issues.
    """
    # Look for the compose file relative to this module's location
    # This works for editable installs where the source tree is available
    current_file = Path(__file__)

    # Navigate from external/saber/src/saber/domain/resources.py
    # to docker/docker-compose.yml
    repo_root = current_file.parents[5]  # Go up to oss_saber root
    compose_path = repo_root / "docker" / "docker-compose.yml"

    if compose_path.exists():
        return compose_path

    # Also check if we're in the wrong location and try to find it
    # by looking for a parent directory containing docker/docker-compose.yml
    for parent in current_file.parents:
        candidate = parent / "docker" / "docker-compose.yml"
        if candidate.exists():
            return candidate

    return None


def _find_development_schema(schema_name: str) -> Path | None:
    """Find schema file in development environment."""
    current_file = Path(__file__)

    # Navigate to schemas/ directory
    repo_root = current_file.parents[5]  # Go up to oss_saber root
    schema_path = repo_root / "schemas" / schema_name

    if schema_path.exists():
        return schema_path

    # Alternative search
    for parent in current_file.parents:
        candidate = parent / "schemas" / schema_name
        if candidate.exists():
            return candidate

    return None


def _find_development_env_example() -> Path | None:
    """Find env.example in development environment."""
    current_file = Path(__file__)

    # Navigate to docker/env.example
    repo_root = current_file.parents[5]  # Go up to oss_saber root
    env_path = repo_root / "docker" / "env.example"

    if env_path.exists():
        return env_path

    # Alternative search
    for parent in current_file.parents:
        candidate = parent / "docker" / "env.example"
        if candidate.exists():
            return candidate

    return None
