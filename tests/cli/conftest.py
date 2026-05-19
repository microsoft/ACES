# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Shared fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from saber.cli.discovery import DiscoveredDomain


def make_domain(
    slug: str = "test",
    name: str = "Test",
    root: Path | None = None,
    permanent_compose: Path | None = None,
    permanent_project: str | None = None,
) -> DiscoveredDomain:
    """Build a ``DiscoveredDomain`` for testing.

    Args:
        slug: Domain slug.
        name: Human-readable name.
        root: Domain root path (defaults to ``/fake/<slug>``).
        permanent_compose: Path to permanent compose file.
        permanent_project: Compose project name for the permanent env.

    Returns:
        A frozen ``DiscoveredDomain`` instance.
    """
    return DiscoveredDomain(
        slug=slug,
        name=name,
        root=root or Path(f"/fake/{slug}"),
        permanent_compose=permanent_compose,
        permanent_project=permanent_project,
    )


def make_domain_with_permanent(
    slug: str = "excytin",
    name: str = "Excytin",
    permanent_env: str = "all_incidents",
) -> DiscoveredDomain:
    """Build a ``DiscoveredDomain`` that has a permanent environment configured.

    Args:
        slug: Domain slug.
        name: Human-readable name.
        permanent_env: Name used for the compose file (without ``.compose.yml``).

    Returns:
        A ``DiscoveredDomain`` with ``permanent_compose`` and ``permanent_project`` set.
    """
    root = Path(f"/fake/{slug}")
    return DiscoveredDomain(
        slug=slug,
        name=name,
        root=root,
        permanent_compose=root / "compose" / f"{permanent_env}.compose.yml",
        permanent_project=f"{slug}-databases",
    )


@pytest.fixture()
def domain() -> DiscoveredDomain:
    """A basic ``DiscoveredDomain`` with no permanent environment."""
    return make_domain()


@pytest.fixture()
def domain_with_permanent() -> DiscoveredDomain:
    """A ``DiscoveredDomain`` with a permanent environment."""
    return make_domain_with_permanent()
