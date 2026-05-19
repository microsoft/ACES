# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Setup hook auto-discovery for SABER domains.

Discovers and loads ``SetupHook`` instances from a domain's ``setup.py``
module. Called by ``saber.task.create_task()`` during Phase 0 (before
config loading) to run data downloads, task generation, etc.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from saber.hooks import SetupHook
from saber.logging import get_logger

logger = get_logger(__name__)


def _discover_setup_hooks(
    domain_root: Path,
    kwargs: dict[str, object],
) -> list[SetupHook]:
    """Auto-discover domain setup hooks.

    Looks for ``<domain_root>/setup.py`` with a
    ``get_hooks(domain_root: Path, **kwargs) -> list[SetupHook]`` function
    and calls it, forwarding any extra keyword arguments (e.g. CLI ``-T``
    flags).

    Named parameters consumed by ``get_hooks()`` are **popped** from
    *kwargs* so they do not leak downstream (e.g. to the agent solver).

    Args:
        domain_root: Domain root directory.
        kwargs: Mutable dict of extra keyword arguments.  Keys consumed
            by the domain's ``get_hooks()`` are removed in-place.

    Returns:
        List of setup hooks, or empty list if none found.
    """
    setup_file = domain_root / "setup.py"
    if not setup_file.is_file():
        return []

    from saber.task import _import_domain_module

    module = _import_domain_module(domain_root, "setup")
    factory = getattr(module, "get_hooks", None)
    if factory is None:
        return []

    # Determine which kwargs the factory consumes by inspecting its
    # signature.  Named parameters (other than ``domain_root``) are
    # extracted from *kwargs* and popped after a successful call.
    sig = inspect.signature(factory)
    consumed: dict[str, object] = {}
    for name, param in sig.parameters.items():
        if name == "domain_root":
            continue
        if (
            param.kind
            in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and name in kwargs
        ):
            consumed[name] = kwargs[name]

    try:
        hooks: list[SetupHook] = factory(domain_root, **consumed)
    except TypeError:
        logger.warning(
            "get_hooks() in %s does not accept named kwargs; calling without extra arguments",
            setup_file,
        )
        hooks = factory(domain_root)
        consumed = {}

    # Pop consumed keys so they don't leak to the agent solver
    for key in consumed:
        kwargs.pop(key, None)

    logger.debug("Auto-discovered %d setup hooks from %s", len(hooks), setup_file)
    return hooks


def _discover_task_filter(
    domain_root: Path,
    dataset: str,
) -> str | None:
    """Discover a domain-specific task filter for a dataset name.

    Looks for ``get_task_filter(dataset: str) -> str | None`` in the
    domain's ``setup.py``.  Returns the filter string if found, or
    ``None`` if the function doesn't exist or returns ``None``.

    Note: Reuses the already-imported setup module from ``sys.modules``
    cache (``_import_domain_module`` was called earlier by
    ``_discover_setup_hooks``).  No redundant file I/O.

    Args:
        domain_root: Domain root directory.
        dataset: Dataset name to resolve (e.g. ``"lite"``).

    Returns:
        Task filter string, or ``None``.
    """
    setup_file = domain_root / "setup.py"
    if not setup_file.is_file():
        return None

    from saber.task import _import_domain_module

    module = _import_domain_module(domain_root, "setup")
    func = getattr(module, "get_task_filter", None)
    if func is None:
        return None

    result: str | None = func(dataset)
    if result is not None:
        logger.debug(
            "Dataset '%s' resolved to task_filter via get_task_filter(): %s...",
            dataset,
            result[:80],
        )
    return result
