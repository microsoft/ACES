"""Setup hook auto-discovery for SABER domains.

TODO: Merge this function into ``saber.task`` alongside ``_discover_tools()``
and ``_discover_strategies()`` when landing Phase 2 (create_task integration).
See ``docs/design/setup_hook_auto_discovery.md`` for the full plan.
"""

from __future__ import annotations

from pathlib import Path

from saber.hooks import SetupHook
from saber.logging import get_logger
from saber.task import _import_domain_module

logger = get_logger(__name__)


def _discover_setup_hooks(
    domain_root: Path, **kwargs: object
) -> list[SetupHook]:
    """Auto-discover domain setup hooks.

    Looks for ``<domain_root>/setup.py`` with a
    ``get_hooks(domain_root: Path, **kwargs) -> list[SetupHook]`` function
    and calls it, forwarding any extra keyword arguments (e.g. CLI ``-T``
    flags).

    Args:
        domain_root: Domain root directory.
        **kwargs: Additional keyword arguments forwarded to the domain's
            ``get_hooks()`` function.  If the function does not accept
            ``**kwargs``, they are silently dropped for backward
            compatibility.

    Returns:
        List of setup hooks, or empty list if none found.
    """
    setup_file = domain_root / "setup.py"
    if not setup_file.is_file():
        return []

    module = _import_domain_module(domain_root, "setup")
    factory = getattr(module, "get_hooks", None)
    if factory is None:
        return []

    try:
        hooks: list[SetupHook] = factory(domain_root, **kwargs)
    except TypeError:
        logger.warning(
            "get_hooks() in %s does not accept **kwargs; "
            "calling without extra arguments",
            setup_file,
        )
        hooks = factory(domain_root)

    logger.debug(
        "Auto-discovered %d setup hooks from %s", len(hooks), setup_file
    )
    return hooks
