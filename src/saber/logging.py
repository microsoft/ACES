"""Thin logging wrapper that delegates to inspect_ai's log handler.

When running under inspect_ai, ``saber.*`` log messages propagate to
the root logger where inspect_ai's ``LogHandler`` routes them to:

* **Console** — controlled by ``INSPECT_LOG_LEVEL`` (default: ``warning``)
* **``.eval`` transcript** — controlled by ``INSPECT_LOG_LEVEL_TRANSCRIPT``
  (default: ``info``), visible in ``inspect view``
* **Trace file** — ``trace-<pid>.log``, always captured at TRACE level

When running standalone (e.g. ``saber`` CLI without inspect_ai), a
basic ``StreamHandler`` is added as fallback so messages reach stderr.

Usage::

    from saber.logging import configure_logging, get_logger

    configure_logging()             # auto-detect from env

    # In saber core (saber/task.py) — __name__ is already saber.task:
    logger = get_logger(__name__)

    # In a domain (domains/crsbench/setup.py) — auto-prefixed:
    logger = get_logger("domains.crsbench.setup")  # → saber.domains.crsbench.setup
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _has_inspect_handler() -> bool:
    """Return True if inspect_ai's ``LogHandler`` is on the root logger."""
    return any(type(h).__name__ == "LogHandler" for h in logging.getLogger().handlers)


def _resolve_level(explicit: int | None) -> int:
    """Resolve log level from *explicit* arg → ``SABER_LOG_LEVEL`` env → default.

    Default is ``DEBUG`` so that inspect_ai's handler decides what to
    display per destination (console=WARNING, transcript=INFO, trace=TRACE).
    """
    if explicit is not None:
        return explicit
    env = os.environ.get("SABER_LOG_LEVEL", "").strip().upper()
    if env and hasattr(logging, env):
        return int(getattr(logging, env))
    return logging.DEBUG


def configure_logging(level: int | None = None) -> None:
    """Configure all ``saber.*`` loggers.

    Under inspect_ai, messages propagate to the root logger where
    inspect_ai's ``LogHandler`` routes them to console, ``.eval``
    transcript, and trace files — so saber messages appear in
    ``inspect view``.

    Without inspect_ai, a ``StreamHandler`` is added as fallback.

    * **Idempotent** — safe to call multiple times; only the first call
      takes effect.

    Args:
        level: Logging level.  When *None*, resolved from
            ``SABER_LOG_LEVEL`` env var or defaults to ``DEBUG``.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    resolved = _resolve_level(level)
    saber_logger = logging.getLogger("saber")
    saber_logger.setLevel(resolved)

    if _has_inspect_handler():
        # Under inspect_ai: propagate to root → LogHandler handles
        # routing to console, .eval transcript, and trace file.
        saber_logger.propagate = True
    else:
        # Standalone: add a basic StreamHandler so messages reach stderr.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
        saber_logger.addHandler(handler)
        saber_logger.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the ``saber.*`` namespace.

    For saber core modules, ``__name__`` is already under ``saber.*``
    so pass it directly.  For domain modules outside the saber package,
    pass a descriptive name — it will be prefixed with ``saber.``
    automatically.

    Examples::

        # In saber core (e.g., saber/task.py):
        logger = get_logger(__name__)  # → "saber.task"

        # In a domain (e.g., domains/crsbench/setup.py):
        logger = get_logger("domains.crsbench.setup")  # → "saber.domains.crsbench.setup"

    Args:
        name: Logger name.  Prefixed with ``saber.`` if not already.

    Returns:
        A ``logging.Logger`` under the ``saber.*`` hierarchy.
    """
    if name.startswith("saber.") or name == "saber":
        return logging.getLogger(name)
    return logging.getLogger(f"saber.{name}")
