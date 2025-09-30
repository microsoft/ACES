"""
SABER Client Evaluation Runner

Reusable evaluation runner that properly integrates with inspect_ai's TUI display.
This module provides the core evaluation execution logic used by both the SABER
client CLI and the saber-domain test command.

Logging category: HARNESS
"""

import asyncio
from typing import Optional

from inspect_ai.log import EvalLog

from ..logging_config import LogCategory, get_saber_logger
from .models import SABERConfig

logger = get_saber_logger(LogCategory.HARNESS, __name__)


def run_saber_evaluation(config: SABERConfig, verbose: bool = False) -> Optional[EvalLog]:
    """
    Run SABER evaluation with proper inspect_ai TUI integration.

    This function handles the complete evaluation lifecycle including:
    - Setting up the inspect_ai task display
    - Running the evaluation through eval_async
    - Handling interrupts and errors gracefully

    This is the canonical way to run SABER evaluations from any CLI tool.

    Args:
        config: Fully validated and hydrated SABERConfig
        verbose: Enable verbose output

    Returns:
        EvalLog if evaluation completes, None otherwise

    Raises:
        Exception: Re-raises unexpected errors for debugging
    """

    if verbose:
        logger.info("Starting SABER evaluation with inspect_ai TUI integration")

    # Validate config has session config
    if not config.session_config:
        raise ValueError(
            "SABERConfig must have session_config initialized. "
            "Ensure server URLs are hydrated before calling run_saber_evaluation."
        )

    eval_result: Optional[EvalLog] = None

    # INSPECT-AI EVAL_ASYNC PATTERN - eval_async controls everything
    async def run_task_app() -> None:
        """Run SABER via inspect_ai eval_async for full UI and dataset iteration."""
        nonlocal eval_result

        logger.info("Starting eval_async task app")

        # Import inspect_ai modules only when needed
        from .inspect_ai import run_saber_eval_async

        # eval_async becomes the main entrypoint - handles UI, dataset iteration, everything
        eval_result = await run_saber_eval_async(config)
        logger.info("eval_async task app completed")

    try:
        logger.info("Starting inspect_ai task display")

        # Import inspect_ai display module only when needed
        from inspect_ai._display.core.active import display as task_display

        # Run the task app through inspect_ai's display system
        # This ensures proper TUI integration with rich progress bars, etc.
        task_display().run_task_app(run_task_app)

        logger.info("inspect_ai task display completed")

    except asyncio.CancelledError:
        logger.info("Task cancelled during shutdown", extra={"cause": "inspect_ai_shutdown"})

    except KeyboardInterrupt:
        logger.info("User interrupted execution", extra={"event": "keyboard_interrupt"})
        # Don't sys.exit here - let caller decide how to handle

    except Exception as exc:
        logger.exception("Unexpected error during execution", extra={"error": str(exc)})
        # Re-raise for debugging
        raise

    return eval_result
