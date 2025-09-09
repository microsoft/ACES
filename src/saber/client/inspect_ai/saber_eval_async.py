"""
SABER eval_async Integration

Main entrypoint for SABER execution using inspect_ai's eval_async framework.
This replaces the old container-based SABER harness with direct agent execution
using the new SABERAgent architecture and direct MCP integration.

Following SABER's new philosophy:
- Direct agent execution without container overhead
- Direct MCP integration with session/episode context
- Tool call limiting via inspect-ai capabilities
- Clean separation between infrastructure and agent logic
- Full backwards incompatibility for clean architecture
"""

import logging
from typing import Union

from inspect_ai import eval_async
from inspect_ai.log import EvalLog

from ..evaluation_orchestrator import SABEREvaluationOrchestrator
from ..models import SABERConfig

logger = logging.getLogger(__name__)


async def run_saber_eval_async(config: SABERConfig) -> Union[EvalLog, None]:
    """
    Main SABER entrypoint using inspect_ai eval_async with new orchestrator architecture.

    This replaces the container-based approach and provides:
    - Direct agent execution without container overhead
    - Direct MCP integration with session/episode context
    - Tool call limiting via inspect-ai capabilities
    - Full textual UI with progress bars and real-time logs
    - Dataset iteration over SABER tasks as inspect_ai Samples

    Uses the SABEREvaluationOrchestrator as a unified interface for all operations.

    Args:
        config: SABER configuration

    Returns:
        EvalLog with evaluation results

    Raises:
        ConfigurationValidationError: If configuration is invalid
        ServerConnectivityError: If SABER server is unreachable
        EvaluationExecutionError: If eval_async execution fails
    """

    logger.info("Starting SABER eval_async execution with orchestrator architecture")
    server_url = config.session_config.base_url if config.session_config else "unknown"
    logger.info(f"Server: {server_url}")
    logger.info(f"Agent: {config.agent_id or config.agent_path}")
    logger.info(f"Tasks: {config.task_ids or 'all available'}")

    # Initialize eval_kwargs early to prevent UnboundLocalError in exception handler
    eval_kwargs = {}

    # Create orchestrator and keep it alive for the entire evaluation
    orchestrator = SABEREvaluationOrchestrator(config)
    await orchestrator.__aenter__()

    try:
        # Discover tasks (either configured or all available)
        logger.info("Discovering tasks")
        task_ids = await orchestrator.discover_tasks()

        # Create inspect_ai dataset from SABER tasks
        logger.info("Creating dataset")
        dataset = await orchestrator.create_dataset(task_ids)
        logger.info(f"Created dataset with {len(dataset)} samples")

        # Create inspect_ai Task with SABER agent
        logger.info("Creating inspect_ai task")
        task = await orchestrator.create_task(dataset)

        # Configure eval_async parameters - simplified approach
        eval_kwargs = {
            "tasks": task,
            "model": config.model,
        }

        # Add optional parameters only if they're specified
        if config.model_args:
            eval_kwargs["model_args"] = config.model_args

        if hasattr(config, "log_dir") and config.log_dir:
            eval_kwargs["log_dir"] = config.log_dir

        if hasattr(config, "log_level") and config.log_level:
            eval_kwargs["log_level"] = config.log_level

        # Configure parallel execution
        if hasattr(config, "parallel_execution") and config.parallel_execution:
            eval_kwargs["max_tasks"] = getattr(config, "max_parallel_tasks", 4)

        # Configure sample limits
        if hasattr(config, "max_samples") and config.max_samples:
            eval_kwargs["max_samples"] = config.max_samples

        # Execute eval_async (inspect_ai specific, stays at entrypoint level)
        logger.info("Starting eval_async execution")
        eval_log = await eval_async(**eval_kwargs)
        logger.info("eval_async execution completed successfully")

        # Handle case where eval_async returns a list of logs
        if isinstance(eval_log, list):
            if len(eval_log) == 1:
                return eval_log[0]
            else:
                # Return the first log or combine them - this depends on your use case
                logger.warning(f"eval_async returned {len(eval_log)} logs, returning the first one")
                return eval_log[0] if eval_log else None

        return eval_log

    except Exception as e:
        logger.error(f"eval_async execution failed: {e}")
        # Use SABER's exception for consistency but don't import in orchestrator
        from ..exceptions import EvaluationExecutionError

        # Build error details with available context
        error_details = {
            "error_type": type(e).__name__,
            "server_url": config.session_config.base_url if config.session_config else "unknown",
            "agent_spec": config.agent_id or config.agent_path,
            "task_ids": config.task_ids,
        }

        # Add eval_config details if available
        if eval_kwargs:
            # Extract eval_kwargs as a separate field to avoid type issues
            eval_config_summary = {k: str(v) for k, v in eval_kwargs.items() if k != "tasks"}
            error_details["eval_config"] = str(eval_config_summary)

        raise EvaluationExecutionError(
            f"eval_async execution failed: {e}",
            details=error_details,
            suggestion="Check eval_async logs for detailed error information",
        ) from e

    finally:
        # Clean up orchestrator and session AFTER eval_async completes
        await orchestrator.__aexit__(None, None, None)
