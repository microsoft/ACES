"""
SABER eval_async Integration

Main entrypoint for SABER execution using inspect_ai's eval_async framework.

Following SABER best practices:
- Clean separation between orchestration (SABER) and evaluation (inspect_ai)
- Stateless orchestration service design
- Per-task agent assignment via meta-agent architecture
- Fail-fast design with clear error messages
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from inspect_ai import eval_async
from inspect_ai.log import EvalLog

from ..evaluation_orchestrator import SABEREvaluationOrchestrator
from ..models import SABERConfig

logger = logging.getLogger(__name__)


def discover_session_log_files(session_log_dir: str, session_id: str) -> list[str]:
    """
    Discover .eval log files in the session-specific log directory.

    Args:
        session_log_dir: Path to the session-specific log directory
        session_id: Session ID for logging context

    Returns:
        List of absolute paths to .eval files found in the directory
    """
    if not session_log_dir or not os.path.exists(session_log_dir):
        logger.warning(f"Session log directory does not exist: {session_log_dir}")
        return []

    try:
        log_dir_path = Path(session_log_dir)
        eval_files = list(log_dir_path.glob("*.eval"))
        eval_file_paths = [str(f.absolute()) for f in eval_files]

        logger.info(f"Discovered {len(eval_file_paths)} .eval files for session {session_id}")
        for file_path in eval_file_paths:
            logger.debug(f"Found eval file: {file_path}")

        return eval_file_paths
    except Exception as e:
        logger.error(f"Error discovering log files in {session_log_dir}: {e}")
        return []


async def upload_file_with_retry(
    session_manager: Any, session_id: str, file_path: str, max_retries: int = 3, timeout: Optional[float] = None
) -> dict:
    """
    Upload a file with retry logic for transient failures.

    Args:
        session_manager: ClientSessionManager instance
        session_id: Session ID for upload
        file_path: Path to file to upload
        max_retries: Maximum number of retry attempts
        timeout: Upload timeout in seconds

    Returns:
        Upload response dict

    Raises:
        Exception: If all retries fail
    """
    import asyncio
    from pathlib import Path

    filename = Path(file_path).name
    last_exception = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if attempt > 0:
                # Wait with exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** (attempt - 1)
                logger.info(
                    f"Retrying upload of {filename} (attempt {attempt + 1}/{max_retries + 1}) after {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

            upload_result = await session_manager.upload_evaluation_file(session_id, file_path, timeout=timeout)

            if attempt > 0:
                logger.info(f"Upload succeeded on retry {attempt} for {filename}")

            return dict(upload_result)

        except FileNotFoundError:
            # File not found errors shouldn't be retried
            raise

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            # Don't retry for certain permanent errors
            if any(
                permanent_error in error_msg
                for permanent_error in [
                    "session not found",
                    "invalid file",
                    "file too large",
                    "unauthorized",
                    "forbidden",
                    "method not allowed",
                ]
            ):
                logger.warning(f"Permanent error uploading {filename}, not retrying: {e}")
                raise

            # Log transient error and continue to retry
            if attempt < max_retries:
                logger.warning(f"Transient error uploading {filename} (attempt {attempt + 1}): {e}")
            else:
                logger.error(f"Upload failed after {max_retries + 1} attempts for {filename}: {e}")

    # All retries exhausted
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(f"Failed to upload {file_path} after {max_retries + 1} attempts")


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

    # Load environment variables early before any Task creation
    # This ensures Azure OpenAI and other LLM configs are available
    # Use explicit parameters to ensure loading works in Docker containers
    load_dotenv(dotenv_path=".env", override=True, verbose=True)

    logger.info("Starting SABER eval_async execution with orchestrator architecture")
    server_url = config.session_config.base_url if config.session_config else "unknown"
    logger.info(f"Server: {server_url}")
    logger.info(f"Agent assignments: {len(config.agents)} configured")
    logger.info(f"Tasks: {config.task_ids or 'all available'}")

    # Initialize eval_kwargs early to prevent UnboundLocalError in exception handler
    eval_kwargs: Dict[str, Any] = {}

    # Create orchestrator and keep it alive for the entire evaluation
    orchestrator = SABEREvaluationOrchestrator(config)
    await orchestrator.__aenter__()

    try:
        # Get session_id for session-specific log directory
        session_id = await orchestrator.get_session_id()
        logger.info(f"Using session ID for log organization: {session_id}")

        # Discover tasks (either configured or all available)
        logger.info("Discovering tasks")
        available_tasks = await orchestrator.discover_tasks()
        task_ids = [task.task_id for task in available_tasks]

        # Create inspect_ai dataset from SABER tasks
        logger.info("Creating dataset")
        dataset = await orchestrator.create_dataset(task_ids)
        logger.info(f"Created dataset with {len(dataset)} samples")

        # Create inspect_ai Tasks with agent-grouped multi-agent support
        logger.info("Creating inspect_ai tasks with agent-grouped architecture")
        tasks_to_run = await orchestrator.create_multi_task_evaluation(task_ids)

        # Configure eval_async parameters - no global model, agents specify their own
        eval_kwargs = {
            "tasks": tasks_to_run,
            # No global model - each agent specifies its own via meta-agent routing
            "log_level": "debug",  # Enable verbose logging to see more details
        }

        # Add optional parameters only if they're specified
        if config.model_args:
            eval_kwargs["model_args"] = config.model_args

        # Configure session-specific log directory
        session_log_dir = None
        if hasattr(config, "log_dir") and config.log_dir:
            # Create session-specific subdirectory
            session_log_dir = str(Path(config.log_dir) / session_id)
            eval_kwargs["log_dir"] = session_log_dir
            logger.info(f"Using session-specific log directory: {session_log_dir}")

            # Ensure the session log directory exists
            Path(session_log_dir).mkdir(parents=True, exist_ok=True)
        else:
            logger.warning("No log directory configured, eval logs will not be saved")

        # Use configured log level from config
        if hasattr(config, "log_level") and config.log_level:
            eval_kwargs["log_level"] = config.log_level.lower()
            logger.info(f"Using configured log level: {config.log_level}")
        else:
            # Use default log level if not specified
            eval_kwargs["log_level"] = "warning"  # Default to warning level
            logger.info("Using default log level: warning")

        # Configure parallel execution
        if hasattr(config, "parallel_execution") and config.parallel_execution:
            eval_kwargs["max_tasks"] = getattr(config, "max_parallel_tasks", 4)

        # Configure sample limits
        if hasattr(config, "max_samples") and config.max_samples:
            eval_kwargs["max_samples"] = config.max_samples

        # Execute eval_async (inspect_ai specific, stays at entrypoint level)
        logger.info("Starting eval_async execution")
        logger.info(f"📊 eval_async parameters: {eval_kwargs}")
        logger.info(f"📊 Number of tasks: {len(tasks_to_run)}")
        for i, task in enumerate(tasks_to_run):
            logger.info(
                f"📊 Task {i}: {getattr(task, 'name', 'unknown')} with {len(getattr(task, 'dataset', []))} samples"
            )
            logger.info(f"📊 Task {i} solver type: {type(getattr(task, 'solver', None))}")

        try:
            eval_log = await eval_async(**eval_kwargs)
            logger.info("eval_async execution completed successfully")
        except Exception as e:
            logger.error(f"❌ eval_async failed with error: {type(e).__name__}: {e}")
            import traceback

            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise

        # Upload log files to server if enabled and session-specific log directory was used
        if config.log_upload_enabled and session_log_dir and session_id:
            try:
                logger.info("Discovering and uploading eval log files...")
                log_files = discover_session_log_files(session_log_dir, session_id)

                if log_files:
                    logger.info(f"Found {len(log_files)} log files to upload")

                    # Upload each log file with retry logic
                    upload_results = []
                    for log_file_path in log_files:
                        logger.info(f"Uploading log file: {log_file_path}")

                        try:
                            # Get the session manager for upload
                            session_manager = orchestrator._session_manager
                            if session_manager:
                                # Upload file using the REST API client method with configured retry count
                                result = await upload_file_with_retry(
                                    session_manager,
                                    session_id,
                                    log_file_path,
                                    max_retries=config.log_upload_max_retries,
                                    timeout=config.log_upload_timeout,
                                )
                                upload_results.append(
                                    {"file_path": log_file_path, "status": "success", "result": result}
                                )
                                logger.info(f"Successfully uploaded: {log_file_path}")
                            else:
                                logger.warning("Session manager not available for upload")
                                upload_results.append(
                                    {
                                        "file_path": log_file_path,
                                        "status": "failed",
                                        "error": "Session manager not available",
                                        "error_type": "configuration",
                                    }
                                )
                        except FileNotFoundError as e:
                            logger.error(f"Eval file not found: {log_file_path}")
                            upload_results.append(
                                {
                                    "file_path": log_file_path,
                                    "status": "failed",
                                    "error": str(e),
                                    "error_type": "file_not_found",
                                }
                            )
                        except ConnectionError as e:
                            logger.error(f"Network error uploading {log_file_path}: {e}")
                            upload_results.append(
                                {
                                    "file_path": log_file_path,
                                    "status": "failed",
                                    "error": str(e),
                                    "error_type": "network",
                                }
                            )
                        except Exception as upload_error:
                            logger.error(f"Failed to upload {log_file_path}: {upload_error}")
                            upload_results.append(
                                {
                                    "file_path": log_file_path,
                                    "status": "failed",
                                    "error": str(upload_error),
                                    "error_type": "unknown",
                                }
                            )

                    # Log upload summary with error type breakdown
                    successful_uploads = [r for r in upload_results if r["status"] == "success"]
                    failed_uploads = [r for r in upload_results if r["status"] == "failed"]

                    logger.info(f"Upload summary: {len(successful_uploads)} successful, {len(failed_uploads)} failed")

                    if failed_uploads:
                        # Categorize failures for better debugging
                        error_types: dict[str, list] = {}
                        for failed in failed_uploads:
                            error_type = str(failed.get("error_type", "unknown"))
                            if error_type not in error_types:
                                error_types[error_type] = []
                            error_types[error_type].append(failed)

                        logger.warning("Log file upload failures by type:")
                        for error_type, failures in error_types.items():
                            logger.warning(f"  {error_type}: {len(failures)} files")
                            for failed in failures:
                                logger.warning(f"    {failed['file_path']}: {failed['error']}")

                        # Handle failure behavior based on configuration
                        total_files = len(upload_results)
                        failure_rate = len(failed_uploads) / total_files

                        if config.log_upload_fail_on_error:
                            # Strict mode - fail evaluation if any upload fails
                            if failed_uploads:
                                error_msg = f"Log upload failed for {len(failed_uploads)}/{total_files} files"
                                logger.error(f"{error_msg} and log_upload_fail_on_error=True")
                                raise RuntimeError(
                                    f"{error_msg}. Set log_upload_fail_on_error=False to ignore upload failures."
                                )
                        else:
                            # Lenient mode - warn but continue
                            if failure_rate > 0.5:  # More than 50% failed
                                logger.warning(f"High failure rate in log upload: {failure_rate:.1%} of files failed")
                                logger.warning("Consider checking network connectivity and server status")
                            else:
                                logger.info(
                                    f"Partial log upload success: "
                                    f"{len(successful_uploads)}/{total_files} files uploaded"
                                )

                else:
                    logger.warning(f"No eval log files found in session directory: {session_log_dir}")
                    logger.info("This may be normal if evaluation completed without generating log files")

            except Exception as e:
                logger.error(f"Error during log file upload process: {e}")

                # Handle failure behavior based on configuration
                if config.log_upload_fail_on_error:
                    logger.error("log_upload_fail_on_error=True, failing evaluation due to upload error")
                    raise RuntimeError(f"Log upload process failed: {e}") from e
                else:
                    # Determine if this is a critical error or can be ignored
                    if "session" in str(e).lower() and "not found" in str(e).lower():
                        logger.error(
                            "Session not found during upload - this indicates a serious session management issue"
                        )
                    elif "permission" in str(e).lower() or "access" in str(e).lower():
                        logger.error("File permission error during upload - check file system permissions")
                    else:
                        logger.warning("Non-critical error in log upload process")

                    # Don't fail the entire evaluation if log upload fails
                    logger.warning("Continuing evaluation despite log upload failure")
        elif not config.log_upload_enabled:
            logger.info("Log upload disabled by configuration (log_upload_enabled=False)")
        else:
            logger.info("Log upload skipped - no session log directory available")

        # Store session context for potential log file discovery
        if session_id and eval_log:
            # Add session metadata to eval_log for future reference
            if isinstance(eval_log, list):
                for log in eval_log:
                    if hasattr(log, "eval") and hasattr(log.eval, "metadata"):
                        log.eval.metadata = log.eval.metadata or {}
                        log.eval.metadata["saber_session_id"] = session_id
                        if session_log_dir:
                            log.eval.metadata["saber_session_log_dir"] = session_log_dir
            else:
                if hasattr(eval_log, "eval") and hasattr(eval_log.eval, "metadata"):
                    eval_log.eval.metadata = eval_log.eval.metadata or {}
                    eval_log.eval.metadata["saber_session_id"] = session_id
                    if session_log_dir:
                        eval_log.eval.metadata["saber_session_log_dir"] = session_log_dir

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
            "agent_assignments": len(config.agents),
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
