"""
SABER eval_async Integration

Main entrypoint for SABER execution using inspect_ai's eval_async framework.

Logging category: EVALUATION.

Following SABER best practices:
- Clean separation between orchestration (SABER) and evaluation (inspect_ai)
- Stateless orchestration service design
- Per-task agent assignment via meta-agent architecture
- Fail-fast design with clear error messages
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from inspect_ai import eval_async
from inspect_ai.log import EvalLog

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_context,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ..evaluation_orchestrator import SABEREvaluationOrchestrator
from ..models import SABERConfig

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


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
        logger.warning(
            "Session log directory missing",
            extra={
                "event": "session_log_directory_missing",
                "session_log_dir": session_log_dir,
                "session_id": session_id,
            },
        )
        return []

    try:
        log_dir_path = Path(session_log_dir)
        eval_files = list(log_dir_path.glob("*.eval"))
        eval_file_paths = [str(f.absolute()) for f in eval_files]

        logger.info(
            "Session eval files discovered",
            extra={
                "event": "session_eval_files_discovered",
                "session_log_dir": session_log_dir,
                "session_id": session_id,
                "file_count": len(eval_file_paths),
            },
        )
        for file_path in eval_file_paths:
            logger.debug(
                "Eval file located",
                extra={
                    "event": "session_eval_file_found",
                    "session_id": session_id,
                    "file_path": file_path,
                },
            )

        return eval_file_paths
    except Exception as exc:
        logger.error(
            "Session eval file discovery failed",
            extra={
                "event": "session_eval_file_discovery_failed",
                "session_log_dir": session_log_dir,
                "session_id": session_id,
                "error": str(exc),
            },
        )
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
    last_exception: Exception | None = None

    log_operation_start(
        logger,
        "log_file_upload",
        session_id=session_id,
        file_path=file_path,
        max_retries=max_retries,
        timeout=timeout,
    )

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if attempt > 0:
                wait_time = 2 ** (attempt - 1)
                logger.info(
                    "Retrying log file upload",
                    extra={
                        "event": "log_file_upload_retry",
                        "session_id": session_id,
                        "file_path": file_path,
                        "filename": filename,
                        "attempt": attempt + 1,
                        "max_attempts": max_retries + 1,
                        "backoff_seconds": wait_time,
                    },
                )
                await asyncio.sleep(wait_time)

            upload_result = await session_manager.upload_evaluation_file(session_id, file_path, timeout=timeout)

            if attempt > 0:
                logger.info(
                    "Log file upload succeeded after retry",
                    extra={
                        "event": "log_file_upload_success_after_retry",
                        "session_id": session_id,
                        "file_path": file_path,
                        "filename": filename,
                        "retry_attempt": attempt,
                    },
                )

            log_operation_success(
                logger,
                "log_file_upload",
                session_id=session_id,
                file_path=file_path,
                filename=filename,
                attempts_used=attempt + 1,
            )
            return dict(upload_result)

        except FileNotFoundError as exc:
            log_operation_failure(
                logger,
                "log_file_upload",
                exc,
                session_id=session_id,
                file_path=file_path,
                filename=filename,
                attempt=attempt + 1,
            )
            raise

        except Exception as exc:
            last_exception = exc
            error_msg = str(exc).lower()

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
                logger.warning(
                    "Permanent log upload error encountered",
                    extra={
                        "event": "log_file_upload_permanent_error",
                        "session_id": session_id,
                        "file_path": file_path,
                        "filename": filename,
                        "attempt": attempt + 1,
                        "error": str(exc),
                    },
                )
                log_operation_failure(
                    logger,
                    "log_file_upload",
                    exc,
                    session_id=session_id,
                    file_path=file_path,
                    filename=filename,
                    attempt=attempt + 1,
                    error_type="permanent",
                )
                raise

            if attempt < max_retries:
                logger.warning(
                    "Transient log upload error",
                    extra={
                        "event": "log_file_upload_transient_error",
                        "session_id": session_id,
                        "file_path": file_path,
                        "filename": filename,
                        "attempt": attempt + 1,
                        "max_attempts": max_retries + 1,
                        "error": str(exc),
                    },
                )
            else:
                logger.error(
                    "Log file upload exhausted retries",
                    extra={
                        "event": "log_file_upload_max_retries_exhausted",
                        "session_id": session_id,
                        "file_path": file_path,
                        "filename": filename,
                        "attempts": max_retries + 1,
                        "error": str(exc),
                    },
                )

    final_error = last_exception or RuntimeError(f"Failed to upload {file_path} after {max_retries + 1} attempts")
    log_operation_failure(
        logger,
        "log_file_upload",
        final_error,
        session_id=session_id,
        file_path=file_path,
        filename=filename,
        attempts_used=max_retries + 1,
    )
    raise final_error


async def run_saber_eval_async(config: SABERConfig) -> Union[EvalLog, None]:
    """
    Execute SABER evaluations through inspect_ai's eval_async pipeline.

    Args:
        config: Fully validated SABER configuration

    Returns:
        EvalLog or None depending on eval_async output format.
    """

    log_operation_start(
        logger,
        "run_eval_async",
        agent_count=len(config.agents),
        requested_tasks=config.task_ids or "all",
        log_upload_enabled=config.log_upload_enabled,
    )

    server_url = config.session_config.base_url if config.session_config else "unknown"
    logger.info(
        "eval_async run initialized",
        extra={
            "event": "eval_async_run_initialized",
            "server_url": server_url,
            "agent_count": len(config.agents),
            "requested_tasks": config.task_ids or "all",
        },
    )

    eval_kwargs: Dict[str, Any] = {}
    orchestrator = SABEREvaluationOrchestrator(config)
    await orchestrator.__aenter__()

    session_id: Optional[str] = None
    session_log_dir: Optional[str] = None
    eval_log: Union[EvalLog, list[EvalLog], None] = None

    try:
        session_id = await orchestrator.get_session_id()

        with log_context(session_id=session_id):
            logger.info(
                "Session context established",
                extra={
                    "event": "eval_async_session_context",
                    "session_id": session_id,
                },
            )

            available_tasks = await orchestrator.discover_tasks()
            task_ids = [task.task_id for task in available_tasks]
            logger.info(
                "Tasks discovered for evaluation",
                extra={
                    "event": "eval_async_tasks_discovered",
                    "task_ids": task_ids,
                    "task_count": len(task_ids),
                },
            )

            dataset = await orchestrator.create_dataset(task_ids)
            logger.info(
                "Dataset materialized",
                extra={
                    "event": "eval_async_dataset_created",
                    "sample_count": len(dataset),
                },
            )

            tasks_to_run = await orchestrator.create_multi_task_evaluation(task_ids)
            logger.info(
                "inspect_ai tasks prepared",
                extra={
                    "event": "eval_async_tasks_prepared",
                    "task_count": len(tasks_to_run),
                },
            )

            eval_kwargs = {
                "tasks": tasks_to_run,
                "log_level": "debug",
            }

            if config.model_args:
                eval_kwargs["model_args"] = config.model_args

            log_dir_value = config.log_dir
            logger.info(
                "Log directory configuration check",
                extra={
                    "event": "eval_async_log_dir_check",
                    "log_dir_value": log_dir_value,
                    "config_log_dir": config.log_dir,
                    "domain": getattr(config, "domain", None),
                    "config_domain": config.domain,
                    "session_id": session_id,
                },
            )
            if log_dir_value:
                base_log_path = Path(log_dir_value)

                # Use domain-aware directory structure if domain is specified
                if config.domain:
                    # For inspect-ai .eval files, we want them in logs/{domain}/{session_id}/
                    # not in logs/{domain}/client-logs/{domain}/{session_id}/
                    # So we use the domain logs root, not the client-logs subdirectory
                    domain_log_root = Path("logs") / config.domain
                    session_log_path = domain_log_root / session_id
                    logger.info(
                        "Domain-aware log directory configured",
                        extra={
                            "event": "eval_async_domain_log_directory",
                            "domain": config.domain,
                            "eval_log_path": str(session_log_path),
                        },
                    )
                else:
                    # logs/{session_id}/ (legacy behavior)
                    session_log_path = base_log_path / session_id

                eval_kwargs["log_dir"] = str(session_log_path)
                session_log_dir = str(session_log_path)  # Set for log upload discovery
                session_log_path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Session-specific log directory prepared",
                    extra={
                        "event": "eval_async_log_directory_prepared",
                        "session_log_dir": str(session_log_path),
                        "domain": config.domain,
                    },
                )
            else:
                logger.warning(
                    "No log directory configured; eval logs will not persist",
                    extra={"event": "eval_async_log_directory_missing"},
                )

            if getattr(config, "log_level", None):
                eval_kwargs["log_level"] = config.log_level.lower()
                logger.info(
                    "Using configured eval_async log level",
                    extra={
                        "event": "eval_async_log_level_configured",
                        "log_level": config.log_level,
                    },
                )
            else:
                eval_kwargs["log_level"] = "warning"
                logger.info(
                    "Default eval_async log level applied",
                    extra={
                        "event": "eval_async_log_level_default",
                        "log_level": "warning",
                    },
                )

            if getattr(config, "parallel_execution", False):
                eval_kwargs["max_tasks"] = getattr(config, "max_parallel_tasks", 4)
                logger.info(
                    "Parallel execution configured",
                    extra={
                        "event": "eval_async_parallel_execution_configured",
                        "max_tasks": eval_kwargs["max_tasks"],
                    },
                )

            if getattr(config, "max_samples", None):
                eval_kwargs["max_samples"] = config.max_samples
                logger.info(
                    "Sample limit applied",
                    extra={
                        "event": "eval_async_sample_limit_set",
                        "max_samples": config.max_samples,
                    },
                )

            # Configure display mode based on ui_enabled setting
            ui_enabled = getattr(config, "ui_enabled", True)
            if ui_enabled:
                eval_kwargs["display"] = "full"  # Enable full TUI display
                logger.info(
                    "Full TUI display enabled",
                    extra={
                        "event": "eval_async_ui_enabled",
                        "display_mode": "full",
                    },
                )
            else:
                eval_kwargs["display"] = "rich"  # Use rich console output without TUI
                logger.info(
                    "Rich console display enabled",
                    extra={
                        "event": "eval_async_ui_disabled",
                        "display_mode": "rich",
                    },
                )

            logger.debug(
                "eval_async invocation parameters prepared",
                extra={
                    "event": "eval_async_parameters_prepared",
                    "eval_kwargs": {k: v for k, v in eval_kwargs.items() if k != "tasks"},
                    "task_count": len(tasks_to_run),
                },
            )

            log_operation_start(
                logger,
                "eval_async_execution",
                session_id=session_id,
                task_count=len(tasks_to_run),
                parameters={k: v for k, v in eval_kwargs.items() if k != "tasks"},
            )

            try:
                eval_log = await eval_async(**eval_kwargs)
                log_operation_success(
                    logger,
                    "eval_async_execution",
                    session_id=session_id,
                    task_count=len(tasks_to_run),
                )
            except Exception as exec_exc:
                log_operation_failure(
                    logger,
                    "eval_async_execution",
                    exec_exc,
                    session_id=session_id,
                    task_count=len(tasks_to_run),
                )
                import traceback

                logger.error(
                    "eval_async execution failed",
                    extra={
                        "event": "eval_async_execution_failed",
                        "error": str(exec_exc),
                        "exception_type": type(exec_exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
                raise

            # Post-evaluation: Move .eval files to domain-specific directory if domain is configured
            # Note: _move_eval_files_to_domain_directory not implemented yet
            # if config.domain and log_dir_value:
            #     _move_eval_files_to_domain_directory(config, session_id, logger)

            if config.log_upload_enabled and session_log_dir:
                try:
                    logger.info(
                        "Uploading eval log files",
                        extra={
                            "event": "eval_async_log_upload_start",
                            "session_log_dir": session_log_dir,
                        },
                    )
                    log_files = discover_session_log_files(session_log_dir, session_id)

                    if log_files:
                        upload_results: list[dict[str, Any]] = []
                        for log_file_path in log_files:
                            logger.debug(
                                "Uploading log file",
                                extra={
                                    "event": "eval_async_log_upload_file",
                                    "file_path": log_file_path,
                                },
                            )
                            session_manager = orchestrator._session_manager
                            if session_manager:
                                try:
                                    result = await upload_file_with_retry(
                                        session_manager,
                                        session_id,
                                        log_file_path,
                                        max_retries=config.log_upload_max_retries,
                                        timeout=config.log_upload_timeout,
                                    )
                                    upload_results.append(
                                        {
                                            "file_path": log_file_path,
                                            "status": "success",
                                            "result": result,
                                        }
                                    )
                                    logger.info(
                                        "Log file uploaded",
                                        extra={
                                            "event": "eval_async_log_upload_success",
                                            "file_path": log_file_path,
                                        },
                                    )
                                except Exception as upload_exc:
                                    upload_results.append(
                                        {
                                            "file_path": log_file_path,
                                            "status": "failed",
                                            "error": str(upload_exc),
                                            "error_type": getattr(upload_exc, "__class__", type(upload_exc)).__name__,
                                        }
                                    )
                                    logger.error(
                                        "Log file upload failed",
                                        extra={
                                            "event": "eval_async_log_upload_failed",
                                            "file_path": log_file_path,
                                            "error": str(upload_exc),
                                        },
                                    )
                            else:
                                upload_results.append(
                                    {
                                        "file_path": log_file_path,
                                        "status": "failed",
                                        "error": "Session manager not available",
                                        "error_type": "configuration",
                                    }
                                )
                                logger.error(
                                    "Session manager not available for log upload",
                                    extra={
                                        "event": "eval_async_log_upload_missing_session_manager",
                                        "file_path": log_file_path,
                                    },
                                )

                        successful_uploads = [r for r in upload_results if r["status"] == "success"]
                        failed_uploads = [r for r in upload_results if r["status"] == "failed"]

                        logger.info(
                            "Log upload summary",
                            extra={
                                "event": "eval_async_log_upload_summary",
                                "successful_count": len(successful_uploads),
                                "failed_count": len(failed_uploads),
                            },
                        )

                        if failed_uploads:
                            error_types: Dict[str, list[dict[str, Any]]] = {}
                            for failed in failed_uploads:
                                error_type = str(failed.get("error_type", "unknown"))
                                error_types.setdefault(error_type, []).append(failed)

                            for error_type, failures in error_types.items():
                                logger.warning(
                                    "Log file upload failures",
                                    extra={
                                        "event": "eval_async_log_upload_failure_type",
                                        "error_type": error_type,
                                        "failure_count": len(failures),
                                        "files": [failure["file_path"] for failure in failures],
                                    },
                                )

                            total_files = len(upload_results)
                            failure_rate = len(failed_uploads) / total_files if total_files else 0

                            if config.log_upload_fail_on_error and failed_uploads:
                                error_msg = f"Log upload failed for {len(failed_uploads)}/{total_files} files"
                                logger.error(
                                    "Log upload strict mode failure",
                                    extra={
                                        "event": "eval_async_log_upload_strict_failure",
                                        "error_message": error_msg,
                                    },
                                )
                                raise RuntimeError(
                                    f"{error_msg}. Set log_upload_fail_on_error=False to continue on failures."
                                )

                            if not config.log_upload_fail_on_error:
                                if failure_rate > 0.5:
                                    logger.warning(
                                        "High log upload failure rate",
                                        extra={
                                            "event": "eval_async_log_upload_high_failure_rate",
                                            "failure_rate": failure_rate,
                                        },
                                    )
                                else:
                                    logger.info(
                                        "Partial log upload success",
                                        extra={
                                            "event": "eval_async_log_upload_partial_success",
                                            "successful_count": len(successful_uploads),
                                            "total_files": total_files,
                                        },
                                    )
                    else:
                        logger.warning(
                            "No eval log files found for upload",
                            extra={
                                "event": "eval_async_log_upload_no_files",
                                "session_log_dir": session_log_dir,
                            },
                        )

                except Exception as upload_process_exc:
                    logger.error(
                        "Log upload process failure",
                        extra={
                            "event": "eval_async_log_upload_process_failure",
                            "error": str(upload_process_exc),
                        },
                    )

                    if config.log_upload_fail_on_error:
                        logger.error(
                            "Failing evaluation due to log upload failure",
                            extra={"event": "eval_async_log_upload_strict_abort"},
                        )
                        raise RuntimeError(f"Log upload process failed: {upload_process_exc}") from upload_process_exc

                    logger.warning(
                        "Continuing evaluation despite log upload failure",
                        extra={"event": "eval_async_log_upload_continuing"},
                    )
            elif not config.log_upload_enabled:
                logger.info(
                    "Log upload disabled by configuration",
                    extra={"event": "eval_async_log_upload_disabled"},
                )
            else:
                logger.info(
                    "Log upload skipped due to missing session log directory",
                    extra={"event": "eval_async_log_upload_skipped"},
                )

            if session_id and eval_log:
                if isinstance(eval_log, list):
                    for log_entry in eval_log:
                        if hasattr(log_entry, "eval") and hasattr(log_entry.eval, "metadata"):
                            log_entry.eval.metadata = log_entry.eval.metadata or {}
                            log_entry.eval.metadata["saber_session_id"] = session_id
                            if session_log_dir:
                                log_entry.eval.metadata["saber_session_log_dir"] = session_log_dir
                else:
                    if hasattr(eval_log, "eval") and hasattr(eval_log.eval, "metadata"):
                        eval_log.eval.metadata = eval_log.eval.metadata or {}
                        eval_log.eval.metadata["saber_session_id"] = session_id
                        if session_log_dir:
                            eval_log.eval.metadata["saber_session_log_dir"] = session_log_dir

            final_eval_log: Union[EvalLog, None] = None
            if isinstance(eval_log, list):
                if len(eval_log) == 1:
                    final_eval_log = eval_log[0]
                else:
                    logger.warning(
                        "eval_async returned multiple logs; returning the first entry",
                        extra={
                            "event": "eval_async_multiple_logs",
                            "log_count": len(eval_log),
                        },
                    )
                    final_eval_log = eval_log[0] if eval_log else None
            else:
                final_eval_log = eval_log

            log_operation_success(
                logger,
                "run_eval_async",
                session_id=session_id,
                task_count=len(task_ids),
                sample_count=len(dataset),
            )

            return final_eval_log

    except Exception as exc:
        from ..exceptions import EvaluationExecutionError

        log_operation_failure(
            logger,
            "run_eval_async",
            exc,
            session_id=session_id,
        )

        error_details = {
            "error_type": type(exc).__name__,
            "server_url": server_url,
            "agent_assignments": len(config.agents),
            "task_ids": config.task_ids,
        }

        if eval_kwargs:
            error_details["eval_config"] = {k: ("<tasks>" if k == "tasks" else str(v)) for k, v in eval_kwargs.items()}

        raise EvaluationExecutionError(
            f"eval_async execution failed: {exc}",
            details=error_details,
            suggestion="Check eval_async logs for detailed error information",
        ) from exc

    finally:
        await orchestrator.__aexit__(None, None, None)
