"""Dedicated debug logging for episode lifecycle investigation.

This module provides a separate debug log file specifically for tracking
episode lifecycle events on the SABER sandbox side. Unlike the main inspect
.eval logs, this captures detailed timing and state information about:
- When sample_cleanup is called and why
- Agent execution start/completion
- Episode creation and termination
- Step counts and timing

Each eval run creates a new timestamped log file in logs/
"""

import logging
import time
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

__all__ = [
    "init_episode_debug_logger",
    "get_episode_debug_logger",
    "set_episode_context",
    "update_episode_context",
    "clear_episode_context",
    "enable_debug_logging",
    "is_debug_logging_enabled",
    "EpisodeDebugLogger",
    "log_sample_init_start",
    "log_sample_init_complete",
    "log_agent_execution_start",
    "log_agent_execution_complete",
    "log_sample_cleanup_called",
    "log_episode_end_request",
    "log_episode_end_complete",
    "log_step_executed",
    "log_mcp_request",
    "log_lifecycle_summary",
    "increment_step_counter",
]

# Context variable to track current episode state
_episode_context: ContextVar[Dict[str, Any]] = ContextVar("episode_debug_context", default={})

# Singleton logger instance
_debug_logger: Optional[logging.Logger] = None
_debug_logging_enabled = False


def enable_debug_logging() -> None:
    """Enable debug logging for the current evaluation run."""
    global _debug_logging_enabled
    _debug_logging_enabled = True


def is_debug_logging_enabled() -> bool:
    """Check if debug logging is enabled."""
    return _debug_logging_enabled


def init_episode_debug_logger(log_dir: Path = Path("logs")) -> logging.Logger:
    """Initialize the dedicated episode lifecycle debug logger.

    Creates a new timestamped log file for each run.

    Args:
        log_dir: Directory where debug log should be written

    Returns:
        Configured logger instance
    """
    global _debug_logger

    if _debug_logger is not None:
        return _debug_logger

    # Create logger
    _debug_logger = logging.Logger("saber.episode_lifecycle_debug")
    _debug_logger.setLevel(logging.DEBUG)
    _debug_logger.propagate = False  # Don't propagate to root logger

    # Create log directory if needed
    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file for this run
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_file = log_dir / f"saber_episode_debug_{timestamp}.log"

    # Create file handler (no rotation - one file per run)
    handler = logging.FileHandler(
        log_file,
        mode="w",  # Write mode - new file each time
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)

    # Format: timestamp | level | component | message | structured data
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(component)-20s | %(message)s%(extra_data)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    _debug_logger.addHandler(handler)

    # Log the file location at startup
    _debug_logger.info(
        f"Debug logging initialized: {log_file}",
        extra={"component": "init", "extra_data": ""},
    )

    return _debug_logger


def get_episode_debug_logger() -> Optional[logging.Logger]:
    """Get the episode debug logger, initializing if needed and enabled.

    Returns None if debug logging is not enabled.
    """
    if not _debug_logging_enabled:
        return None
    if _debug_logger is None:
        return init_episode_debug_logger()
    return _debug_logger


def set_episode_context(**context: Any) -> None:
    """Set episode context for subsequent log messages."""
    _episode_context.set(dict(context))


def update_episode_context(**context: Any) -> None:
    """Update episode context with new values."""
    current = dict(_episode_context.get())
    current.update(context)
    _episode_context.set(current)


def clear_episode_context() -> None:
    """Clear episode context."""
    _episode_context.set({})


def _format_extra(extra: Optional[Dict[str, Any]]) -> str:
    """Format extra data for logging."""
    if not extra:
        return ""

    # Merge with context
    context = dict(_episode_context.get())
    merged = {**context, **extra}

    if not merged:
        return ""

    pairs = []
    for key in sorted(merged.keys()):
        value = merged[key]
        if isinstance(value, float):
            pairs.append(f"{key}={value:.3f}")
        else:
            pairs.append(f"{key}={value}")

    return " | " + " ".join(pairs)


class EpisodeDebugLogger:
    """Wrapper class for easier episode lifecycle logging."""

    def __init__(self, component: str):
        self.component = component
        self.logger = get_episode_debug_logger()

    def _log(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Internal logging method that adds component and formats extra data."""
        if self.logger is None:
            return  # Debug logging not enabled
        extra_str = _format_extra(extra)
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(internal)",
            0,
            msg,
            (),
            None,
            extra={
                "component": self.component,
                "extra_data": extra_str,
            },
        )
        self.logger.handle(record)

    def debug(self, msg: str, **extra: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, extra)

    def info(self, msg: str, **extra: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, msg, extra)

    def warning(self, msg: str, **extra: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, extra)

    def error(self, msg: str, **extra: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, extra)

    def critical(self, msg: str, **extra: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, msg, extra)


# Convenience functions for common lifecycle events


def log_sample_init_start(task_id: str, episode_id: str, session_id: str, **extra: Any) -> None:
    """Log the start of sample initialization."""
    logger = EpisodeDebugLogger("sample_init")
    logger.info(
        "🎬 SAMPLE_INIT: Starting sample initialization",
        task_id=task_id,
        episode_id=episode_id,
        session_id=session_id,
        timestamp=time.time(),
        **extra,
    )
    set_episode_context(
        task_id=task_id,
        episode_id=episode_id,
        session_id=session_id,
        sample_init_time=time.time(),
    )


def log_sample_init_complete(duration: float, **extra: Any) -> None:
    """Log the completion of sample initialization."""
    logger = EpisodeDebugLogger("sample_init")
    logger.info(
        "✅ SAMPLE_INIT: Sample initialization complete",
        duration=duration,
        timestamp=time.time(),
        **extra,
    )
    update_episode_context(sample_init_complete_time=time.time())


def log_agent_execution_start(agent_type: str, **extra: Any) -> None:
    """Log the start of agent execution."""
    logger = EpisodeDebugLogger("agent_execution")
    logger.info(
        "🤖 AGENT_START: Agent execution starting",
        agent_type=agent_type,
        timestamp=time.time(),
        **extra,
    )
    update_episode_context(agent_start_time=time.time(), agent_type=agent_type)


def log_agent_execution_complete(result_type: str, steps_executed: Optional[int] = None, **extra: Any) -> None:
    """Log the completion of agent execution."""
    logger = EpisodeDebugLogger("agent_execution")
    context = _episode_context.get()
    start_time = context.get("agent_start_time")
    duration = time.time() - start_time if start_time else None

    logger.info(
        "✅ AGENT_COMPLETE: Agent execution completed",
        result_type=result_type,
        steps_executed=steps_executed,
        duration=duration,
        timestamp=time.time(),
        **extra,
    )
    update_episode_context(
        agent_complete_time=time.time(),
        agent_result_type=result_type,
        agent_steps_executed=steps_executed,
    )


def log_sample_cleanup_called(interrupted: bool, caller: str, **extra: Any) -> None:
    """Log when sample_cleanup is called."""
    logger = EpisodeDebugLogger("sample_cleanup")
    context = _episode_context.get()
    sample_init_time = context.get("sample_init_time")
    duration_since_init = time.time() - sample_init_time if sample_init_time else None

    logger.info(
        "🧹 CLEANUP_CALLED: sample_cleanup invoked",
        interrupted=interrupted,
        caller=caller,
        duration_since_init=duration_since_init,
        timestamp=time.time(),
        **extra,
    )

    update_episode_context(
        cleanup_called_time=time.time(),
        cleanup_interrupted=interrupted,
        cleanup_caller=caller,
    )


def log_episode_end_request(reason: str, **extra: Any) -> None:
    """Log when an episode end request is made."""
    logger = EpisodeDebugLogger("episode_end")
    context = _episode_context.get()
    sample_init_time = context.get("sample_init_time")
    agent_start_time = context.get("agent_start_time")
    agent_complete_time = context.get("agent_complete_time")

    duration_since_init = time.time() - sample_init_time if sample_init_time else None
    duration_since_agent_start = time.time() - agent_start_time if agent_start_time else None
    duration_since_agent_complete = time.time() - agent_complete_time if agent_complete_time else None

    logger.info(
        "🛑 EPISODE_END: Episode end request",
        reason=reason,
        duration_since_init=duration_since_init,
        duration_since_agent_start=duration_since_agent_start,
        duration_since_agent_complete=duration_since_agent_complete,
        timestamp=time.time(),
        **extra,
    )
    update_episode_context(episode_end_time=time.time(), episode_end_reason=reason)


def log_episode_end_complete(**extra: Any) -> None:
    """Log when episode end is complete."""
    logger = EpisodeDebugLogger("episode_end")
    context = _episode_context.get()
    episode_end_time = context.get("episode_end_time")
    duration = time.time() - episode_end_time if episode_end_time else None

    logger.info(
        "✅ EPISODE_END_COMPLETE: Episode ended successfully",
        duration=duration,
        timestamp=time.time(),
        **extra,
    )


def log_step_executed(step_number: int, tool_name: Optional[str] = None, **extra: Any) -> None:
    """Log when an agent step is executed."""
    logger = EpisodeDebugLogger("step_execution")

    # Update step count in context
    update_episode_context(current_step=step_number)

    logger.debug(
        "▶️  STEP_EXECUTED: Agent step completed",
        step_number=step_number,
        tool_name=tool_name,
        timestamp=time.time(),
        **extra,
    )


def increment_step_counter() -> int:
    """Increment and return the current step counter."""
    context = _episode_context.get()
    current_step = cast(int, context.get("current_step", 0))
    new_step = current_step + 1
    update_episode_context(current_step=new_step)
    return new_step


def log_mcp_request(operation: str, **extra: Any) -> None:
    """Log MCP tool requests."""
    logger = EpisodeDebugLogger("mcp_tools")
    logger.debug(
        "🔧 MCP_REQUEST: MCP tool operation",
        operation=operation,
        timestamp=time.time(),
        **extra,
    )


def log_lifecycle_summary() -> None:
    """Log a summary of the episode lifecycle timing."""
    logger = EpisodeDebugLogger("lifecycle_summary")
    context = _episode_context.get()

    if not context:
        logger.warning("No episode context available for summary")
        return

    sample_init_time = context.get("sample_init_time")
    agent_start_time = context.get("agent_start_time")
    agent_complete_time = context.get("agent_complete_time")
    cleanup_called_time = context.get("cleanup_called_time")
    episode_end_time = context.get("episode_end_time")

    if sample_init_time and cleanup_called_time:
        total_duration = cleanup_called_time - sample_init_time

        timings = {
            "total_sample_duration": total_duration,
        }

        if agent_start_time:
            timings["time_to_agent_start"] = agent_start_time - sample_init_time

        if agent_complete_time and agent_start_time:
            timings["agent_execution_duration"] = agent_complete_time - agent_start_time

        if cleanup_called_time and agent_complete_time:
            timings["agent_complete_to_cleanup"] = cleanup_called_time - agent_complete_time

        if episode_end_time and cleanup_called_time:
            timings["cleanup_to_end"] = episode_end_time - cleanup_called_time

        logger.info(
            "📊 LIFECYCLE_SUMMARY: Episode lifecycle complete",
            task_id=context.get("task_id"),
            episode_id=context.get("episode_id"),
            steps_executed=context.get("agent_steps_executed"),
            result_type=context.get("agent_result_type"),
            interrupted=context.get("cleanup_interrupted"),
            **timings,
        )
