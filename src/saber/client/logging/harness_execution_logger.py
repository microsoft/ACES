"""
Harness Execution Logger

Provides structured logging for harness execution events, lifecycle, and debugging.
Follows the same patterns as container logging but focuses on orchestration events.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from .config import ClientLoggingConfig

if TYPE_CHECKING:
    from .events import (
        CleanupEvent,
        EpisodeQueueEvent,
        ErrorEvent,
        ExecutionCompleteEvent,
        ExecutionStartEvent,
        HarnessInitializationEvent,
        InfrastructureEvent,
        SessionCreationEvent,
        TaskResolutionEvent,
        UIEvent,
    )

logger = logging.getLogger(__name__)


class HarnessExecutionLogger:
    """Manages structured logging for harness execution events."""

    def __init__(self, config: ClientLoggingConfig, session_log_dir: Path, client_id: str):
        """Initialize harness execution logger."""
        self.config = config
        self.session_log_dir = session_log_dir
        self.client_id = client_id
        self.session_id: Optional[str] = None  # Set when session is created

        # Create harness logs subdirectory
        self.harness_log_dir = session_log_dir / "harness-execution"
        self.harness_log_dir.mkdir(parents=True, exist_ok=True)

        # System log file for main harness events
        self.system_log_file = session_log_dir / "system.log"

        # Daily events file for structured events
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        self.events_file = self.harness_log_dir / f"harness-events-{date_str}.jsonl"

        logger.info(f"📋 Harness execution logging initialized: {self.harness_log_dir}")

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID after session creation."""
        self.session_id = session_id
        logger.debug(f"📋 Updated harness logger session ID: {session_id}")

    def log_harness_event(
        self,
        event_type: str,
        message: str,
        level: str = "info",
        details: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        """
        Log a structured harness execution event.

        Args:
            event_type: Type of event (harness_init, episode_start, container_launch, etc.)
            message: Human-readable summary of the event
            level: Log level (info, warn, error, debug)
            details: Additional structured data for the event
            task_id: Task ID if applicable
            episode_id: Episode ID if applicable
        """
        if not self.config.enable_container_logging:
            return

        try:
            timestamp = datetime.utcnow().isoformat() + "Z"

            event_data = {
                "timestamp": timestamp,
                "client_id": self.client_id,
                "session_id": self.session_id,
                "task_id": task_id,
                "episode_id": episode_id,
                "event_type": event_type,
                "level": level,
                "message": message,
                "details": details or {},
            }

            # Write to structured events file
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")

            # Also write to system log in human-readable format
            self._write_system_log(timestamp, level, event_type, message, details)

            logger.debug(f"📋 Logged harness event: {event_type} - {message}")

        except Exception as e:
            logger.error(f"❌ Failed to log harness event: {e}")

    def _write_system_log(
        self, timestamp: str, level: str, event_type: str, message: str, details: Optional[Dict[str, Any]]
    ) -> None:
        """Write event to human-readable system log."""
        try:
            level_symbol = {"debug": "🔍", "info": "ℹ️", "warn": "⚠️", "error": "❌"}.get(level, "📋")

            log_line = f"[{timestamp}] {level_symbol} [{level.upper()}] {event_type}: {message}\n"

            if details:
                # Add key details on separate lines
                for key, value in details.items():
                    log_line += f"  {key}: {value}\n"

            with open(self.system_log_file, "a", encoding="utf-8") as f:
                f.write(log_line)

        except Exception as e:
            logger.warning(f"⚠️ Failed to write system log: {e}")

    def log_initialization(self, config_details: Dict[str, Any]) -> None:
        """Log harness initialization."""
        self.log_harness_event(
            event_type="harness_initialization",
            message="SABER harness execution started",
            level="info",
            details={
                "parallelism": config_details.get("parallelism"),
                "client_id": config_details.get("client_id"),
                "base_image": config_details.get("base_image"),
                "mcp_base_url": config_details.get("mcp_base_url"),
                "rest_base_url": config_details.get("rest_base_url"),
                "logging_enabled": config_details.get("enable_container_logging"),
                "log_directory": str(config_details.get("client_log_dir")),
            },
        )

    def log_infrastructure_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log infrastructure-related events (sidecar start, network setup, etc.)."""
        self.log_harness_event(
            event_type=f"infrastructure_{event_type}", message=message, level="info", details=details
        )

    def log_episode_lifecycle(
        self,
        event_type: str,
        task_id: str,
        episode_id: Optional[str] = None,
        attempt: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log episode lifecycle events."""
        message = f"Episode {event_type} for task {task_id}"
        if attempt is not None:
            message += f" (attempt {attempt})"

        event_details = {"task_id": task_id, "attempt": attempt}
        if details:
            event_details.update(details)

        self.log_harness_event(
            event_type=f"episode_{event_type}",
            message=message,
            level="info",
            details=event_details,
            task_id=task_id,
            episode_id=episode_id,
        )

    def log_agent_execution(
        self, event_type: str, task_id: str, episode_id: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log agent execution events."""
        self.log_harness_event(
            event_type=f"agent_{event_type}",
            message=message,
            level="info",
            details=details,
            task_id=task_id,
            episode_id=episode_id,
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        task_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log error events with optional exception details."""
        error_details = details or {}

        if exception:
            error_details.update(
                {
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                    "traceback": str(exception.__traceback__) if hasattr(exception, "__traceback__") else None,
                }
            )

        self.log_harness_event(
            event_type=f"error_{error_type}",
            message=message,
            level="error",
            details=error_details,
            task_id=task_id,
            episode_id=episode_id,
        )

    def log_performance_metrics(
        self,
        metrics_type: str,
        metrics: Dict[str, Any],
        task_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        """Log performance and timing metrics."""
        self.log_harness_event(
            event_type=f"metrics_{metrics_type}",
            message=f"Performance metrics: {metrics_type}",
            level="info",
            details=metrics,
            task_id=task_id,
            episode_id=episode_id,
        )

    def log_cleanup(self, cleanup_details: Dict[str, Any]) -> None:
        """Log cleanup and finalization events."""
        self.log_harness_event(
            event_type="harness_cleanup",
            message="SABER harness execution cleanup completed",
            level="info",
            details=cleanup_details,
        )

    # Structured Event Logging Methods
    # These methods accept structured event objects instead of raw dictionaries

    def log_harness_initialization_event(self, event: "HarnessInitializationEvent") -> None:
        """Log harness initialization using structured event."""
        self.log_harness_event(
            event_type="harness_initialization",
            message="SABER harness execution started",
            level="info",
            details=event.to_dict(),
        )

    def log_session_creation_event(self, event: "SessionCreationEvent") -> None:
        """Log session creation attempt using structured event."""
        if event.success:
            self.log_harness_event(
                event_type="session_creation",
                message=f"Created SABER session: {event.session_id}",
                level="info",
                details=event.to_dict(),
            )
        else:
            self.log_harness_event(
                event_type="session_creation_failed",
                message=f"Failed to create SABER session: {event.error_message}",
                level="error",
                details=event.to_dict(),
            )

    def log_task_resolution_event(self, event: "TaskResolutionEvent") -> None:
        """Log task resolution using structured event."""
        self.log_harness_event(
            event_type="task_resolution",
            message=f"Resolved {event.total_tasks} tasks for execution",
            level="info",
            details=event.to_dict(),
        )

    def log_episode_queue_event(self, event: "EpisodeQueueEvent") -> None:
        """Log episode queue building using structured event."""
        self.log_harness_event(
            event_type="episode_queue_built",
            message=f"Built episode queue with {event.episode_count} episodes",
            level="info",
            details=event.to_dict(),
        )

    def log_execution_start_event(self, event: "ExecutionStartEvent") -> None:
        """Log execution start using structured event."""
        self.log_harness_event(
            event_type="execution_start", message="Starting episode execution", level="info", details=event.to_dict()
        )

    def log_execution_complete_event(self, event: "ExecutionCompleteEvent") -> None:
        """Log execution completion using structured event."""
        self.log_harness_event(
            event_type="execution_complete",
            message=f"Episode execution complete: {event.successful_episodes}/{event.total_episodes} successful",
            level="info",
            details=event.to_dict(),
        )

    def log_infrastructure_event_structured(self, event: "InfrastructureEvent") -> None:
        """Log infrastructure event using structured event."""
        level = "info" if event.success else "error"
        message = f"Infrastructure {event.operation}: {event.component}"
        if not event.success and event.error_message:
            message += f" - {event.error_message}"

        self.log_harness_event(
            event_type=f"infrastructure_{event.operation}", message=message, level=level, details=event.to_dict()
        )

    def log_ui_event_structured(self, event: "UIEvent") -> None:
        """Log UI event using structured event."""
        level = "info" if event.success else "error"
        self.log_harness_event(
            event_type=f"ui_{event.event_type}",
            message=f"UI {event.event_type} with backend: {event.ui_backend}",
            level=level,
            details=event.to_dict(),
        )

    def log_cleanup_event(self, event: "CleanupEvent") -> None:
        """Log cleanup using structured event."""
        level = "info" if event.cleanup_successful else "error"
        self.log_harness_event(
            event_type="harness_cleanup",
            message="SABER harness execution cleanup completed",
            level=level,
            details=event.to_dict(),
        )

    def log_error_event(self, event: "ErrorEvent") -> None:
        """Log error using structured event."""
        self.log_harness_event(
            event_type=f"error_{event.error_type}",
            message=event.error_message,
            level="error",
            details=event.to_dict(),
            task_id=event.task_id,
            episode_id=event.episode_id,
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session's harness events."""
        try:
            summary: Dict[str, Any] = {
                "session_id": self.session_id,
                "client_id": self.client_id,
                "events_file": str(self.events_file),
                "system_log": str(self.system_log_file),
                "event_count": 0,
                "error_count": 0,
                "events_by_type": {},
            }

            if self.events_file.exists():
                with open(self.events_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            summary["event_count"] += 1

                            if event.get("level") == "error":
                                summary["error_count"] += 1

                            event_type = event.get("event_type", "unknown")
                            if isinstance(event_type, str):
                                current_count = summary["events_by_type"].get(event_type, 0)
                                if isinstance(current_count, int):
                                    summary["events_by_type"][event_type] = current_count + 1

                        except json.JSONDecodeError:
                            continue

            return summary

        except Exception as e:
            logger.error(f"❌ Failed to generate session summary: {e}")
            return {"error": str(e)}
