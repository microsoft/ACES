"""
Unified container cleanup manager for SABER.

This module provides the single entry point for all container cleanup operations,
with comprehensive logging, state tracking, and debugging capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ....logging_config import get_cleanup_logger
from ..sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from .cleanup_reason import CleanupReason

logger = get_cleanup_logger(__name__)


@dataclass
class CleanupOperation:
    """Tracks a single cleanup operation for debugging and metrics."""

    session_id: str
    reason: CleanupReason
    context: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None
    steps_completed: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate cleanup duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ContainerCleanupManager:
    """
    Centralized container cleanup manager.

    This class is the single entry point for all container cleanup operations
    in SABER, providing unified logging, state tracking, and debugging capabilities.
    """

    def __init__(self, sandbox_manager: SandboxEnvironmentManager):
        """
        Initialize the cleanup manager.

        Args:
            sandbox_manager: The sandbox manager to use for actual cleanup operations
        """
        self.sandbox_manager = sandbox_manager
        self._active_cleanups: Set[str] = set()
        self._cleanup_history: List[CleanupOperation] = []
        self._max_history_size = 1000  # Keep last 1000 cleanup operations

        logger.info("🧹 ContainerCleanupManager initialized")

    def cleanup_session(self, session_id: str, reason: CleanupReason, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clean up containers for a specific session.

        This is the single entry point for all session cleanup operations.

        Args:
            session_id: ID of the session to clean up
            reason: Standardized reason for cleanup
            context: Additional context for debugging (e.g., error details)

        Returns:
            True if cleanup was successful, False otherwise
        """
        if context is None:
            context = {}

        # Prevent duplicate cleanup operations for the same session
        if session_id in self._active_cleanups:
            logger.warning(
                f"🧹 DUPLICATE CLEANUP PREVENTED: Session {session_id} cleanup already in progress, reason: {reason}",
                session_id,
            )
            return False

        # Create cleanup operation for tracking
        operation = CleanupOperation(
            session_id=session_id, reason=reason, context=context, start_time=datetime.utcnow()
        )

        # Mark cleanup as active
        self._active_cleanups.add(session_id)

        logger.warning(
            f"🧹 [{session_id}] 🧹 CLEANUP INITIATED: Starting container cleanup for session {session_id}, "
            f"reason: {reason}, context: {context}, is_error_scenario: {reason.is_error_scenario}",
            session_id,
        )

        try:
            # Step 1: Check if session has active environment
            operation.steps_completed.append("environment_check")
            environment = self.sandbox_manager.get_session_environment(session_id)

            if not environment:
                logger.info(
                    f"🧹 [{session_id}] 🧹 NO ENVIRONMENT: No active environment found for session {session_id}, "
                    f"checking orphaned resources, reason: {reason}",
                    session_id,
                )
                operation.steps_completed.append("no_environment_found")

                # Step 2: Clean up any orphaned resources
                operation.steps_completed.append("orphaned_cleanup_start")
                self.sandbox_manager._cleanup_orphaned_session_resources(session_id)
                operation.steps_completed.append("orphaned_cleanup_complete")

            else:
                logger.info(
                    f"🧹 [{session_id}] 🧹 ENVIRONMENT FOUND: Active environment found for session {session_id}, "
                    f"stopping containers, reason: {reason}",
                    session_id,
                )
                operation.steps_completed.append("environment_found")

                # Step 2: Stop the environment
                operation.steps_completed.append("environment_stop_start")
                environment.stop()
                operation.steps_completed.append("environment_stop_complete")

                # Step 3: Remove from sandbox manager tracking
                operation.steps_completed.append("tracking_removal_start")
                if session_id in self.sandbox_manager.active_sessions:
                    del self.sandbox_manager.active_sessions[session_id]
                operation.steps_completed.append("tracking_removal_complete")

            # Mark operation as successful
            operation.success = True
            operation.end_time = datetime.utcnow()

            logger.info(
                f"🧹 [{session_id}] 🧹 CLEANUP SUCCESS: Container cleanup completed for session {session_id}, "
                f"reason: {reason}, duration: {operation.duration_seconds:.3f}s, steps: {operation.steps_completed}",
                session_id,
            )

            return True

        except Exception as e:
            # Mark operation as failed
            operation.error = str(e)
            operation.end_time = datetime.utcnow()

            logger.error(
                f"🧹 [{session_id}] 🧹 CLEANUP FAILED: Container cleanup failed for session {session_id}: {e}, "
                f"reason: {reason}, duration: {operation.duration_seconds:.3f}s, steps: {operation.steps_completed}",
                session_id,
            )

            # Still try fallback cleanup to remove from tracking
            try:
                if session_id in self.sandbox_manager.active_sessions:
                    del self.sandbox_manager.active_sessions[session_id]
                    logger.warning(
                        f"🧹 FALLBACK SUCCESS: Removed session {session_id} from tracking after cleanup failure",
                        session_id,
                    )
            except Exception as fallback_error:
                logger.error(
                    f"🧹 FALLBACK FAILED: Could not remove session {session_id} from tracking: {fallback_error}",
                    session_id,
                )

            return False

        finally:
            # Always clean up tracking state
            self._active_cleanups.discard(session_id)
            self._add_to_history(operation)

    def cleanup_all_sessions(self, reason: CleanupReason, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Clean up containers for all active sessions.

        Args:
            reason: Standardized reason for cleanup (typically SERVER_SHUTDOWN)
            context: Additional context for debugging

        Returns:
            Number of sessions successfully cleaned up
        """
        if context is None:
            context = {}

        active_session_ids = list(self.sandbox_manager.active_sessions.keys())

        if not active_session_ids:
            logger.info("🧹 NO SESSIONS: No active sessions to clean up")
            return 0

        logger.warning(
            f"🧹 MASS CLEANUP INITIATED: Cleaning up {len(active_session_ids)} active sessions, "
            f"reason: {reason}, session_ids: {active_session_ids}"
        )

        successful_cleanups = 0

        for session_id in active_session_ids:
            try:
                logger.info(f"🧹 BATCH CLEANUP: Processing session {session_id}", session_id)
                if self.cleanup_session(session_id, reason, context):
                    successful_cleanups += 1
            except Exception as e:
                logger.error(f"🧹 BATCH CLEANUP FAILED: Error cleaning up session {session_id}: {e}", session_id)

        logger.info(
            f"🧹 MASS CLEANUP COMPLETE: Cleaned up {successful_cleanups}/{len(active_session_ids)} sessions, "
            f"reason: {reason}"
        )

        return successful_cleanups

    def get_cleanup_history(self, session_id: Optional[str] = None) -> List[CleanupOperation]:
        """
        Get cleanup history for debugging.

        Args:
            session_id: If provided, filter history to this session only

        Returns:
            List of cleanup operations
        """
        if session_id:
            return [op for op in self._cleanup_history if op.session_id == session_id]
        return self._cleanup_history.copy()

    def get_active_cleanups(self) -> Set[str]:
        """Get currently active cleanup operations."""
        return self._active_cleanups.copy()

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics for monitoring."""
        total_cleanups = len(self._cleanup_history)
        successful_cleanups = sum(1 for op in self._cleanup_history if op.success)
        failed_cleanups = total_cleanups - successful_cleanups

        # Calculate average cleanup duration
        completed_operations = [op for op in self._cleanup_history if op.duration_seconds is not None]
        avg_duration = (
            sum(op.duration_seconds for op in completed_operations if op.duration_seconds is not None)
            / len(completed_operations)
            if completed_operations
            else 0
        )

        # Count cleanups by reason
        reason_counts: Dict[str, int] = {}
        for op in self._cleanup_history:
            reason_str = str(op.reason)
            reason_counts[reason_str] = reason_counts.get(reason_str, 0) + 1

        return {
            "total_cleanups": total_cleanups,
            "successful_cleanups": successful_cleanups,
            "failed_cleanups": failed_cleanups,
            "success_rate": successful_cleanups / total_cleanups if total_cleanups > 0 else 0,
            "average_duration_seconds": avg_duration,
            "active_cleanups": len(self._active_cleanups),
            "cleanup_reasons": reason_counts,
        }

    def _add_to_history(self, operation: CleanupOperation) -> None:
        """Add cleanup operation to history with size limiting."""
        self._cleanup_history.append(operation)

        # Limit history size to prevent memory issues
        if len(self._cleanup_history) > self._max_history_size:
            # Remove oldest operations
            self._cleanup_history = self._cleanup_history[-self._max_history_size :]
