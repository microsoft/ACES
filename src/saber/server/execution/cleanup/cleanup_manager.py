"""
Unified container cleanup manager for SABER.

This module provides the single entry point for all container cleanup operations,
with comprehensive logging, state tracking, and debugging capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ....logging_config import get_cleanup_logger
from ..sandbox.permanent_environment_manager import PermanentEnvironmentManager
from ..sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from .cleanup_reason import CleanupReason

logger = get_cleanup_logger(__name__)


@dataclass
class CleanupOperation:
    """Tracks a single cleanup operation for debugging and metrics."""

    identifier: str  # episode_id or "all"
    operation_type: str  # "episode" or "all"
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

    def __init__(
        self,
        sandbox_manager: SandboxEnvironmentManager,
        permanent_manager: Optional[PermanentEnvironmentManager] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the cleanup manager.

        Args:
            sandbox_manager: The sandbox manager to use for ephemeral container cleanup operations
            permanent_manager: Optional permanent environment manager for persistent container lifecycle
            debug_mode: If True, skip container cleanup to allow manual debugging
        """
        self.sandbox_manager = sandbox_manager
        self.permanent_manager = permanent_manager
        self.debug_mode = debug_mode
        self._active_cleanups: Set[str] = set()
        self._cleanup_history: List[CleanupOperation] = []
        self._max_history_size = 1000  # Keep last 1000 cleanup operations

        logger.info("🧹 ContainerCleanupManager initialized")
        if debug_mode:
            logger.warning("🐛 DEBUG MODE ENABLED: Container cleanup will be SKIPPED")
        if permanent_manager:
            logger.info("🧹 ContainerCleanupManager configured with permanent environment support")

    def cleanup_episode(self, episode_id: str, reason: CleanupReason, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clean up containers for a specific episode.

        This is the primary entry point for episode-based cleanup operations.

        Args:
            episode_id: ID of the episode to clean up
            reason: Standardized reason for cleanup
            context: Additional context for debugging (e.g., error details)

        Returns:
            True if cleanup was successful, False otherwise
        """
        if context is None:
            context = {}

        # Prevent duplicate cleanup operations for the same episode
        if episode_id in self._active_cleanups:
            logger.warning(
                f"🧹 DUPLICATE CLEANUP PREVENTED: Episode {episode_id} cleanup already in progress, reason: {reason}",
                episode_id,
            )
            return False

        # Create cleanup operation for tracking
        operation = CleanupOperation(
            identifier=episode_id,
            operation_type="episode",
            reason=reason,
            context=context,
            start_time=datetime.utcnow(),
        )

        # Mark cleanup as active
        self._active_cleanups.add(episode_id)

        logger.warning(
            f"🧹 [{episode_id}] 🧹 EPISODE CLEANUP INITIATED: Starting container cleanup for episode {episode_id}, "
            f"reason: {reason}, context: {context}, is_error_scenario: {reason.is_error_scenario}",
            episode_id,
        )

        try:
            # Check if debug mode is enabled - skip cleanup if so
            if self.debug_mode:
                logger.warning(
                    f"🐛 [{episode_id}] DEBUG MODE: Skipping container cleanup for episode {episode_id}, "
                    f"reason: {reason}. Containers left running for manual debugging.",
                    episode_id,
                )
                operation.steps_completed.append("debug_mode_skip")
                operation.success = True
                operation.end_time = datetime.utcnow()

                # Still mark cleanup as complete so episode tracking is updated
                self._active_cleanups.discard(episode_id)
                self._add_to_history(operation)

                logger.info(
                    f"🐛 [{episode_id}] DEBUG MODE: Episode cleanup skipped successfully. "
                    f"Containers remain active for debugging purposes.",
                    episode_id,
                )
                return True

            # Step 1: Check if episode has active environment
            operation.steps_completed.append("environment_check")
            environment = self.sandbox_manager.get_episode_environment(episode_id)

            if not environment:
                logger.info(
                    f"🧹 [{episode_id}] 🧹 NO ENVIRONMENT: No active environment found for episode {episode_id}, "
                    f"checking orphaned resources, reason: {reason}",
                    episode_id,
                )
                operation.steps_completed.append("no_environment_found")

                # Step 2: Clean up any orphaned resources
                operation.steps_completed.append("orphaned_cleanup_start")
                self.sandbox_manager._cleanup_orphaned_episode_resources(episode_id)
                operation.steps_completed.append("orphaned_cleanup_complete")

            else:
                logger.info(
                    f"🧹 [{episode_id}] 🧹 ENVIRONMENT FOUND: Active environment found for episode {episode_id}, "
                    f"stopping containers, reason: {reason}",
                    episode_id,
                )
                operation.steps_completed.append("environment_found")

                # Step 2: Stop the environment
                operation.steps_completed.append("environment_stop_start")
                environment.stop()
                operation.steps_completed.append("environment_stop_complete")

                # Step 3: Remove from sandbox manager tracking
                operation.steps_completed.append("tracking_removal_start")
                if episode_id in self.sandbox_manager.active_environments:
                    del self.sandbox_manager.active_environments[episode_id]
                operation.steps_completed.append("tracking_removal_complete")

            # Mark operation as successful
            operation.success = True
            operation.end_time = datetime.utcnow()

            logger.info(
                f"🧹 [{episode_id}] 🧹 EPISODE CLEANUP SUCCESS: Container cleanup completed for episode {episode_id}, "
                f"reason: {reason}, duration: {operation.duration_seconds:.3f}s, steps: {operation.steps_completed}",
                episode_id,
            )

            return True

        except Exception as e:
            # Mark operation as failed
            operation.error = str(e)
            operation.end_time = datetime.utcnow()

            logger.error(
                f"🧹 [{episode_id}] 🧹 EPISODE CLEANUP FAILED: Container cleanup failed for episode {episode_id}: {e}, "
                f"reason: {reason}, duration: {operation.duration_seconds:.3f}s, steps: {operation.steps_completed}",
                episode_id,
            )

            # Still try fallback cleanup to remove from tracking
            try:
                if episode_id in self.sandbox_manager.active_environments:
                    del self.sandbox_manager.active_environments[episode_id]
                operation.steps_completed.append("tracking_removal_fallback")
            except Exception as cleanup_error:
                logger.error(f"🧹 [{episode_id}] Failed fallback tracking removal: {cleanup_error}", episode_id)

            return False

        finally:
            # Always remove from active cleanups and add to history
            self._active_cleanups.discard(episode_id)
            self._add_to_history(operation)

    def cleanup_all_episodes(self, reason: CleanupReason, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Clean up containers for all active episodes.

        Args:
            reason: Standardized reason for cleanup (typically SERVER_SHUTDOWN)
            context: Additional context for debugging

        Returns:
            Number of episodes successfully cleaned up
        """
        if context is None:
            context = {}

        active_episode_ids = list(self.sandbox_manager.active_environments.keys())

        if not active_episode_ids:
            logger.info("🧹 NO EPISODES: No active episodes to clean up")
            return 0

        logger.warning(
            f"🧹 MASS EPISODE CLEANUP INITIATED: Cleaning up {len(active_episode_ids)} active episodes, "
            f"reason: {reason}, episode_ids: {active_episode_ids}"
        )

        if self.debug_mode:
            logger.warning(
                f"🐛 DEBUG MODE: cleanup_all_episodes called for {len(active_episode_ids)} episodes, "
                f"but containers will not be cleaned up"
            )

        successful_cleanups = 0

        for episode_id in active_episode_ids:
            try:
                logger.info(f"🧹 BATCH EPISODE CLEANUP: Processing episode {episode_id}", episode_id)
                if self.cleanup_episode(episode_id, reason, context):
                    successful_cleanups += 1
            except Exception as e:
                logger.error(
                    f"🧹 BATCH EPISODE CLEANUP FAILED: Error cleaning up episode {episode_id}: {e}", episode_id
                )

        logger.info(
            f"🧹 MASS EPISODE CLEANUP COMPLETE: Cleaned up {successful_cleanups}/{len(active_episode_ids)} episodes, "
            f"reason: {reason}"
        )

        return successful_cleanups

    def start_permanent_environment(self, environment_spec: Any) -> bool:
        """
        Start permanent environment through unified lifecycle management.

        Args:
            environment_spec: Permanent environment specification

        Returns:
            True if permanent environment started successfully, False otherwise
        """
        if not self.permanent_manager:
            logger.error("🧹 PERMANENT ENV ERROR: No permanent environment manager configured")
            return False

        try:
            logger.info("🧹 PERMANENT ENV START: Starting permanent environment services")
            # Use ensure_permanent_environments_current for configuration change detection
            self.permanent_manager.ensure_permanent_environments_current(environment_spec)
            logger.info("🧹 PERMANENT ENV SUCCESS: Permanent environment started successfully")
            return True

        except Exception as e:
            logger.error(f"🧹 PERMANENT ENV FAILED: Failed to start permanent environment: {e}")
            return False

    def stop_permanent_environment(self) -> bool:
        """
        Stop permanent environment through unified lifecycle management.

        Returns:
            True if permanent environment stopped successfully, False otherwise
        """
        if not self.permanent_manager:
            logger.warning("🧹 PERMANENT ENV WARNING: No permanent environment manager configured")
            return True  # Return True since there's nothing to stop

        if not self.permanent_manager.is_running():
            logger.info("🧹 PERMANENT ENV INFO: Permanent environment not running")
            return True

        try:
            logger.info("🧹 PERMANENT ENV STOP: Stopping permanent environment services")
            self.permanent_manager.stop_permanent_environment()
            logger.info("🧹 PERMANENT ENV STOPPED: Permanent environment stopped successfully")
            return True

        except Exception as e:
            logger.error(f"🧹 PERMANENT ENV STOP FAILED: Failed to stop permanent environment: {e}")
            return False

    def is_permanent_environment_running(self) -> bool:
        """
        Check if permanent environment is running.

        Returns:
            True if permanent environment is running, False otherwise
        """
        if not self.permanent_manager:
            return False
        return self.permanent_manager.is_running()

    def cleanup_all_containers(self, reason: CleanupReason, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Clean up all containers (both ephemeral and permanent) for server shutdown.

        Args:
            reason: Standardized reason for cleanup (typically SERVER_SHUTDOWN)
            context: Additional context for debugging

        Returns:
            Dictionary with cleanup results for both ephemeral and permanent containers
        """
        if context is None:
            context = {}

        logger.info("🧹 FULL CLEANUP START: Starting cleanup of all containers (ephemeral + permanent)")

        if self.debug_mode:
            logger.warning("🐛 DEBUG MODE: cleanup_all_containers called but containers will not be cleaned up")

        # Clean up all ephemeral episode containers
        ephemeral_episodes_cleaned = self.cleanup_all_episodes(reason, context)

        # Clean up permanent environment
        permanent_cleaned = self.stop_permanent_environment()

        result = {
            "ephemeral_episodes_cleaned": ephemeral_episodes_cleaned,
            "permanent_environment_stopped": permanent_cleaned,
            "total_cleanup_success": permanent_cleaned,  # Overall success depends on both
            "reason": reason,
            "context": context,
        }

        logger.info(
            f"🧹 FULL CLEANUP COMPLETE: Ephemeral episodes: {ephemeral_episodes_cleaned}, "
            f"Permanent stopped: {permanent_cleaned}, Overall success: {result['total_cleanup_success']}"
        )

        return result

    def get_cleanup_history(self, identifier: Optional[str] = None) -> List[CleanupOperation]:
        """
        Get cleanup history for debugging.

        Args:
            identifier: If provided, filter history to this identifier only (episode_id, session_id, etc.)

        Returns:
            List of cleanup operations
        """
        if identifier:
            return [op for op in self._cleanup_history if op.identifier == identifier]
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
