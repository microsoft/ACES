"""Task handlers for orchestrated benchmark execution.

This module provides polymorphic handlers for different benchmark task types,
abstracting the complexity of episode lifecycle management for single vs orchestrated tasks.

Key Design:
- Handler pattern separates execution strategy from sandbox lifecycle
- Semaphore management matches current SABER behavior (held from init through cleanup)
- Best-effort cleanup with logged failures
- Polymorphic dispatch via factory function

Logging category: AGENT
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from saber.client.client_session import ClientSessionManager
from saber.logging_config import LogCategory, get_saber_logger
from saber.models import BenchmarkTask, OrchestratedTask, OrchestrationStrategy, SingleEpisodeTask

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Configuration constants
CLEANUP_ERROR_THRESHOLD = 5  # Maximum cleanup errors before critical alert
EPISODE_CREATE_TIMEOUT = 300  # Seconds to wait for episode creation (5 minutes)


@dataclass
class CleanupResult:
    """Result of cleanup operation with error tracking.

    Attributes:
        success: True if cleanup completed without errors
        error_count: Number of errors encountered during cleanup
        errors: List of error messages
        semaphore_released: Whether semaphore was successfully released
        threshold_exceeded: Whether error count exceeded critical threshold
    """

    success: bool
    error_count: int
    errors: List[str] = field(default_factory=list)
    semaphore_released: bool = False
    threshold_exceeded: bool = False

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return self.error_count > 0


class BenchmarkTaskHandler(ABC):
    """Base class for benchmark task execution handlers.

    Handlers manage the complete lifecycle of episode(s) for a benchmark task:
    - Initialization: Create episodes and acquire resources
    - Cleanup: End episodes and release resources

    Semaphore management follows current SABER semantics:
    - Acquire during initialization (tracked in instance)
    - Hold throughout episode execution
    - Release during cleanup (guaranteed via instance tracking)
    """

    def __init__(self) -> None:
        """Initialize handler with semaphore tracking.

        Note: Caller MUST call cleanup() to release semaphore. The cleanup()
        method is designed for best-effort execution and will log errors without
        re-raising. Semaphore leaks are tracked via _cleanup_error_count counter.
        """
        self._semaphore_acquired = False
        self._acquired_semaphore_ref: Optional[asyncio.Semaphore] = None
        self._cleanup_error_count = 0
        self._cleanup_lock = asyncio.Lock()

    def __del__(self) -> None:
        """Destructor to detect semaphore leaks.

        If the handler is destroyed without cleanup() being called, this will
        log a critical error to alert about the resource leak.
        """
        if self._semaphore_acquired and self._acquired_semaphore_ref:
            logger.error(
                f"SEMAPHORE LEAK DETECTED: {self.__class__.__name__} destroyed without cleanup(). "
                f"This indicates a bug in the caller code. Semaphore slot permanently lost.",
                extra={"cleanup_error_count": self._cleanup_error_count},
            )

    @abstractmethod
    async def initialize(
        self,
        benchmark_task: BenchmarkTask,
        session_id: str,
        session_manager: ClientSessionManager,
        semaphore: Optional[asyncio.Semaphore],
    ) -> Dict[str, Any]:
        """Initialize episodes for this benchmark task.

        Args:
            benchmark_task: Task definition (SingleEpisodeTask or OrchestratedTask)
            session_id: SABER session ID
            session_manager: Client session manager for API calls
            semaphore: Optional semaphore for concurrency control

        Returns:
            State dictionary with episode_ids and other handler-specific data

        Raises:
            Exception: If episode creation fails (partial cleanup performed)
        """
        pass

    @abstractmethod
    async def cleanup(
        self,
        state: Dict[str, Any],
        session_id: str,
        session_manager: ClientSessionManager,
        semaphore: Optional[asyncio.Semaphore],
    ) -> CleanupResult:
        """Clean up episodes for this benchmark task.

        Best-effort cleanup: logs failures but doesn't re-raise exceptions.
        Semaphore is released AFTER episode cleanup completes.

        Args:
            state: State dictionary from initialize()
            session_id: SABER session ID
            session_manager: Client session manager for API calls
            semaphore: Optional semaphore to release after cleanup

        Returns:
            CleanupResult with error tracking and semaphore release status
        """
        pass


class SingleEpisodeTaskHandler(BenchmarkTaskHandler):
    """Handler for traditional single-episode tasks.

    Manages one episode per task with standard semaphore lifecycle:
    - Acquire semaphore
    - Create episode
    - Wait for READY
    - (Agent execution happens externally)
    - End episode
    - Release semaphore
    """

    async def initialize(
        self,
        benchmark_task: BenchmarkTask,
        session_id: str,
        session_manager: ClientSessionManager,
        semaphore: Optional[asyncio.Semaphore],
    ) -> Dict[str, Any]:
        """Initialize single episode.

        Args:
            benchmark_task: SingleEpisodeTask instance
            session_id: SABER session ID
            session_manager: Client session manager
            semaphore: Optional concurrency semaphore

        Returns:
            State dict with episode_ids, primary_episode_id
        """
        if not isinstance(benchmark_task, SingleEpisodeTask):
            raise TypeError(f"SingleEpisodeTaskHandler requires SingleEpisodeTask, got {type(benchmark_task)}")

        episode_id = None

        try:
            # Acquire semaphore slot for episode lifetime (tracked in instance)
            if semaphore:
                await semaphore.acquire()
                self._semaphore_acquired = True
                self._acquired_semaphore_ref = semaphore
                logger.debug(
                    f"Acquired semaphore slot for single episode task {benchmark_task.task_id}",
                    extra={
                        "task_id": benchmark_task.task_id,
                        "semaphore_available": semaphore._value,
                    },
                )

            # Create episode with timeout
            try:
                episode_response = await asyncio.wait_for(
                    session_manager.create_episode(session_id, benchmark_task.task_id), timeout=EPISODE_CREATE_TIMEOUT
                )
                episode_id = episode_response.episode_id
            except asyncio.TimeoutError:
                logger.error(
                    f"Episode creation timed out after {EPISODE_CREATE_TIMEOUT}s",
                    extra={"task_id": benchmark_task.task_id, "timeout": EPISODE_CREATE_TIMEOUT},
                )
                raise
            logger.debug(
                f"Created episode for single task {benchmark_task.task_id}",
                extra={
                    "task_id": benchmark_task.task_id,
                    "episode_id": episode_id,
                },
            )

            # Wait for episode ready
            await session_manager.wait_for_episode_ready(session_id, episode_id)
            logger.debug(
                f"Episode ready for task {benchmark_task.task_id}",
                extra={
                    "task_id": benchmark_task.task_id,
                    "episode_id": episode_id,
                },
            )

            return {
                "episode_ids": [episode_id],
                "primary_episode_id": episode_id,
                "semaphore_acquired": self._semaphore_acquired,
            }

        except Exception:
            # Cleanup on initialization failure
            if episode_id:
                try:
                    await session_manager.end_episode(session_id, episode_id)
                except Exception as cleanup_err:
                    self._cleanup_error_count += 1
                    logger.error(
                        "Failed to cleanup episode during initialization rollback",
                        extra={
                            "task_id": benchmark_task.task_id,
                            "episode_id": episode_id,
                            "error": str(cleanup_err),
                            "total_cleanup_errors": self._cleanup_error_count,
                        },
                    )

            # Release semaphore on failure (guaranteed via instance tracking)
            if self._semaphore_acquired and self._acquired_semaphore_ref:
                self._acquired_semaphore_ref.release()
                logger.debug(
                    "Released semaphore after initialization failure",
                    extra={
                        "task_id": benchmark_task.task_id,
                        "semaphore_available": self._acquired_semaphore_ref._value,
                    },
                )
                self._semaphore_acquired = False
                self._acquired_semaphore_ref = None

            raise

    async def cleanup(
        self,
        state: Dict[str, Any],
        session_id: str,
        session_manager: ClientSessionManager,
        semaphore: Optional[asyncio.Semaphore],
    ) -> CleanupResult:
        """Clean up single episode and release semaphore.

        Cleanup errors are tracked and returned in CleanupResult. If error threshold
        is exceeded, raises RuntimeError after logging critical alert.

        Args:
            state: State from initialize() with episode_ids
            session_id: SABER session ID
            session_manager: Client session manager
            semaphore: Optional semaphore (ignored - uses instance tracking)

        Returns:
            CleanupResult with success status and error details

        Raises:
            RuntimeError: If cleanup error threshold exceeded
        """
        errors: List[str] = []
        episode_id = state.get("primary_episode_id")

        # End episode
        if episode_id:
            try:
                await session_manager.end_episode(session_id, episode_id)
                logger.debug(
                    f"Ended episode {episode_id}",
                    extra={"episode_id": episode_id},
                )
            except Exception as e:
                self._cleanup_error_count += 1
                error_msg = f"Failed to end episode {episode_id}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    "Failed to end episode during cleanup",
                    extra={
                        "episode_id": episode_id,
                        "error": str(e),
                        "total_cleanup_errors": self._cleanup_error_count,
                        "task_type": "single_episode",
                    },
                )

        # Release semaphore AFTER episode cleanup (protected by lock to prevent double-release)
        semaphore_released = False
        async with self._cleanup_lock:
            should_release = self._semaphore_acquired or state.get("semaphore_acquired", False)
            sem_to_release = self._acquired_semaphore_ref or semaphore

            if should_release and sem_to_release:
                try:
                    sem_to_release.release()
                    semaphore_released = True
                    logger.debug(
                        "Released semaphore slot after single episode cleanup",
                        extra={
                            "semaphore_available": sem_to_release._value,
                            "cleanup_errors": self._cleanup_error_count,
                        },
                    )
                    self._semaphore_acquired = False
                    self._acquired_semaphore_ref = None
                    # Clear state flag to prevent double-release on subsequent cleanup calls
                    state["semaphore_acquired"] = False
                except Exception as e:
                    error_msg = f"Failed to release semaphore: {str(e)}"
                    errors.append(error_msg)
                    logger.error(
                        "Failed to release semaphore during cleanup",
                        extra={"error": str(e)},
                    )

        # Build result
        threshold_exceeded = self._cleanup_error_count > CLEANUP_ERROR_THRESHOLD
        result = CleanupResult(
            success=len(errors) == 0,
            error_count=self._cleanup_error_count,
            errors=errors,
            semaphore_released=semaphore_released,
            threshold_exceeded=threshold_exceeded,
        )

        # Log warnings/errors based on error count
        if threshold_exceeded:
            logger.critical(
                f"CRITICAL: Cleanup error threshold exceeded ({self._cleanup_error_count}/{CLEANUP_ERROR_THRESHOLD})",
                extra={
                    "cleanup_error_count": self._cleanup_error_count,
                    "threshold": CLEANUP_ERROR_THRESHOLD,
                    "task_type": "single_episode",
                    "errors": errors,
                },
            )
            raise RuntimeError(
                f"Cleanup error threshold exceeded: {self._cleanup_error_count} errors "
                f"(threshold: {CLEANUP_ERROR_THRESHOLD})"
            )
        elif self._cleanup_error_count > 0:
            logger.warning(
                f"Single episode cleanup completed with {self._cleanup_error_count} errors",
                extra={
                    "cleanup_error_count": self._cleanup_error_count,
                    "task_type": "single_episode",
                    "errors": errors,
                },
            )

        return result


class OrchestratedTaskHandler(BenchmarkTaskHandler):
    """Handler for orchestrated multi-episode tasks.

    Manages multiple episodes as a coordinated group:
    - Acquire ONE semaphore slot for entire orchestration
    - Create episodes in order (root first, then dependents)
    - Wait for each to reach READY before creating next
    - (Agent execution happens externally)
    - End all episodes
    - Release semaphore

    Key insight: Orchestrated tasks consume ONE semaphore slot total,
    not one per sub-task. This prevents resource exhaustion from paired episodes.
    """

    async def initialize(
        self,
        benchmark_task: BenchmarkTask,
        session_id: str,
        session_manager: ClientSessionManager,
        semaphore: Optional[asyncio.Semaphore],
    ) -> Dict[str, Any]:
        """Initialize orchestrated task with multiple episodes.

        Creates episodes sequentially in dependency order, waiting for
        each to reach READY before creating the next. This ensures
        dependency attachment works correctly.

        Args:
            benchmark_task: OrchestratedTask instance
            session_id: SABER session ID
            session_manager: Client session manager
            semaphore: Optional concurrency semaphore

        Returns:
            State dict with episode_ids, episodes (with role info), primary_episode_id
        """
        if not isinstance(benchmark_task, OrchestratedTask):
            raise TypeError(f"OrchestratedTaskHandler requires OrchestratedTask, got {type(benchmark_task)}")

        # Validate that the orchestration strategy is implemented
        if benchmark_task.orchestration_strategy != OrchestrationStrategy.SEQUENTIAL_PAIRED:
            raise NotImplementedError(
                f"Orchestration strategy '{benchmark_task.orchestration_strategy}' is not yet implemented. "
                f"Only '{OrchestrationStrategy.SEQUENTIAL_PAIRED}' is currently supported. "
                f"See https://github.com/your-repo/issues/XXX for parallel/conditional orchestration roadmap."
            )

        created_episodes: List[Dict[str, Any]] = []

        try:
            # Acquire ONE semaphore slot for entire orchestration lifetime (tracked in instance)
            if semaphore:
                await semaphore.acquire()
                self._semaphore_acquired = True
                self._acquired_semaphore_ref = semaphore
                logger.debug(
                    f"Acquired semaphore slot for orchestration {benchmark_task.benchmark_task_id}",
                    extra={
                        "orchestration_id": benchmark_task.benchmark_task_id,
                        "sub_task_count": len(benchmark_task.sub_tasks),
                        "semaphore_available": semaphore._value,
                    },
                )

            # Create episodes in order (sequential paired creation)
            for sub_task in sorted(benchmark_task.sub_tasks, key=lambda x: x.order):
                try:
                    episode_response = await asyncio.wait_for(
                        session_manager.create_episode(session_id, sub_task.task_id), timeout=EPISODE_CREATE_TIMEOUT
                    )
                    episode_id = episode_response.episode_id
                except asyncio.TimeoutError:
                    logger.error(
                        f"Episode creation timed out after {EPISODE_CREATE_TIMEOUT}s for sub-task",
                        extra={
                            "task_id": sub_task.task_id,
                            "orchestration_id": benchmark_task.benchmark_task_id,
                            "timeout": EPISODE_CREATE_TIMEOUT,
                        },
                    )
                    raise
                created_episodes.append(
                    {
                        "episode_id": episode_id,
                        "task_id": sub_task.task_id,
                        "role": sub_task.role,
                    }
                )

                logger.debug(
                    f"Created episode for sub-task {sub_task.task_id} (role: {sub_task.role})",
                    extra={
                        "orchestration_id": benchmark_task.benchmark_task_id,
                        "task_id": sub_task.task_id,
                        "episode_id": episode_id,
                        "role": sub_task.role,
                        "order": sub_task.order,
                    },
                )

                # Wait for ready before creating next (ensures dependency attachment)
                await session_manager.wait_for_episode_ready(session_id, episode_id)
                logger.debug(
                    f"Episode ready for sub-task {sub_task.task_id}",
                    extra={
                        "orchestration_id": benchmark_task.benchmark_task_id,
                        "task_id": sub_task.task_id,
                        "episode_id": episode_id,
                    },
                )

            # All episodes created successfully
            logger.info(
                f"Orchestration initialized successfully: {benchmark_task.benchmark_task_id}",
                extra={
                    "orchestration_id": benchmark_task.benchmark_task_id,
                    "episode_count": len(created_episodes),
                    "episode_ids": [ep["episode_id"] for ep in created_episodes],
                },
            )

            return {
                "episode_ids": [ep["episode_id"] for ep in created_episodes],
                "episodes": created_episodes,
                "primary_episode_id": created_episodes[0]["episode_id"],
                "semaphore_acquired": self._semaphore_acquired,
            }

        except Exception:
            # Best-effort cleanup on initialization failure
            for ep_info in created_episodes:
                try:
                    await session_manager.end_episode(session_id, ep_info["episode_id"])
                except Exception as cleanup_err:
                    self._cleanup_error_count += 1
                    logger.error(
                        "Failed to cleanup episode during orchestration rollback",
                        extra={
                            "orchestration_id": benchmark_task.benchmark_task_id,
                            "episode_id": ep_info["episode_id"],
                            "task_id": ep_info["task_id"],
                            "error": str(cleanup_err),
                            "total_cleanup_errors": self._cleanup_error_count,
                        },
                    )

            if self._cleanup_error_count > 0:
                logger.error(
                    f"Orchestration initialization failed with {self._cleanup_error_count} cleanup errors",
                    extra={
                        "orchestration_id": benchmark_task.benchmark_task_id,
                        "cleanup_errors": self._cleanup_error_count,
                    },
                )

            # Release semaphore on initialization failure (guaranteed via instance tracking)
            if self._semaphore_acquired and self._acquired_semaphore_ref:
                self._acquired_semaphore_ref.release()
                logger.debug(
                    "Released semaphore after orchestration initialization failure",
                    extra={
                        "orchestration_id": benchmark_task.benchmark_task_id,
                        "semaphore_available": self._acquired_semaphore_ref._value,
                    },
                )
                self._semaphore_acquired = False
                self._acquired_semaphore_ref = None

            raise

    async def cleanup(
        self,
        state: Dict[str, Any],
        session_id: str,
        session_manager: ClientSessionManager,
        semaphore: Optional[asyncio.Semaphore],
    ) -> CleanupResult:
        """Clean up all episodes in orchestration and release semaphore.

        Cleanup errors are tracked and returned in CleanupResult. If error threshold
        is exceeded, raises RuntimeError after logging critical alert.

        Args:
            state: State from initialize() with episode_ids
            session_id: SABER session ID
            session_manager: Client session manager
            semaphore: Optional semaphore (parameter unused - instance tracking only)

        Returns:
            CleanupResult with success status and error details

        Raises:
            RuntimeError: If cleanup error threshold exceeded
        """
        errors: List[str] = []

        # End all episodes in orchestration
        for episode_id in state.get("episode_ids", []):
            try:
                await session_manager.end_episode(session_id, episode_id)
                logger.debug(
                    f"Ended orchestration episode {episode_id}",
                    extra={"episode_id": episode_id},
                )
            except Exception as e:
                self._cleanup_error_count += 1
                error_msg = f"Failed to end episode {episode_id}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    "Failed to end episode during orchestration cleanup",
                    extra={
                        "episode_id": episode_id,
                        "error": str(e),
                        "total_cleanup_errors": self._cleanup_error_count,
                        "task_type": "orchestrated",
                    },
                )

        # Release semaphore AFTER all episodes cleaned up (protected by lock to prevent double-release)
        semaphore_released = False
        async with self._cleanup_lock:
            should_release = self._semaphore_acquired or state.get("semaphore_acquired", False)
            sem_to_release = self._acquired_semaphore_ref or semaphore

            if should_release and sem_to_release:
                try:
                    sem_to_release.release()
                    semaphore_released = True
                    logger.debug(
                        "Released semaphore slot after orchestration cleanup",
                        extra={
                            "semaphore_available": sem_to_release._value,
                            "cleanup_errors": self._cleanup_error_count,
                        },
                    )
                    self._semaphore_acquired = False
                    self._acquired_semaphore_ref = None
                    # Clear state flag to prevent double-release on subsequent cleanup calls
                    state["semaphore_acquired"] = False
                except Exception as e:
                    error_msg = f"Failed to release semaphore: {str(e)}"
                    errors.append(error_msg)
                    logger.error(
                        "Failed to release semaphore during cleanup",
                        extra={"error": str(e)},
                    )

        # Build result
        threshold_exceeded = self._cleanup_error_count > CLEANUP_ERROR_THRESHOLD
        result = CleanupResult(
            success=len(errors) == 0,
            error_count=self._cleanup_error_count,
            errors=errors,
            semaphore_released=semaphore_released,
            threshold_exceeded=threshold_exceeded,
        )

        # Log warnings/errors based on error count
        if threshold_exceeded:
            logger.critical(
                f"CRITICAL: Cleanup error threshold exceeded ({self._cleanup_error_count}/{CLEANUP_ERROR_THRESHOLD})",
                extra={
                    "cleanup_error_count": self._cleanup_error_count,
                    "threshold": CLEANUP_ERROR_THRESHOLD,
                    "task_type": "orchestrated",
                    "errors": errors,
                },
            )
            raise RuntimeError(
                f"Cleanup error threshold exceeded: {self._cleanup_error_count} errors "
                f"(threshold: {CLEANUP_ERROR_THRESHOLD})"
            )
        elif self._cleanup_error_count > 0:
            logger.warning(
                f"Orchestration cleanup completed with {self._cleanup_error_count} errors",
                extra={
                    "cleanup_error_count": self._cleanup_error_count,
                    "task_type": "orchestrated",
                    "errors": errors,
                },
            )

        return result


def get_benchmark_task_handler(benchmark_task: BenchmarkTask) -> BenchmarkTaskHandler:
    """Factory function for task type handlers.

    Args:
        benchmark_task: BenchmarkTask instance (SingleEpisodeTask or OrchestratedTask)

    Returns:
        Appropriate handler instance

    Raises:
        ValueError: If unknown task type
    """
    if isinstance(benchmark_task, SingleEpisodeTask):
        return SingleEpisodeTaskHandler()
    elif isinstance(benchmark_task, OrchestratedTask):
        return OrchestratedTaskHandler()
    else:
        raise ValueError(f"Unknown benchmark task type: {type(benchmark_task)}")


__all__ = [
    "BenchmarkTaskHandler",
    "CleanupResult",
    "SingleEpisodeTaskHandler",
    "OrchestratedTaskHandler",
    "get_benchmark_task_handler",
]
