"""Orchestration coordinator for managing multi-sample orchestrated tasks.

This module provides a singleton coordinator that manages the lifecycle of orchestrated
sub-tasks where multiple samples (e.g., blue team, red team) need to be coordinated
and executed in a specific order with dependency management.

The coordinator handles:
- Registration of root and dependent samples
- Dependency tracking and signaling
- Episode ID coordination across samples
- Cascade termination of entire orchestration groups
- Semaphore management for cleanup
"""

import asyncio
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

from ...logging_config import LogCategory, get_saber_logger
from ..constants import SandboxTimeouts

logger = get_saber_logger(LogCategory.HARNESS, __name__)


@dataclass
class SampleRegistration:
    """Registration info for a sample in an orchestration."""

    sample_id: str
    role: str
    depends_on_role: Optional[str]
    order: int
    episode_id: Optional[str] = None
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    termination_requested: bool = False
    active_sample: Optional[Any] = None  # Reference to Inspect AI's ActiveSample for interrupt


@dataclass
class OrchestrationGroup:
    """Group of samples that are orchestrated together."""

    orchestration_id: str
    samples: dict[str, SampleRegistration] = field(default_factory=dict)
    root_role: Optional[str] = None
    semaphore: Optional[asyncio.Semaphore] = None
    terminated: bool = False

    # Score coordination fields
    all_scored_event: Optional[asyncio.Event] = None
    scored_samples: set[str] = field(default_factory=set)
    sample_scores: dict[str, float] = field(default_factory=dict)  # Track scores for logging


class OrchestrationCoordinator:
    """Singleton coordinator for orchestrated multi-sample tasks.

    This coordinator manages groups of samples that need to be executed in a coordinated
    manner with dependency tracking. It uses a singleton pattern to ensure all samples
    within an orchestration can communicate through the same coordinator instance.

    Thread-safe for registration operations.
    """

    _instance: Optional["OrchestrationCoordinator"] = None
    _lock = Lock()

    def __init__(self) -> None:
        """Initialize orchestration coordinator (singleton pattern)."""
        # Only initialize once
        if not hasattr(self, "_initialized"):
            self._orchestrations: dict[str, OrchestrationGroup] = {}
            self._init_lock = Lock()
            self._initialized = True

    def __new__(cls) -> "OrchestrationCoordinator":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def register_root_sample(
        self,
        orchestration_id: str,
        role: str,
        sample_id: str,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> bool:
        """Register a root sample (no dependency).

        Args:
            orchestration_id: Unique ID for this orchestration group
            role: Role of this sample (e.g., "blue", "red")
            sample_id: Unique sample ID
            semaphore: Optional semaphore for concurrency control

        Returns:
            True if registration succeeded, False otherwise
        """
        with self._init_lock:
            if orchestration_id not in self._orchestrations:
                self._orchestrations[orchestration_id] = OrchestrationGroup(
                    orchestration_id=orchestration_id,
                    root_role=role,
                    semaphore=semaphore,
                )

            group = self._orchestrations[orchestration_id]

            # Check if already registered
            if role in group.samples:
                logger.warning(
                    f"Sample {role} already registered in orchestration {orchestration_id}",
                    extra={"orchestration_id": orchestration_id, "role": role},
                )
                return False

            # Register root sample
            registration = SampleRegistration(
                sample_id=sample_id,
                role=role,
                depends_on_role=None,
                order=1,
            )
            group.samples[role] = registration

            logger.info(
                f"Registered root sample {role} in orchestration {orchestration_id}",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "sample_id": sample_id,
                },
            )
            return True

    async def register_dependent_sample(
        self,
        orchestration_id: str,
        role: str,
        sample_id: str,
        depends_on_role: str,
        order: int,
        timeout: float = SandboxTimeouts.ORCHESTRATION_REGISTRATION_SECONDS,
    ) -> bool:
        """Register a dependent sample that waits for another sample.

        Args:
            orchestration_id: Unique ID for this orchestration group
            role: Role of this sample
            sample_id: Unique sample ID
            depends_on_role: Role this sample depends on
            order: Execution order
            timeout: Timeout for registration (not used currently)

        Returns:
            True if registration succeeded, False otherwise
        """
        with self._init_lock:
            if orchestration_id not in self._orchestrations:
                logger.error(
                    f"Orchestration {orchestration_id} not found for dependent sample {role}",
                    extra={"orchestration_id": orchestration_id, "role": role},
                )
                return False

            group = self._orchestrations[orchestration_id]

            # Check if already registered
            if role in group.samples:
                logger.warning(
                    f"Sample {role} already registered in orchestration {orchestration_id}",
                    extra={"orchestration_id": orchestration_id, "role": role},
                )
                return False

            # Verify dependency exists
            if depends_on_role not in group.samples:
                logger.error(
                    f"Dependency {depends_on_role} not found for sample {role}",
                    extra={
                        "orchestration_id": orchestration_id,
                        "role": role,
                        "depends_on_role": depends_on_role,
                    },
                )
                return False

            # Register dependent sample
            registration = SampleRegistration(
                sample_id=sample_id,
                role=role,
                depends_on_role=depends_on_role,
                order=order,
            )
            group.samples[role] = registration

            logger.info(
                f"Registered dependent sample {role} in orchestration {orchestration_id}",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "sample_id": sample_id,
                    "depends_on_role": depends_on_role,
                    "order": order,
                },
            )
            return True

    async def wait_for_dependency_ready(
        self,
        orchestration_id: str,
        role: str,
        session_manager: Any,
        session_id: str,
        timeout: float = SandboxTimeouts.ORCHESTRATION_DEPENDENCY_WAIT_SECONDS,
    ) -> str:
        """Wait for dependency to be ready and return its episode ID.

        Args:
            orchestration_id: Orchestration ID
            role: This sample's role
            session_manager: Session manager (unused, for compatibility)
            session_id: Session ID (unused, for compatibility)
            timeout: Maximum time to wait

        Returns:
            Episode ID of the dependency

        Raises:
            TimeoutError: If dependency doesn't become ready in time
            ValueError: If dependency not found or no episode ID
        """
        group = self._orchestrations.get(orchestration_id)
        if not group:
            raise ValueError(f"Orchestration {orchestration_id} not found")

        sample = group.samples.get(role)
        if not sample or not sample.depends_on_role:
            raise ValueError(f"Sample {role} has no dependency")

        dependency = group.samples.get(sample.depends_on_role)
        if not dependency:
            raise ValueError(f"Dependency {sample.depends_on_role} not found")

        # Wait for dependency to signal ready
        logger.info(
            f"Sample {role} waiting for dependency {sample.depends_on_role}",
            extra={
                "orchestration_id": orchestration_id,
                "role": role,
                "depends_on_role": sample.depends_on_role,
            },
        )

        try:
            await asyncio.wait_for(dependency.ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout waiting for dependency {sample.depends_on_role}",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "depends_on_role": sample.depends_on_role,
                    "timeout": timeout,
                },
            )
            raise TimeoutError(f"Dependency {sample.depends_on_role} did not become ready within {timeout}s")

        # Check if termination was requested
        if dependency.termination_requested or group.terminated:
            raise RuntimeError(f"Orchestration {orchestration_id} was terminated before dependency became ready")

        # Return dependency's episode ID
        if not dependency.episode_id:
            raise ValueError(f"Dependency {sample.depends_on_role} has no episode ID")

        logger.info(
            f"Dependency {sample.depends_on_role} is ready with episode {dependency.episode_id}",
            extra={
                "orchestration_id": orchestration_id,
                "role": role,
                "depends_on_role": sample.depends_on_role,
                "dependency_episode_id": dependency.episode_id,
            },
        )

        return dependency.episode_id

    def set_episode_id(
        self,
        orchestration_id: str,
        role: str,
        episode_id: str,
    ) -> None:
        """Set episode ID for a sample and signal dependents.

        Args:
            orchestration_id: Orchestration ID
            role: Sample's role
            episode_id: Episode ID to set
        """
        group = self._orchestrations.get(orchestration_id)
        if not group:
            logger.warning(
                f"Orchestration {orchestration_id} not found when setting episode ID",
                extra={"orchestration_id": orchestration_id, "role": role},
            )
            return

        sample = group.samples.get(role)
        if not sample:
            logger.warning(
                f"Sample {role} not found in orchestration {orchestration_id}",
                extra={"orchestration_id": orchestration_id, "role": role},
            )
            return

        sample.episode_id = episode_id
        sample.ready_event.set()

        logger.info(
            f"Set episode ID for {role} and signaled dependents",
            extra={
                "orchestration_id": orchestration_id,
                "role": role,
                "episode_id": episode_id,
            },
        )

    def set_active_sample(
        self,
        orchestration_id: str,
        role: str,
        active_sample: Any,
    ) -> None:
        """Store reference to Inspect AI's ActiveSample for interrupt capability.

        This allows the coordinator to interrupt sibling samples when one sample
        completes, triggering their natural cleanup and scoring paths.

        Args:
            orchestration_id: Orchestration ID
            role: Sample's role
            active_sample: The Inspect AI ActiveSample instance
        """
        group = self._orchestrations.get(orchestration_id)
        if not group:
            logger.warning(
                f"Orchestration {orchestration_id} not found when setting active_sample",
                extra={"orchestration_id": orchestration_id, "role": role},
            )
            return

        sample = group.samples.get(role)
        if not sample:
            logger.warning(
                f"Sample {role} not found in orchestration {orchestration_id}",
                extra={"orchestration_id": orchestration_id, "role": role},
            )
            return

        sample.active_sample = active_sample

        logger.debug(
            f"Set active_sample reference for {role}",
            extra={
                "orchestration_id": orchestration_id,
                "role": role,
            },
        )

    def trigger_termination(self, orchestration_id: str, skip_role: Optional[str] = None) -> list[tuple[str, str]]:
        """Trigger cascade termination for entire orchestration.

        Marks all samples in the orchestration for termination, interrupts sibling
        samples (except skip_role), and returns episode IDs that need cleanup.

        Args:
            orchestration_id: Orchestration ID to terminate
            skip_role: Role of the sample initiating termination (won't be interrupted)

        Returns:
            List of (role, episode_id) tuples for cleanup
        """
        group = self._orchestrations.get(orchestration_id)
        if not group:
            logger.warning(
                f"Orchestration {orchestration_id} not found for termination",
                extra={"orchestration_id": orchestration_id},
            )
            return []

        group.terminated = True
        episodes_to_cleanup = []

        for role, sample in group.samples.items():
            sample.termination_requested = True
            # Signal any waiting dependents
            sample.ready_event.set()

            if sample.episode_id:
                episodes_to_cleanup.append((role, sample.episode_id))

                # Signal the wrapper's shutdown event to unblock wait_for_injection_event
                # This is critical for blue team agents that are waiting indefinitely
                try:
                    from ..integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper

                    wrapper = WebSocketTranscriptSyncingModelWrapper.get_wrapper_for_episode(sample.episode_id)
                    if wrapper:
                        wrapper._events.signal_shutdown()
                        logger.info(
                            f"Signaled shutdown to wrapper for {role} in orchestration {orchestration_id}",
                            extra={
                                "orchestration_id": orchestration_id,
                                "role": role,
                                "episode_id": sample.episode_id,
                            },
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not signal wrapper shutdown for {role}: {e}",
                        extra={"orchestration_id": orchestration_id, "role": role},
                    )

            # Interrupt sibling samples (not the one initiating cleanup)
            if role != skip_role and sample.active_sample is not None:
                try:
                    sample.active_sample.interrupt("score")
                    logger.info(
                        f"Interrupted sibling sample {role} in orchestration {orchestration_id}",
                        extra={"orchestration_id": orchestration_id, "role": role},
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to interrupt sample {role}: {e}",
                        extra={"orchestration_id": orchestration_id, "role": role},
                    )

        logger.info(
            f"Triggered termination for orchestration {orchestration_id}",
            extra={
                "orchestration_id": orchestration_id,
                "episodes_to_cleanup": len(episodes_to_cleanup),
                "skip_role": skip_role,
            },
        )

        return episodes_to_cleanup

    def cleanup_sample(
        self,
        orchestration_id: str,
        role: str,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> bool:
        """Cleanup a sample from orchestration.

        Args:
            orchestration_id: Orchestration ID
            role: Sample's role
            semaphore: Semaphore (unused, for compatibility)

        Returns:
            True if this is the last sample and semaphore should be released
        """
        group = self._orchestrations.get(orchestration_id)
        if not group:
            logger.warning(
                f"Orchestration {orchestration_id} not found for cleanup",
                extra={"orchestration_id": orchestration_id, "role": role},
            )
            return False

        # Remove sample
        if role in group.samples:
            del group.samples[role]

        # Check if this is the last sample
        is_last = len(group.samples) == 0

        if is_last:
            # Remove orchestration group
            del self._orchestrations[orchestration_id]
            logger.info(
                f"Cleaned up last sample in orchestration {orchestration_id}",
                extra={"orchestration_id": orchestration_id, "role": role},
            )
            return True
        else:
            logger.info(
                f"Cleaned up sample {role} in orchestration {orchestration_id}",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "remaining_samples": len(group.samples),
                },
            )
            return False

    async def wait_for_all_scored(
        self,
        orchestration_id: str,
        role: str,
        score: float,
        timeout: float = SandboxTimeouts.ORCHESTRATION_SCORE_SYNC_SECONDS,
    ) -> None:
        """Wait for all samples in orchestration to complete scoring.

        This method should be called in the scorer BEFORE returning Score to Inspect AI.
        It blocks until all sibling samples have also reached this point.

        Args:
            orchestration_id: Unique orchestration identifier
            role: This sample's role (e.g., "blue", "red")
            score: This sample's calculated score (for logging)
            timeout: Maximum time to wait for siblings (default 300s = 5min)

        Raises:
            asyncio.TimeoutError: If siblings don't complete within timeout

        Example:
            >>> coordinator = OrchestrationCoordinator()
            >>> # In scorer, after calculating score:
            >>> if orchestration_id:
            >>>     await coordinator.wait_for_all_scored(orchestration_id, role, total_score)
            >>> return Score(value=total_score, ...)  # Now all siblings ready
        """
        # Get orchestration group
        group = self._orchestrations.get(orchestration_id)
        if not group:
            # Not an orchestrated sample, return immediately
            logger.debug(
                f"No orchestration found for {orchestration_id}, skipping coordination",
                extra={"orchestration_id": orchestration_id, "role": role},
            )
            return

        sample = group.samples.get(role)
        if not sample:
            logger.warning(
                f"Sample {role} not registered in orchestration {orchestration_id}",
                extra={"orchestration_id": orchestration_id, "role": role},
            )
            return

        # Initialize coordination primitives if first sample
        with self._init_lock:
            if group.all_scored_event is None:
                group.all_scored_event = asyncio.Event()
                group.scored_samples = set()
                group.sample_scores = {}

        # Mark this sample as scored
        group.scored_samples.add(role)
        group.sample_scores[role] = score

        logger.info(
            f"Sample {role} completed scoring with score {score} ({len(group.scored_samples)}/{len(group.samples)})",
            extra={
                "orchestration_id": orchestration_id,
                "role": role,
                "score": score,
                "completed_samples": list(group.scored_samples),
                "total_samples": len(group.samples),
                "event": "orchestration_sample_scored",
            },
        )

        # When the first sample finishes scoring, trigger termination for siblings
        # This is needed for domains like airt_demo where one role (blue) monitors
        # indefinitely and needs to be terminated when the other role (red) finishes
        if len(group.scored_samples) == 1 and len(group.samples) > 1:
            logger.info(
                f"First sample {role} scored - triggering termination cascade for siblings",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "siblings": list(set(group.samples.keys()) - {role}),
                },
            )
            self.trigger_termination(orchestration_id, skip_role=role)

        # Check if all samples have scored
        if len(group.scored_samples) == len(group.samples):
            logger.info(
                f"All {len(group.samples)} samples scored in orchestration {orchestration_id}, releasing",
                extra={
                    "orchestration_id": orchestration_id,
                    "sample_scores": group.sample_scores,
                    "event": "orchestration_all_scored",
                },
            )
            group.all_scored_event.set()
            return  # Last sample, no need to wait

        # Wait for all siblings to complete scoring
        logger.info(
            f"Sample {role} waiting for {len(group.samples) - len(group.scored_samples)} siblings to complete scoring",
            extra={
                "orchestration_id": orchestration_id,
                "role": role,
                "waiting_for": list(set(group.samples.keys()) - group.scored_samples),
                "timeout": timeout,
                "event": "orchestration_waiting_for_siblings",
            },
        )

        try:
            await asyncio.wait_for(group.all_scored_event.wait(), timeout=timeout)
            logger.info(
                f"Sample {role} proceeding after all siblings scored",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "event": "orchestration_coordination_complete",
                },
            )
        except asyncio.TimeoutError:
            missing_roles = set(group.samples.keys()) - group.scored_samples
            logger.error(
                f"Orchestration timeout after {timeout}s - proceeding with partial results",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "completed_samples": list(group.scored_samples),
                    "missing_samples": list(missing_roles),
                    "timeout": timeout,
                    "event": "orchestration_timeout",
                },
            )
            # Set event to release other waiting samples
            group.all_scored_event.set()
            raise

    def reset(self) -> None:
        """Reset coordinator state (for testing)."""
        with self._init_lock:
            self._orchestrations.clear()
            logger.debug("Reset orchestration coordinator")


# Expose singleton instance
__all__ = ["OrchestrationCoordinator"]
