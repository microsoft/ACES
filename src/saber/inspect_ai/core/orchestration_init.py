"""Orchestrated sub-task initialization for multi-sample coordination.

This module handles initialization and cleanup of orchestrated sub-tasks in the
new multi-sample orchestration approach where each sub-task (e.g., blue, red) is
a separate sample coordinated via OrchestrationCoordinator.

Key features:
- Root vs dependent sample handling
- Semaphore acquisition for root samples
- Dependency waiting and episode coordination
- Cascade termination for cleanup
"""

import asyncio
from typing import Optional

from saber.client.client_session import ClientSessionManager
from saber.logging_config import LogCategory, get_saber_logger
from saber.models import MetadataKeys

from ..constants import SandboxTimeouts
from .orchestration_coordinator import OrchestrationCoordinator

logger = get_saber_logger(LogCategory.AGENT, __name__)


class SandboxError(Exception):
    """Exception raised for SABER sandbox lifecycle errors."""

    pass


class OrchestrationInitializer:
    """Handles orchestrated sub-task initialization and cleanup.

    Complements OrchestrationCoordinator (coordination) with
    initialization logic (episode setup, dependency waiting).
    """

    async def init_orchestrated_sub_task(
        self,
        metadata: dict[str, str],
        session_id: str,
        session_manager: ClientSessionManager,
        sample_id: str,
        semaphore: Optional[asyncio.Semaphore],
    ) -> dict:
        """Initialize episode for an orchestrated sub-task sample.

        This method handles the new multi-sample orchestration approach where each
        sub-task (e.g., blue, red) is a separate sample coordinated via OrchestrationCoordinator.

        Key behaviors:
        - Root sample (depends_on_role=None): Acquires semaphore, registers orchestration
        - Dependent sample: Joins orchestration, waits for dependency to be READY
        - All samples: Create single episode, record episode_id with coordinator

        Args:
            metadata: Sample metadata with orchestration info
            session_id: SABER session ID
            session_manager: Client session manager for API calls
            sample_id: Unique sample identifier
            semaphore: Optional episode concurrency semaphore

        Returns:
            Handler state dict with orchestration metadata

        Raises:
            SandboxError: If initialization fails
        """
        orchestration_id = metadata[MetadataKeys.ORCHESTRATION_ID]
        role = metadata[MetadataKeys.SUB_TASK_ROLE]
        task_id = metadata[MetadataKeys.TASK_ID]
        depends_on_role = metadata.get(MetadataKeys.DEPENDS_ON_ROLE)
        order = int(metadata[MetadataKeys.ORDER])

        coordinator = OrchestrationCoordinator()
        semaphore_acquired = False

        try:
            if depends_on_role is None:
                # Root sample - acquire semaphore and register orchestration
                if semaphore:
                    await semaphore.acquire()
                    semaphore_acquired = True
                    logger.debug(
                        f"Acquired semaphore for root sample {role} in orchestration {orchestration_id}",
                        extra={
                            "orchestration_id": orchestration_id,
                            "role": role,
                            "semaphore_available": semaphore._value if hasattr(semaphore, "_value") else "unknown",
                        },
                    )

                # Register root sample with coordinator
                success = coordinator.register_root_sample(
                    orchestration_id=orchestration_id,
                    role=role,
                    sample_id=sample_id,
                    semaphore=semaphore,
                )

                if not success:
                    raise SandboxError(f"Failed to register root sample for orchestration {orchestration_id}")
            else:
                # Dependent sample - register and wait for dependency
                success = await coordinator.register_dependent_sample(
                    orchestration_id=orchestration_id,
                    role=role,
                    sample_id=sample_id,
                    depends_on_role=depends_on_role,
                    order=order,
                    timeout=SandboxTimeouts.ORCHESTRATION_REGISTRATION_SECONDS,
                )

                if not success:
                    raise SandboxError(
                        f"Failed to register dependent sample {role} for orchestration {orchestration_id}"
                    )

                # Wait for dependency to be ready
                logger.info(
                    f"Sample {role} waiting for dependency {depends_on_role}",
                    extra={
                        "orchestration_id": orchestration_id,
                        "role": role,
                        "depends_on_role": depends_on_role,
                    },
                )

                dependency_episode_id = await coordinator.wait_for_dependency_ready(
                    orchestration_id=orchestration_id,
                    role=role,
                    session_manager=session_manager,
                    session_id=session_id,
                    timeout=SandboxTimeouts.ORCHESTRATION_DEPENDENCY_WAIT_SECONDS,
                )

                logger.info(
                    f"Dependency {depends_on_role} is ready with episode {dependency_episode_id}",
                    extra={
                        "orchestration_id": orchestration_id,
                        "role": role,
                        "depends_on_role": depends_on_role,
                        "dependency_episode_id": dependency_episode_id,
                    },
                )

            # Create single episode for this sub-task
            episode_response = await session_manager.create_episode(
                session_id,
                task_id,
            )
            episode_id = episode_response.episode_id

            logger.debug(
                f"Created episode for orchestrated sub-task {role}",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "task_id": task_id,
                    "episode_id": episode_id,
                },
            )

            # Wait for episode to be ready
            await session_manager.wait_for_episode_ready(
                session_id,
                episode_id,
            )

            # Record episode_id with coordinator (signals dependents)
            coordinator.set_episode_id(
                orchestration_id=orchestration_id,
                role=role,
                episode_id=episode_id,
            )

            # Create handler state for cleanup
            handler_state = {
                "episode_ids": [episode_id],
                "primary_episode_id": episode_id,
                "semaphore_acquired": semaphore_acquired,
                "orchestration_id": orchestration_id,
                "sub_task_role": role,
            }

            logger.info(
                f"Orchestrated sub-task {role} ready for execution",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "episode_id": episode_id,
                },
            )

            return handler_state

        except Exception as e:
            # Cleanup on failure
            if semaphore_acquired and semaphore:
                semaphore.release()
                logger.debug(
                    "Released semaphore after orchestrated sub-task init failure",
                    extra={
                        "orchestration_id": orchestration_id,
                        "role": role,
                    },
                )

            # Trigger cascade termination if orchestration was started
            try:
                coordinator.trigger_termination(orchestration_id)
            except Exception as term_err:
                logger.warning(
                    f"Failed to trigger termination during init failure: {term_err}",
                    extra={"orchestration_id": orchestration_id},
                )

            raise SandboxError(f"Failed to initialize orchestrated sub-task {role}: {e}") from e

    async def cleanup_orchestrated_sub_task(
        self,
        handler_state: dict,
        session_id: str,
        session_manager: ClientSessionManager,
        semaphore: Optional[asyncio.Semaphore],
    ) -> None:
        """Cleanup orchestrated sub-task sample with cascade termination.

        This method:
        1. Triggers cascade termination (marks all samples in orchestration for cleanup)
        2. Ends all episodes returned by coordinator
        3. Cleans up this sample
        4. Releases semaphore if this is the last sample

        Args:
            handler_state: Handler state dict with orchestration metadata
            session_id: SABER session ID
            session_manager: Client session manager for API calls
            semaphore: Optional episode concurrency semaphore

        Raises:
            SandboxError: If handler state is invalid
        """
        orchestration_id = handler_state["orchestration_id"]
        role = handler_state["sub_task_role"]
        coordinator = OrchestrationCoordinator()

        # Trigger cascade termination for entire orchestration
        episodes_to_cleanup = coordinator.trigger_termination(orchestration_id)

        logger.info(
            f"Triggered cascade termination for orchestration {orchestration_id}",
            extra={
                "orchestration_id": orchestration_id,
                "role": role,
                "episodes_to_cleanup": len(episodes_to_cleanup),
            },
        )

        # End all episodes in the orchestration
        for cleanup_role, episode_id in episodes_to_cleanup:
            try:
                await session_manager.end_episode(session_id, episode_id)
                logger.debug(
                    f"Ended episode for role {cleanup_role}",
                    extra={
                        "orchestration_id": orchestration_id,
                        "role": cleanup_role,
                        "episode_id": episode_id,
                    },
                )
            except Exception as e:
                logger.error(
                    f"Failed to end episode {episode_id} for role {cleanup_role}",
                    extra={
                        "orchestration_id": orchestration_id,
                        "role": cleanup_role,
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )

        # Cleanup this sample and check if semaphore should be released
        should_release = coordinator.cleanup_sample(
            orchestration_id=orchestration_id,
            role=role,
            semaphore=semaphore,
        )

        # Release semaphore if this is the last sample
        if should_release and semaphore:
            semaphore.release()
            logger.debug(
                f"Released semaphore after last sample cleanup in orchestration {orchestration_id}",
                extra={
                    "orchestration_id": orchestration_id,
                    "role": role,
                    "semaphore_available": semaphore._value if hasattr(semaphore, "_value") else "unknown",
                },
            )

        logger.info(
            f"Cleaned up orchestrated sub-task {role}",
            extra={
                "orchestration_id": orchestration_id,
                "role": role,
                "semaphore_released": should_release,
            },
        )
