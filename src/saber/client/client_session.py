"""
Client Session Manager for SABER client-server communication.

Manages sessions, episodes, and policy interactions from the client side.
MCP tools are now handled natively by inspect_ai via mcp_server_http().
"""

import asyncio
import random
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import aiohttp

from ..logging_config import get_session_manager_logger
from ..models import (  # Use shared api models directly
    BenchmarkInfo,
    EpisodeCreateResponse,
    EpisodeStatusResponse,
    EvalSubmission,
    PolicyResponse,
    SessionCreateResponse,
    TaskInfo,
)
from ..models.rest.evaluation import (
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    EvaluationCriteriaResponse,
    EvaluationOverrideRequest,
    EvaluationResultResponse,
    EvaluationResultSubmission,
    StepEvaluationCriteriaResponse,
    SubmissionEvaluationCriteriaResponse,
)
from ..server import EpisodeState
from .models import SessionManagerConfig

logger = get_session_manager_logger(__name__)


class ClientSessionManager:
    """
    Manages SABER sessions and episodes for client-server communication.

    Handles:
    - Session creation and management via REST API
    - Episode lifecycle via REST API
    - Policy endpoint integration
    - MCP tool discovery and execution
    """

    def __init__(self, config: SessionManagerConfig):
        """
        Initialize session manager with unified configuration.

        Args:
            config: Unified SessionManagerConfig
        """
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.client_id = config.client_id
        self.timeout = config.rest_timeout
        self._current_session_id: Optional[str] = None

        logger.debug(
            "ClientSessionManager initialized",
            extra={"base_url": self.base_url, "client_id": self.client_id},
        )

    async def _retry_request(
        self,
        operation_name: str,
        request_func: Callable[[], Awaitable[tuple[int, Any]]],
        max_retries: Optional[int] = None,
    ) -> tuple[int, Any]:
        """
        Execute HTTP request with retry logic for transient failures.

        Args:
            operation_name: Name of the operation for logging
            request_func: Async function that returns (status_code, response_data)
            max_retries: Maximum retry attempts (defaults to config.rest_max_retries)

        Returns:
            Tuple of (status_code, response_data)

        Raises:
            Exception: After max_retries attempts have failed
        """
        if max_retries is None:
            max_retries = self.config.rest_max_retries

        last_exception = None

        for attempt in range(max_retries):
            try:
                status, data = await request_func()
                if status == 200:
                    return status, data

                # Non-200 status, retry on server errors (5xx) or timeouts
                if status >= 500 or status == 408:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            f"{operation_name} failed with status {status}, retrying in {wait_time}s",
                            extra={
                                "event": f"{operation_name}_retry",
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "status_code": status,
                                "wait_time": wait_time,
                            },
                        )
                        await asyncio.sleep(wait_time)
                        continue

                # Client error (4xx) or final attempt, don't retry
                return status, data

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"{operation_name} failed with {type(e).__name__}, retrying in {wait_time}s",
                        extra={
                            "event": f"{operation_name}_retry",
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "wait_time": wait_time,
                        },
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"{operation_name} failed after all retries",
                        extra={
                            "event": f"{operation_name}_failed",
                            "attempts": max_retries,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

        # All retries exhausted
        error_msg = str(last_exception) if last_exception else "Unknown error"
        raise Exception(
            f"{operation_name} failed after {max_retries} attempts. "
            f"Last error: {type(last_exception).__name__}: {error_msg}"
        ) from last_exception

    async def create_session(self) -> str:
        """
        Create a new SABER session via REST API with retry logic.

        Returns:
            Session ID

        Raises:
            Exception: If session creation fails after retries
        """
        logger.info(
            "Creating SABER session",
            extra={"event": "session_create_requested", "client_id": self.client_id},
        )

        url = f"{self.base_url}/api/v1/session"
        params = {"client_id": self.client_id}

        async def _do_request() -> tuple[int, Any]:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.post(url, params=params, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return response.status, data
                    else:
                        error_text = await response.text()
                        return response.status, error_text

        status, result = await self._retry_request("create_session", _do_request)

        if status == 200:
            session_response = SessionCreateResponse(**result)
            self._current_session_id = session_response.session_id

            logger.info(
                "SABER session created",
                extra={
                    "event": "session_created",
                    "session_id": session_response.session_id,
                    "client_id": self.client_id,
                },
            )
            return session_response.session_id
        else:
            logger.error(
                "Session creation failed",
                extra={
                    "event": "session_create_failed",
                    "client_id": self.client_id,
                    "status_code": status,
                    "response_text": result,
                },
            )
            raise Exception(f"Failed to create session: {status} - {result}")

    async def create_episode(self, session_id: str, task_id: str) -> EpisodeCreateResponse:
        """
        Create a new episode within a session via REST API with retry logic.

        Args:
            session_id: Session ID
            task_id: Task ID for the episode

        Returns:
            EpisodeCreateResponse with episode details

        Raises:
            Exception: If episode creation fails after retries
        """
        url = f"{self.base_url}/api/v1/session/{session_id}/episodes"
        params = {"task_id": task_id}

        logger.info(
            "Creating episode",
            extra={
                "event": "episode_create_request",
                "session_id": session_id,
                "task_id": task_id,
            },
        )

        async def _do_request() -> tuple[int, Any]:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.post(url, params=params, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return response.status, data
                    else:
                        error_text = await response.text()
                        return response.status, error_text

        status, result = await self._retry_request("create_episode", _do_request)

        if status == 200:
            episode_response = EpisodeCreateResponse(**result)

            log_msg = (
                f"Episode creation request acknowledged - episode {episode_response.episode_id} "
                "initialization in progress"
            )
            log_data = {
                "event": "episode_creation_acknowledged",
                "session_id": session_id,
                "task_id": task_id,
                "episode_id": episode_response.episode_id,
                "state": episode_response.state,
            }

            if episode_response.attached_to_episode_id:
                log_msg += f" (attached to {episode_response.attached_to_episode_id})"
                log_data["attached_to_episode_id"] = episode_response.attached_to_episode_id

            logger.info(log_msg, extra=log_data)

            return episode_response
        else:
            logger.error(
                "Episode creation failed",
                extra={
                    "event": "episode_create_failed",
                    "session_id": session_id,
                    "task_id": task_id,
                    "status_code": status,
                    "error": result,
                },
            )
            raise Exception(f"Failed to create episode: {status} - {result}")

    async def get_episode_status(self, session_id: str, episode_id: str) -> EpisodeStatusResponse:
        """
        Get episode status for readiness polling.

        Args:
            session_id: Session ID
            episode_id: Episode ID

        Returns:
            EpisodeStatusResponse with current episode state

        Raises:
            Exception: If status request fails
        """
        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/status"

        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    return EpisodeStatusResponse(**data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get episode status: {response.status} - {error_text}")

    async def wait_for_episode_ready(
        self,
        session_id: str,
        episode_id: str,
        timeout_seconds: int = 300,
        min_poll_interval: float = 10.0,
    ) -> EpisodeStatusResponse:
        """
        Poll episode status until it becomes ready or fails.

        Args:
            session_id: Session ID
            episode_id: Episode ID
            timeout_seconds: Maximum time to wait (default 300s)
            min_poll_interval: Minimum seconds between status checks (default 10s)

        Returns:
            EpisodeStatusResponse when episode is ready

        Raises:
            TimeoutError: If episode not ready within timeout
            Exception: If episode creation failed
        """
        import time

        start_time = time.time()

        logger.info(
            f"Waiting for episode {episode_id} to become ready...",
            extra={
                "event": "wait_for_episode_ready_start",
                "session_id": session_id,
                "episode_id": episode_id,
                "timeout_seconds": timeout_seconds,
                "min_poll_interval": min_poll_interval,
            },
        )

        while True:
            status = await self.get_episode_status(session_id, episode_id)
            elapsed = time.time() - start_time

            if status.is_ready:
                logger.info(
                    f"Episode {episode_id} is ready (elapsed: {elapsed:.1f}s)",
                    extra={
                        "event": "wait_for_episode_ready_success",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "state": status.state,
                        "elapsed_seconds": elapsed,
                    },
                )
                return status

            if status.state == EpisodeState.FAILED_CREATION.value:
                error_msg = status.creation_error or "Unknown error"
                logger.error(
                    f"Episode {episode_id} creation failed: {error_msg}",
                    extra={
                        "event": "wait_for_episode_ready_failed",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "error": error_msg,
                    },
                )
                raise Exception(f"Episode creation failed: {error_msg}")

            if status.state in {EpisodeState.FAILED.value, EpisodeState.TIMEOUT.value}:
                raise Exception(f"Episode entered terminal state '{status.state}' before readiness")

            if status.state == EpisodeState.COMPLETED.value:
                raise Exception("Episode completed before readiness")

            if elapsed >= timeout_seconds:
                logger.error(
                    f"Episode {episode_id} readiness timeout after {elapsed:.1f}s",
                    extra={
                        "event": "wait_for_episode_ready_timeout",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": timeout_seconds,
                        "last_state": status.state,
                    },
                )
                raise TimeoutError(
                    f"Episode {episode_id} not ready after {timeout_seconds}s (current state: {status.state})"
                )

            # Log polling status update
            logger.info(
                f"Episode {episode_id} status: {status.state} (elapsed: {elapsed:.1f}s)",
                extra={
                    "event": "episode_status_poll",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "state": status.state,
                    "elapsed_seconds": elapsed,
                    "is_ready": status.is_ready,
                },
            )

            await asyncio.sleep(min_poll_interval)

    async def create_episode_and_wait(
        self, session_id: str, task_id: str, timeout_seconds: int = 300
    ) -> EpisodeCreateResponse:
        """
        Create episode and wait for it to become ready (convenience method).

        Args:
            session_id: Session ID
            task_id: Task ID
            timeout_seconds: Maximum time to wait for readiness

        Returns:
            EpisodeCreateResponse when episode is ready

        Raises:
            Exception: If creation or readiness check fails
        """
        # Create episode (returns immediately)
        response = await self.create_episode(session_id, task_id)

        # Wait for ready
        try:
            await self.wait_for_episode_ready(session_id, response.episode_id, timeout_seconds=timeout_seconds)
        except TimeoutError as e:
            # Re-raise with more context about which episode timed out
            logger.error(
                f"Episode {response.episode_id} failed to become ready within {timeout_seconds}s",
                extra={
                    "event": "create_episode_and_wait_timeout",
                    "session_id": session_id,
                    "task_id": task_id,
                    "episode_id": response.episode_id,
                    "timeout_seconds": timeout_seconds,
                },
            )
            raise TimeoutError(
                f"Episode {response.episode_id} for task '{task_id}' failed to become ready within {timeout_seconds}s. "
                f"Original error: {str(e)}"
            ) from e
        except Exception as e:
            # Re-raise other exceptions with context
            logger.error(
                f"Episode {response.episode_id} readiness check failed",
                extra={
                    "event": "create_episode_and_wait_failed",
                    "session_id": session_id,
                    "task_id": task_id,
                    "episode_id": response.episode_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise Exception(
                f"Episode {response.episode_id} for task '{task_id}' readiness check failed: "
                f"{type(e).__name__}: {str(e)}"
            ) from e

        return response

    async def get_policy_response(self, session_id: str, episode_id: str) -> Optional[PolicyResponse]:
        """
        Get policy response for dynamic prompt generation via REST API.

        Args:
            session_id: Session ID
            episode_id: Episode ID

        Returns:
            Policy response with prompt, or None if not available

        Raises:
            Exception: If policy request fails
        """
        logger.debug(
            "Fetching policy response",
            extra={
                "event": "policy_request_started",
                "session_id": session_id,
                "episode_id": episode_id,
            },
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/policy"

        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        policy_response = PolicyResponse(prompt=data.get("prompt", ""), domain=data.get("domain"))

                        logger.debug(
                            "Policy response received",
                            extra={
                                "event": "policy_request_completed",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "domain": policy_response.domain,
                            },
                        )
                        return policy_response
                    elif response.status == 404:
                        logger.debug(
                            "Policy not available",
                            extra={
                                "event": "policy_not_found",
                                "session_id": session_id,
                                "episode_id": episode_id,
                            },
                        )
                        return None
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get policy: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            logger.warning(
                "Policy request timed out",
                extra={
                    "event": "policy_request_timeout",
                    "session_id": session_id,
                    "episode_id": episode_id,
                },
            )
            return None
        except Exception as exc:
            logger.exception(
                "Policy request failed",
                extra={
                    "event": "policy_request_failed",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "error": str(exc),
                },
            )
            raise

    async def get_available_tasks(self) -> List[TaskInfo]:
        """
        Get available tasks from SABER server.

        Returns:
            List of TaskInfo objects from server

        Raises:
            Exception: If task discovery fails
        """
        logger.info(
            "Discovering available tasks",
            extra={"event": "task_discovery_started", "base_url": self.base_url},
        )

        url = f"{self.base_url}/api/v1/tasks"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Parse the BenchmarkInfo response
                    benchmark_info = BenchmarkInfo(**data)
                    logger.info(
                        "Tasks discovered",
                        extra={
                            "event": "task_discovery_completed",
                            "task_count": len(benchmark_info.tasks),
                        },
                    )
                    return benchmark_info.tasks
                else:
                    error_text = await response.text()
                    logger.error(
                        "Task discovery failed",
                        extra={
                            "event": "task_discovery_failed",
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(f"Failed to get available tasks: {response.status} - {error_text}")

    async def get_tasks(self, task_ids: List[str]) -> List[TaskInfo]:
        """
        Get full task data for specific task IDs.

        Args:
            task_ids: List of task IDs to retrieve

        Returns:
            List of TaskInfo objects

        Raises:
            Exception: If task retrieval fails
        """
        logger.info(
            "Retrieving tasks",
            extra={
                "event": "task_retrieval_requested",
                "requested_task_count": len(task_ids),
                "task_ids": task_ids,
            },
        )

        # Get all available tasks
        all_tasks = await self.get_available_tasks()

        # Filter to requested task IDs
        requested_tasks = [task for task in all_tasks if task.task_id in task_ids]

        # Check if all requested tasks were found
        found_ids = {task.task_id for task in requested_tasks}
        missing_ids = set(task_ids) - found_ids

        if missing_ids:
            available_ids = {task.task_id for task in all_tasks}
            logger.error(
                "Requested tasks not found",
                extra={
                    "event": "task_retrieval_missing",
                    "missing_task_ids": sorted(missing_ids),
                    "available_task_ids": sorted(available_ids),
                },
            )
            raise Exception(f"Tasks not found: {missing_ids}. Available tasks: {sorted(available_ids)}")

        logger.info(
            "Tasks retrieved",
            extra={
                "event": "task_retrieval_completed",
                "retrieved_task_count": len(requested_tasks),
            },
        )
        return requested_tasks

    async def terminate_session(self, session_id: str) -> None:
        """
        Terminate a session via REST API.

        Args:
            session_id: Session ID to terminate

        Raises:
            Exception: If session termination fails
        """
        logger.info(
            "Terminating session",
            extra={"event": "session_termination_requested", "session_id": session_id},
        )

        url = f"{self.base_url}/api/v1/session/{session_id}"

        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.delete(url, timeout=timeout) as response:
                if response.status == 200:
                    logger.info(
                        "Session terminated",
                        extra={"event": "session_terminated", "session_id": session_id},
                    )
                    if self._current_session_id == session_id:
                        self._current_session_id = None
                else:
                    error_text = await response.text()
                    logger.warning(
                        "Session termination failed",
                        extra={
                            "event": "session_termination_failed",
                            "session_id": session_id,
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    # Don't raise - termination failures shouldn't break cleanup

    async def verify_episode_ended(
        self,
        session_id: str,
        episode_id: str,
        timeout: float = 10.0,
    ) -> Tuple[bool, str]:
        """
        Verify if an episode has actually ended by checking its status.

        Args:
            session_id: Session ID
            episode_id: Episode ID to verify
            timeout: Timeout for verification request in seconds (default: 10.0)

        Returns:
            Tuple of (is_ended, state_description)
            - (True, "COMPLETED") - Episode successfully ended
            - (True, "FAILED") - Episode ended with failure
            - (False, "ACTIVE") - Episode still active
            - (False, "CREATING") - Episode still creating
            - (False, "UNKNOWN") - Cannot determine state
        """
        try:
            # Use the proper status endpoint
            url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/status"
            async with aiohttp.ClientSession() as session:
                client_timeout = aiohttp.ClientTimeout(total=timeout)
                async with session.get(url, timeout=client_timeout) as response:
                    if response.status == 404:
                        # Episode not found - likely moved to completed episodes
                        logger.debug(
                            "Episode not found via status endpoint (likely completed)",
                            extra={
                                "event": "episode_verification_not_found",
                                "session_id": session_id,
                                "episode_id": episode_id,
                            },
                        )
                        return (True, "COMPLETED")

                    if response.status == 200:
                        data = await response.json()
                        state = data.get("state", "UNKNOWN")

                        # Episode states that indicate completion
                        # Compare against enum values (handling case-insensitivity)
                        state_upper = state.upper()
                        terminal_states = {
                            EpisodeState.COMPLETED.value.upper(),
                            EpisodeState.FAILED.value.upper(),
                            EpisodeState.FAILED_CREATION.value.upper(),
                        }

                        if state_upper in terminal_states:
                            logger.debug(
                                f"Episode verified as ended with state: {state}",
                                extra={
                                    "event": "episode_verification_ended",
                                    "session_id": session_id,
                                    "episode_id": episode_id,
                                    "state": state,
                                },
                            )
                            return (True, state)
                        else:
                            logger.debug(
                                f"Episode still active with state: {state}",
                                extra={
                                    "event": "episode_verification_active",
                                    "session_id": session_id,
                                    "episode_id": episode_id,
                                    "state": state,
                                },
                            )
                            return (False, state)

                    # Unexpected status code
                    logger.warning(
                        f"Unexpected status code during verification: {response.status}",
                        extra={
                            "event": "episode_verification_unexpected_status",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "status_code": response.status,
                        },
                    )
                    # Try to get response text for debugging
                    try:
                        response_text = await response.text()
                        logger.warning(
                            f"Verification response body: {response_text[:500]}",
                            extra={
                                "event": "episode_verification_response_body",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "response_text": response_text[:500],
                            },
                        )
                    except Exception:
                        pass
                    return (False, "UNKNOWN")

        except asyncio.TimeoutError:
            logger.warning(
                "Timeout during episode verification",
                extra={
                    "event": "episode_verification_timeout",
                    "session_id": session_id,
                    "episode_id": episode_id,
                },
                exc_info=True,
            )
            return (False, "UNKNOWN")
        except aiohttp.ClientError as e:
            logger.warning(
                "Network error during episode verification",
                extra={
                    "event": "episode_verification_network_error",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return (False, "UNKNOWN")
        except Exception as e:
            logger.error(
                f"Failed to verify episode state: {type(e).__name__}: {str(e)}",
                extra={
                    "event": "episode_verification_error",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_module": type(e).__module__,
                },
                exc_info=True,  # This will include full traceback
            )
            return (False, "UNKNOWN")

    async def end_episode_with_retry(
        self,
        session_id: str,
        episode_id: str,
        reason: str = "completed",
        result: Optional[EvalSubmission] = None,
        cascade_end_attached_episodes: bool = False,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> bool:
        """
        End an episode with retry logic and verification.

        This method implements robust episode ending with:
        - Multiple retry attempts with exponential backoff
        - Verification of episode state after attempts
        - Detection of already-ended episodes (idempotent)
        - Comprehensive error logging

        Args:
            session_id: Session ID
            episode_id: Episode ID
            reason: Completion reason
            result: Optional EvalSubmission data
            cascade_end_attached_episodes: If True, also end attached episodes
            max_retries: Maximum number of retry attempts (default: 5)
            initial_backoff: Initial backoff delay in seconds (default: 1.0)
            max_backoff: Maximum backoff delay in seconds (default: 30.0)

        Returns:
            True if episode was successfully ended and verified, False otherwise
        """
        backoff = initial_backoff
        last_error = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                logger.info(
                    f"Attempting to end episode (attempt {attempt + 1}/{max_retries + 1})",
                    extra={
                        "event": "episode_end_attempt",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                        "backoff": backoff if attempt > 0 else 0,
                    },
                )

                # Attempt to end the episode
                await self.end_episode(
                    session_id=session_id,
                    episode_id=episode_id,
                    reason=reason,
                    result=result,
                    cascade_end_attached_episodes=cascade_end_attached_episodes,
                )

                # Verify the episode actually ended
                # Add small delay before verification to let server process
                await asyncio.sleep(0.5)

                # Try verification with a couple of quick retries for transient issues
                is_ended = False
                state = "UNKNOWN"
                for verify_attempt in range(2):  # 2 attempts max
                    is_ended, state = await self.verify_episode_ended(
                        session_id, episode_id, timeout=10.0  # Longer timeout for verification
                    )

                    if is_ended or state != "UNKNOWN":
                        break  # Got a definitive answer

                    if verify_attempt < 1:  # Only sleep before second attempt
                        await asyncio.sleep(1.0)

                if is_ended:
                    logger.info(
                        f"Episode successfully ended and verified (state: {state})",
                        extra={
                            "event": "episode_end_verified_success",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "state": state,
                            "attempts_required": attempt + 1,
                        },
                    )
                    return True
                elif state == "UNKNOWN":
                    # If we got 200 OK from DELETE but verification is uncertain,
                    # treat as success rather than retrying endlessly
                    # (verification might fail due to timing or endpoint issues)
                    logger.warning(
                        "Episode end succeeded (200 OK) but verification inconclusive - treating as success",
                        extra={
                            "event": "episode_end_verification_inconclusive",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "state": state,
                            "attempt": attempt + 1,
                            "note": "Check logs above for verification error details",
                        },
                    )
                    return True
                else:
                    logger.warning(
                        f"Episode end succeeded but verification shows state: {state}",
                        extra={
                            "event": "episode_end_verification_mismatch",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "state": state,
                            "attempt": attempt + 1,
                        },
                    )
                    # Continue to retry - episode is still ACTIVE/CREATING
                    last_error = f"Episode still in state {state} after end request"

            except RuntimeError as e:
                error_str = str(e)

                # Check if episode is already ended (400 or 404 errors are OK)
                if "400" in error_str or "not active" in error_str.lower():
                    # Might be already ended, verify
                    is_ended, state = await self.verify_episode_ended(session_id, episode_id)
                    if is_ended:
                        logger.info(
                            f"Episode already ended (idempotent success, state: {state})",
                            extra={
                                "event": "episode_end_already_ended",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "state": state,
                            },
                        )
                        return True
                    else:
                        logger.error(
                            "Episode returned 'not active' but verification shows still active",
                            extra={
                                "event": "episode_end_inconsistent_state",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "state": state,
                            },
                        )
                        last_error = f"Inconsistent state: {error_str}"
                else:
                    # Server error, retryable
                    last_error = error_str
                    logger.warning(
                        f"Episode end attempt {attempt + 1} failed: {error_str}",
                        extra={
                            "event": "episode_end_attempt_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "attempt": attempt + 1,
                            "error": error_str,
                        },
                    )

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                # Network/timeout errors are retryable
                last_error = f"{type(e).__name__}: {str(e)}"
                logger.warning(
                    f"Episode end attempt {attempt + 1} failed with network error",
                    extra={
                        "event": "episode_end_network_error",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "attempt": attempt + 1,
                        "error": last_error,
                    },
                )

                # After timeout, verify if episode ended anyway
                is_ended, state = await self.verify_episode_ended(session_id, episode_id)
                if is_ended:
                    logger.info(
                        f"Episode ended despite timeout (state: {state})",
                        extra={
                            "event": "episode_end_timeout_but_verified",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "state": state,
                        },
                    )
                    return True

            except Exception as e:
                # Unexpected error
                last_error = f"Unexpected {type(e).__name__}: {str(e)}"
                logger.error(
                    f"Episode end attempt {attempt + 1} failed with unexpected error",
                    extra={
                        "event": "episode_end_unexpected_error",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "attempt": attempt + 1,
                        "error": last_error,
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

            # If not last attempt, apply backoff
            if attempt < max_retries:
                # Add jitter to prevent thundering herd
                jitter = random.uniform(-0.1 * backoff, 0.1 * backoff)
                sleep_time = min(backoff + jitter, max_backoff)

                logger.debug(
                    f"Retrying episode end after {sleep_time:.2f}s",
                    extra={
                        "event": "episode_end_retry_backoff",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "backoff_seconds": sleep_time,
                        "next_attempt": attempt + 2,
                    },
                )

                await asyncio.sleep(sleep_time)
                backoff *= 2.0  # Exponential backoff

        # All retries exhausted - final verification
        is_ended, state = await self.verify_episode_ended(session_id, episode_id)

        if is_ended:
            logger.warning(
                f"Episode ended but required all {max_retries + 1} attempts (state: {state})",
                extra={
                    "event": "episode_end_retries_exhausted_but_verified",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "state": state,
                    "last_error": last_error,
                },
            )
            return True
        else:
            logger.error(
                f"Failed to end episode after {max_retries + 1} attempts (state: {state})",
                extra={
                    "event": "episode_end_failed_all_retries",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "state": state,
                    "attempts": max_retries + 1,
                    "last_error": last_error,
                },
            )
            return False

    async def end_episode(
        self,
        session_id: str,
        episode_id: str,
        reason: str = "completed",
        result: Optional[EvalSubmission] = None,
        cascade_end_attached_episodes: bool = False,
    ) -> None:
        """
        End an episode via REST API (single attempt, no retry).

        For robust episode ending with retry logic, use end_episode_with_retry() instead.

        Args:
            session_id: Session ID
            episode_id: Episode ID
            reason: Completion reason
            result: Optional EvalSubmission data
            cascade_end_attached_episodes: If True, also end episodes that this episode is attached to
        """
        logger.debug(
            "Ending episode",
            extra={
                "event": "episode_end_requested",
                "session_id": session_id,
                "episode_id": episode_id,
                "reason": reason,
                "cascade_end_attached_episodes": cascade_end_attached_episodes,
                "has_result": result is not None,
            },
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}"
        params = {"reason": reason, "cascade_end_attached_episodes": str(cascade_end_attached_episodes).lower()}
        data = None
        headers = {}

        if result:
            result_json = result.model_dump_json()
            # Send as request body instead of query parameter for large data
            data = result_json
            headers["Content-Type"] = "application/json"

        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.delete(url, params=params, data=data, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        logger.debug(
                            "Episode ended",
                            extra={
                                "event": "episode_end_completed",
                                "session_id": session_id,
                                "episode_id": episode_id,
                            },
                        )
                    else:
                        error_text = await response.text()
                        logger.error(
                            "Failed to end episode - server returned error",
                            extra={
                                "event": "episode_end_failed",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "status_code": response.status,
                                "response_text": error_text,
                                "url": url,
                                "reason": reason,
                            },
                        )
                        # Raise exception so caller knows the operation failed
                        raise RuntimeError(
                            f"Failed to end episode {episode_id}: " f"HTTP {response.status} - {error_text}"
                        )
        except asyncio.TimeoutError:
            # Timeout during episode end - the server likely received and processed the request
            # but the response didn't arrive in time. This is usually not a critical error.
            logger.warning(
                "Timeout waiting for episode end response (episode likely ended successfully on server)",
                extra={
                    "event": "episode_end_timeout",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "timeout": self.timeout,
                    "reason": reason,
                    "url": url,
                    "note": "Server likely processed the request successfully despite timeout",
                },
            )
            # Don't re-raise - treat as non-fatal since episode likely ended on server
        except aiohttp.ClientError as exc:
            logger.error(
                "Episode end request error",
                extra={
                    "event": "episode_end_request_error",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "error": str(exc),
                },
            )
            raise  # Re-raise so caller knows the operation failed
        except RuntimeError:
            # Re-raise RuntimeError from non-200 response above
            raise
        except Exception as exc:
            logger.error(
                f"Unexpected error during episode end: {type(exc).__name__}: {exc}",
                extra={
                    "event": "episode_end_unexpected_error",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "error_module": type(exc).__module__,
                    "url": url,
                    "reason": reason,
                    "has_result": result is not None,
                    "cascade_end_attached_episodes": cascade_end_attached_episodes,
                },
                exc_info=True,  # This will include the full traceback in the logs
            )
            raise

    async def update_episode_status(self, episode_id: str, status: str) -> None:
        """
        Update episode status.

        Args:
            episode_id: Episode ID
            status: New status
        """
        logger.debug(
            "Updating episode status",
            extra={
                "event": "episode_status_update_requested",
                "episode_id": episode_id,
                "status": status,
            },
        )

        # TODO: Implement if server supports status updates
        await asyncio.sleep(0.01)

    async def get_episode_evaluation(self, session_id: str, episode_id: str) -> EvaluationResultResponse:
        """
        Get evaluation result for a specific episode.

        Args:
            session_id: Session ID containing the episode
            episode_id: Episode ID to get evaluation for

        Returns:
            Evaluation result data

        Raises:
            Exception: If evaluation retrieval fails
        """
        logger.debug(
            "Fetching episode evaluation",
            extra={
                "event": "evaluation_fetch_requested",
                "session_id": session_id,
                "episode_id": episode_id,
            },
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/evaluations/{episode_id}"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(
                        "Episode evaluation retrieved",
                        extra={
                            "event": "evaluation_fetch_completed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                    return EvaluationResultResponse(**data["evaluation_result"])
                elif response.status == 404:
                    logger.warning(
                        "Episode evaluation not found",
                        extra={
                            "event": "evaluation_not_found",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                    raise Exception(f"Evaluation not found for episode {episode_id}")
                else:
                    error_text = await response.text()
                    logger.error(
                        "Failed to fetch episode evaluation",
                        extra={
                            "event": "evaluation_fetch_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(
                        f"Failed to get evaluation for episode {episode_id}: {response.status} - {error_text}"
                    )

    async def get_session_evaluations(
        self, session_id: str, task_id: Optional[str] = None
    ) -> List[EvaluationResultResponse]:
        """
        Get all evaluation results for a session.

        Args:
            session_id: Session ID to get evaluations for
            task_id: Optional task ID filter

        Returns:
            List of evaluation result data

        Raises:
            Exception: If evaluation retrieval fails
        """
        logger.debug(
            "Fetching session evaluations",
            extra={
                "event": "session_evaluations_fetch_requested",
                "session_id": session_id,
                "task_id": task_id,
            },
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/evaluations"
        params = {}
        if task_id:
            params["task_id"] = task_id

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(
                        "Session evaluations retrieved",
                        extra={
                            "event": "session_evaluations_fetch_completed",
                            "session_id": session_id,
                            "task_id": task_id,
                            "evaluation_count": data["total_count"],
                        },
                    )
                    return [EvaluationResultResponse(**eval_data) for eval_data in data["evaluations"]]
                else:
                    error_text = await response.text()
                    logger.error(
                        "Failed to fetch session evaluations",
                        extra={
                            "event": "session_evaluations_fetch_failed",
                            "session_id": session_id,
                            "task_id": task_id,
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(
                        f"Failed to get evaluations for session {session_id}: {response.status} - {error_text}"
                    )

    async def get_evaluation_criteria(self, session_id: str, episode_id: str) -> EvaluationCriteriaResponse:
        """
        Get evaluation criteria package for client-side evaluation.

        Args:
            session_id: Session ID containing the episode
            episode_id: Episode ID to get evaluation criteria for

        Returns:
            Evaluation criteria package containing task context, evaluation config, and submission

        Raises:
            Exception: If evaluation criteria retrieval fails
        """
        logger.debug(
            "Fetching evaluation criteria",
            extra={
                "event": "evaluation_criteria_fetch_requested",
                "session_id": session_id,
                "episode_id": episode_id,
            },
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/evaluation-criteria"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(
                        "Evaluation criteria retrieved",
                        extra={
                            "event": "evaluation_criteria_fetch_completed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                    # Parse the response into the proper model
                    return EvaluationCriteriaResponse(**data)
                elif response.status == 404:
                    logger.warning(
                        "Evaluation criteria not found",
                        extra={
                            "event": "evaluation_criteria_not_found",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                    raise Exception(f"Episode {episode_id} not found or no evaluation criteria available")
                elif response.status == 400:
                    error_text = await response.text()
                    logger.error(
                        "Invalid evaluation criteria request",
                        extra={
                            "event": "evaluation_criteria_invalid_request",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(f"Invalid request for evaluation criteria: {error_text}")
                else:
                    error_text = await response.text()
                    logger.error(
                        "Failed to fetch evaluation criteria",
                        extra={
                            "event": "evaluation_criteria_fetch_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(
                        f"Failed to get evaluation criteria for episode {episode_id}: {response.status} - {error_text}"
                    )

    async def override_episode_evaluation(
        self, session_id: str, episode_id: str, override_request: EvaluationOverrideRequest
    ) -> dict:
        """
        Submit evaluation override to server via REST API.

        Args:
            session_id: Session ID containing the episode
            episode_id: Episode ID to override evaluation for
            override_request: Strongly typed override request object

        Returns:
            Server response dict confirming override submission

        Raises:
            Exception: If override submission fails
        """
        logger.debug(
            "Submitting evaluation override",
            extra={
                "event": "evaluation_override_submitted",
                "session_id": session_id,
                "episode_id": episode_id,
            },
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/evaluations/{episode_id}/override"

        # Convert override request to JSON format for HTTP transmission
        request_data = override_request.model_dump()

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.put(url, json=request_data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    logger.debug(
                        "Evaluation override accepted",
                        extra={
                            "event": "evaluation_override_completed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                    return dict(response_data)
                elif response.status == 404:
                    logger.warning(
                        "Evaluation override target not found",
                        extra={
                            "event": "evaluation_override_not_found",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                    raise Exception(f"Episode {episode_id} not found for override submission")
                elif response.status == 400:
                    error_text = await response.text()
                    logger.error(
                        "Invalid evaluation override payload",
                        extra={
                            "event": "evaluation_override_invalid_request",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(f"Invalid override request for episode {episode_id}: {error_text}")
                else:
                    error_text = await response.text()
                    logger.error(
                        "Evaluation override submission failed",
                        extra={
                            "event": "evaluation_override_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(
                        f"Failed to submit override for episode {episode_id}: {response.status} - {error_text}"
                    )

    def get_current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session_id

    def get_or_create_session_id(self) -> str:
        """Get current session ID or raise if none exists."""
        if not self._current_session_id:
            raise RuntimeError("No active session - call create_session() first")
        return self._current_session_id

    async def upload_evaluation_file(self, session_id: str, file_path: str, timeout: Optional[float] = None) -> dict:
        """
        Upload an evaluation file to the server via REST API.

        Args:
            session_id: Session ID to upload file for
            file_path: Absolute path to the .eval file to upload
            timeout: Upload timeout in seconds (defaults to instance timeout)

        Returns:
            dict: Upload response containing file info and status

        Raises:
            Exception: If file doesn't exist or upload fails
        """
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Eval file not found: {file_path}")

        filename = os.path.basename(file_path)
        logger.info(
            "Uploading evaluation file",
            extra={
                "event": "evaluation_file_upload_started",
                "session_id": session_id,
                "filename": filename,
                "file_path": file_path,
            },
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/evaluations/upload"

        # Read file for upload
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Create form data for file upload
        data = aiohttp.FormData()
        data.add_field("file", file_content, filename=filename, content_type="application/octet-stream")

        # Use provided timeout or fall back to instance timeout
        upload_timeout = timeout if timeout is not None else self.timeout

        async with aiohttp.ClientSession() as session:
            timeout_config = aiohttp.ClientTimeout(total=upload_timeout)
            async with session.post(url, data=data, timeout=timeout_config) as response:
                if response.status == 200:
                    response_data = await response.json()
                    logger.info(
                        "Evaluation file uploaded",
                        extra={
                            "event": "evaluation_file_upload_completed",
                            "session_id": session_id,
                            "filename": filename,
                        },
                    )
                    return dict(response_data)
                else:
                    error_text = await response.text()
                    logger.error(
                        "Evaluation file upload failed",
                        extra={
                            "event": "evaluation_file_upload_failed",
                            "session_id": session_id,
                            "filename": filename,
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(f"Failed to upload evaluation file {filename}: {response.status} - {error_text}")

    # ============================================================================
    # NEW CLIENT-SIDE EVALUATION METHODS (Breaking Change Migration)
    # ============================================================================

    async def get_episode_submission(self, session_id: str, episode_id: str) -> EpisodeSubmissionResponse:
        """
        Get episode submission data for client-side evaluation.

        Args:
            session_id: Session ID
            episode_id: Episode ID

        Returns:
            EpisodeSubmissionResponse with submission data

        Raises:
            Exception: If request fails
        """
        logger.debug(
            "Fetching episode submission",
            extra={"event": "get_episode_submission", "session_id": session_id, "episode_id": episode_id},
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/submission"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return EpisodeSubmissionResponse(**data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get episode submission: {response.status} - {error_text}")

    async def get_episode_steps(self, session_id: str, episode_id: str) -> EpisodeStepsResponse:
        """
        Get episode step history for client-side evaluation.

        Args:
            session_id: Session ID
            episode_id: Episode ID

        Returns:
            EpisodeStepsResponse with step data

        Raises:
            Exception: If request fails
        """
        logger.debug(
            "Fetching episode steps",
            extra={"event": "get_episode_steps", "session_id": session_id, "episode_id": episode_id},
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/steps"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return EpisodeStepsResponse(**data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get episode steps: {response.status} - {error_text}")

    async def get_submission_evaluation_criteria(
        self, session_id: str, episode_id: str
    ) -> SubmissionEvaluationCriteriaResponse:
        """
        Get submission evaluation criteria (template paths only, no rendering).

        Args:
            session_id: Session ID
            episode_id: Episode ID

        Returns:
            SubmissionEvaluationCriteriaResponse with template paths

        Raises:
            Exception: If request fails
        """
        logger.debug(
            "Fetching submission evaluation criteria",
            extra={"event": "get_submission_evaluation_criteria", "session_id": session_id, "episode_id": episode_id},
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/submission-evaluation-criteria"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return SubmissionEvaluationCriteriaResponse(**data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get submission evaluation criteria: {response.status} - {error_text}")

    async def get_step_evaluation_criteria(
        self, session_id: str, episode_id: str
    ) -> Optional[StepEvaluationCriteriaResponse]:
        """
        Get step evaluation criteria (template paths only, no rendering).
        Returns None if step evaluation is not configured for the task.

        Args:
            session_id: Session ID
            episode_id: Episode ID

        Returns:
            StepEvaluationCriteriaResponse with template paths, or None if not configured

        Raises:
            Exception: If request fails (except 404 which returns None)
        """
        logger.debug(
            "Fetching step evaluation criteria",
            extra={"event": "get_step_evaluation_criteria", "session_id": session_id, "episode_id": episode_id},
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/step-evaluation-criteria"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return StepEvaluationCriteriaResponse(**data)
                elif response.status == 404:
                    # Step evaluation is optional
                    return None
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get step evaluation criteria: {response.status} - {error_text}")

    async def get_template_content(self, template_path: str) -> str:
        """
        Get raw template content by path.

        Args:
            template_path: Relative template path (e.g., 'judge/submission/system.md')

        Returns:
            Raw template content string

        Raises:
            Exception: If request fails
        """
        logger.debug(
            "Fetching template content",
            extra={"event": "get_template_content", "template_path": template_path},
        )

        url = f"{self.base_url}/api/v1/templates/{template_path}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["content"]
                    assert isinstance(content, str), "Template content must be a string"
                    return content
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get template content: {response.status} - {error_text}")

    async def submit_evaluation_result(
        self, session_id: str, episode_id: str, evaluation_data: EvaluationResultSubmission
    ) -> EvaluationResultResponse:
        """
        Submit client-side evaluation result.

        Args:
            session_id: Session ID
            episode_id: Episode ID
            evaluation_data: Evaluation result data

        Returns:
            EvaluationResultResponse

        Raises:
            Exception: If request fails
        """
        logger.info(
            "Submitting evaluation result",
            extra={"event": "submit_evaluation_result", "session_id": session_id, "episode_id": episode_id},
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/evaluation"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=evaluation_data.model_dump()) as response:
                if response.status == 200:
                    data = await response.json()
                    response_obj = data.get("evaluation_result", data)
                    return EvaluationResultResponse(**response_obj)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to submit evaluation result: {response.status} - {error_text}")

    async def cleanup(self) -> None:
        """Cleanup session manager resources."""
        if self._current_session_id:
            await self.terminate_session(self._current_session_id)

        logger.info(
            "ClientSessionManager cleanup completed",
            extra={"event": "client_session_manager_cleanup", "session_id": self._current_session_id},
        )
