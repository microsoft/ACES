"""
Client Session Manager for SABER client-server communication.

Manages sessions, episodes, and policy interactions from the client side.
MCP tools are now handled natively by inspect_ai via mcp_server_http().
"""

import asyncio
from typing import List, Optional

import aiohttp

from ..logging_config import get_session_manager_logger
from ..models import (  # Use shared api models directly
    BenchmarkInfo,
    EpisodeCreateResponse,
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

    async def create_session(self) -> str:
        """
        Create a new SABER session via REST API.

        Returns:
            Session ID

        Raises:
            Exception: If session creation fails
        """
        logger.info(
            "Creating SABER session",
            extra={"event": "session_create_requested", "client_id": self.client_id},
        )

        url = f"{self.base_url}/api/v1/session"
        params = {"client_id": self.client_id}

        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    session_response = SessionCreateResponse(**data)
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
                    error_text = await response.text()
                    logger.error(
                        "Session creation failed",
                        extra={
                            "event": "session_create_failed",
                            "client_id": self.client_id,
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(f"Failed to create session: {response.status} - {error_text}")

    async def create_episode(self, session_id: str, task_id: str) -> EpisodeCreateResponse:
        """
        Create a new episode within a session via REST API.

        Args:
            session_id: Session ID
            task_id: Task ID for the episode

        Returns:
            EpisodeCreateResponse object from server

        Raises:
            Exception: If episode creation fails
        """
        logger.info(
            "Creating episode",
            extra={
                "event": "episode_create_requested",
                "session_id": session_id,
                "task_id": task_id,
            },
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes"
        params = {"task_id": task_id}

        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    episode_response = EpisodeCreateResponse(**data)

                    logger.info(
                        "Episode created",
                        extra={
                            "event": "episode_created",
                            "session_id": session_id,
                            "episode_id": episode_response.episode_id,
                            "task_id": task_id,
                            "attached_to_episode_id": episode_response.attached_to_episode_id,
                        },
                    )

                    if episode_response.attached_to_episode_id:
                        logger.info(
                            "Episode attached to dependency",
                            extra={
                                "event": "episode_dependency_attached",
                                "episode_id": episode_response.episode_id,
                                "attached_to_episode_id": episode_response.attached_to_episode_id,
                            },
                        )

                    return episode_response
                else:
                    error_text = await response.text()
                    logger.error(
                        "Episode creation failed",
                        extra={
                            "event": "episode_create_failed",
                            "session_id": session_id,
                            "task_id": task_id,
                            "status_code": response.status,
                            "response_text": error_text,
                        },
                    )
                    raise Exception(f"Failed to create episode: {response.status} - {error_text}")

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

    async def end_episode(
        self,
        session_id: str,
        episode_id: str,
        reason: str = "completed",
        result: Optional[EvalSubmission] = None,
        cascade_end_attached_episodes: bool = False,
    ) -> None:
        """
        End an episode via REST API.

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
                        logger.warning(
                            "Failed to end episode",
                            extra={
                                "event": "episode_end_failed",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "status_code": response.status,
                                "response_text": error_text,
                            },
                        )
        except Exception as exc:
            logger.warning(
                "Episode end request error",
                extra={
                    "event": "episode_end_request_error",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "error": str(exc),
                },
            )
            # Don't raise - episode end failures shouldn't break cleanup

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
