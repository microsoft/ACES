"""
Client Session Manager for SABER client-server communication.

Manages sessions, episodes, and policy interactions from the client side.
MCP tools are now handled natively by inspect_ai via mcp_server_http().
"""

import asyncio
import logging
from typing import List, Optional

import aiohttp

from ..models import (  # Use shared api models directly
    BenchmarkInfo,
    EpisodeCreateResponse,
    EvalSubmission,
    PolicyResponse,
    SessionCreateResponse,
    TaskInfo,
)
from ..models.rest.evaluation import EvaluationCriteriaResponse, EvaluationOverrideRequest, EvaluationResultResponse
from .models import SessionManagerConfig

logger = logging.getLogger(__name__)


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

        logger.debug(f"Initialized ClientSessionManager for: {self.base_url}")

    async def create_session(self) -> str:
        """
        Create a new SABER session via REST API.

        Returns:
            Session ID

        Raises:
            Exception: If session creation fails
        """
        logger.info(f"Creating new SABER session for client: {self.client_id}")

        url = f"{self.base_url}/api/v1/session"
        params = {"client_id": self.client_id}

        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    session_response = SessionCreateResponse(**data)
                    self._current_session_id = session_response.session_id

                    logger.info(f"Created session: {session_response.session_id}")
                    return session_response.session_id
                else:
                    error_text = await response.text()
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
        logger.info(f"Creating episode for session {session_id}, task {task_id}")

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes"
        params = {"task_id": task_id}

        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    episode_response = EpisodeCreateResponse(**data)

                    logger.info(f"Created episode: {episode_response.episode_id}")
                    return episode_response
                else:
                    error_text = await response.text()
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
        logger.debug(f"Getting policy response for episode {episode_id}")

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/policy"

        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        policy_response = PolicyResponse(prompt=data.get("prompt", ""), domain=data.get("domain"))

                        logger.debug(f"Retrieved policy response for episode {episode_id}")
                        return policy_response
                    elif response.status == 404:
                        logger.debug(f"No policy available for episode {episode_id}")
                        return None
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get policy: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            logger.warning(f"Policy request timeout for episode {episode_id}")
            return None
        except Exception as e:
            logger.error(f"Policy request failed for episode {episode_id}: {e}")
            raise

    async def get_available_tasks(self) -> List[TaskInfo]:
        """
        Get all available tasks from SABER server.

        Returns:
            List of TaskInfo objects from server

        Raises:
            Exception: If task discovery fails
        """
        logger.info("Discovering all available tasks via REST API")

        url = f"{self.base_url}/api/v1/benchmark"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Parse the BenchmarkInfo response
                    benchmark_info = BenchmarkInfo(**data)
                    logger.info(f"Discovered {len(benchmark_info.tasks)} available tasks")
                    return benchmark_info.tasks
                else:
                    error_text = await response.text()
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
        logger.info(f"Retrieving {len(task_ids)} tasks via REST API")

        # Get all available tasks as TaskInfo objects
        all_tasks = await self.get_available_tasks()

        # Filter to requested task IDs
        requested_tasks = [task for task in all_tasks if task.task_id in task_ids]

        # Check if all requested tasks were found
        found_ids = {task.task_id for task in requested_tasks}
        missing_ids = set(task_ids) - found_ids

        if missing_ids:
            available_ids = {task.task_id for task in all_tasks}
            raise Exception(f"Tasks not found: {missing_ids}. " f"Available tasks: {sorted(available_ids)}")

        logger.info(f"Retrieved {len(requested_tasks)} tasks successfully as TaskInfo objects")
        return requested_tasks

    async def terminate_session(self, session_id: str) -> None:
        """
        Terminate a session via REST API.

        Args:
            session_id: Session ID to terminate

        Raises:
            Exception: If session termination fails
        """
        logger.info(f"Terminating session: {session_id}")

        url = f"{self.base_url}/api/v1/session/{session_id}"

        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.delete(url, timeout=timeout) as response:
                if response.status == 200:
                    logger.info(f"Session {session_id} terminated successfully")
                    if self._current_session_id == session_id:
                        self._current_session_id = None
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to terminate session {session_id}: {response.status} - {error_text}")
                    # Don't raise - termination failures shouldn't break cleanup

    async def end_episode(
        self, session_id: str, episode_id: str, reason: str = "completed", result: Optional[EvalSubmission] = None
    ) -> None:
        """
        End an episode via REST API.

        Args:
            session_id: Session ID
            episode_id: Episode ID
            reason: Completion reason
            result: Optional EvalSubmission data
        """
        logger.debug(f"Ending episode {episode_id} with reason: {reason}")

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}"
        params = {"reason": reason}
        if result:
            params["result"] = result.model_dump_json()

        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.delete(url, params=params, timeout=timeout) as response:
                    if response.status == 200:
                        logger.debug(f"Episode {episode_id} ended successfully")
                    else:
                        error_text = await response.text()
                        logger.warning(f"Failed to end episode {episode_id}: {response.status} - {error_text}")
        except Exception as e:
            logger.warning(f"Episode end request failed for {episode_id}: {e}")
            # Don't raise - episode end failures shouldn't break cleanup

    async def update_episode_status(self, episode_id: str, status: str) -> None:
        """
        Update episode status.

        Args:
            episode_id: Episode ID
            status: New status
        """
        logger.debug(f"Updating episode {episode_id} status to: {status}")

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
        logger.debug(f"Getting evaluation for episode {episode_id} in session {session_id}")

        url = f"{self.base_url}/api/v1/session/{session_id}/evaluations/{episode_id}"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Retrieved evaluation for episode {episode_id}")
                    return EvaluationResultResponse(**data["evaluation_result"])
                elif response.status == 404:
                    raise Exception(f"Evaluation not found for episode {episode_id}")
                else:
                    error_text = await response.text()
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
            f"Getting evaluations for session {session_id}" + (f" with task filter {task_id}" if task_id else "")
        )

        url = f"{self.base_url}/api/v1/session/{session_id}/evaluations"
        params = {}
        if task_id:
            params["task_id"] = task_id

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Retrieved {data['total_count']} evaluations for session {session_id}")
                    return [EvaluationResultResponse(**eval_data) for eval_data in data["evaluations"]]
                else:
                    error_text = await response.text()
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
        logger.debug(f"Getting evaluation criteria for episode {episode_id} in session {session_id}")

        url = f"{self.base_url}/api/v1/session/{session_id}/episodes/{episode_id}/evaluation-criteria"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Retrieved evaluation criteria for episode {episode_id}")
                    # Parse the response into the proper model
                    return EvaluationCriteriaResponse(**data)
                elif response.status == 404:
                    raise Exception(f"Episode {episode_id} not found or no evaluation criteria available")
                elif response.status == 400:
                    error_text = await response.text()
                    raise Exception(f"Invalid request for evaluation criteria: {error_text}")
                else:
                    error_text = await response.text()
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
        logger.debug(f"Submitting evaluation override for episode {episode_id} in session {session_id}")

        url = f"{self.base_url}/api/v1/session/{session_id}/evaluations/{episode_id}/override"

        # Convert override request to JSON format for HTTP transmission
        request_data = override_request.model_dump()

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.put(url, json=request_data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    logger.debug(f"Successfully submitted override for episode {episode_id}")
                    return dict(response_data)
                elif response.status == 404:
                    raise Exception(f"Episode {episode_id} not found for override submission")
                elif response.status == 400:
                    error_text = await response.text()
                    raise Exception(f"Invalid override request for episode {episode_id}: {error_text}")
                else:
                    error_text = await response.text()
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
        logger.info(f"Uploading evaluation file {filename} for session {session_id}")

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
                    logger.info(f"Successfully uploaded evaluation file {filename}")
                    return dict(response_data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to upload evaluation file {filename}: {response.status} - {error_text}")

    async def cleanup(self) -> None:
        """Cleanup session manager resources."""
        if self._current_session_id:
            await self.terminate_session(self._current_session_id)

        logger.info("ClientSessionManager cleanup completed")
