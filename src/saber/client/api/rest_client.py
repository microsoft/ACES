"""
SABER REST Client - Dataset Support & Evaluation Retrieval

REST client for SABER operations including task data fetching and evaluation retrieval.
Follows SABER fail-fast principles with no defensive programming fallbacks.
"""

import logging
from typing import Any, Dict, Optional, cast

import aiohttp

from ...models import BenchmarkInfo
from ...models.rest.evaluation import EvaluationListResponse, EvaluationResponse, EvaluationSummaryResponse
from ..exceptions import (
    EvaluationNotFoundError,
    EvaluationRetrievalError,
    InvalidEvaluationRequestError,
    SessionEvaluationError,
)

logger = logging.getLogger(__name__)


class SABERRestClient:
    """
    REST client for SABER client operations.
    """

    def __init__(self, saber_server_url: str, request_timeout: float = 30.0):
        """
        Initialize REST client.

        Args:
            saber_server_url: SABER server URL
            request_timeout: Request timeout in seconds
        """
        self.saber_server_url = saber_server_url.rstrip("/")
        self.request_timeout = request_timeout

    async def get_benchmark_info(self) -> BenchmarkInfo:
        """
        Get complete benchmark information from SABER server.

        Returns:
            BenchmarkInfo object with all benchmark data

        Raises:
            Exception: If request fails or server returns error
        """
        url = f"{self.saber_server_url}/api/v1/benchmark"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    benchmark_info = BenchmarkInfo(**data)
                    logger.debug(
                        f"Retrieved benchmark info: {benchmark_info.domain} with {benchmark_info.total_tasks} tasks"
                    )
                    return benchmark_info
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get benchmark info: {response.status} - {error_text}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check against the server.

        Returns:
            Health status data

        Raises:
            Exception: If health check fails
        """
        url = f"{self.saber_server_url}/api/v1/health"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    return cast(Dict[str, Any], result)
                else:
                    raise Exception(f"Health check failed: {response.status}")

    async def get_evaluation(self, session_id: str, episode_id: str) -> EvaluationResponse:
        """
        Get evaluation result for specific episode.

        Args:
            session_id: Session ID containing the episode
            episode_id: Episode ID to get evaluation for

        Returns:
            EvaluationResponse with evaluation result

        Raises:
            EvaluationNotFoundError: Evaluation not found (404)
            InvalidEvaluationRequestError: Invalid request parameters (422)
            SessionEvaluationError: Session-level access error (500)
            EvaluationRetrievalError: Other retrieval errors
        """
        url = f"{self.saber_server_url}/api/v1/session/{session_id}/evaluations/{episode_id}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                response_text = await response.text()

                if response.status == 200:
                    data = await response.json()
                    return EvaluationResponse(**data)
                elif response.status == 404:
                    raise EvaluationNotFoundError(
                        f"Evaluation not found for session {session_id}, episode {episode_id}",
                        details={"session_id": session_id, "episode_id": episode_id},
                    )
                elif response.status == 422:
                    raise InvalidEvaluationRequestError(
                        f"Invalid evaluation request: {response_text}",
                        details={"session_id": session_id, "episode_id": episode_id, "status_code": response.status},
                    )
                elif response.status == 500:
                    raise SessionEvaluationError(
                        f"Session evaluation error: {response_text}",
                        details={"session_id": session_id, "episode_id": episode_id, "status_code": response.status},
                    )
                else:
                    raise EvaluationRetrievalError(
                        f"Failed to get evaluation: HTTP {response.status} - {response_text}",
                        details={"session_id": session_id, "episode_id": episode_id, "status_code": response.status},
                    )

    async def list_evaluations(self, session_id: str, task_id: Optional[str] = None) -> EvaluationListResponse:
        """
        List evaluation results for session.

        Args:
            session_id: Session ID to list evaluations for
            task_id: Optional task ID filter

        Returns:
            EvaluationListResponse with list of evaluations

        Raises:
            InvalidEvaluationRequestError: Invalid request parameters (422)
            SessionEvaluationError: Session-level access error (500)
            EvaluationRetrievalError: Other retrieval errors
        """
        url = f"{self.saber_server_url}/api/v1/session/{session_id}/evaluations"
        params = {}
        if task_id:
            params["task_id"] = task_id

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=self.request_timeout) as response:
                response_text = await response.text()

                if response.status == 200:
                    data = await response.json()
                    return EvaluationListResponse(**data)
                elif response.status == 422:
                    raise InvalidEvaluationRequestError(
                        f"Invalid evaluation list request: {response_text}",
                        details={"session_id": session_id, "task_id": task_id, "status_code": response.status},
                    )
                elif response.status == 500:
                    raise SessionEvaluationError(
                        f"Session evaluation error: {response_text}",
                        details={"session_id": session_id, "task_id": task_id, "status_code": response.status},
                    )
                else:
                    raise EvaluationRetrievalError(
                        f"Failed to list evaluations: HTTP {response.status} - {response_text}",
                        details={"session_id": session_id, "task_id": task_id, "status_code": response.status},
                    )

    async def get_evaluation_summary(self, session_id: str) -> EvaluationSummaryResponse:
        """
        Get aggregate evaluation summary for session.

        Args:
            session_id: Session ID to get summary for

        Returns:
            EvaluationSummaryResponse with session summary

        Raises:
            InvalidEvaluationRequestError: Invalid request parameters (422)
            SessionEvaluationError: Session-level access error (500)
            EvaluationRetrievalError: Other retrieval errors
        """
        url = f"{self.saber_server_url}/api/v1/session/{session_id}/evaluations/summary"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                response_text = await response.text()

                if response.status == 200:
                    data = await response.json()
                    return EvaluationSummaryResponse(**data)
                elif response.status == 422:
                    raise InvalidEvaluationRequestError(
                        f"Invalid evaluation summary request: {response_text}",
                        details={"session_id": session_id, "status_code": response.status},
                    )
                elif response.status == 500:
                    raise SessionEvaluationError(
                        f"Session evaluation error: {response_text}",
                        details={"session_id": session_id, "status_code": response.status},
                    )
                else:
                    raise EvaluationRetrievalError(
                        f"Failed to get evaluation summary: HTTP {response.status} - {response_text}",
                        details={"session_id": session_id, "status_code": response.status},
                    )
