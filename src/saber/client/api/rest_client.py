"""
SABER REST Client - Dataset Support & Evaluation Retrieval

REST client for SABER operations including task data fetching and evaluation retrieval.
Logging category: COMMUNICATION
Follows SABER fail-fast principles with no defensive programming fallbacks.
"""

from typing import Any, Dict, Optional, cast

import aiohttp

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...models import BenchmarkInfo
from ...models.rest.evaluation import EvaluationListResponse, EvaluationResponse, EvaluationSummaryResponse
from ..exceptions import (
    EvaluationNotFoundError,
    EvaluationRetrievalError,
    InvalidEvaluationRequestError,
    SessionEvaluationError,
)

logger = get_saber_logger(LogCategory.COMMUNICATION, __name__)


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

        operation = "fetch_benchmark_info"
        log_operation_start(logger, operation, url=url)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    benchmark_info = BenchmarkInfo(**data)
                    log_operation_success(
                        logger,
                        operation,
                        domain=benchmark_info.domain,
                        task_count=benchmark_info.total_tasks,
                    )
                    return benchmark_info

                error_text = await response.text()
                log_operation_failure(
                    logger,
                    operation,
                    f"HTTP {response.status}",
                    url=url,
                    status_code=response.status,
                    response_text=error_text,
                )
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

        operation = "client_health_check"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    return cast(Dict[str, Any], result)

                log_operation_failure(
                    logger,
                    operation,
                    f"HTTP {response.status}",
                    url=url,
                    status_code=response.status,
                )
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

        operation = "fetch_evaluation"
        log_operation_start(
            logger,
            operation,
            session_id=session_id,
            episode_id=episode_id,
            url=url,
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                response_text = await response.text()

                if response.status == 200:
                    data = await response.json()
                    log_operation_success(
                        logger,
                        operation,
                        session_id=session_id,
                        episode_id=episode_id,
                        status_code=response.status,
                    )
                    return EvaluationResponse(**data)

                failure_extra = {
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "status_code": response.status,
                    "response_text": response_text,
                    "url": url,
                }

                if response.status == 404:
                    log_operation_failure(logger, operation, "evaluation_not_found", **failure_extra)
                    raise EvaluationNotFoundError(
                        f"Evaluation not found for session {session_id}, episode {episode_id}",
                        details={"session_id": session_id, "episode_id": episode_id},
                    )

                if response.status == 422:
                    log_operation_failure(logger, operation, "invalid_evaluation_request", **failure_extra)
                    raise InvalidEvaluationRequestError(
                        f"Invalid evaluation request: {response_text}",
                        details={"session_id": session_id, "episode_id": episode_id, "status_code": response.status},
                    )

                if response.status == 500:
                    log_operation_failure(logger, operation, "session_evaluation_error", **failure_extra)
                    raise SessionEvaluationError(
                        f"Session evaluation error: {response_text}",
                        details={"session_id": session_id, "episode_id": episode_id, "status_code": response.status},
                    )

                log_operation_failure(logger, operation, "evaluation_retrieval_error", **failure_extra)
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

        operation = "list_evaluations"
        log_operation_start(
            logger,
            operation,
            session_id=session_id,
            task_id=task_id,
            url=url,
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=self.request_timeout) as response:
                response_text = await response.text()

                if response.status == 200:
                    data = await response.json()
                    payload = EvaluationListResponse(**data)
                    log_operation_success(
                        logger,
                        operation,
                        session_id=session_id,
                        task_id=task_id,
                        status_code=response.status,
                        evaluation_count=payload.total_count,
                    )
                    return payload

                failure_extra = {
                    "session_id": session_id,
                    "task_id": task_id,
                    "status_code": response.status,
                    "response_text": response_text,
                    "url": url,
                }

                if response.status == 422:
                    log_operation_failure(logger, operation, "invalid_evaluation_list_request", **failure_extra)
                    raise InvalidEvaluationRequestError(
                        f"Invalid evaluation list request: {response_text}",
                        details={"session_id": session_id, "task_id": task_id, "status_code": response.status},
                    )

                if response.status == 500:
                    log_operation_failure(logger, operation, "session_evaluation_error", **failure_extra)
                    raise SessionEvaluationError(
                        f"Session evaluation error: {response_text}",
                        details={"session_id": session_id, "task_id": task_id, "status_code": response.status},
                    )

                log_operation_failure(logger, operation, "evaluation_list_retrieval_error", **failure_extra)
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

        operation = "fetch_evaluation_summary"
        log_operation_start(logger, operation, session_id=session_id, url=url)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                response_text = await response.text()

                if response.status == 200:
                    data = await response.json()
                    payload = EvaluationSummaryResponse(**data)
                    log_operation_success(
                        logger,
                        operation,
                        session_id=session_id,
                        status_code=response.status,
                        total_episodes=payload.total_episodes,
                    )
                    return payload

                failure_extra = {
                    "session_id": session_id,
                    "status_code": response.status,
                    "response_text": response_text,
                    "url": url,
                }

                if response.status == 422:
                    log_operation_failure(logger, operation, "invalid_evaluation_summary_request", **failure_extra)
                    raise InvalidEvaluationRequestError(
                        f"Invalid evaluation summary request: {response_text}",
                        details={"session_id": session_id, "status_code": response.status},
                    )

                if response.status == 500:
                    log_operation_failure(logger, operation, "session_evaluation_error", **failure_extra)
                    raise SessionEvaluationError(
                        f"Session evaluation error: {response_text}",
                        details={"session_id": session_id, "status_code": response.status},
                    )

                log_operation_failure(logger, operation, "evaluation_summary_retrieval_error", **failure_extra)
                raise EvaluationRetrievalError(
                    f"Failed to get evaluation summary: HTTP {response.status} - {response_text}",
                    details={"session_id": session_id, "status_code": response.status},
                )
