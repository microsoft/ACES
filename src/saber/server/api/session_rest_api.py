"""
SessionRestAPI implementation for SABER domain server.

The SessionRestAPI handles REST API endpoints for session management, episodes,
policy, status, and events. Tool execution is handled by SessionMCPAPI.

Logging category: REST_API.
"""

import uuid

# Forward declaration to avoid circular imports
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect

from ...logging_config import get_api_logger, log_operation_failure, log_operation_start, log_operation_success
from ...models import (
    APIEndpoints,
    BenchmarkInfo,
    EpisodeContext,
    EpisodeCreateResponse,
    EpisodeEndResponse,
    EpisodeStatusResponse,
    EpisodeTaskResponse,
    EvalSubmission,
    HealthResponse,
    MessageInjectRequest,
    MessageInjectResponse,
    MetadataKeys,
    PendingMessagesResponse,
    PolicyResponse,
    SessionCreateResponse,
    SessionTerminateResponse,
    StepEvaluationStrategy,
    SubmissionEvaluationStrategy,
    TranscriptGetResponse,
    TranscriptPushRequest,
    TranscriptPushResponse,
    TranscriptSyncConfig,
    TranscriptSyncRequest,
)
from ...models.rest.evaluation import (
    EpisodeStepData,
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    EvaluationFileUploadResponse,
    EvaluationListResponse,
    EvaluationResponse,
    EvaluationSummaryResponse,
    SubmissionEvaluationCriteriaResponse,
    SubtaskEvaluationCriteriaResponse,
    TaskEvaluationContext,
    TemplateContentResponse,
)
from ...models.rest.websocket_constants import WebSocketCloseCode, WebSocketMessageType
from ...models.rest.websocket_messages import (
    PongMessage,
    PushAckData,
    PushAckMessage,
    StateEventData,
    StateEventMessage,
    SyncMode,
    SyncResponseData,
    SyncResponseMessage,
    TranscriptOperation,
    TranscriptVersion,
)
from ...models.rest.websocket_requests import PushMessageRequestData, SyncRequestData
from ..evaluation.exceptions import EvaluationNotFoundError, InvalidEvaluationRequestError, SessionEvaluationError

if TYPE_CHECKING:
    from ..session_manager import SessionManager

logger = get_api_logger(__name__)


class SessionRestAPI:
    """
    REST API layer for SessionManager.

    Handles session management, episodes, policy, status, and events.
    Tool execution is handled by SessionMCPAPI.
    """

    def __init__(self, session_manager: "SessionManager", host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Initialize the SessionRestAPI.

        Args:
            session_manager: SessionManager instance to delegate to
            host: Server host address
            port: Server port
        """
        self.session_manager = session_manager
        self.host = host
        self.port = port

        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"SABER {session_manager.domain_name.title()} Domain Server",
            description=f"SessionManager REST API for {session_manager.domain_name} security domain",
            version="1.0.0",
        )

        # Setup API routes
        self._setup_routes()

        logger.info(
            "REST API initialized",
            extra={
                "event": "rest_api_initialized",
                "domain": session_manager.domain_name,
                "host": host,
                "port": port,
            },
        )

    def _setup_routes(self) -> None:
        """Setup FastAPI routes for the session management API."""

        @self.app.post(APIEndpoints.SESSION, response_model=SessionCreateResponse)
        async def create_session_endpoint(client_id: str) -> SessionCreateResponse:
            """Create a new client session."""
            session = await self.session_manager.create_session(client_id)
            return SessionCreateResponse(session_id=session.session_id, message="Session created successfully")

        @self.app.delete(APIEndpoints.SESSION_BY_ID, response_model=SessionTerminateResponse)
        async def terminate_session_endpoint(session_id: str) -> SessionTerminateResponse:
            """Terminate a client session."""
            logger.warning(
                "Terminate session requested",
                extra={
                    "event": "session_termination_requested",
                    "session_id": session_id,
                    "http_method": "DELETE",
                    "path": "/api/v1/session/{session_id}",
                },
            )
            await self.session_manager.terminate_session(session_id)
            return SessionTerminateResponse(message="Session terminated successfully")

        @self.app.get(APIEndpoints.EPISODE_TASK, response_model=EpisodeTaskResponse)
        async def get_episode_task_endpoint(session_id: str, episode_id: str) -> EpisodeTaskResponse:
            """Get task information for a specific episode."""
            task = await self.session_manager.get_current_task(session_id, episode_id)

            # Get episode to access its configuration
            episode = self.session_manager.get_episode_by_id(episode_id)

            # Build episode context
            if episode:
                # Get episode configuration from the task using BenchmarkManager
                episode_config = self.session_manager.benchmark_manager.get_episode_config(episode.task_id)
                episode_context = EpisodeContext(
                    session_id=session_id,
                    task_timeout=episode_config.get("task_timeout") if episode_config else None,
                    max_steps=episode_config.get("max_steps") if episode_config else None,
                    metadata=episode_config if episode_config else {},
                )
            else:
                episode_context = EpisodeContext(
                    session_id=session_id,
                    task_timeout=None,
                    max_steps=None,
                    metadata={},
                )

            # Convert task to dict to get attributes
            task_dict = task.to_dict()

            return EpisodeTaskResponse(
                task_id=task_dict["task_id"],
                title=task_dict["title"],
                description=task_dict["description"],
                episode_id=episode_id,
                episode_context=episode_context,
                initial_context=task_dict.get("initial_context"),
            )

        @self.app.get(APIEndpoints.EPISODE_POLICY, response_model=PolicyResponse)
        async def get_policy_endpoint(session_id: str, episode_id: str) -> PolicyResponse:
            """Get policy information for a specific episode."""
            episode = self.session_manager.get_episode_by_id(episode_id)
            if not episode:
                raise HTTPException(status_code=404, detail="Episode not found")

            policy = self.session_manager.get_policy(session_id, episode_id)
            policy_dict = policy.to_dict()

            return PolicyResponse(prompt=policy_dict.get("prompt", ""), domain=policy_dict.get("domain"))

        @self.app.post(APIEndpoints.EPISODES, response_model=EpisodeCreateResponse)
        async def create_episode_endpoint(session_id: str, task_id: str) -> EpisodeCreateResponse:
            """Create a new episode for a specific task with automatic dependency resolution."""
            try:
                log_operation_start(
                    logger,
                    "create_episode",
                    session_id=session_id,
                    task_id=task_id,
                )

                # Create episode with automatic dependency resolution
                episode = await self.session_manager.start_episode(session_id, task_id)

                # Create episode context with limits and metadata
                episode_context = EpisodeContext(
                    session_id=session_id,
                    task_timeout=None,
                    max_steps=episode.max_steps,
                    metadata=episode.metadata,
                )

                response = EpisodeCreateResponse(
                    episode_id=episode.episode_id,
                    task_id=task_id,
                    session_id=session_id,
                    state=episode.state.value,
                    message="Episode created successfully"
                    + (f" (attached to {episode.attached_to_episode_id})" if episode.attached_to_episode_id else ""),
                    episode_context=episode_context,
                    attached_to_episode_id=episode.attached_to_episode_id,
                )
                log_operation_success(
                    logger,
                    "create_episode",
                    session_id=session_id,
                    task_id=task_id,
                    episode_id=episode.episode_id,
                    episode_state=episode.state.value,
                )
                return response
            except HTTPException as exc:
                logger.warning(
                    "Episode creation failed",
                    extra={
                        "event": "create_episode_failed",
                        "session_id": session_id,
                        "task_id": task_id,
                        "status_code": exc.status_code,
                        "detail": exc.detail,
                    },
                )
                raise
            except Exception as exc:
                log_operation_failure(
                    logger,
                    "create_episode",
                    exc,
                    session_id=session_id,
                    task_id=task_id,
                )
                raise HTTPException(status_code=500, detail=f"Failed to create episode: {exc}") from exc

        @self.app.get(APIEndpoints.EPISODE_STATUS, response_model=EpisodeStatusResponse)
        async def get_episode_status_endpoint(session_id: str, episode_id: str) -> EpisodeStatusResponse:
            """Get episode status for readiness polling."""
            log_operation_start(logger, "get_episode_status", session_id=session_id, episode_id=episode_id)
            try:
                episode = self.session_manager.get_episode_status(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                # Build episode context if episode is ready
                episode_context = None
                if episode.is_ready:
                    episode_config = self.session_manager.benchmark_manager.get_episode_config(episode.task_id)
                    episode_context = EpisodeContext(
                        session_id=session_id,
                        task_timeout=episode_config.get("task_timeout") if episode_config else None,
                        max_steps=episode.max_steps,
                        metadata=episode.metadata,
                    )

                # Determine readiness
                is_ready = episode.is_ready

                # Build status message
                if is_ready:
                    message = "Episode is ready for execution"
                elif episode.state.value == "creating":
                    message = "Episode is being created"
                elif episode.state.value == "failed_creation":
                    message = "Episode creation failed"
                else:
                    message = f"Episode is in {episode.state.value} state"

                response = EpisodeStatusResponse(
                    episode_id=episode.episode_id,
                    task_id=episode.task_id,
                    session_id=session_id,
                    state=episode.state.value,
                    is_ready=is_ready,
                    message=message,
                    creation_error=getattr(episode, "creation_error", None),
                    episode_context=episode_context,
                    attached_to_episode_id=episode.attached_to_episode_id,
                )

                log_operation_success(
                    logger,
                    "get_episode_status",
                    session_id=session_id,
                    episode_id=episode_id,
                    state=episode.state.value,
                    is_ready=is_ready,
                )
                return response

            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(logger, "get_episode_status", exc, session_id=session_id, episode_id=episode_id)
                raise HTTPException(status_code=500, detail=f"Failed to get episode status: {exc}") from exc

        @self.app.delete(APIEndpoints.EPISODE_BY_ID, response_model=EpisodeEndResponse)
        async def end_episode_endpoint(
            session_id: str,
            episode_id: str,
            request: Request,
            reason: str = "manual_termination",
            cascade_end_attached_episodes: str = "false",
        ) -> EpisodeEndResponse:
            """End a specific episode."""

            # Parse EvalSubmission from query parameter OR request body
            eval_submission = None
            result_data = None

            # First try to get from query parameters (legacy method)
            result_param = request.query_params.get("result")
            if result_param:
                result_data = result_param
            else:
                # Try to get from request body
                try:
                    body = await request.body()
                    if body:
                        body_text = body.decode("utf-8")
                        result_data = body_text
                except Exception as exc:
                    logger.warning(
                        "Failed to read request body",
                        extra={
                            "event": "request_body_read_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "error": str(exc),
                        },
                    )

            if result_data:
                try:
                    import json

                    result_dict = json.loads(result_data)
                    eval_submission = EvalSubmission(**result_dict)
                except Exception as exc:
                    logger.error(
                        "Failed to parse evaluation submission",
                        extra={
                            "event": "evaluation_submission_parse_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "error": str(exc),
                        },
                    )
                    # Continue without eval_submission

            # Convert string boolean parameter to actual boolean
            cascade_bool = cascade_end_attached_episodes.lower() in ("true", "1", "yes", "on")

            response = await self.session_manager.end_episode(
                session_id, episode_id, reason, eval_submission, cascade_bool
            )
            return response

        @self.app.get(APIEndpoints.TASKS, response_model=BenchmarkInfo)
        async def get_tasks_endpoint() -> BenchmarkInfo:
            """
            Get task list with episode attempts for client orchestration.
            Returns BenchmarkInfo object with all configured tasks.
            Each task reports its own configured episode_attempts.
            """
            logger.debug("Tasks endpoint requested", extra={"event": "tasks_requested"})
            try:
                benchmark_info: BenchmarkInfo = self.session_manager.get_benchmark_info()
                logger.debug(
                    "Tasks retrieved",
                    extra={
                        "event": "tasks_retrieved",
                        "task_count": benchmark_info.total_tasks,
                        "episode_count": benchmark_info.total_episodes,
                    },
                )

                # Test serialization before returning
                try:
                    benchmark_info.model_dump()
                except Exception as serialization_exc:
                    logger.exception(
                        "Benchmark info serialization failed",
                        extra={
                            "event": "benchmark_info_serialization_failed",
                            "task_count": benchmark_info.total_tasks,
                            "episode_count": benchmark_info.total_episodes,
                        },
                    )
                    raise HTTPException(
                        status_code=500, detail="Failed to serialize benchmark info"
                    ) from serialization_exc

                return benchmark_info
            except HTTPException:
                # Re-raise HTTPException to preserve status codes
                raise
            except Exception as exc:
                logger.exception("Failed to retrieve tasks", extra={"event": "tasks_retrieval_failed"})
                raise HTTPException(status_code=500, detail=f"Failed to get tasks: {exc}") from exc

        @self.app.get(APIEndpoints.HEALTH, response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Enhanced health check endpoint with manifest metadata and dependency validation."""
            health_data = self.session_manager.get_health_metadata()

            # Return 503 Service Unavailable if server or dependencies are unhealthy
            if health_data.get("status") != "healthy":
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": "Server unhealthy - permanent environment dependencies failed",
                        "health_data": health_data,
                    },
                )

            return HealthResponse(**health_data)

        # Evaluation endpoints
        @self.app.get(APIEndpoints.EVALUATION_BY_EPISODE, response_model=EvaluationResponse)
        async def get_evaluation_endpoint(session_id: str, episode_id: str) -> EvaluationResponse:
            """Get evaluation result for specific episode."""
            try:
                # Validate session exists first
                self.session_manager._get_session(session_id)

                evaluation_service = self.session_manager.get_evaluation_service()
                result = await evaluation_service.get_evaluation(session_id, episode_id)

                # Convert to response model
                from ...models.rest.evaluation import EvaluationResultResponse

                evaluation_response = EvaluationResultResponse(
                    episode_id=result.episode_id,
                    task_id=result.task_id,
                    strategy=result.strategy,
                    raw_score=result.raw_score,
                    max_score=result.max_score,
                    score=result.score,
                    success=result.success,
                    timestamp=result.timestamp,
                    details=result.details,
                )

                return EvaluationResponse(evaluation_result=evaluation_response, session_id=session_id)
            except HTTPException:
                # Re-raise HTTPException to preserve status codes (404, 422, etc.)
                raise
            except EvaluationNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except InvalidEvaluationRequestError as e:
                raise HTTPException(status_code=422, detail=str(e))
            except SessionEvaluationError as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(APIEndpoints.EVALUATIONS_LIST, response_model=EvaluationListResponse)
        async def list_evaluations_endpoint(session_id: str, task_id: Optional[str] = None) -> EvaluationListResponse:
            """List evaluation results for session."""
            try:
                # Validate session exists first
                self.session_manager._get_session(session_id)

                evaluation_service = self.session_manager.get_evaluation_service()
                evaluations = await evaluation_service.list_session_evaluations(session_id, task_id)

                # Convert to response models
                from ...models.rest.evaluation import EvaluationResultResponse

                evaluation_responses = [
                    EvaluationResultResponse(
                        episode_id=result.episode_id,
                        task_id=result.task_id,
                        strategy=result.strategy,
                        raw_score=result.raw_score,
                        max_score=result.max_score,
                        score=result.score,
                        success=result.success,
                        timestamp=result.timestamp,
                        details=result.details,
                    )
                    for result in evaluations
                ]

                return EvaluationListResponse(
                    evaluations=evaluation_responses,
                    total_count=len(evaluation_responses),
                    session_id=session_id,
                    task_filter=task_id,
                )
            except HTTPException:
                # Re-raise HTTPException to preserve status codes (404, 422, etc.)
                raise
            except InvalidEvaluationRequestError as e:
                raise HTTPException(status_code=422, detail=str(e))
            except SessionEvaluationError as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(APIEndpoints.EVALUATIONS_SUMMARY, response_model=EvaluationSummaryResponse)
        async def get_evaluation_summary_endpoint(session_id: str) -> EvaluationSummaryResponse:
            """Get aggregate evaluation summary for session."""
            try:
                # Validate session exists first
                self.session_manager._get_session(session_id)

                evaluation_service = self.session_manager.get_evaluation_service()
                summary = await evaluation_service.get_session_summary(session_id)
                return EvaluationSummaryResponse(**summary)
            except HTTPException:
                # Re-raise HTTPException to preserve status codes (404, 422, etc.)
                raise
            except InvalidEvaluationRequestError as e:
                raise HTTPException(status_code=422, detail=str(e))
            except SessionEvaluationError as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Evaluation file upload endpoint
        @self.app.post(APIEndpoints.EVALUATIONS_UPLOAD, response_model=EvaluationFileUploadResponse)
        async def upload_evaluation_file_endpoint(
            session_id: str, file: UploadFile = File(...)
        ) -> EvaluationFileUploadResponse:
            """Upload external evaluation file (.eval) to session directory."""
            log_operation_start(
                logger,
                "upload_evaluation_file",
                session_id=session_id,
                filename=file.filename,
            )

            try:
                # Validate session exists first
                self.session_manager._get_session(session_id)

                # Validate file extension
                if not file.filename or not file.filename.endswith(".eval"):
                    logger.warning(
                        "Invalid evaluation file extension",
                        extra={
                            "event": "evaluation_file_extension_invalid",
                            "session_id": session_id,
                            "filename": file.filename,
                        },
                    )
                    raise HTTPException(status_code=422, detail="File must have .eval extension")

                # Save file via evaluation manager
                file_size = await self.session_manager.save_evaluation_file(session_id, file)

                log_operation_success(
                    logger,
                    "upload_evaluation_file",
                    session_id=session_id,
                    filename=file.filename,
                    file_size=file_size,
                )

                return EvaluationFileUploadResponse(
                    message="Evaluation file uploaded successfully",
                    session_id=session_id,
                    filename=file.filename,
                    file_size=file_size,
                )

            except HTTPException:
                # Re-raise HTTPException to preserve status codes (404, 422, etc.)
                raise
            except Exception as exc:
                log_operation_failure(
                    logger,
                    "upload_evaluation_file",
                    exc,
                    session_id=session_id,
                    filename=file.filename,
                )
                raise HTTPException(status_code=500, detail=f"Failed to upload evaluation file: {exc}") from exc

        # ============================================================================
        # CLIENT-SIDE EVALUATION: OLD ENDPOINTS REMOVED
        # - DELETE: GET /api/v1/session/{session_id}/episodes/{episode_id}/evaluation-criteria
        #   Replaced by separate endpoints for submission/steps/criteria
        # - DELETE: PUT /api/v1/session/{session_id}/evaluations/{episode_id}/override
        #   Replaced by POST /api/v1/session/{session_id}/episodes/{episode_id}/evaluation
        # ============================================================================

        # ============================================================================
        # NEW CLIENT-SIDE EVALUATION ENDPOINTS (Breaking Change Migration)
        # ============================================================================

        @self.app.get(APIEndpoints.EPISODE_SUBMISSION)
        async def get_episode_submission_endpoint(session_id: str, episode_id: str) -> EpisodeSubmissionResponse:
            """Get episode submission data for client-side evaluation."""
            log_operation_start(logger, "get_episode_submission", session_id=session_id, episode_id=episode_id)
            try:
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                return EpisodeSubmissionResponse(
                    session_id=session_id,
                    episode_id=episode_id,
                    task_id=episode.task_id,
                    submission=episode.submission or "",
                    model=episode.eval_submission.model if episode.eval_submission else None,
                    tokens=episode.eval_submission.tokens if episode.eval_submission else {},
                    execution_time=episode.eval_submission.time if episode.eval_submission else None,
                )
            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(
                    logger, "get_episode_submission", exc, session_id=session_id, episode_id=episode_id
                )
                raise HTTPException(status_code=500, detail=f"Failed to get episode submission: {exc}") from exc

        @self.app.post(APIEndpoints.EPISODE_SUBMISSION)
        async def post_episode_submission_endpoint(
            session_id: str, episode_id: str, submission: EvalSubmission
        ) -> dict:
            """Store episode submission without ending the episode."""
            log_operation_start(logger, "post_episode_submission", session_id=session_id, episode_id=episode_id)
            try:
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                # Store the submission data on the episode
                episode.eval_submission = submission
                episode.submission = submission.submission

                log_operation_success(
                    logger,
                    "post_episode_submission",
                    session_id=session_id,
                    episode_id=episode_id,
                    submission_length=len(submission.submission) if submission.submission else 0,
                )

                return {"message": "Submission stored successfully"}

            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(
                    logger, "post_episode_submission", exc, session_id=session_id, episode_id=episode_id
                )
                raise HTTPException(status_code=500, detail=f"Failed to store episode submission: {exc}") from exc

        @self.app.get(APIEndpoints.EPISODE_STEPS)
        async def get_episode_steps_endpoint(session_id: str, episode_id: str) -> EpisodeStepsResponse:
            """Get episode step history for client-side evaluation."""
            log_operation_start(logger, "get_episode_steps", session_id=session_id, episode_id=episode_id)
            try:
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                steps_data = [
                    EpisodeStepData(
                        step_number=step.step_number,
                        tool_name=step.action.tool_name,
                        tool_input=step.action.parameters,
                        tool_output=str(step.response),  # Convert response dict to string
                        timestamp=step.timestamp,
                        assistant_message=step.action.assistant_message,
                        reasoning=step.action.reasoning,
                    )
                    for step in episode.steps
                ]

                return EpisodeStepsResponse(
                    session_id=session_id,
                    episode_id=episode_id,
                    task_id=episode.task_id,
                    steps=steps_data,
                    total_steps=len(steps_data),
                )
            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(logger, "get_episode_steps", exc, session_id=session_id, episode_id=episode_id)
                raise HTTPException(status_code=500, detail=f"Failed to get episode steps: {exc}") from exc

        @self.app.post(APIEndpoints.EPISODE_TRANSCRIPT)
        async def push_transcript_endpoint(
            session_id: str, episode_id: str, transcript_request: TranscriptPushRequest
        ) -> TranscriptPushResponse:
            """Push conversation transcript from client to server.

            Supports two modes:
            - 'replace': Replace entire transcript (default, for backward compatibility)
            - 'append': Append new messages to existing transcript (differential sync)

            This endpoint allows post-completion pushes for audit trail and debugging.
            """
            import json
            from datetime import datetime

            from pydantic import ValidationError

            # Validate mode parameter
            mode = transcript_request.mode
            if mode not in ["replace", "append"]:
                raise HTTPException(status_code=422, detail=f"Invalid mode '{mode}'. Must be 'replace' or 'append'.")

            log_operation_start(
                logger,
                "push_transcript",
                session_id=session_id,
                episode_id=episode_id,
                mode=mode,
                message_count=len(transcript_request.messages),
            )
            try:
                # Get episode
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

                # Note: Removed is_complete check to allow post-completion pushes
                # Rationale: Transcript is audit/debugging data, clients may push after task completion

                # Check payload size using actual JSON byte length (not sys.getsizeof)
                messages_json = json.dumps([msg.dict() for msg in transcript_request.messages])
                payload_size = len(messages_json.encode("utf-8"))
                if payload_size > TranscriptSyncConfig.MAX_PAYLOAD_SIZE_BYTES:
                    max_size = TranscriptSyncConfig.MAX_PAYLOAD_SIZE_BYTES
                    raise HTTPException(
                        status_code=413,
                        detail=f"Transcript payload too large: {payload_size} bytes (max: {max_size} bytes)",
                    )

                # Store transcript in episode context atomically (thread-safe)
                timestamp = datetime.utcnow().isoformat()

                # Handle append vs replace mode
                if mode == "append":
                    # Append new messages to existing transcript
                    existing_messages = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
                    updated_messages = existing_messages + [msg.dict() for msg in transcript_request.messages]
                else:
                    # Replace entire transcript
                    updated_messages = [msg.dict() for msg in transcript_request.messages]

                context_updates: Dict[str, Any] = {
                    MetadataKeys.CLIENT_TRANSCRIPT.value: updated_messages,
                    MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT.value: timestamp,  # Auto-set on push
                }
                if transcript_request.metadata:
                    context_updates[MetadataKeys.TRANSCRIPT_METADATA.value] = transcript_request.metadata

                await episode.update_context_atomic(context_updates)

                log_operation_success(
                    logger,
                    "push_transcript",
                    session_id=session_id,
                    episode_id=episode_id,
                    mode=mode,
                    message_count=len(transcript_request.messages),
                    total_messages=len(updated_messages),
                )

                return TranscriptPushResponse(
                    success=True,
                    episode_id=episode_id,
                    message_count=len(transcript_request.messages),
                    stored_at=timestamp,
                )

            except HTTPException:
                raise
            except ValidationError as exc:
                # Return 422 for validation errors (not 500)
                log_operation_failure(
                    logger,
                    "push_transcript",
                    exc,
                    session_id=session_id,
                    episode_id=episode_id,
                    error_type="validation_error",
                )
                raise HTTPException(
                    status_code=422,
                    detail={"message": "Validation failed", "errors": exc.errors()},
                )
            except Exception as exc:
                log_operation_failure(logger, "push_transcript", exc, session_id=session_id, episode_id=episode_id)
                raise HTTPException(status_code=500, detail=f"Failed to push transcript: {exc}") from exc

        @self.app.get(APIEndpoints.EPISODE_TRANSCRIPT)
        async def get_transcript_endpoint(session_id: str, episode_id: str) -> TranscriptGetResponse:
            """Get conversation transcript for episode (for red team access)."""
            log_operation_start(logger, "get_transcript", session_id=session_id, episode_id=episode_id)
            try:
                # Get episode
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

                # Retrieve transcript from episode context
                messages = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
                last_updated = episode.context.get(MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT)
                metadata = episode.context.get(MetadataKeys.TRANSCRIPT_METADATA)

                log_operation_success(
                    logger, "get_transcript", session_id=session_id, episode_id=episode_id, message_count=len(messages)
                )

                return TranscriptGetResponse(
                    episode_id=episode_id,
                    messages=messages,
                    message_count=len(messages),
                    last_updated=last_updated,
                    metadata=metadata,
                )

            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(logger, "get_transcript", exc, session_id=session_id, episode_id=episode_id)
                raise HTTPException(status_code=500, detail=f"Failed to get transcript: {exc}") from exc

        @self.app.post(APIEndpoints.EPISODE_MESSAGES_INJECT)
        async def inject_message_endpoint(
            session_id: str, episode_id: str, inject_request: "MessageInjectRequest"
        ) -> "MessageInjectResponse":
            """Inject a red team message into the episode conversation.

            The message is queued in pending_injections and will be retrieved by the client
            on the next pull. This enables server-side message injection for adversarial testing.
            """
            import uuid
            from datetime import datetime

            log_operation_start(logger, "inject_message", session_id=session_id, episode_id=episode_id)
            try:
                # Get episode
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

                # Create injection record
                injection_id = str(uuid.uuid4())
                injected_at = datetime.utcnow().isoformat()

                injection_record = {
                    "injection_id": injection_id,
                    "role": inject_request.role,
                    "content": inject_request.content,
                    "metadata": inject_request.metadata,
                    "injected_at": injected_at,
                }

                # Append to pending injections list atomically
                pending_injections = episode.context.get(MetadataKeys.PENDING_INJECTIONS, [])
                pending_injections.append(injection_record)

                context_updates: Dict[str, Any] = {
                    MetadataKeys.PENDING_INJECTIONS.value: pending_injections,
                }

                await episode.update_context_atomic(context_updates)

                log_operation_success(
                    logger,
                    "inject_message",
                    session_id=session_id,
                    episode_id=episode_id,
                    injection_id=injection_id,
                    role=inject_request.role,
                )

                from ...models.rest import MessageInjectResponse

                return MessageInjectResponse(
                    success=True,
                    episode_id=episode_id,
                    injection_id=injection_id,
                    injected_at=injected_at,
                )

            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(logger, "inject_message", exc, session_id=session_id, episode_id=episode_id)
                raise HTTPException(status_code=500, detail=f"Failed to inject message: {exc}") from exc

        @self.app.get(APIEndpoints.EPISODE_MESSAGES_INJECT)
        async def get_pending_messages_endpoint(session_id: str, episode_id: str) -> "PendingMessagesResponse":
            """Retrieve and clear pending injected messages for the episode.

            This endpoint returns all pending messages and atomically clears the pending list,
            moving the messages to injection history for audit purposes.
            """
            from datetime import datetime

            log_operation_start(logger, "get_pending_messages", session_id=session_id, episode_id=episode_id)
            try:
                # Get episode
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

                # Retrieve pending injections atomically
                pending_injections = episode.context.get(MetadataKeys.PENDING_INJECTIONS, [])
                injection_history = episode.context.get(MetadataKeys.INJECTION_HISTORY, [])

                # Convert to ChatMessage format
                from ...models.rest import ChatMessage

                messages = [
                    ChatMessage(
                        role=record["role"],
                        content=record["content"],
                        tool_calls=None,
                        tool_call_id=None,
                        name=None,
                        reasoning=None,
                        # Store injection metadata in a way that client can identify injected messages
                    )
                    for record in pending_injections
                ]

                # Move pending to history and clear pending list
                retrieved_at = datetime.utcnow().isoformat()
                for record in pending_injections:
                    record["retrieved_at"] = retrieved_at
                    injection_history.append(record)

                context_updates: Dict[str, Any] = {
                    MetadataKeys.PENDING_INJECTIONS.value: [],  # Clear pending
                    MetadataKeys.INJECTION_HISTORY.value: injection_history,  # Update history
                }

                await episode.update_context_atomic(context_updates)

                log_operation_success(
                    logger,
                    "get_pending_messages",
                    session_id=session_id,
                    episode_id=episode_id,
                    pending_count=len(messages),
                )

                from ...models.rest import PendingMessagesResponse

                return PendingMessagesResponse(
                    episode_id=episode_id,
                    messages=messages,
                    pending_count=len(messages),
                    retrieved_at=retrieved_at,
                )

            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(logger, "get_pending_messages", exc, session_id=session_id, episode_id=episode_id)
                raise HTTPException(status_code=500, detail=f"Failed to get pending messages: {exc}") from exc

        # ===== Blocking Transcript Solver Endpoints =====

        @self.app.get("/api/v1/session/{session_id}/episodes/{episode_id}/transcript/metadata")
        async def get_transcript_metadata_endpoint(session_id: str, episode_id: str) -> Dict[str, Any]:
            """Get transcript metadata including timestamps for change detection.

            This endpoint is lightweight and designed for frequent polling by blue team.
            Blue team compares last_modified_at with its own last_pull timestamp to detect changes.

            Returns:
                last_pushed_at: ISO timestamp of last blue team push
                last_modified_at: ISO timestamp of last red team modification (or None)
                modification_count: Monotonic counter
                message_count: Number of messages in transcript
                last_updated: ISO timestamp of last transcript update
            """
            log_operation_start(logger, "get_transcript_metadata", session_id=session_id, episode_id=episode_id)
            try:
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

                transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])

                metadata = {
                    "last_pushed_at": episode.context.get(MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT),
                    "last_modified_at": episode.context.get(MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT),
                    "modification_count": episode.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0),
                    "message_count": len(transcript),
                    "last_updated": episode.context.get(MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT),
                }

                log_operation_success(
                    logger,
                    "get_transcript_metadata",
                    session_id=session_id,
                    episode_id=episode_id,
                    modification_count=metadata["modification_count"],
                )

                return metadata

            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(
                    logger, "get_transcript_metadata", exc, session_id=session_id, episode_id=episode_id
                )
                raise HTTPException(status_code=500, detail=f"Failed to get transcript metadata: {exc}") from exc

        # ===== End Blocking Transcript Solver Endpoints =====

        @self.app.get(APIEndpoints.EPISODE_SUBMISSION_EVALUATION_CRITERIA)
        async def get_submission_evaluation_criteria_endpoint(
            session_id: str, episode_id: str
        ) -> SubmissionEvaluationCriteriaResponse:
            """Get submission evaluation criteria (template paths only, no rendering)."""
            log_operation_start(
                logger, "get_submission_evaluation_criteria", session_id=session_id, episode_id=episode_id
            )
            try:
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                task = self.session_manager.benchmark_manager.get_task(episode.task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")

                if not task.submission_evaluation_config:
                    raise HTTPException(status_code=400, detail="Task has no submission evaluation config")

                task_context = TaskEvaluationContext(
                    task_id=task.task_id,
                    title=task.title,
                    description=task.description,
                    domain=task.domain,
                    subtasks=[],  # Not needed for submission evaluation
                )

                # Build criteria with template content (not paths)
                raw_criteria = task.submission_evaluation_config.get("criteria", {})
                criteria_with_content = raw_criteria.copy()

                # If LLM strategy, fetch template content
                if task.submission_evaluation_config.get("strategy") == SubmissionEvaluationStrategy.LLM_JUDGE:
                    system_template_path = raw_criteria.get("judge_system_template")
                    user_template_path = raw_criteria.get("judge_user_template")

                    if system_template_path and user_template_path:
                        try:
                            system_content = self.session_manager.benchmark_manager.get_template_content(
                                system_template_path
                            )
                            user_content = self.session_manager.benchmark_manager.get_template_content(
                                user_template_path
                            )

                            criteria_with_content["judge_system_template"] = system_content
                            criteria_with_content["judge_user_template"] = user_content
                        except Exception as e:
                            logger.warning(
                                f"Failed to load templates for submission evaluation: {e}",
                                extra={"system_path": system_template_path, "user_path": user_template_path},
                            )

                return SubmissionEvaluationCriteriaResponse(
                    session_id=session_id,
                    episode_id=episode_id,
                    task_id=task.task_id,
                    strategy=task.submission_evaluation_config.get("strategy", SubmissionEvaluationStrategy.LLM_JUDGE),
                    criteria=criteria_with_content,
                    scoring=task.submission_evaluation_config.get("scoring", {}),
                    task_context=task_context,
                )
            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(
                    logger, "get_submission_evaluation_criteria", exc, session_id=session_id, episode_id=episode_id
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to get submission evaluation criteria: {exc}"
                ) from exc

        @self.app.get(APIEndpoints.EPISODE_SUBTASK_EVALUATION_CRITERIA)
        async def get_subtask_evaluation_criteria_endpoint(
            session_id: str, episode_id: str
        ) -> List[SubtaskEvaluationCriteriaResponse]:
            """Get subtask evaluation criteria (template paths only, no rendering)."""
            log_operation_start(logger, "get_subtask_evaluation_criteria", session_id=session_id, episode_id=episode_id)
            try:
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                task = self.session_manager.benchmark_manager.get_task(episode.task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")

                """if not task.step_evaluation_config:
                    # Step evaluation is optional
                    raise HTTPException(status_code=404, detail="Task has no step evaluation config")"""

                # Build subtasks data with max_score
                subtasks_data = [
                    {
                        "subtask_id": st.subtask_id,
                        "title": st.title,
                        "description": st.description,
                        "objective": st.objective,
                        "max_score": st.subtask_max_score,  # Direct field from SubTask
                        "subtask_strategy": st.subtask_strategy,
                        "subtask_weight": st.subtask_weight,
                        "subtask_criteria": st.subtask_criteria,
                    }
                    for st in task.subtasks
                ]

                task_context = TaskEvaluationContext(
                    task_id=task.task_id,
                    title=task.title,
                    description=task.description,
                    domain=task.domain,
                    subtasks=subtasks_data,
                )

                # Build criteria with template content (not paths)
                subtask_evaluation_criteria_list = []
                for st in task.subtasks:
                    # If LLM strategy, fetch template content
                    if st.subtask_strategy == StepEvaluationStrategy.LLM_JUDGE:
                        if st.subtask_criteria is not None:
                            system_template_path = st.subtask_criteria.get("judge_system_template")
                            user_template_path = st.subtask_criteria.get("judge_user_template")

                            if system_template_path and user_template_path:
                                try:
                                    system_content = self.session_manager.benchmark_manager.get_template_content(
                                        system_template_path
                                    )
                                    user_content = self.session_manager.benchmark_manager.get_template_content(
                                        user_template_path
                                    )

                                    st.subtask_criteria["judge_system_template"] = system_content
                                    st.subtask_criteria["judge_user_template"] = user_content
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to load templates for step evaluation: {e}",
                                        extra={"system_path": system_template_path, "user_path": user_template_path},
                                    )
                    subtask_evaluation_criteria_list.append(
                        SubtaskEvaluationCriteriaResponse(
                            session_id=session_id,
                            episode_id=episode_id,
                            task_id=task.task_id,
                            subtask_id=st.subtask_id,
                            strategy=st.subtask_strategy or "",
                            criteria=st.subtask_criteria or {},
                            max_score=st.subtask_max_score,
                            weight=st.subtask_weight,
                            objective=st.objective,
                            title=st.title,
                            description=st.description,
                            task_context=task_context,
                        )
                    )

                return subtask_evaluation_criteria_list
            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(
                    logger, "get_subtask_evaluation_criteria", exc, session_id=session_id, episode_id=episode_id
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to get subtask evaluation criteria: {exc}"
                ) from exc

        @self.app.get(APIEndpoints.TEMPLATE_CONTENT)
        async def get_template_content_endpoint(template_path: str) -> TemplateContentResponse:
            """Get raw template content by path."""
            log_operation_start(logger, "get_template_content", template_path=template_path)
            try:
                # Security: validate path
                if ".." in template_path or template_path.startswith("/"):
                    raise HTTPException(status_code=400, detail="Invalid template path")

                # Construct full path
                prompts_dir = Path(self.session_manager.config_dir) / "prompts"
                full_path = prompts_dir / template_path

                if not full_path.exists():
                    raise HTTPException(status_code=404, detail=f"Template not found: {template_path}")

                if not full_path.is_file():
                    raise HTTPException(status_code=400, detail="Path is not a file")

                # Read template content
                content = full_path.read_text()

                return TemplateContentResponse(template_path=template_path, content=content)
            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(logger, "get_template_content", exc, template_path=template_path)
                raise HTTPException(status_code=500, detail=f"Failed to get template content: {exc}") from exc

        # NOTE: POST /evaluation endpoint REMOVED - evaluation computed client-side only
        # Server no longer needs to store evaluation results

        # ===== WebSocket Endpoint for Real-Time Transcript Notifications =====

        @self.app.websocket("/api/v1/episodes/{episode_id}/ws")
        async def websocket_endpoint(websocket: WebSocket, episode_id: str) -> None:
            """
            WebSocket endpoint for bidirectional transcript synchronization.

            Protocol (Server → Client):
            - Client connects → server sends {"type": "connected"}
            - Red team injects → server broadcasts {"type": "transcript_modified", "data": {...}}

            Protocol (Client → Server):
            - {"type": "ping"} → server responds {"type": "pong"}
            - {"type": "sync_request", "data": {version, checksum}} → server responds {"type": "sync_response"}
            - {"type": "push_message", "data": {message, version, checksum}} → server responds {"type": "push_ack"}

            Flow:
            1. Agent connects on first generate() call
            2. Connection persists for entire episode (bidirectional)
            3. Agent pushes messages via WebSocket (replaces REST sync API)
            4. Red team injects → server broadcasts event → Blue agent pulls delta via WebSocket
            """
            # Validate episode exists
            episode = self.session_manager.episode_manager.get_episode_by_id(episode_id)
            if not episode:
                await websocket.close(code=WebSocketCloseCode.INTERNAL_ERROR, reason=f"Episode {episode_id} not found")
                return

            # Register connection
            role = episode.context.get(MetadataKeys.ORCHESTRATION_ROLE)
            await self.session_manager.episode_manager.connection_manager.connect(
                episode_id=episode_id,
                websocket=websocket,
                metadata={
                    "role": role if role is not None else "",
                    "session_id": episode.session_id,
                },
            )

            # Send initial state event after connection
            # This unblocks the client's first generate() call which waits for a state event
            try:
                from ..episodes.transcript_state_machine import TranscriptStateMachine

                coordinator = self.session_manager.episode_manager.transcript_coordinator
                state_machine = coordinator._state_machine

                # Re-fetch episode to get current transcript state
                episode = self.session_manager.episode_manager.get_episode_by_id(episode_id)
                if episode:
                    current_state = state_machine.get_state(episode)
                    event_type = TranscriptStateMachine.state_to_event_type(current_state)
                    current_version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)
                    modification_count = episode.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0)

                    initial_state_event = StateEventMessage(
                        type=event_type,
                        data=StateEventData(
                            version=current_version,
                            operation=TranscriptOperation.INIT,
                            modification_count=modification_count,
                            state=current_state.value,
                        ),
                        id=str(uuid.uuid4()),
                        timestamp=datetime.utcnow().isoformat(),
                    )
                    await websocket.send_json(initial_state_event.model_dump())

                    logger.debug(
                        "Sent initial state event after WebSocket connect",
                        extra={
                            "episode_id": episode_id,
                            "state": current_state.value,
                            "event_type": event_type,
                            "version": current_version,
                        },
                    )
            except Exception as e:
                logger.error(
                    "Failed to send initial state event - closing connection",
                    extra={"episode_id": episode_id, "error": str(e)},
                )
                await websocket.close(code=WebSocketCloseCode.INTERNAL_ERROR, reason="Initial state event failed")
                return

            try:
                # Handle bidirectional WebSocket messages
                while True:
                    data = await websocket.receive_json()
                    message_type = data.get("type")

                    if message_type == WebSocketMessageType.PING:
                        # Keepalive
                        pong = PongMessage(timestamp=datetime.utcnow().isoformat())
                        await websocket.send_json(pong.model_dump())

                    elif message_type == WebSocketMessageType.SYNC_REQUEST:
                        # Client requesting transcript sync
                        request_data = SyncRequestData(**data.get("data", {}))

                        sync_request = TranscriptSyncRequest(
                            episode_id=episode_id,
                            since_version=request_data.since_version,
                            client_checksum=request_data.client_checksum,
                        )

                        coordinator = self.session_manager.episode_manager.transcript_coordinator
                        sync_response = await coordinator.sync(sync_request)

                        # Cast sync_mode to SyncMode (post_init normalizes it but mypy doesn't know)
                        sync_mode = (
                            sync_response.sync_mode
                            if isinstance(sync_response.sync_mode, SyncMode)
                            else SyncMode(sync_response.sync_mode)
                        )
                        response_data = SyncResponseData(
                            current_version=TranscriptVersion(
                                sequence=sync_response.current_version.sequence,
                                checksum=sync_response.current_version.checksum,
                                message_count=sync_response.current_version.message_count,
                                last_operation=sync_response.current_version.last_operation,
                            ),
                            delta=sync_response.delta,
                            full_transcript=sync_response.full_transcript,
                            sync_mode=sync_mode,
                            modified=sync_response.modified,
                        )

                        message = SyncResponseMessage(
                            data=response_data,
                            id=data.get("id", ""),
                            timestamp=datetime.utcnow().isoformat(),
                        )
                        await websocket.send_json(message.model_dump())

                    elif message_type == WebSocketMessageType.PUSH_MESSAGE:
                        # Client pushing new message (supports both normal and injection mode)
                        push_data = PushMessageRequestData(**data.get("data", {}))

                        # Support cross-episode pushes (red team targeting blue team)
                        target_episode_id = push_data.target_episode_id or episode_id

                        sync_request = TranscriptSyncRequest(
                            episode_id=target_episode_id,
                            since_version=push_data.since_version,
                            client_checksum=push_data.client_checksum,
                            messages_to_push=[push_data.message],
                            # "strategy" in client message maps to "operation" in backend
                            operation=push_data.strategy,
                            rewind_count=push_data.rewind_count,
                            insert_position=push_data.insert_position,
                        )

                        coordinator = self.session_manager.episode_manager.transcript_coordinator
                        sync_response = await coordinator.sync(sync_request)

                        # Build response with optional injection metadata
                        push_response: Dict[str, Any] = {
                            "version": sync_response.current_version.sequence,
                            "checksum": sync_response.current_version.checksum,
                        }

                        # Get current transcript state after push
                        target_episode = self.session_manager.episode_manager.get_episode_by_id(target_episode_id)
                        target_state_value: str = "modified"
                        modification_count = 0
                        if target_episode:
                            state_machine = coordinator._state_machine
                            target_state_value = state_machine.get_state(target_episode).value
                            modification_count = target_episode.context.get(
                                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0
                            )

                        # Prepare state event for broadcast (only for cross-episode/injection)
                        state_event = None
                        is_cross_episode = target_episode_id != episode_id

                        if is_cross_episode and target_episode:
                            # Cross-episode push (injection): update modification count and add metadata
                            modification_count += 1
                            await target_episode.update_context_atomic(
                                {MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: modification_count}
                            )
                            push_response["modification_count"] = modification_count
                            push_response["target_episode_id"] = target_episode_id

                            # Convert strategy string to TranscriptOperation enum
                            try:
                                operation = TranscriptOperation(push_data.strategy)
                            except ValueError:
                                operation = TranscriptOperation.APPEND

                            state_event = StateEventMessage(
                                type="transcript_modified",
                                data=StateEventData(
                                    version=sync_response.current_version.sequence,
                                    operation=operation,
                                    modification_count=modification_count,
                                    injected_by=episode_id,
                                    state=target_state_value,
                                ),
                                id=str(uuid.uuid4()),
                                timestamp=datetime.utcnow().isoformat(),
                            )
                        # For same-episode pushes, no state event needed - client knows it pushed

                        # 1. Send push_ack FIRST (client is waiting for this)
                        ack_data = PushAckData(
                            version=push_response["version"],
                            checksum=push_response["checksum"],
                            modification_count=push_response.get("modification_count"),
                            target_episode_id=push_response.get("target_episode_id"),
                        )

                        ack_message = PushAckMessage(
                            data=ack_data,
                            id=data.get("id", ""),
                            timestamp=datetime.utcnow().isoformat(),
                        )
                        await websocket.send_json(ack_message.model_dump())

                        # 2. Then broadcast state event (for next generate() call to proceed)
                        if state_event:
                            await coordinator.connection_manager.broadcast_to_episode(
                                episode_id=target_episode_id,
                                message=state_event,
                            )

                    else:
                        logger.warning(
                            "Unknown WebSocket message type",
                            extra={"episode_id": episode_id, "type": message_type},
                        )

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected", extra={"episode_id": episode_id})
            except Exception as e:
                logger.error("WebSocket error", extra={"episode_id": episode_id, "error": str(e)})
            finally:
                await self.session_manager.episode_manager.connection_manager.disconnect(episode_id, websocket)

    async def start_server(self) -> None:
        """Start the SessionRestAPI server."""
        logger.info(
            "Starting REST server",
            extra={
                "event": "rest_server_starting",
                "domain": self.session_manager.domain_name,
                "host": self.host,
                "port": self.port,
            },
        )

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
