"""
SessionRestAPI implementation for SABER domain server.

The SessionRestAPI handles REST API endpoints for session management, episodes,
policy, status, and events. Tool execution is handled by SessionMCPAPI.

Logging category: REST_API.
"""

# Forward declaration to avoid circular imports
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from ...logging_config import get_api_logger, log_operation_failure, log_operation_start, log_operation_success
from ...models import (
    BenchmarkInfo,
    EpisodeContext,
    EpisodeCreateResponse,
    EpisodeEndResponse,
    EpisodeStatusResponse,
    EpisodeTaskResponse,
    EvalSubmission,
    HealthResponse,
    PolicyResponse,
    SessionCreateResponse,
    SessionTerminateResponse,
)
from ...models.rest.evaluation import (
    EpisodeStepData,
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    EvaluationFileUploadResponse,
    EvaluationListResponse,
    EvaluationResponse,
    EvaluationResultSubmission,
    EvaluationSummaryResponse,
    StepEvaluationCriteriaResponse,
    SubmissionEvaluationCriteriaResponse,
    TaskEvaluationContext,
    TemplateContentResponse,
)
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

        @self.app.post("/api/v1/session", response_model=SessionCreateResponse)
        async def create_session_endpoint(client_id: str) -> SessionCreateResponse:
            """Create a new client session."""
            session = await self.session_manager.create_session(client_id)
            return SessionCreateResponse(session_id=session.session_id, message="Session created successfully")

        @self.app.delete("/api/v1/session/{session_id}", response_model=SessionTerminateResponse)
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

        @self.app.get("/api/v1/session/{session_id}/episodes/{episode_id}/task", response_model=EpisodeTaskResponse)
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

        @self.app.get("/api/v1/session/{session_id}/episodes/{episode_id}/policy", response_model=PolicyResponse)
        async def get_policy_endpoint(session_id: str, episode_id: str) -> PolicyResponse:
            """Get policy information for a specific episode."""
            episode = self.session_manager.get_episode_by_id(episode_id)
            if not episode:
                raise HTTPException(status_code=404, detail="Episode not found")

            policy = self.session_manager.get_policy(session_id, episode_id)
            policy_dict = policy.to_dict()

            return PolicyResponse(prompt=policy_dict.get("prompt", ""), domain=policy_dict.get("domain"))

        @self.app.post("/api/v1/session/{session_id}/episodes", response_model=EpisodeCreateResponse)
        async def create_episode_endpoint(session_id: str, task_id: str) -> EpisodeCreateResponse:
            """Create a new episode for a specific task with automatic dependency resolution (async)."""
            try:
                log_operation_start(
                    logger,
                    "create_episode",
                    session_id=session_id,
                    task_id=task_id,
                )

                # Initiate episode creation (returns immediately with CREATING state)
                episode = await self.session_manager.initiate_episode(session_id, task_id)

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
                    message="Episode creation initiated (poll status endpoint for readiness)"
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

        @self.app.get("/api/v1/session/{session_id}/episodes/{episode_id}/status", response_model=EpisodeStatusResponse)
        async def get_episode_status_endpoint(session_id: str, episode_id: str) -> EpisodeStatusResponse:
            """Get episode status for readiness polling."""
            try:
                episode = self.session_manager.get_episode_status(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

                # Check if episode belongs to this session
                if episode.session_id != session_id:
                    raise HTTPException(
                        status_code=403, detail=f"Episode {episode_id} does not belong to session {session_id}"
                    )

                # Create episode context if episode is ready
                episode_context = None
                if episode.is_ready:
                    episode_context = EpisodeContext(
                        session_id=session_id,
                        task_timeout=None,
                        max_steps=episode.max_steps,
                        metadata=episode.metadata,
                    )

                return EpisodeStatusResponse(
                    episode_id=episode.episode_id,
                    task_id=episode.task_id,
                    session_id=session_id,
                    state=episode.state.value,
                    is_ready=episode.is_ready,
                    message=f"Episode {episode.state.value}",
                    creation_error=episode.creation_error,
                    episode_context=episode_context,
                    attached_to_episode_id=episode.attached_to_episode_id,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.error(
                    "Failed to get episode status",
                    extra={
                        "event": "get_episode_status_failed",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "error": str(exc),
                    },
                )
                raise HTTPException(status_code=500, detail=f"Failed to get episode status: {exc}") from exc

        @self.app.delete("/api/v1/session/{session_id}/episodes/{episode_id}", response_model=EpisodeEndResponse)
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

        @self.app.get("/api/v1/tasks", response_model=BenchmarkInfo)
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

        @self.app.get("/api/v1/health", response_model=HealthResponse)
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
        @self.app.get("/api/v1/session/{session_id}/evaluations/{episode_id}", response_model=EvaluationResponse)
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

        @self.app.get("/api/v1/session/{session_id}/evaluations", response_model=EvaluationListResponse)
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

        @self.app.get("/api/v1/session/{session_id}/evaluations/summary", response_model=EvaluationSummaryResponse)
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
        @self.app.post("/api/v1/session/{session_id}/evaluations/upload", response_model=EvaluationFileUploadResponse)
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

        @self.app.get("/api/v1/session/{session_id}/episodes/{episode_id}/submission")
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

        @self.app.post("/api/v1/session/{session_id}/episodes/{episode_id}/submission")
        async def post_episode_submission_endpoint(
            session_id: str, episode_id: str, submission: EvalSubmission
        ) -> dict:
            """
            Store episode submission without ending the episode.

            Args:
                session_id: Session ID
                episode_id: Episode ID
                submission: EvalSubmission with answer and metadata

            Returns:
                Success response
            """
            log_operation_start(logger, "post_episode_submission", session_id=session_id, episode_id=episode_id)
            try:
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                # Store the submission on the episode (doesn't end it)
                episode.eval_submission = submission
                episode.submission = submission.submission

                logger.info(
                    "Episode submission stored (episode remains active)",
                    extra={
                        "event": "episode_submission_stored",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "submission_length": len(submission.submission),
                    },
                )

                return {"success": True, "message": "Submission stored"}
            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(
                    logger, "post_episode_submission", exc, session_id=session_id, episode_id=episode_id
                )
                raise HTTPException(status_code=500, detail=f"Failed to store episode submission: {exc}") from exc

        @self.app.get("/api/v1/session/{session_id}/episodes/{episode_id}/steps")
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

        @self.app.get("/api/v1/session/{session_id}/episodes/{episode_id}/submission-evaluation-criteria")
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
                if task.submission_evaluation_config.get("strategy") == "llm_judge":
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
                    strategy=task.submission_evaluation_config.get("strategy", "llm_judge"),
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

        @self.app.get("/api/v1/session/{session_id}/episodes/{episode_id}/step-evaluation-criteria")
        async def get_step_evaluation_criteria_endpoint(
            session_id: str, episode_id: str
        ) -> StepEvaluationCriteriaResponse:
            """Get step evaluation criteria (template paths only, no rendering)."""
            log_operation_start(logger, "get_step_evaluation_criteria", session_id=session_id, episode_id=episode_id)
            try:
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                task = self.session_manager.benchmark_manager.get_task(episode.task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")

                if not task.step_evaluation_config:
                    # Step evaluation is optional
                    raise HTTPException(status_code=404, detail="Task has no step evaluation config")

                # Build subtasks data with max_score and weight
                subtasks_data = [
                    {
                        "subtask_id": st.subtask_id,
                        "title": st.title,
                        "description": st.description,
                        "objective": st.objective,
                        "max_score": st.max_score,  # Direct field from SubTask
                        "weight": st.weight,  # Weight for graded scoring
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
                raw_criteria = task.step_evaluation_config.get("criteria", {})
                criteria_with_content = raw_criteria.copy()

                # If LLM strategy, fetch template content
                if task.step_evaluation_config.get("strategy") == "llm_judge":
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
                                f"Failed to load templates for step evaluation: {e}",
                                extra={"system_path": system_template_path, "user_path": user_template_path},
                            )

                return StepEvaluationCriteriaResponse(
                    session_id=session_id,
                    episode_id=episode_id,
                    task_id=task.task_id,
                    strategy=task.step_evaluation_config.get("strategy", "llm_judge"),
                    criteria=criteria_with_content,
                    subtasks=subtasks_data,
                    task_context=task_context,
                )
            except HTTPException:
                raise
            except Exception as exc:
                log_operation_failure(
                    logger, "get_step_evaluation_criteria", exc, session_id=session_id, episode_id=episode_id
                )
                raise HTTPException(status_code=500, detail=f"Failed to get step evaluation criteria: {exc}") from exc

        @self.app.get("/api/v1/templates/{template_path:path}")
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

        @self.app.post("/api/v1/session/{session_id}/episodes/{episode_id}/evaluation")
        async def submit_evaluation_endpoint(
            session_id: str, episode_id: str, request: EvaluationResultSubmission
        ) -> EvaluationResponse:
            """Submit client-side evaluation result."""
            log_operation_start(logger, "submit_evaluation", session_id=session_id, episode_id=episode_id)
            try:
                from datetime import datetime, timezone

                from ...models.rest.evaluation import EvaluationResultResponse
                from ..evaluation.models import EvaluationResult

                # Get episode to extract required fields
                episode = self.session_manager.get_episode_by_id(episode_id)
                if not episode:
                    raise HTTPException(status_code=404, detail="Episode not found")

                # Build EvaluationResult from submission
                result = EvaluationResult(
                    episode_id=episode_id,
                    task_id=request.details.get("task_id", episode.task_id),
                    strategy=request.strategy,
                    raw_score=request.raw_score,
                    max_score=request.max_score,
                    score=request.score,
                    success=request.success,
                    timestamp=datetime.now(timezone.utc),
                    details=request.details,
                    # Required fields from episode
                    submission=episode.submission if hasattr(episode, "submission") and episode.submission else "",
                    step_count=len(episode.steps),
                    # Optional fields from episode
                    executed_commands=getattr(episode, "executed_commands", []),
                    completion_reason=getattr(episode, "completion_reason", None),
                    model=getattr(episode, "model", None),
                    choices=getattr(episode, "choices", []),
                    tokens=getattr(episode, "tokens", {}),
                    execution_time=getattr(episode, "execution_time", None),
                )

                # Store evaluation
                await self.session_manager.evaluation_manager.store.save(result, session_id=session_id)

                # Build response
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
                raise
            except Exception as exc:
                log_operation_failure(logger, "submit_evaluation", exc, session_id=session_id, episode_id=episode_id)
                raise HTTPException(status_code=500, detail=f"Failed to submit evaluation: {exc}") from exc

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
