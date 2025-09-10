"""
SessionRestAPI implementation for SABER domain server.

The SessionRestAPI handles REST API endpoints for session management, episodes,
policy, status, and events. Tool execution is handled by SessionMCPAPI.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

# Forward declaration to avoid circular imports
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ...models import (
    ActionExecutionResponse,
    ActiveCleanupInfo,
    ActiveCleanupsResponse,
    ActiveEpisodeInfo,
    ActiveEpisodesResponse,
    BenchmarkInfo,
    CleanupHistoryEntry,
    CleanupHistoryResponse,
    EpisodeContext,
    EpisodeCreateResponse,
    EpisodeDetailResponse,
    EpisodeEndResponse,
    EpisodeListResponse,
    EpisodeTaskResponse,
    HealthResponse,
    HTTPHeaders,
    PolicyResponse,
    SessionCleanupHistoryResponse,
    SessionCreateResponse,
    SessionListResponse,
    SessionStatsResponse,
    SessionSummary,
    SessionTerminateResponse,
    TaskOrchestrationResponse,
)
from ...models.rest.evaluation import EvaluationListResponse, EvaluationResponse, EvaluationSummaryResponse
from ..evaluation.exceptions import EvaluationNotFoundError, InvalidEvaluationRequestError, SessionEvaluationError
from .events.tool_event_publisher import ToolEventPublisher

if TYPE_CHECKING:
    from ..session_manager import SessionManager

logger = logging.getLogger(__name__)


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

        # Initialize tool event publisher for real-time SSE events
        self.tool_event_publisher = ToolEventPublisher(max_queue_size=100)

        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"SABER {session_manager.domain_name.title()} Domain Server",
            description=f"SessionManager REST API for {session_manager.domain_name} security domain",
            version="1.0.0",
        )

        # Setup API routes
        self._setup_routes()

        logger.info(f"SessionRestAPI initialized for domain '{session_manager.domain_name}' on {host}:{port}")

    def _setup_routes(self) -> None:
        """Setup FastAPI routes for the session management API."""

        @self.app.post("/session", response_model=SessionCreateResponse)
        async def create_session_endpoint(client_id: str) -> SessionCreateResponse:
            """Create a new client session."""
            session = await self.session_manager.create_session(client_id)
            return SessionCreateResponse(session_id=session.session_id, message="Session created successfully")

        @self.app.delete("/session/{session_id}", response_model=SessionTerminateResponse)
        async def terminate_session_endpoint(session_id: str) -> SessionTerminateResponse:
            """Terminate a client session."""
            logger.warning(f"🔥 REST API TERMINATION: DELETE /session/{session_id} endpoint called")
            await self.session_manager.terminate_session(session_id)
            return SessionTerminateResponse(message="Session terminated successfully")

        @self.app.get("/session/{session_id}/episodes/{episode_id}/task", response_model=EpisodeTaskResponse)
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
            )

        @self.app.get("/session/{session_id}/episodes/{episode_id}/policy", response_model=PolicyResponse)
        async def get_policy_endpoint(session_id: str, episode_id: str) -> PolicyResponse:
            """Get policy information for a specific episode."""
            episode = self.session_manager.get_episode_by_id(episode_id)
            if not episode:
                raise HTTPException(status_code=404, detail="Episode not found")

            policy = self.session_manager.get_policy(session_id, episode_id)
            policy_dict = policy.to_dict()

            return PolicyResponse(prompt=policy_dict.get("prompt", ""), domain=policy_dict.get("domain"))

        @self.app.get("/tool-events/stream")
        async def tool_events_stream(request: Request) -> EventSourceResponse:
            """Server-Sent Events stream for real-time tool call events with episode filtering."""
            session_id = request.headers.get(HTTPHeaders.SESSION_ID)
            episode_id = request.headers.get(HTTPHeaders.EPISODE_ID)  # EPISODE-FIRST: Support episode filtering

            if not session_id:
                raise HTTPException(status_code=400, detail=f"Missing {HTTPHeaders.SESSION_ID} header")

            logger.info(f"Starting tool events SSE stream for session: {session_id}, episode: {episode_id}")

            # Check if session exists
            if not hasattr(self, "tool_event_publisher"):
                raise HTTPException(status_code=500, detail="Tool event publisher not available")

            # Subscribe to tool events for this session (and optionally episode)
            event_queue = await self.tool_event_publisher.subscribe(session_id, episode_id=episode_id)

            async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
                """Generate SSE events from the tool event queue."""
                try:
                    # Send initial connection event
                    yield {
                        "data": json.dumps(
                            {
                                "type": "connection",
                                "message": "Tool events stream connected",
                                "session_id": session_id,
                                "episode_id": episode_id,  # EPISODE-FIRST: Include episode_id in events
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    }

                    # Stream tool events
                    while True:
                        try:
                            # Wait for events with timeout for heartbeat
                            event = await asyncio.wait_for(event_queue.get(), timeout=30.0)

                            # EPISODE-FIRST: Filter events by episode_id if specified
                            if episode_id and event.get("episode_id") != episode_id:
                                logger.debug(
                                    f"Skipping event for different episode: {event.get('episode_id')} != {episode_id}"
                                )
                                continue

                            yield {"data": json.dumps(event)}
                            logger.debug(
                                f"Sent tool event: {event['type']} for session {session_id}, episode {episode_id}"
                            )
                        except asyncio.TimeoutError:
                            # Send heartbeat to keep connection alive
                            yield {
                                "data": json.dumps(
                                    {
                                        "type": "heartbeat",
                                        "session_id": session_id,
                                        "episode_id": episode_id,  # EPISODE-FIRST: Include episode_id in heartbeat
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                    }
                                )
                            }
                            logger.debug(f"Sent heartbeat for session {session_id}, episode {episode_id}")
                except Exception as e:
                    logger.error(f"Tool events stream error for session {session_id}, episode {episode_id}: {e}")
                    yield {
                        "data": json.dumps(
                            {
                                "type": "error",
                                "message": str(e),
                                "session_id": session_id,
                                "episode_id": episode_id,  # EPISODE-FIRST: Include episode_id in error
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    }
                finally:
                    # Clean up subscription
                    await self.tool_event_publisher.unsubscribe(session_id, episode_id=episode_id)
                    logger.info(f"Tool events stream closed for session: {session_id}, episode: {episode_id}")

            return EventSourceResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )

        @self.app.post("/session/{session_id}/episodes", response_model=EpisodeCreateResponse)
        async def create_episode_endpoint(session_id: str, task_id: str) -> EpisodeCreateResponse:
            """Create a new episode for a specific task."""
            try:
                logger.info(f"🔄 Episode creation endpoint called: session_id={session_id}, task_id={task_id}")
                episode = await self.session_manager.start_episode(session_id, task_id)
                logger.info(f"✅ Episode created successfully: episode_id={episode.episode_id}")

                # Create episode context with limits and metadata
                episode_context = EpisodeContext(
                    session_id=session_id,
                    task_timeout=None,
                    max_steps=episode.max_steps,
                    metadata=episode.metadata,
                )
                logger.info("✅ Episode context created successfully")

                response = EpisodeCreateResponse(
                    episode_id=episode.episode_id,
                    task_id=task_id,
                    session_id=session_id,
                    state=episode.state.value,
                    message="Episode created successfully",
                    episode_context=episode_context,
                )
                logger.info("✅ Episode response created successfully, returning to client")
                return response
            except Exception as e:
                logger.error(f"❌ Error in episode creation endpoint: {type(e).__name__}: {str(e)}")
                logger.error("❌ Full traceback:", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to create episode: {str(e)}")

        @self.app.get("/session/{session_id}/episodes", response_model=EpisodeListResponse)
        async def list_episodes_endpoint(session_id: str, include_completed: bool = False) -> EpisodeListResponse:
            """List all episodes for a session."""
            session = self.session_manager._get_session(session_id)

            return EpisodeListResponse(
                session_id=session_id,
                active_episodes=session.active_episode_ids,
                episode_history=session.episode_history if include_completed else [],
                episode_counts=session.get_episode_count(),
                task_queue=session.task_queue,
            )

        @self.app.get("/session/{session_id}/episodes/active", response_model=ActiveEpisodesResponse)
        async def list_active_episodes_endpoint(session_id: str) -> ActiveEpisodesResponse:
            """List only active episodes for a session."""
            session = self.session_manager._get_session(session_id)

            # Get episode details from episode manager
            active_episodes = []
            for episode_id in session.active_episode_ids:
                episode = self.session_manager.episode_manager.get_episode_by_id(episode_id)
                if episode:
                    active_episodes.append(
                        ActiveEpisodeInfo(
                            episode_id=episode.episode_id,
                            task_id=episode.task_id,
                            state=episode.state.value,
                            step_count=len(episode.steps),
                            start_time=episode.start_time.isoformat(),
                            duration=episode.duration,
                        )
                    )

            return ActiveEpisodesResponse(
                session_id=session_id,
                active_episodes=active_episodes,
                count=len(active_episodes),
            )

        @self.app.get("/session/{session_id}/episodes/{episode_id}", response_model=EpisodeDetailResponse)
        async def get_episode_endpoint(session_id: str, episode_id: str) -> EpisodeDetailResponse:
            """Get detailed information about a specific episode."""
            # Validate episode belongs to session
            session = self.session_manager._get_session(session_id)
            if episode_id not in session.active_episode_ids and episode_id not in session.episode_history:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found in session {session_id}")

            episode = self.session_manager.episode_manager.get_episode_by_id(episode_id)
            if not episode:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

            return EpisodeDetailResponse(
                episode_id=episode.episode_id,
                task_id=episode.task_id,
                session_id=episode.session_id,
                state=episode.state.value,
                step_count=len(episode.steps),
                start_time=episode.start_time.isoformat(),
                end_time=episode.end_time.isoformat() if episode.end_time else None,
                duration=episode.duration,
                completion_reason=episode.completion_reason,
                context=episode.context,
                metadata=episode.metadata,
            )

        @self.app.delete("/session/{session_id}/episodes/{episode_id}", response_model=EpisodeEndResponse)
        async def end_episode_endpoint(
            session_id: str, episode_id: str, reason: str = "manual_termination", result: Optional[str] = None
        ) -> EpisodeEndResponse:
            """End a specific episode."""
            response = await self.session_manager.end_episode(session_id, episode_id, reason, result)
            return response

        @self.app.post("/session/{session_id}/episodes/{episode_id}/actions", response_model=ActionExecutionResponse)
        async def execute_episode_action_endpoint(
            session_id: str, episode_id: str, action_data: Dict[str, Any]
        ) -> ActionExecutionResponse:
            """Execute action in specific episode context."""
            from ..base import Action

            try:
                action = Action(**action_data)
                result = await self.session_manager.execute_episode_action(session_id, episode_id, action)
                return ActionExecutionResponse(
                    success=result.success,
                    data=result.data,
                    execution_time=result.execution_time,
                    error=result.error,
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid action data: {str(e)}")

        @self.app.post("/session/{session_id}/orchestrate", response_model=TaskOrchestrationResponse)
        async def orchestrate_tasks_endpoint(session_id: str, task_ids: str) -> TaskOrchestrationResponse:
            """Queue multiple tasks for orchestration in a session."""
            task_id_list = [tid.strip() for tid in task_ids.split(",")]
            session = self.session_manager._get_session(session_id)

            for task_id in task_id_list:
                session.add_task_to_queue(task_id)

            return TaskOrchestrationResponse(
                session_id=session_id,
                queued_tasks=task_id_list,
                message=f"Queued {len(task_id_list)} tasks for orchestration",
            )

        @self.app.get("/benchmark", response_model=BenchmarkInfo)
        async def get_benchmark_endpoint() -> BenchmarkInfo:
            """
            Get complete benchmark task list with episode attempts for client orchestration.
            Returns BenchmarkInfo object with typed data structure.
            Each task reports its own configured episode_attempts.
            """
            logger.debug("Benchmark endpoint called")
            try:
                benchmark_info: BenchmarkInfo = self.session_manager.get_benchmark_info()
                logger.debug(
                    f"Got benchmark info: {benchmark_info.total_tasks} tasks, {benchmark_info.total_episodes} episodes"
                )

                # Test serialization before returning
                try:
                    benchmark_info.model_dump()
                    logger.debug("Benchmark info serialization successful")
                except Exception as ser_e:
                    logger.error(f"Benchmark info serialization failed: {ser_e}")
                    raise

                return benchmark_info
            except Exception as e:
                logger.error(f"Error in benchmark endpoint: {e}")
                logger.error(f"Exception type: {type(e)}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(status="healthy", domain=self.session_manager.domain_name)

        @self.app.get("/sessions", response_model=SessionListResponse)
        async def list_sessions() -> SessionListResponse:
            """List all active sessions."""
            sessions = [
                SessionSummary(
                    session_id=session_id,
                    client_id=session.client_id,
                    created_at=session.created_at.isoformat(),
                    last_activity=session.last_activity.isoformat(),
                    is_active=session.is_active,
                    active_episodes=len(session.active_episode_ids),
                    total_episodes=session.get_episode_count().get("total", 0),
                )
                for session_id, session in self.session_manager.active_sessions.items()
            ]
            return SessionListResponse(
                sessions=sessions,
                total_count=len(self.session_manager.active_sessions),
                active_count=len([s for s in sessions if s.is_active]),
            )

        @self.app.get("/sessions/stats", response_model=SessionStatsResponse)
        async def get_session_stats() -> SessionStatsResponse:
            """Get detailed session statistics including timeout information."""
            stats = self.session_manager.get_session_stats()
            return SessionStatsResponse(
                total_sessions=stats.get("total_sessions", 0),
                active_sessions=stats.get("active_sessions", 0),
                total_episodes=stats.get("total_episodes", 0),
                active_episodes=stats.get("active_episodes", 0),
                average_session_duration=stats.get("average_session_duration"),
                oldest_session_age=stats.get("oldest_session_age"),
            )

        @self.app.get("/debug/cleanup-history", response_model=CleanupHistoryResponse)
        async def get_cleanup_history() -> CleanupHistoryResponse:
            """Get complete cleanup history for debugging."""
            if hasattr(self.session_manager, "cleanup_manager"):
                history = self.session_manager.cleanup_manager.get_cleanup_history()
                return CleanupHistoryResponse(
                    cleanup_history=[
                        CleanupHistoryEntry(
                            session_id=op.session_id,
                            cleanup_time=op.start_time.isoformat(),
                            reason=str(op.reason),
                            episode_count=op.steps_completed or 0,
                        )
                        for op in history
                    ],
                    total_cleanups=len(history),
                )
            raise HTTPException(status_code=500, detail="Cleanup manager not available")

        @self.app.get("/debug/cleanup-history/{session_id}", response_model=SessionCleanupHistoryResponse)
        async def get_session_cleanup_history(session_id: str) -> SessionCleanupHistoryResponse:
            """Get cleanup history for a specific session."""
            if hasattr(self.session_manager, "cleanup_manager"):
                history = self.session_manager.cleanup_manager.get_cleanup_history(session_id)
                return SessionCleanupHistoryResponse(
                    session_id=session_id,
                    cleanup_entries=[
                        CleanupHistoryEntry(
                            session_id=op.session_id,
                            cleanup_time=op.start_time.isoformat(),
                            reason=str(op.reason),
                            episode_count=op.steps_completed or 0,
                        )
                        for op in history
                    ],
                    total_cleanups=len(history),
                )
            raise HTTPException(status_code=500, detail="Cleanup manager not available")

        @self.app.get("/debug/active-cleanups", response_model=ActiveCleanupsResponse)
        async def get_active_cleanups() -> ActiveCleanupsResponse:
            """Get currently active cleanup operations."""
            if hasattr(self.session_manager, "cleanup_manager"):
                active = self.session_manager.cleanup_manager.get_active_cleanups()
                return ActiveCleanupsResponse(
                    active_cleanups=[
                        ActiveCleanupInfo(
                            session_id=session_id,
                            cleanup_type="session_cleanup",
                            start_time="Unknown",  # Would need to track this in cleanup_manager
                            progress=None,
                        )
                        for session_id in active
                    ],
                    count=len(active),
                )
            raise HTTPException(status_code=500, detail="Cleanup manager not available")

        # Evaluation endpoints
        @self.app.get("/api/v1/session/{session_id}/evaluations/{episode_id}", response_model=EvaluationResponse)
        async def get_evaluation_endpoint(session_id: str, episode_id: str) -> EvaluationResponse:
            """Get evaluation result for specific episode."""
            try:
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
            except InvalidEvaluationRequestError as e:
                raise HTTPException(status_code=422, detail=str(e))
            except SessionEvaluationError as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/session/{session_id}/evaluations/summary", response_model=EvaluationSummaryResponse)
        async def get_evaluation_summary_endpoint(session_id: str) -> EvaluationSummaryResponse:
            """Get aggregate evaluation summary for session."""
            try:
                evaluation_service = self.session_manager.get_evaluation_service()
                summary = await evaluation_service.get_session_summary(session_id)
                return EvaluationSummaryResponse(**summary)
            except InvalidEvaluationRequestError as e:
                raise HTTPException(status_code=422, detail=str(e))
            except SessionEvaluationError as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def start_server(self) -> None:
        """Start the SessionRestAPI server."""
        logger.info(f"Starting SABER {self.session_manager.domain_name} domain REST server on {self.host}:{self.port}")

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
