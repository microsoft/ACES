"""
SessionRestAPI implementation for SABER domain server.

The SessionRestAPI handles REST API endpoints for session management, episodes,
policy, status, and events. Tool execution is handled by SessionMCPAPI.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ...base import MCPHeaders
from .events.tool_event_publisher import ToolEventPublisher

logger = logging.getLogger(__name__)


class SessionRestAPI:
    """
    REST API layer for SessionManager.

    Handles session management, episodes, policy, status, and events.
    Tool execution is handled by SessionMCPAPI.
    """

    def __init__(self, session_manager: Any, host: str = "0.0.0.0", port: int = 8000) -> None:
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

        @self.app.post("/session", response_model=Dict[str, str])
        async def create_session_endpoint(client_id: str) -> Dict[str, str]:
            """Create a new client session."""
            session = await self.session_manager.create_session(client_id)
            result: Dict[str, str] = {"session_id": session.session_id, "message": "Session created successfully"}
            return result

        @self.app.delete("/session/{session_id}")
        async def terminate_session_endpoint(session_id: str) -> Dict[str, str]:
            """Terminate a client session."""
            logger.warning(f"🔥 REST API TERMINATION: DELETE /session/{session_id} endpoint called")
            await self.session_manager.terminate_session(session_id)
            result: Dict[str, str] = {"message": "Session terminated successfully"}
            return result

        @self.app.get("/session/{session_id}/episodes/{episode_id}/task")
        async def get_episode_task_endpoint(session_id: str, episode_id: str) -> Dict[str, Any]:
            """Get task information for a specific episode."""
            task = await self.session_manager.get_current_task(session_id, episode_id)
            task_dict = cast(Dict[str, Any], task.to_dict())

            # Get episode to access its configuration
            episode = self.session_manager.get_episode_by_id(episode_id)
            if episode:
                # Get episode configuration from the task using BenchmarkManager
                episode_config = self.session_manager.benchmark_manager.get_episode_config(episode.task_id)
                task_dict["episode_context"] = episode_config
            else:
                task_dict["episode_context"] = {"session_id": session_id}

            task_dict["episode_id"] = episode_id

            return task_dict

        @self.app.get("/session/{session_id}/episodes/{episode_id}/policy")
        async def get_policy_endpoint(session_id: str, episode_id: str) -> Dict[str, Any]:
            """Get policy information for a specific episode."""
            episode = self.session_manager.get_episode_by_id(episode_id)
            if not episode:
                raise HTTPException(status_code=404, detail="Episode not found")

            policy = self.session_manager.get_policy(session_id, episode_id)
            policy_dict = policy.to_dict()
            return dict(policy_dict) if policy_dict else {}

        @self.app.get("/tool-events/stream")
        async def tool_events_stream(request: Request) -> EventSourceResponse:
            """Server-Sent Events stream for real-time tool call events with episode filtering."""
            session_id = request.headers.get(MCPHeaders.SESSION_ID)
            episode_id = request.headers.get(MCPHeaders.EPISODE_ID)  # EPISODE-FIRST: Support episode filtering

            if not session_id:
                raise HTTPException(status_code=400, detail=f"Missing {MCPHeaders.SESSION_ID} header")

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

        @self.app.post("/session/{session_id}/episodes")
        async def create_episode_endpoint(session_id: str, task_id: str) -> Dict[str, Any]:
            """Create a new episode for a specific task."""
            episode = await self.session_manager.start_episode(session_id, task_id)
            result: Dict[str, Any] = {
                "episode_id": episode.episode_id,
                "task_id": task_id,
                "session_id": session_id,
                "state": episode.state.value,
                "message": "Episode created successfully",
            }
            return result

        @self.app.get("/session/{session_id}/episodes")
        async def list_episodes_endpoint(session_id: str, include_completed: bool = False) -> Dict[str, Any]:
            """List all episodes for a session."""
            session = self.session_manager._get_session(session_id)

            result: Dict[str, Any] = {
                "session_id": session_id,
                "active_episodes": session.active_episode_ids,
                "episode_history": session.episode_history if include_completed else [],
                "episode_counts": session.get_episode_count(),
                "task_queue": session.task_queue,
            }
            return result

        @self.app.get("/session/{session_id}/episodes/active")
        async def list_active_episodes_endpoint(session_id: str) -> Dict[str, Any]:
            """List only active episodes for a session."""
            session = self.session_manager._get_session(session_id)

            # Get episode details from episode manager
            active_episodes = []
            for episode_id in session.active_episode_ids:
                episode = self.session_manager.episode_manager.get_episode_by_id(episode_id)
                if episode:
                    active_episodes.append(
                        {
                            "episode_id": episode.episode_id,
                            "task_id": episode.task_id,
                            "state": episode.state.value,
                            "step_count": len(episode.steps),
                            "start_time": episode.start_time.isoformat(),
                            "duration": episode.duration,
                        }
                    )

            result: Dict[str, Any] = {
                "session_id": session_id,
                "active_episodes": active_episodes,
                "count": len(active_episodes),
            }
            return result

        @self.app.get("/session/{session_id}/episodes/{episode_id}")
        async def get_episode_endpoint(session_id: str, episode_id: str) -> Dict[str, Any]:
            """Get detailed information about a specific episode."""
            # Validate episode belongs to session
            session = self.session_manager._get_session(session_id)
            if episode_id not in session.active_episode_ids and episode_id not in session.episode_history:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found in session {session_id}")

            episode = self.session_manager.episode_manager.get_episode_by_id(episode_id)
            if not episode:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

            result: Dict[str, Any] = {
                "episode_id": episode.episode_id,
                "task_id": episode.task_id,
                "session_id": episode.session_id,
                "state": episode.state.value,
                "step_count": len(episode.steps),
                "start_time": episode.start_time.isoformat(),
                "end_time": episode.end_time.isoformat() if episode.end_time else None,
                "duration": episode.duration,
                "completion_reason": episode.completion_reason,
                "context": episode.context,
                "metadata": episode.metadata,
            }
            return result

        @self.app.delete("/session/{session_id}/episodes/{episode_id}")
        async def end_episode_endpoint(
            session_id: str, episode_id: str, reason: str = "manual_termination", result: Optional[str] = None
        ) -> Dict[str, Any]:
            """End a specific episode."""
            response = await self.session_manager.end_episode(session_id, episode_id, reason, result)
            return cast(Dict[str, Any], response.model_dump())

        @self.app.post("/session/{session_id}/episodes/{episode_id}/actions")
        async def execute_episode_action_endpoint(
            session_id: str, episode_id: str, action_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Execute action in specific episode context."""
            from ..base import Action

            try:
                action = Action(**action_data)
                result = await self.session_manager.execute_episode_action(session_id, episode_id, action)
                return {
                    "success": result.success,
                    "data": result.data,
                    "execution_time": result.execution_time,
                    "error": result.error,
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid action data: {str(e)}")

        @self.app.post("/session/{session_id}/orchestrate")
        async def orchestrate_tasks_endpoint(session_id: str, task_ids: str) -> Dict[str, Any]:
            """Queue multiple tasks for orchestration in a session."""
            task_id_list = [tid.strip() for tid in task_ids.split(",")]
            session = self.session_manager._get_session(session_id)

            for task_id in task_id_list:
                session.add_task_to_queue(task_id)

            result: Dict[str, Any] = {
                "session_id": session_id,
                "queued_tasks": task_id_list,
                "task_queue": session.task_queue,
                "message": f"Queued {len(task_id_list)} tasks for orchestration",
            }
            return result

        @self.app.get("/benchmark")
        async def get_benchmark_endpoint() -> Dict[str, Any]:
            """
            Get complete benchmark task list with episode attempts for client orchestration.
            Each task reports its own configured episode_attempts.
            """
            benchmark_info = self.session_manager.get_benchmark_info()
            return cast(Dict[str, Any], benchmark_info.to_dict())

        @self.app.get("/health")
        async def health_check() -> Dict[str, str]:
            """Health check endpoint."""
            result: Dict[str, str] = {"status": "healthy", "domain": self.session_manager.domain_name}
            return result

        @self.app.get("/sessions")
        async def list_sessions() -> Dict[str, Any]:
            """List all active sessions."""
            sessions = {
                session_id: {
                    "client_id": session.client_id,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "active_episode_ids": session.active_episode_ids,
                    "episode_counts": session.get_episode_count(),
                    "is_active": session.is_active,
                }
                for session_id, session in self.session_manager.active_sessions.items()
            }
            result: Dict[str, Any] = {
                "active_session_count": len(self.session_manager.active_sessions),
                "sessions": sessions,
            }
            return result

        @self.app.get("/sessions/stats")
        async def get_session_stats() -> Dict[str, Any]:
            """Get detailed session statistics including timeout information."""
            stats: Dict[str, Any] = self.session_manager.get_session_stats()
            return stats

        @self.app.get("/debug/cleanup-history")
        async def get_cleanup_history() -> Dict[str, Any]:
            """Get complete cleanup history for debugging."""
            if hasattr(self.session_manager, "cleanup_manager"):
                history = self.session_manager.cleanup_manager.get_cleanup_history()
                stats = self.session_manager.cleanup_manager.get_cleanup_stats()
                return {
                    "history": [
                        {
                            "session_id": op.session_id,
                            "reason": str(op.reason),
                            "context": op.context,
                            "start_time": op.start_time.isoformat(),
                            "end_time": op.end_time.isoformat() if op.end_time else None,
                            "success": op.success,
                            "error": op.error,
                            "duration_seconds": op.duration_seconds,
                            "steps_completed": op.steps_completed,
                        }
                        for op in history
                    ],
                    "stats": stats,
                }
            return {"error": "Cleanup manager not available"}

        @self.app.get("/debug/cleanup-history/{session_id}")
        async def get_session_cleanup_history(session_id: str) -> Dict[str, Any]:
            """Get cleanup history for a specific session."""
            if hasattr(self.session_manager, "cleanup_manager"):
                history = self.session_manager.cleanup_manager.get_cleanup_history(session_id)
                return {
                    "session_id": session_id,
                    "history": [
                        {
                            "reason": str(op.reason),
                            "context": op.context,
                            "start_time": op.start_time.isoformat(),
                            "end_time": op.end_time.isoformat() if op.end_time else None,
                            "success": op.success,
                            "error": op.error,
                            "duration_seconds": op.duration_seconds,
                            "steps_completed": op.steps_completed,
                        }
                        for op in history
                    ],
                }
            return {"error": "Cleanup manager not available"}

        @self.app.get("/debug/active-cleanups")
        async def get_active_cleanups() -> Dict[str, Any]:
            """Get currently active cleanup operations."""
            if hasattr(self.session_manager, "cleanup_manager"):
                active = self.session_manager.cleanup_manager.get_active_cleanups()
                return {"active_cleanup_count": len(active), "active_session_ids": list(active)}
            return {"error": "Cleanup manager not available"}

    async def start_server(self) -> None:
        """Start the SessionRestAPI server."""
        logger.info(f"Starting SABER {self.session_manager.domain_name} domain REST server on {self.host}:{self.port}")

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
