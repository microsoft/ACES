"""
SessionRestAPI implementation for SABER domain server.

The SessionRestAPI handles REST API endpoints for session management, episodes,
policy, status, and events. Tool execution is handled by SessionMCPAPI.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, cast

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

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

        @self.app.get("/session/{session_id}/current-task")
        async def get_current_task_endpoint(session_id: str) -> Dict[str, Any]:
            """Get current task information."""
            task = await self.session_manager.get_current_task(session_id)
            task_dict = cast(Dict[str, Any], task.to_dict())

            # Add episode context
            episode_config = self.session_manager.get_episode_config(session_id)
            task_dict["episode_context"] = episode_config

            return task_dict

        @self.app.get("/session/{session_id}/policy")
        async def get_policy_endpoint(session_id: str) -> Dict[str, Any]:
            """Get domain policy document."""
            policy = self.session_manager.get_policy(session_id)
            policy_dict = policy.to_dict()
            return dict(policy_dict) if policy_dict else {}

        @self.app.get("/session/{session_id}/events")
        async def get_events_endpoint(session_id: str, request: Request) -> StreamingResponse:
            """SSE endpoint for real-time updates."""
            result: StreamingResponse = await self.get_events_stream(session_id, request)
            return result

        @self.app.get("/session/{session_id}/episodes/{episode_id}/events")
        async def get_episode_events_endpoint(session_id: str, episode_id: str, request: Request) -> StreamingResponse:
            """SSE endpoint for episode-specific real-time events."""
            result: StreamingResponse = await self.get_episode_events_stream(session_id, episode_id, request)
            return result

        @self.app.post("/session/{session_id}/start-benchmark")
        async def start_benchmark_endpoint(session_id: str, benchmark_config: Dict[str, Any] = {}) -> Dict[str, Any]:
            """Start a full benchmark with all tasks for the session."""
            benchmark_session = await self.session_manager.start_benchmark(session_id, benchmark_config)
            result: Dict[str, Any] = {
                "benchmark_session": benchmark_session,
                "message": "Benchmark started successfully",
            }
            return result

        @self.app.get("/tasks")
        async def list_tasks_endpoint() -> Dict[str, Any]:
            """List all available tasks for benchmarking."""
            tasks = self.session_manager.benchmark_manager.list_benchmark_tasks()
            result: Dict[str, Any] = {
                "tasks": tasks,
                "total_tasks": len(tasks),
            }
            return result

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
                    "current_episode_id": session.current_episode_id,
                    "current_task_id": session.current_task_id,
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

    async def get_events_stream(self, session_id: str, request: Request) -> StreamingResponse:
        """
        SSE endpoint for real-time updates.

        Args:
            session_id: ID of the client session
            request: FastAPI request object

        Returns:
            StreamingResponse with SSE events
        """
        session = self.session_manager._get_session(session_id)

        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate SSE events for the session."""
            try:
                while session.is_active:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break

                    # Send heartbeat event
                    yield f'event: heartbeat\ndata: {{"timestamp": "{datetime.utcnow().isoformat()}"}}\n\n'

                    # Wait before next heartbeat
                    await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"SSE stream error for session {session_id}: {e}")
            finally:
                logger.info(f"SSE stream ended for session {session_id}")

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    async def get_episode_events_stream(self, session_id: str, episode_id: str, request: Request) -> StreamingResponse:
        """
        SSE endpoint for episode-specific real-time events.

        Args:
            session_id: ID of the client session
            episode_id: ID of the episode to monitor
            request: FastAPI request object

        Returns:
            StreamingResponse with episode SSE events
        """
        session = self.session_manager._get_session(session_id)
        episode = self.session_manager.episode_manager.get_current_episode(session_id)

        if not episode or episode.episode_id != episode_id:
            # Return empty stream if episode not found or doesn't match
            async def empty_generator() -> AsyncGenerator[str, None]:
                yield 'event: error\ndata: {"message": "Episode not found or inactive"}\n\n'
                return

            return StreamingResponse(
                empty_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control",
                },
            )

        async def episode_event_generator() -> AsyncGenerator[str, None]:
            """Generate SSE events for the episode - simplified for essential termination signaling."""
            try:
                logger.info(f"🔗 SSE: Monitoring episode {episode_id} for session {session_id}")

                while session.is_active:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        logger.info(f"🔗 SSE: Client disconnected for {session_id}/{episode_id}")
                        break

                    # Check if episode should be terminated by server
                    should_terminate, reason = self.session_manager.should_terminate_episode(session_id)
                    if should_terminate:
                        logger.warning(f"🔗 SSE: Server terminating episode {episode_id} - {reason}")
                        # Let the session manager handle the actual termination
                        await self.session_manager.end_episode(session_id, reason)
                        yield (
                            f"event: episode_terminated\n"
                            f'data: {{"episode_id": "{episode_id}", "reason": "{reason}"}}\n\n'
                        )
                        break

                    # Check if episode still exists (client may have ended it naturally)
                    current_episode = self.session_manager.episode_manager.get_current_episode(session_id)
                    if not current_episode or current_episode.episode_id != episode_id:
                        logger.info(f"🔗 SSE: Episode {episode_id} ended naturally")
                        yield (
                            f"event: episode_complete\n"
                            f'data: {{"episode_id": "{episode_id}", "reason": "completed"}}\n\n'
                        )
                        break

                    # Send heartbeat every 10 seconds
                    current_steps = len(current_episode.steps)
                    timestamp = datetime.utcnow().isoformat()
                    yield (
                        f"event: heartbeat\n"
                        f'data: {{"episode_id": "{episode_id}", "steps": {current_steps}, '
                        f'"timestamp": "{timestamp}"}}\n\n'
                    )

                    await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"🔗 SSE: Episode monitoring error for {session_id}/{episode_id}: {e}")
                yield f'event: error\ndata: {{"message": "Monitoring error: {str(e)}"}}\n\n'
            finally:
                logger.info(f"🔗 SSE: Stopped monitoring episode {episode_id}")

        return StreamingResponse(
            episode_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    async def start_server(self) -> None:
        """Start the SessionRestAPI server."""
        logger.info(f"Starting SABER {self.session_manager.domain_name} domain REST server on {self.host}:{self.port}")

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
