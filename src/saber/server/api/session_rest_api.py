"""
SessionRestAPI implementation for SABER domain server.

The SessionRestAPI handles REST API endpoints for session management, episodes,
policy, status, and events. Tool execution is handled by SessionMCPAPI.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict

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
            await self.session_manager.terminate_session(session_id)
            result: Dict[str, str] = {"message": "Session terminated successfully"}
            return result

        @self.app.post("/session/{session_id}/start-episode")
        async def start_episode_endpoint(session_id: str, task_id: str) -> Dict[str, Any]:
            """Initialize episode for a task."""
            episode = await self.session_manager.start_episode(session_id, task_id)
            result: Dict[str, Any] = {
                "episode_id": episode.episode_id,
                "task_id": episode.task_id,
                "message": "Episode started successfully",
            }
            return result

        @self.app.get("/session/{session_id}/current-task")
        async def get_current_task_endpoint(session_id: str) -> Dict[str, Any]:
            """Get current task information."""
            task_info: Dict[str, Any] = await self.session_manager.get_current_task(session_id)
            return task_info

        @self.app.get("/session/{session_id}/policy")
        async def get_policy_endpoint(session_id: str) -> Dict[str, Any]:
            """Get domain policy document."""
            policy = self.session_manager.get_policy(session_id)
            policy_dict: Dict[str, Any] = policy.to_dict()
            return policy_dict

        @self.app.get("/session/{session_id}/events")
        async def get_events_endpoint(session_id: str, request: Request) -> StreamingResponse:
            """SSE endpoint for real-time updates."""
            result: StreamingResponse = await self.get_events_stream(session_id, request)
            return result

        @self.app.get("/internal/episode-status/{session_id}")
        async def check_episode_status_endpoint(session_id: str, token: str) -> Dict[str, bool]:
            """
            Internal endpoint for container polling - check if episode is still active.

            Containers use this endpoint to determine if they should continue running.
            If this returns {"active": false}, containers should self-terminate.

            Args:
                session_id: Session identifier
                token: Cleanup token for authentication

            Returns:
                {"active": true/false} indicating if episode is active
            """
            is_active = self.session_manager.episode_manager.is_episode_active(session_id, token)
            result: Dict[str, bool] = {"active": is_active}
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

    async def start_server(self) -> None:
        """Start the SessionRestAPI server."""
        logger.info(f"Starting SABER {self.session_manager.domain_name} domain REST server on {self.host}:{self.port}")

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
