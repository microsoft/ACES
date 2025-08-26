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
            logger.warning(f"🔥 REST API TERMINATION: DELETE /session/{session_id} endpoint called")
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

        @self.app.get("/session/{session_id}/episodes/{episode_id}/events")
        async def get_episode_events_endpoint(session_id: str, episode_id: str, request: Request) -> StreamingResponse:
            """SSE endpoint for episode-specific real-time events."""
            result: StreamingResponse = await self.get_episode_events_stream(session_id, episode_id, request)
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
            """Generate SSE events for the episode."""
            try:
                # Get episode configuration from SessionManager
                episode_config = self.session_manager.get_episode_config(session_id)
                max_steps = episode_config["max_steps"]
                last_step_count = 0

                logger.info(
                    f"🔗 DEBUG: SSE Started episode monitoring for {session_id}/{episode_id} (max_steps: {max_steps})"
                )
                loop_count = 0

                while session.is_active:
                    loop_count += 1

                    # Check if client disconnected
                    if await request.is_disconnected():
                        logger.warning(
                            f"🔗 DEBUG: SSE Client disconnected after {loop_count} loops for {session_id}/{episode_id}"
                        )
                        break

                    # Check if episode should be terminated using SessionManager logic
                    should_terminate, termination_reason = self.session_manager.is_episode_over(session_id)

                    if should_terminate:
                        logger.warning(
                            f"🔥 SSE EPISODE TERMINATION: Episode {episode_id} for session {session_id} "
                            f"should terminate - reason: {termination_reason} (loop {loop_count})"
                        )
                        if termination_reason.startswith("max_steps_reached"):
                            yield (
                                f"event: max_steps_reached\n"
                                f'data: {{"current_steps": {last_step_count}, "max_steps": {max_steps}, '
                                f'"episode_id": "{episode_id}"}}\n\n'
                            )
                            # End the episode
                            logger.warning(
                                f"🔥 SSE MAX STEPS END: Ending episode {episode_id} for session {session_id} "
                                f"due to max steps"
                            )
                            self.session_manager.episode_manager.end_episode(session_id, termination_reason)
                        else:
                            yield (
                                f"event: episode_complete\n"
                                f'data: {{"episode_id": "{episode_id}", "reason": "{termination_reason}"}}\n\n'
                            )
                        break

                    # Get current episode (refresh in case it was updated)
                    current_episode = self.session_manager.episode_manager.get_current_episode(session_id)
                    if not current_episode or current_episode.episode_id != episode_id:
                        # Episode ended or changed
                        logger.warning(
                            f"🔗 DEBUG: SSE Episode ended or changed for {session_id}/{episode_id} (loop {loop_count})"
                        )
                        yield (
                            f"event: episode_complete\n"
                            f'data: {{"episode_id": "{episode_id}", "reason": "episode_ended"}}\n\n'
                        )
                        break

                    current_steps = len(current_episode.steps)

                    # Send step count update if it changed
                    if current_steps != last_step_count:
                        logger.debug(
                            f"🔗 DEBUG: SSE Step count changed from {last_step_count} to {current_steps} "
                            f"for {session_id}/{episode_id}"
                        )
                        yield (
                            f"event: step_count_update\n"
                            f'data: {{"current_steps": {current_steps}, "max_steps": {max_steps}, '
                            f'"episode_id": "{episode_id}"}}\n\n'
                        )
                        last_step_count = current_steps

                    # Send periodic heartbeat
                    timestamp = datetime.utcnow().isoformat()
                    if loop_count % 5 == 0:  # Log every 5th heartbeat
                        logger.debug(
                            f"🔗 DEBUG: SSE Heartbeat #{loop_count} for {session_id}/{episode_id} "
                            f"- steps: {current_steps}"
                        )
                    yield (
                        f"event: heartbeat\n"
                        f'data: {{"timestamp": "{timestamp}", "episode_id": "{episode_id}", '
                        f'"steps": {current_steps}}}\n\n'
                    )

                    # Wait before next check
                    await asyncio.sleep(2)  # Check every 2 seconds for responsiveness

            except Exception as e:
                logger.error(
                    f"🔗 DEBUG: SSE Episode SSE stream error for {session_id}/{episode_id}: {type(e).__name__}: {e}"
                )
                import traceback

                logger.error(f"🔗 DEBUG: SSE Traceback: {traceback.format_exc()}")
                yield f'event: error\ndata: {{"message": "Stream error: {str(e)}"}}\n\n'
            finally:
                logger.info(
                    f"🔗 DEBUG: SSE Episode SSE stream ended for {session_id}/{episode_id} "
                    f"(total loops: {loop_count if 'loop_count' in locals() else 'unknown'})"
                )

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
