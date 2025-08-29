"""
SessionRestAPI implementation for SABER domain server.

The SessionRestAPI handles REST API endpoints for session management, episodes,
policy, status, and events. Tool execution is handled by SessionMCPAPI.
"""

import logging
from typing import Any, Dict, cast

import uvicorn
from fastapi import FastAPI

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

        @self.app.post("/session/{session_id}/start-episode")
        async def start_episode_endpoint(session_id: str, task_id: str) -> Dict[str, Any]:
            """Start an episode for a specific task."""
            episode = await self.session_manager.start_episode(session_id, task_id)
            result: Dict[str, Any] = {
                "episode_id": episode.episode_id,
                "task_id": task_id,
                "session_id": session_id,
                "message": "Episode started successfully",
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

    async def start_server(self) -> None:
        """Start the SessionRestAPI server."""
        logger.info(f"Starting SABER {self.session_manager.domain_name} domain REST server on {self.host}:{self.port}")

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
