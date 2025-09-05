"""
Session Manager for MCP Service

Manages session registration and routing for containerized agents.
Maps agent container sessions to SABER server sessions for proper request routing.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    """Represents an active agent container session."""

    agent_id: str
    saber_session_id: str
    saber_episode_id: str  # REQUIRED: Episode-first architecture requires saber_episode_id
    task_id: Optional[str] = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)


class AgentSessionRegistry:
    """
    Registry for agent container sessions and routing to SABER sessions.

    This class is responsible for:
    - Registering agent containers with their SABER session mappings
    - Routing MCP requests based on agent identity
    - Managing session lifecycle and cleanup
    - Providing session context for MCP proxy operations
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def register_session(
        self,
        agent_id: str,
        saber_session_id: str,
        saber_episode_id: str,  # REQUIRED: Episode-first architecture requires saber_episode_id
        task_id: Optional[str] = None,
    ) -> AgentSession:
        """
        Register a new agent container session.

        Args:
            agent_id: Unique identifier for the agent container
            saber_session_id: SABER server session ID to route requests to
            saber_episode_id: Episode ID for episode-first execution (REQUIRED)
            task_id: Optional task ID for context

        Returns:
            AgentSession: The registered session object

        Raises:
            ValueError: If agent_id is already registered with different session or saber_episode_id is missing
        """
        if not saber_episode_id:
            raise ValueError("saber_episode_id is required for episode-first architecture")

        async with self._lock:
            if agent_id in self._sessions:
                existing = self._sessions[agent_id]
                if existing.saber_session_id != saber_session_id:
                    raise ValueError(
                        f"Agent {agent_id} already registered with different session "
                        f"{existing.saber_session_id}, cannot register with {saber_session_id}"
                    )
                # Update existing session with new episode info
                existing.saber_episode_id = saber_episode_id
                existing.task_id = task_id
                existing.update_activity()
                existing.is_active = True
                logger.info(f"Updated existing session for agent {agent_id} with episode {saber_episode_id}")
                return existing

            # Create new session
            session = AgentSession(
                agent_id=agent_id, saber_session_id=saber_session_id, saber_episode_id=saber_episode_id, task_id=task_id
            )
            self._sessions[agent_id] = session

            logger.info(
                f"Registered new session: agent={agent_id}, "
                f"saber_session={saber_session_id}, episode={saber_episode_id}, task={task_id}"
            )
            return session

    async def unregister_session(self, agent_id: str) -> bool:
        """
        Unregister an agent container session.

        Args:
            agent_id: Agent container ID to unregister

        Returns:
            bool: True if session was found and removed, False otherwise
        """
        async with self._lock:
            if agent_id in self._sessions:
                session = self._sessions.pop(agent_id)
                logger.info(f"Unregistered session: agent={agent_id}, " f"saber_session={session.saber_session_id}")
                return True
            return False

    async def get_session(self, agent_id: str) -> Optional[AgentSession]:
        """
        Get session information for an agent.

        Args:
            agent_id: Agent container ID

        Returns:
            AgentSession if found, None otherwise
        """
        async with self._lock:
            session = self._sessions.get(agent_id)
            if session:
                session.update_activity()
            return session

    async def get_saber_session_id(self, agent_id: str) -> Optional[str]:
        """
        Get SABER session ID for routing requests.

        Args:
            agent_id: Agent container ID

        Returns:
            SABER session ID if agent is registered, None otherwise
        """
        session = await self.get_session(agent_id)
        return session.saber_session_id if session else None

    async def get_agent_id_by_saber_session(self, saber_session_id: str) -> Optional[str]:
        """
        Find agent_id for a given SABER session id.

        Performs a linear scan over active sessions (sufficient for small N, revisit if needed).

        Args:
            saber_session_id: SABER server session ID

        Returns:
            agent_id if found, otherwise None
        """
        async with self._lock:
            for agent_id, session in self._sessions.items():
                if session.saber_session_id == saber_session_id and session.is_active:
                    session.update_activity()
                    return agent_id
        return None

    async def list_active_sessions(self) -> Dict[str, AgentSession]:
        """
        Get all active sessions.

        Returns:
            Dictionary mapping agent IDs to their sessions
        """
        async with self._lock:
            return {agent_id: session for agent_id, session in self._sessions.items() if session.is_active}

    async def deactivate_session(self, agent_id: str) -> bool:
        """
        Mark a session as inactive without removing it.

        Args:
            agent_id: Agent container ID

        Returns:
            bool: True if session was found and deactivated, False otherwise
        """
        async with self._lock:
            if agent_id in self._sessions:
                self._sessions[agent_id].is_active = False
                logger.info(f"Deactivated session for agent {agent_id}")
                return True
            return False

    async def cleanup_inactive_sessions(self, max_age_minutes: int = 60) -> int:
        """
        Remove inactive sessions older than specified age.

        Args:
            max_age_minutes: Maximum age for inactive sessions

        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_minutes * 60)

        async with self._lock:
            to_remove = []
            for agent_id, session in self._sessions.items():
                if not session.is_active and session.last_activity.timestamp() < cutoff_time:
                    to_remove.append(agent_id)

            for agent_id in to_remove:
                del self._sessions[agent_id]
                logger.info(f"Cleaned up inactive session for agent {agent_id}")

            return len(to_remove)

    async def get_session_stats(self) -> Dict[str, int]:
        """
        Get statistics about current sessions.

        Returns:
            Dictionary with session statistics
        """
        async with self._lock:
            total = len(self._sessions)
            active = sum(1 for s in self._sessions.values() if s.is_active)
            inactive = total - active

            return {"total_sessions": total, "active_sessions": active, "inactive_sessions": inactive}
