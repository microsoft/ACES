"""Type-safe data structures for SABER Inspect AI integration.

This module provides strongly-typed dataclasses to replace Dict[str, Any] patterns
throughout the inspect_ai integration, improving type safety and developer experience.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from saber.client.client_session import ClientSessionManager

# ============================================================================
# Handler State Types
# ============================================================================


@dataclass
class HandlerState:
    """Type-safe handler state for episode lifecycle management.

    Base class for all handler state objects, providing common fields and methods
    for episode tracking and semaphore management.
    """

    episode_ids: List[str]
    primary_episode_id: str
    semaphore_acquired: bool

    def has_multiple_episodes(self) -> bool:
        """Check if this state manages multiple episodes."""
        return len(self.episode_ids) > 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility.

        This allows gradual migration from Dict[str, Any] to typed classes.
        """
        return {
            "episode_ids": self.episode_ids,
            "primary_episode_id": self.primary_episode_id,
            "semaphore_acquired": self.semaphore_acquired,
        }


@dataclass
class OrchestratedHandlerState(HandlerState):
    """Extended state for orchestrated tasks with multiple episodes.

    Used by OrchestratedTaskHandler to track state for orchestrations where
    a single task creates multiple episodes that execute together.
    """

    episodes: List[Dict[str, Any]] = field(default_factory=list)

    def get_episode_by_role(self, role: str) -> Optional[Dict[str, Any]]:
        """Find episode info by role."""
        for ep in self.episodes:
            if ep.get("role") == role:
                return ep
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including episodes."""
        base = super().to_dict()
        base["episodes"] = self.episodes
        return base


@dataclass
class OrchestrationSubTaskState(HandlerState):
    """State for orchestrated sub-tasks in multi-sample coordination.

    Used in the new multi-sample orchestration approach where each sub-task
    (e.g., blue team, red team) is a separate sample coordinated via
    OrchestrationCoordinator.
    """

    orchestration_id: str
    sub_task_role: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including orchestration metadata."""
        base = super().to_dict()
        base["orchestration_id"] = self.orchestration_id
        base["sub_task_role"] = self.sub_task_role
        return base


# ============================================================================
# Registry Entry Types
# ============================================================================


@dataclass
class DomainRegistryEntry:
    """Type-safe domain registry entry for sandbox ownership tracking.

    Replaces Dict[str, Any] in SandboxRegistry with a properly typed structure
    that provides clear contracts and better IDE support.
    """

    domain_slug: str
    owner: str
    controller: Any  # DomainController - avoid circular import
    context: Any  # DomainContext - avoid circular import
    ownership: bool
    rest_port: int
    mcp_port: int
    rest_url: str
    mcp_url: str
    session_id: str

    def can_stop_domain(self) -> bool:
        """Check if this owner can stop the domain.

        Only domains started by the current owner (ownership=True) can be stopped.
        Domains started externally or by factory (ownership=False) are not stopped
        during cleanup.
        """
        return self.ownership

    def get_api_endpoints(self) -> tuple[str, str]:
        """Get REST and MCP endpoints as tuple.

        Returns:
            Tuple of (rest_url, mcp_url)
        """
        return (self.rest_url, self.mcp_url)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "domain_slug": self.domain_slug,
            "owner": self.owner,
            "controller": self.controller,
            "context": self.context,
            "ownership": self.ownership,
            "rest_port": self.rest_port,
            "mcp_port": self.mcp_port,
            "rest_url": self.rest_url,
            "mcp_url": self.mcp_url,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainRegistryEntry":
        """Create from dictionary (for migration from old format)."""
        return cls(
            domain_slug=data["domain_slug"],
            owner=data["owner"],
            controller=data["controller"],
            context=data["context"],
            ownership=data["ownership"],
            rest_port=data["rest_port"],
            mcp_port=data["mcp_port"],
            rest_url=data["rest_url"],
            mcp_url=data["mcp_url"],
            session_id=data["session_id"],
        )


# ============================================================================
# Episode Mapping Types
# ============================================================================


@dataclass
class EpisodeMapping:
    """Type-safe episode mapping for sample tracking.

    Replaces the anonymous type() objects used in sandbox_registry with a
    proper dataclass that provides type safety and clear contracts.
    """

    episode_id: str
    session_id: str
    task_id: str
    sample_id: str
    attached_to_episode_id: Optional[str] = None

    def is_attached(self) -> bool:
        """Check if this episode is attached to another episode."""
        return self.attached_to_episode_id is not None

    def get_primary_episode_id(self) -> str:
        """Get primary episode ID.

        Returns:
            attached_to_episode_id if attached, otherwise own episode_id
        """
        return self.attached_to_episode_id or self.episode_id


# ============================================================================
# Session Context Types
# ============================================================================


@dataclass
class SessionContext:
    """Type-safe session context (always valid after initialization).

    Replaces multiple Optional fields (session_manager, session_id) with a
    single non-optional context object. This eliminates repeated None checks
    and provides a clean interface for session operations.
    """

    session_manager: ClientSessionManager
    session_id: str
    domain_slug: str

    async def create_episode(self, task_id: str) -> str:
        """Create new episode in this session.

        Args:
            task_id: Task identifier

        Returns:
            Episode ID
        """
        response = await self.session_manager.create_episode(
            self.session_id,
            task_id,
        )
        episode_id: str = response.episode_id
        return episode_id

    async def end_episode(self, episode_id: str) -> None:
        """End episode in this session.

        Args:
            episode_id: Episode to end
        """
        await self.session_manager.end_episode(
            self.session_id,
            episode_id,
        )


__all__ = [
    "HandlerState",
    "OrchestratedHandlerState",
    "OrchestrationSubTaskState",
    "DomainRegistryEntry",
    "EpisodeMapping",
    "SessionContext",
]
