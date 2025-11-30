"""Sandbox-specific registry for domain ownership and episode mappings.

This module provides thread-safe registry operations for:
- Domain ownership tracking (per-task ownership of domains)
- Episode mapping storage (sample_id -> episode metadata in Inspect AI store)
- Stale ownership cleanup (for debugging/recovery)

Complements domain_manager.py:
- domain_manager.py: Process-wide domain registry (factory pattern)
- sandbox_registry.py: Per-task ownership + episode mappings (sandbox pattern)
"""

import threading
from typing import Any, Dict, Optional

from inspect_ai.util import store

from saber.logging_config import LogCategory, get_saber_logger
from saber.models import CleanupReason

from ..constants import InspectStoreKeys

logger = get_saber_logger(LogCategory.AGENT, __name__)


class SandboxRegistry:
    """Thread-safe registry for sandbox domain ownership and episode mappings.

    This class manages:
    1. Domain ownership: Ensures only one task owns a domain at a time
    2. Episode mappings: Tracks sample_id -> episode metadata for scorers

    Thread Safety:
    - _lock: Protects _registry read-modify-write operations
    - _episode_mapping_lock: Protects episode_mapping in Inspect AI store
    """

    # Class-level registry: domain_slug -> metadata
    _registry: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()
    _episode_mapping_lock = threading.Lock()  # Protects episode_mapping read-modify-write

    @classmethod
    def register_domain(
        cls,
        domain_slug: str,
        task_name: str,
        controller: Any,
        context: Any,
        session_id: str,
        rest_port: int,
        mcp_port: int,
        rest_url: str,
        mcp_url: str,
        ownership: bool = True,
    ) -> None:
        """Register domain with sandbox ownership.

        Args:
            domain_slug: Domain identifier
            task_name: Task that owns this domain
            controller: DomainController instance
            context: DomainContext instance
            session_id: SABER session ID
            rest_port: REST API port
            mcp_port: MCP API port
            rest_url: Full REST API URL
            mcp_url: Full MCP API URL
            ownership: True if we started the domain (can stop it)
        """
        with cls._lock:
            cls._registry[domain_slug] = {
                "owner": task_name,
                "domain_slug": domain_slug,
                "controller": controller,
                "context": context,
                "ownership": ownership,
                "rest_port": rest_port,
                "mcp_port": mcp_port,
                "rest_url": rest_url,
                "mcp_url": mcp_url,
                "session_id": session_id,
            }
            logger.debug(
                f"Registered domain '{domain_slug}' in sandbox registry",
                extra={
                    "domain": domain_slug,
                    "task": task_name,
                    "ownership": ownership,
                },
            )

    @classmethod
    def unregister_domain(cls, domain_slug: str) -> Optional[Dict[str, Any]]:
        """Remove domain from sandbox registry and return entry.

        Args:
            domain_slug: Domain to unregister

        Returns:
            Registry entry dict or None if not found
        """
        with cls._lock:
            entry = cls._registry.pop(domain_slug, None)
            if entry:
                logger.debug(
                    f"Unregistered domain '{domain_slug}' from sandbox registry",
                    extra={"domain": domain_slug, "owner": entry.get("owner")},
                )
            return entry

    @classmethod
    def get_domain_entry(cls, domain_slug: str) -> Optional[dict]:
        """Get domain entry (thread-safe).

        Args:
            domain_slug: Domain to lookup

        Returns:
            Registry entry dict or None if not found
        """
        with cls._lock:
            return cls._registry.get(domain_slug)

    @classmethod
    def update_session_id(cls, domain_slug: str, session_id: str) -> None:
        """Update session ID for a registered domain.

        Args:
            domain_slug: Domain to update
            session_id: New session ID
        """
        with cls._lock:
            if domain_slug in cls._registry:
                cls._registry[domain_slug]["session_id"] = session_id
                logger.debug(
                    f"Updated session_id for domain '{domain_slug}'",
                    extra={"domain": domain_slug, "session_id": session_id},
                )

    @classmethod
    def clear_stale_ownership(cls, domain_slug: str, force: bool = False) -> bool:
        """Clear stale ownership for a domain (useful for debugging/recovery).

        This method allows manual cleanup of domain ownership in cases where:
        - A previous eval was interrupted and didn't clean up properly
        - eval-retry is failing due to ownership conflicts
        - Manual intervention is needed for development/debugging

        Args:
            domain_slug: The domain to clear ownership for
            force: If True, clear ownership even if entry looks valid

        Returns:
            True if ownership was cleared, False if domain wasn't in registry
        """
        with cls._lock:
            if domain_slug not in cls._registry:
                return False

            entry = cls._registry[domain_slug]
            logger.warning(
                f"Clearing ownership for domain '{domain_slug}' (owner: {entry.get('owner')})",
                extra={
                    "domain": domain_slug,
                    "owner": entry.get("owner"),
                    "force": force,
                    "reason": CleanupReason.MANUAL_CLEANUP,
                },
            )
            del cls._registry[domain_slug]
            return True

    @classmethod
    def store_episode_mapping(
        cls,
        sample_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
    ) -> None:
        """Store episode mapping in Inspect AI store with thread-safe locking.

        This centralizes all episode mapping storage to prevent race conditions
        from concurrent read-modify-write operations on the shared store.

        Args:
            sample_id: Unique sample identifier (includes attempt suffix)
            episode_id: SABER episode ID
            session_id: SABER session ID
            task_id: Task ID (for debugging)
        """
        task_store = store()
        with cls._episode_mapping_lock:
            episode_mapping = task_store.get(InspectStoreKeys.EPISODE_MAPPING, {})
            episode_mapping[sample_id] = type(
                "Episode",
                (),
                {
                    "episode_id": episode_id,
                    "session_id": session_id,
                    "attached_to_episode_id": None,
                    "task_id": task_id,
                    "sample_id": sample_id,
                },
            )()
            task_store.set(InspectStoreKeys.EPISODE_MAPPING, episode_mapping)

            logger.debug(
                f"Stored episode mapping for sample {sample_id}",
                extra={
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "total_mappings": len(episode_mapping),
                    "event": "episode_mapping_stored",
                },
            )

    @classmethod
    def remove_episode_mapping(cls, sample_id: str, task_id: str) -> None:
        """Remove episode mapping from Inspect AI store with thread-safe locking.

        This centralizes all episode mapping cleanup to prevent race conditions.

        Args:
            sample_id: Unique sample identifier to remove
            task_id: Task ID (for debugging/logging)
        """
        task_store = store()
        with cls._episode_mapping_lock:
            episode_mapping = task_store.get(InspectStoreKeys.EPISODE_MAPPING, {})
            if sample_id in episode_mapping:
                del episode_mapping[sample_id]
                task_store.set(InspectStoreKeys.EPISODE_MAPPING, episode_mapping)
                logger.debug(
                    f"Removed episode mapping for sample {sample_id}",
                    extra={
                        "sample_id": sample_id,
                        "task_id": task_id,
                        "remaining_mappings": len(episode_mapping),
                        "event": "episode_mapping_removed",
                    },
                )
