"""Domain lifecycle management for SABER task execution.

This module manages the process-wide registry of active SABER domains,
handling domain startup, reuse detection, and cleanup.

Key features:
- Process-wide domain registry with thread-safe access
- Domain ownership tracking (who started it)
- Reuse detection (same process vs external)
- Cleanup coordination

Thread Safety:
- All registry operations are protected by _active_domains_lock
- Safe for concurrent access from multiple task factories
"""

import threading
from typing import Any

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Process-wide registry tracking active domains
# Key: domain_slug
# Value: Dict with controller, context, ownership, rest_url, mcp_url, etc.
_active_domains: dict[str, dict[str, Any]] = {}
_active_domains_lock = threading.Lock()


def get_active_domain(domain_slug: str) -> dict[str, Any] | None:
    """Get active domain registry entry if exists.

    Used by SABERSandboxEnvironment to detect if the factory already
    started the server, enabling ownership transfer.

    Args:
        domain_slug: Domain to lookup

    Returns:
        Registry entry dict or None if not active
    """
    with _active_domains_lock:
        return _active_domains.get(domain_slug)


def register_domain(
    domain_slug: str,
    controller: Any,
    context: Any,
    ownership: bool,
    rest_port: int,
    mcp_port: int,
) -> None:
    """Register a domain in the active domains registry.

    Args:
        domain_slug: Domain identifier
        controller: DomainController instance
        context: DomainContext instance
        ownership: True if we started it (can stop it), False if external
        rest_port: REST API port
        mcp_port: MCP API port
    """
    with _active_domains_lock:
        _active_domains[domain_slug] = {
            "owner": f"domain_task_{domain_slug}",
            "domain_slug": domain_slug,
            "controller": controller,
            "context": context,
            "ownership": ownership,
            "rest_port": rest_port,
            "mcp_port": mcp_port,
            "rest_url": context.rest_url,
            "mcp_url": context.mcp_url,
        }
    logger.debug(
        f"Registered domain '{domain_slug}' in active registry",
        extra={
            "domain": domain_slug,
            "ownership": ownership,
            "rest_port": rest_port,
            "mcp_port": mcp_port,
        },
    )


def remove_active_domain(domain_slug: str) -> None:
    """Remove domain from active registry.

    Called by SABERSandboxEnvironment.task_cleanup() to clean up
    after evaluation completes.

    Args:
        domain_slug: Domain to remove
    """
    with _active_domains_lock:
        if domain_slug in _active_domains:
            del _active_domains[domain_slug]
            logger.debug(f"Removed '{domain_slug}' from active domains registry")


def check_domain_ownership(domain_slug: str) -> bool:
    """Check if we own the domain (started it ourselves).

    Args:
        domain_slug: Domain to check

    Returns:
        True if we started the domain, False otherwise
    """
    with _active_domains_lock:
        domain_entry = _active_domains.get(domain_slug)
        if domain_entry:
            return bool(domain_entry.get("ownership", False))
        return False


def unregister_domain_on_failure(domain_slug: str) -> bool:
    """Remove domain from registry and return ownership status.

    Used during cleanup after startup failures.

    Args:
        domain_slug: Domain to unregister

    Returns:
        True if we owned the domain (should clean up), False otherwise
    """
    with _active_domains_lock:
        if domain_slug in _active_domains:
            ownership = bool(_active_domains[domain_slug].get("ownership", False))
            del _active_domains[domain_slug]
            logger.debug(f"Unregistered '{domain_slug}' from active domains after failure")
            return ownership
        return False
