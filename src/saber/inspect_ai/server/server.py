"""Domain orchestrator adapter providing async-compatible API for SABER sandbox.

This module delegates to SABER's DomainOrchestrator to provide:
- All production-tested health checking and error handling
- Proper log output on failures
- Container cleanup on errors
- Consistent behavior with standalone SABER usage

Following DRY principles by calling the orchestrator directly (same code path as CLI).
Component 1 of the SABER sandbox integration (Phase 2).
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from inspect_ai._util.error import PrerequisiteError

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


@dataclass
class DomainContext:
    """Context information returned after starting a domain.

    Contains all metadata needed for sandbox operation and cleanup.
    """

    domain: str
    """Domain slug that was started."""

    rest_url: str
    """URL for SABER REST API (session/episode management)."""

    mcp_url: str
    """URL for SABER MCP API (tool execution)."""

    rest_port: int
    """REST API port number."""

    mcp_port: int
    """MCP API port number."""

    project_slug: str
    """Docker Compose project slug for cleanup."""

    domains_root: Path
    """Path to domains directory."""


def _create_orchestrator(domains_root: Path) -> Any:
    """Create orchestrator instance using SABER's standard pattern.

    This uses the same code path as the saber-domain CLI to ensure
    consistent behavior.
    """
    try:
        from saber.domain.orchestrator import DomainOrchestrator
        from saber.domain.resources import resolve_compose_file
    except ImportError as e:
        raise PrerequisiteError(
            "SABER is not installed or cannot be imported.\n\n"
            "To use the SABER sandbox, ensure the SABER submodule is initialized:\n"
            "  git submodule update --init --recursive external/saber\n\n"
            "Then install SABER in editable mode:\n"
            "  uv pip install -e external/saber\n\n"
            f"Import error: {e}"
        )

    try:
        # Use SABER's compose file resolution (same as CLI)
        with resolve_compose_file() as compose_path:
            return DomainOrchestrator(domains_root, compose_path)
    except Exception as e:
        raise PrerequisiteError(
            f"Failed to create SABER DomainOrchestrator for domains_root={domains_root}\n\n" f"Error: {e}"
        )


class DomainController:
    """Async-compatible controller for SABER domain operations.

    This class calls SABER's DomainOrchestrator directly (same code path as CLI)
    to ensure all production-tested behaviors are preserved:
    - Health checking with timeout
    - Error log output on failures
    - Automatic cleanup on startup errors
    - Consistent environment handling

    Following DRY principles by reusing the orchestrator logic.
    """

    def __init__(self, domains_root: Path):
        """Initialize domain controller.

        Args:
            domains_root: Path to SABER domains directory
        """
        self._domains_root = domains_root
        # Create orchestrator once and reuse (DRY principle)
        self._orchestrator = _create_orchestrator(domains_root)

    async def check_running_domain(
        self,
        rest_port: int = 8000,
        mcp_port: int = 8001,
    ) -> Optional[str]:
        """Check if a SABER domain is running on the specified ports.

        Uses the DomainOrchestrator to check domain status via docker compose.

        Args:
            rest_port: Port to check for REST API
            mcp_port: Port to check for MCP API

        Returns:
            Domain name if running, None otherwise
        """
        # Check all possible domains in the domains directory
        try:
            for domain_path in self._domains_root.iterdir():
                if not domain_path.is_dir() or domain_path.name.startswith("."):
                    continue

                domain_slug = domain_path.name

                # Run status check in thread executor (blocking call)
                status = await asyncio.to_thread(self._orchestrator.get_domain_status, domain_slug)

                # Check if domain is running
                if status.get("running", False):
                    return domain_slug

        except Exception as e:
            logger.debug(f"Error checking running domains: {e}")

        return None

    async def start(
        self,
        domain: str,
        rest_port: int = 8000,
        mcp_port: int = 8001,
        log_level: str = "INFO",
        build: Optional[str] = None,
        rebuild: Optional[str] = None,
    ) -> DomainContext:
        """Start a SABER domain asynchronously.

        Calls DomainOrchestrator.start_domain() in a thread executor (blocking I/O)
        which handles:
        - Domain validation
        - Image building (if needed)
        - Service startup via docker compose
        - Health checking with timeout (_wait_for_services)
        - Error log output on failure

        Args:
            domain: Domain slug to start (e.g., "excytin_demo")
            rest_port: Port for REST API (default: 8000)
            mcp_port: Port for MCP API (default: 8001)
            log_level: Logging level for domain services (default: "INFO")
            build: Optional image filter for building images (e.g., "server")
            rebuild: Optional image filter for rebuilding images

        Returns:
            DomainContext with URLs, ports, and metadata for the started domain.

        Raises:
            PrerequisiteError: If domain validation or startup fails.
        """
        logger.info(
            "Starting SABER domain via orchestrator",
            extra={
                "event": "start_domain",
                "domain": domain,
                "rest_port": rest_port,
                "mcp_port": mcp_port,
                "domains_root": str(self._domains_root),
            },
        )

        try:
            # Run blocking orchestrator operation in thread executor
            # This includes all the CLI logic: validation, building, starting, health checks
            await asyncio.to_thread(
                self._orchestrator.start_domain,
                domain=domain,
                rest_port=rest_port,
                mcp_port=mcp_port,
                log_level=log_level,
                build=build,
                rebuild=rebuild,
                dry_run=False,
            )

        except Exception as e:
            logger.error(
                "Failed to start SABER domain",
                extra={
                    "event": "start_domain_failed",
                    "domain": domain,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise PrerequisiteError(
                f"Failed to start SABER domain '{domain}'.\n\n"
                f"Error: {e}\n\n"
                f"Ensure the domain is properly configured:\n"
                f"  uv run saber-domain validate {domain} --domains-root {self._domains_root}"
            )

        # Construct URLs and context
        rest_url = f"http://localhost:{rest_port}"
        mcp_url = f"http://localhost:{mcp_port}"
        project_slug = domain  # Docker compose project name

        context = DomainContext(
            domain=domain,
            rest_url=rest_url,
            mcp_url=mcp_url,
            rest_port=rest_port,
            mcp_port=mcp_port,
            project_slug=project_slug,
            domains_root=self._domains_root,
        )

        logger.info(
            "SABER domain started successfully",
            extra={
                "event": "start_domain_success",
                "domain": domain,
                "rest_url": rest_url,
                "mcp_url": mcp_url,
                "project_slug": project_slug,
            },
        )

        return context

    async def stop(self, domain: str) -> None:
        """Stop a SABER domain asynchronously.

        Calls DomainOrchestrator.stop_domain() which handles proper compose teardown.

        Args:
            domain: Domain slug to stop (e.g., "excytin_demo")

        Raises:
            PrerequisiteError: If domain stopping fails.
        """
        logger.info(
            "Stopping SABER domain via orchestrator",
            extra={
                "event": "stop_domain",
                "domain": domain,
                "domains_root": str(self._domains_root),
            },
        )

        try:
            # Run blocking orchestrator operation in thread executor
            await asyncio.to_thread(
                self._orchestrator.stop_domain,
                domain=domain,
                dry_run=False,
            )

            logger.info(
                "SABER domain stopped successfully",
                extra={
                    "event": "stop_domain_success",
                    "domain": domain,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to stop SABER domain",
                extra={
                    "event": "stop_domain_failed",
                    "domain": domain,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise PrerequisiteError(f"Failed to stop SABER domain '{domain}'.\n\n" f"Error: {e}")


async def start_domain(
    domain: str,
    domains_root: Path,
    rest_port: int = 8000,
    mcp_port: int = 8001,
    log_level: str = "INFO",
    build: Optional[str] = None,
    rebuild: Optional[str] = None,
) -> DomainContext:
    """Convenience function to start a SABER domain.

    This is a simplified interface for starting domains without directly
    instantiating a DomainController.

    Args:
        domain: Domain slug to start
        domains_root: Path to SABER domains directory
        rest_port: Port for REST API (default: 8000)
        mcp_port: Port for MCP API (default: 8001)
        log_level: Logging level (default: "INFO")
        build: Optional image filter for building
        rebuild: Optional image filter for rebuilding

    Returns:
        DomainContext with connection details and metadata
    """
    controller = DomainController(domains_root)
    return await controller.start(
        domain=domain,
        rest_port=rest_port,
        mcp_port=mcp_port,
        log_level=log_level,
        build=build,
        rebuild=rebuild,
    )


async def stop_domain(
    domain: str,
    domains_root: Path,
) -> None:
    """Convenience function to stop a SABER domain.

    Args:
        domain: Domain slug to stop
        domains_root: Path to SABER domains directory
    """
    controller = DomainController(domains_root)
    await controller.stop(domain)
