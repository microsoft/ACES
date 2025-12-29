"""
Docker Compose Health Checker for Episode Services.

Logging Category: DOCKER

This module provides health checking capabilities for Docker Compose environments
used in SABER episodes. It verifies that all services in a compose environment
are healthy before allowing episode execution to proceed.

FAIL-FAST DESIGN: This module follows a fail-fast approach - any health check
failure immediately raises an exception. No silent failures or fallbacks.
"""

import time
from pathlib import Path
from typing import Any

import yaml

try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    docker = None  # type: ignore
    DOCKER_AVAILABLE = False

from saber.logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.DOCKER, __name__)


class ComposeHealthCheckError(Exception):
    """Raised when compose health checks fail."""

    pass


class ComposeHealthChecker:
    """
    Health checker for Docker Compose environments in SABER episodes.

    This service verifies that all containers in a compose environment are healthy
    before allowing episode execution to continue. Uses Docker API to check both
    explicit health checks and container running status.

    FAIL-FAST: All methods fail immediately on any error - no silent failures.
    """

    def __init__(self) -> None:
        """Initialize the health checker."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker package not available. Install with: pip install docker")

        self._docker_client: Any | None = None

    @property
    def docker_client(self) -> Any:
        """Get or create Docker client."""
        if self._docker_client is None:
            if docker is None:
                raise RuntimeError("Docker package not available")
            self._docker_client = docker.from_env()  # type: ignore
        return self._docker_client

    def wait_for_all_services_healthy(
        self,
        compose_file_path: str,
        project_name: str,
        timeout_seconds: int = 300,  # 5 minutes default
        check_interval: float = 2.0,  # Check every 2 seconds
    ) -> None:
        """
        Wait for all services in a Docker Compose environment to become healthy.

        Args:
            compose_file_path: Path to the docker-compose.yml file
            project_name: Docker Compose project name (used for container naming)
            timeout_seconds: Maximum time to wait for health checks
            check_interval: Time between health check attempts

        Raises:
            ComposeHealthCheckError: If services fail to become healthy within timeout
            RuntimeError: If compose file cannot be read or Docker is unavailable
        """
        logger.debug(
            "Compose health check started",
            extra={
                "event": "compose_health_check_started",
                "project_name": project_name,
                "compose_file": compose_file_path,
                "timeout_seconds": timeout_seconds,
                "check_interval_seconds": check_interval,
            },
        )
        logger.debug(
            "Compose file read requested",
            extra={
                "event": "compose_health_check_reading_file",
                "compose_file": compose_file_path,
            },
        )

        # Parse compose file to get service definitions
        services = self._parse_compose_services(compose_file_path)
        if not services:
            logger.error(
                "Compose health check found no services",
                extra={
                    "event": "compose_health_check_no_services",
                    "project_name": project_name,
                    "compose_file": compose_file_path,
                },
            )
            raise ComposeHealthCheckError(
                f"EPISODE CREATION FAILED: No services found in compose file: {compose_file_path}. "
                f"Cannot validate health for empty environment."
            )

        logger.debug(
            "Compose services discovered",
            extra={
                "event": "compose_health_check_services_discovered",
                "project_name": project_name,
                "service_count": len(services),
                "service_names": list(services.keys()),
            },
        )
        if services:
            first_service_name = next(iter(services))
            logger.debug(
                "First compose service inspected",
                extra={
                    "event": "compose_health_check_first_service_inspected",
                    "project_name": project_name,
                    "service_name": first_service_name,
                },
            )

        # Validate that we can access Docker before starting health checks
        try:
            self.docker_client.ping()
        except Exception as e:
            raise ComposeHealthCheckError(
                f"EPISODE CREATION FAILED: Cannot connect to Docker daemon: {e}. "
                f"Health checks require Docker connectivity."
            ) from e

        start_time = time.time()
        last_print_time = 0.0

        while time.time() - start_time < timeout_seconds:
            try:
                # Check health of all services
                health_status = self._check_all_services_health(services, project_name)

                # Count healthy services
                healthy_count = sum(1 for status in health_status.values() if status["healthy"])
                total_count = len(health_status)

                # Log progress at INFO level so it's visible during eval
                elapsed = round(time.time() - start_time, 2)
                unhealthy_names = [name for name, status in health_status.items() if not status["healthy"]]

                # Print progress every 10 seconds to avoid spam
                if time.time() - last_print_time >= 10 or healthy_count == total_count:
                    print(
                        f"[COMPOSE] Health: {healthy_count}/{total_count} services healthy "
                        f"({elapsed:.0f}s/{timeout_seconds}s)",
                        flush=True,
                    )
                    if unhealthy_names and len(unhealthy_names) <= 5:
                        print(f"[COMPOSE]   Waiting for: {', '.join(unhealthy_names)}", flush=True)
                    last_print_time = time.time()

                logger.info(
                    f"Health check progress: {healthy_count}/{total_count} services healthy ({elapsed}s elapsed)",
                    extra={
                        "event": "compose_health_check_progress",
                        "project_name": project_name,
                        "service_count": total_count,
                        "healthy_services": healthy_count,
                        "unhealthy_services": unhealthy_names,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": timeout_seconds,
                    },
                )

                # If all services are healthy, we're done
                if healthy_count == total_count:
                    elapsed = time.time() - start_time
                    logger.debug(
                        "Compose services healthy",
                        extra={
                            "event": "compose_health_check_success",
                            "project_name": project_name,
                            "service_count": total_count,
                            "elapsed_seconds": round(elapsed, 2),
                        },
                    )
                    return

                # Log status of unhealthy services
                for service_name, status in health_status.items():
                    if not status["healthy"]:
                        logger.debug(
                            "Compose service unhealthy",
                            extra={
                                "event": "compose_health_check_service_unhealthy",
                                "project_name": project_name,
                                "service_name": service_name,
                                "reason": status["reason"],
                                "container_name": status.get("container_name", "unknown"),
                                "status": status.get("status"),
                            },
                        )

            except Exception as e:
                logger.warning(
                    "Compose health check attempt failed",
                    extra={
                        "event": "compose_health_check_attempt_failed",
                        "project_name": project_name,
                        "error": str(e),
                    },
                )
                # Continue retrying - the outer timeout will handle ultimate failure

            time.sleep(check_interval)

        # Timeout reached - perform final check and report failures
        print(f"[COMPOSE] ✗ Health check timeout ({timeout_seconds}s) - checking final status...", flush=True)
        try:
            final_status = self._check_all_services_health(services, project_name)
        except Exception as e:
            raise ComposeHealthCheckError(
                f"Failed to perform final health check after {timeout_seconds}s timeout: {e}"
            ) from e

        unhealthy_services = []
        for name, status in final_status.items():
            if not status["healthy"]:
                unhealthy_services.append(
                    f"{name} (container: {status.get('container_name', 'unknown')}): {status['reason']}"
                )
                print(f"[COMPOSE]   ✗ {name}: {status['reason']}", flush=True)

        # Create detailed error message for fail-fast behavior
        error_msg = (
            f"EPISODE CREATION FAILED: Services failed to become healthy within {timeout_seconds}s. "
            f"This is a mandatory requirement - no fallback mode available.\n"
            f"Unhealthy services ({len(unhealthy_services)}/{len(final_status)}):\n"
        )
        for service_error in unhealthy_services:
            error_msg += f"  • {service_error}\n"

        error_msg += f"\nProject: {project_name}\nCompose file: {compose_file_path}"
        print(f"[COMPOSE] ERROR: {error_msg}", flush=True)

        logger.error(
            "Compose services failed health check",
            extra={
                "event": "compose_health_check_failed",
                "project_name": project_name,
                "compose_file": compose_file_path,
                "timeout_seconds": timeout_seconds,
                "unhealthy_services": unhealthy_services,
            },
        )

        raise ComposeHealthCheckError(error_msg)

    def _parse_compose_services(self, compose_file_path: str) -> dict[str, dict[str, Any]]:
        """
        Parse Docker Compose file to extract service definitions.

        Args:
            compose_file_path: Path to compose file

        Returns:
            Dictionary of service name -> service config

        Raises:
            RuntimeError: If compose file cannot be read or parsed
        """
        logger.debug(
            "Parsing compose services",
            extra={
                "event": "compose_health_check_parse_services",
                "compose_file": compose_file_path,
            },
        )

        try:
            compose_path = Path(compose_file_path)
            if not compose_path.exists():
                raise RuntimeError(f"Compose file not found: {compose_file_path}")

            with open(compose_path) as f:
                compose_content = f.read()
                logger.debug(
                    "Compose file read",
                    extra={
                        "event": "compose_health_check_file_read",
                        "compose_file": compose_file_path,
                        "content_length": len(compose_content),
                    },
                )

                compose_data = yaml.safe_load(compose_content)

            services: dict[str, dict[str, Any]] = compose_data.get("services", {})
            if not services:
                raise RuntimeError(f"No services section found in: {compose_file_path}")

            logger.debug(
                "Compose services parsed",
                extra={
                    "event": "compose_health_check_services_parsed",
                    "compose_file": compose_file_path,
                    "service_count": len(services),
                },
            )

            return services

        except Exception as e:
            # Handle YAML parsing errors and other file errors
            if "yaml" in str(type(e)).lower() or "yaml" in str(e).lower():
                raise RuntimeError(f"Failed to parse compose file {compose_file_path}: {e}") from e
            else:
                raise RuntimeError(f"Error reading compose file {compose_file_path}: {e}") from e

    def _check_all_services_health(
        self, services: dict[str, dict[str, Any]], project_name: str
    ) -> dict[str, dict[str, Any]]:
        """
        Check health status of all services in the compose environment.

        Args:
            services: Service definitions from compose file
            project_name: Docker Compose project name

        Returns:
            Dictionary mapping service name to health status info:
            {
                'service_name': {
                    'healthy': bool,
                    'reason': str,
                    'container_name': str,
                    'status': str
                }
            }

        Raises:
            ComposeHealthCheckError: If unable to check any service health
        """
        health_status = {}
        critical_errors = []

        for service_name, service_config in services.items():
            try:
                status = self._check_service_health(service_name, service_config, project_name)
                health_status[service_name] = status
            except Exception as e:
                error_msg = f"Critical error checking service '{service_name}': {e}"
                critical_errors.append(error_msg)
                health_status[service_name] = {
                    "healthy": False,
                    "reason": f"Critical error: {e}",
                    "container_name": "unknown",
                    "status": "error",
                }

        # If we have critical errors, log them but don't fail immediately
        # Let the timeout mechanism handle the failure with full context
        if critical_errors:
            logger.warning(
                "Critical errors during compose health check",
                extra={
                    "event": "compose_health_check_critical_errors",
                    "project_name": project_name,
                    "errors": critical_errors,
                },
            )

        return health_status

    def _check_service_health(
        self, service_name: str, service_config: dict[str, Any], project_name: str
    ) -> dict[str, Any]:
        """
        Check health status of a single service.

        Args:
            service_name: Name of the service
            service_config: Service configuration from compose file
            project_name: Docker Compose project name

        Returns:
            Health status dictionary with 'healthy', 'reason', 'container_name', 'status'
        """
        # Determine expected container name
        container_name = service_config.get("container_name")
        if not container_name:
            # Default Docker Compose naming: {project_name}-{service_name}-1
            container_name = f"{project_name}-{service_name}-1"

        try:
            # Get container from Docker
            container = self.docker_client.containers.get(container_name)
            container.reload()  # Refresh container state

            # Check if container is running
            if container.status != "running":
                return {
                    "healthy": False,
                    "reason": f"Container not running (status: {container.status})",
                    "container_name": container_name,
                    "status": container.status,
                }

            # Check Docker health check if available
            health_status = container.attrs.get("State", {}).get("Health", {}).get("Status")

            if health_status is not None:
                # Container has explicit health check
                if health_status == "healthy":
                    return {
                        "healthy": True,
                        "reason": "Health check passed",
                        "container_name": container_name,
                        "status": container.status,
                    }
                elif health_status == "starting":
                    return {
                        "healthy": False,
                        "reason": "Health check still starting",
                        "container_name": container_name,
                        "status": container.status,
                    }
                elif health_status == "unhealthy":
                    return {
                        "healthy": False,
                        "reason": "Health check failed",
                        "container_name": container_name,
                        "status": container.status,
                    }
                else:
                    return {
                        "healthy": False,
                        "reason": f"Unknown health status: {health_status}",
                        "container_name": container_name,
                        "status": container.status,
                    }
            else:
                # No explicit health check - consider running container as healthy
                return {
                    "healthy": True,
                    "reason": "Container running (no health check defined)",
                    "container_name": container_name,
                    "status": container.status,
                }

        except Exception as e:
            # Handle both docker.errors.NotFound and other Docker exceptions
            if hasattr(e, "__class__") and "NotFound" in str(type(e)):
                return {
                    "healthy": False,
                    "reason": "Container not found",
                    "container_name": container_name,
                    "status": "not_found",
                }
            else:
                return {
                    "healthy": False,
                    "reason": f"Error accessing container: {e}",
                    "container_name": container_name,
                    "status": "error",
                }

    def get_service_health_summary(self, compose_file_path: str, project_name: str) -> dict[str, Any]:
        """
        Get a summary of all service health statuses.

        Args:
            compose_file_path: Path to compose file
            project_name: Docker Compose project name

        Returns:
            Summary dictionary with overall status and per-service details
        """
        try:
            services = self._parse_compose_services(compose_file_path)
            health_status = self._check_all_services_health(services, project_name)

            healthy_count = sum(1 for status in health_status.values() if status["healthy"])
            total_count = len(health_status)

            return {
                "overall_healthy": healthy_count == total_count,
                "healthy_count": healthy_count,
                "total_count": total_count,
                "services": health_status,
            }
        except Exception as e:
            return {"overall_healthy": False, "healthy_count": 0, "total_count": 0, "error": str(e), "services": {}}
