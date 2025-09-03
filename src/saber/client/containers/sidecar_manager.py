"""
Shared MCP Sidecar Manager

Manages the shared MCP sidecar container that acts as a proxy between
agent containers and the SABER MCP server. Handles lifecycle, health monitoring,
and session registration coordination.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Set

import aiohttp
import docker.types
from docker.models.containers import Container

import docker
from docker import errors as docker_errors

logger = logging.getLogger(__name__)


@dataclass
class SidecarConfig:
    """Configuration for the shared MCP sidecar."""

    image: str = "saber/mcp-service:latest"
    container_name: str = "saber-mcp-sidecar"
    network_name: str = "saber-network"
    port: int = 8002
    saber_server_url: str = "http://host.docker.internal:8000"
    saber_mcp_url: str = "http://host.docker.internal:8001"
    harness_url: str = "http://host.docker.internal:8765"
    health_check_timeout: int = 30
    health_check_interval: float = 0.5
    memory_limit: str = "256m"
    cpu_limit: float = 0.5


class SidecarManager:
    """
    Manages the shared MCP sidecar container.

    Responsibilities:
    - Start/stop shared sidecar container
    - Health monitoring and readiness checks
    - Session registration coordination
    - Network management
    - Resource cleanup
    """

    def __init__(self, config: Optional[SidecarConfig] = None):
        """Initialize sidecar manager with configuration."""
        self.config = config or SidecarConfig()
        self.docker_client = docker.DockerClient.from_env()  # type: ignore
        self.container: Optional[Container] = None
        self.is_running = False
        self._startup_lock = asyncio.Lock()
        # Track registered agent IDs for cleanup; use plain set for 3.10 compatibility
        self._registered_agent_ids: Set[str] = set()

        logger.info(f"🔧 Sidecar manager initialized for container: {self.config.container_name}")

    async def start_sidecar(self) -> bool:
        """
        Start the shared MCP sidecar container.

        Returns:
            bool: True if sidecar started successfully, False otherwise
        """
        async with self._startup_lock:
            if self.is_running:
                logger.info("📍 Sidecar already running")
                return True

            try:
                logger.info("🚀 Starting shared MCP sidecar...")

                # Ensure network exists
                await self._ensure_network()

                # Stop any existing container
                await self._cleanup_existing_container()

                # Start new container
                container = await self._create_and_start_container()

                # Wait for readiness
                if await self._wait_for_readiness():
                    self.container = container
                    self.is_running = True
                    logger.info(f"✅ Sidecar started successfully on port {self.config.port}")
                    return True
                else:
                    logger.error("❌ Sidecar failed readiness check")
                    await self._cleanup_container(container)
                    return False

            except Exception as e:
                logger.error(f"❌ Failed to start sidecar: {e}")
                return False

    async def stop_sidecar(self) -> None:
        """Stop the shared MCP sidecar container."""
        try:
            logger.info("🛑 Stopping shared MCP sidecar...")

            if self.container:
                await self._cleanup_container(self.container)
                self.container = None

            self.is_running = False
            logger.info("✅ Sidecar stopped successfully")

        except Exception as e:
            logger.error(f"❌ Failed to stop sidecar: {e}")

    async def register_agent_session(
        self, session_id: str, task_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Register an agent session with the sidecar.

        Args:
            session_id: SABER server session ID
            task_id: Optional task ID for context

        Returns:
            Agent ID if registration successful, None otherwise
        """
        if not self.is_running:
            logger.error("❌ Cannot register session - sidecar not running")
            return None

        try:
            # Use container name for network communication instead of localhost
            url = f"http://{self.config.container_name}:{self.config.port}/admin/sessions"
            agent_identifier = agent_id or f"agent-{session_id}"
            payload = {
                "agent_id": agent_identifier,
                "saber_session_id": session_id,
                "task_id": task_id,
            }

            async with aiohttp.ClientSession() as client:
                async with client.post(url, json=payload, timeout=self.config.health_check_timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        agent_id = result.get("agent_id")
                        logger.info(f"✅ Registered agent session: {agent_identifier}")
                        self._registered_agent_ids.add(agent_identifier)
                        return agent_identifier
                    else:
                        logger.error(f"❌ Failed to register session: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"❌ Session registration failed: {e}")
            return None

    async def unregister_agent_session(self, agent_id: str) -> bool:
        """
        Unregister an agent session from the sidecar.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            bool: True if unregistration successful, False otherwise
        """
        if not self.is_running:
            logger.warning("⚠️ Cannot unregister session - sidecar not running")
            return False

        try:
            # Use container name for network communication instead of localhost
            url = f"http://{self.config.container_name}:{self.config.port}/admin/sessions/{agent_id}"
            async with aiohttp.ClientSession() as client:
                async with client.delete(url) as response:
                    if response.status == 200:
                        logger.info(f"✅ Unregistered agent session: {agent_id}")
                        self._registered_agent_ids.discard(agent_id)
                        return True
                    else:
                        logger.warning(f"⚠️ Failed to unregister session: {response.status}")
                        return False

        except Exception as e:
            logger.warning(f"⚠️ Session unregistration failed: {e}")
            return False

    async def health_check(self) -> bool:
        """
        Check if the sidecar is healthy and responsive.

        Returns:
            bool: True if healthy, False otherwise
        """
        if not self.is_running or not self.container:
            return False

        try:
            # Check container status
            self.container.reload()
            if self.container.status != "running":
                return False

            # Check HTTP readiness endpoint (does not require server health)
            async with aiohttp.ClientSession() as client:
                # Use container name for network communication instead of localhost
                sidecar_url = f"http://{self.config.container_name}:{self.config.port}/ready"
                async with client.get(sidecar_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return bool(response.status == 200)

        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def get_sidecar_url(self) -> str:
        """Get the sidecar URL for agent containers."""
        # For containers on the same user-defined bridge network, use container DNS name
        # Publish host port primarily for local debugging.
        return f"http://{self.config.container_name}:{self.config.port}"

    async def _ensure_network(self) -> None:
        """Ensure the Docker network exists for container communication."""
        try:
            self.docker_client.networks.get(self.config.network_name)
            logger.debug(f"📡 Network {self.config.network_name} already exists")
        except docker_errors.NotFound:
            logger.info(f"📡 Creating network: {self.config.network_name}")
            self.docker_client.networks.create(self.config.network_name, driver="bridge", check_duplicate=True)

    async def _cleanup_existing_container(self) -> None:
        """Clean up any existing sidecar container."""
        try:
            existing = self.docker_client.containers.get(self.config.container_name)
            logger.info(f"🧹 Cleaning up existing container: {self.config.container_name}")
            existing.stop(timeout=10)
            existing.remove()
        except docker.errors.NotFound:
            pass  # No existing container
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup existing container: {e}")

    async def _create_and_start_container(self) -> docker.models.containers.Container:
        """Create and start the sidecar container."""
        logger.info(f"📦 Creating sidecar container: {self.config.image}")

        # Environment variables for sidecar configuration
        environment = {
            "SABER_SERVER_URL": self.config.saber_server_url,
            "SABER_MCP_URL": self.config.saber_mcp_url,
            "HARNESS_URL": self.config.harness_url,
            "PORT": str(self.config.port),
            "LOG_LEVEL": "INFO",
        }

        # Resource limits
        mem_limit = self.config.memory_limit
        cpu_quota = int(self.config.cpu_limit * 100000)  # Convert to microseconds

        # Host port mapping (allow dynamic port if configured port is in use)
        ports_map = {f"{self.config.port}/tcp": self.config.port}

        container = self.docker_client.containers.run(
            image=self.config.image,
            name=self.config.container_name,
            ports=ports_map,
            environment=environment,
            network=self.config.network_name,
            detach=True,
            remove=False,  # Keep for debugging and log collection
            mem_limit=mem_limit,
            cpu_quota=cpu_quota,
            cpu_period=100000,  # 100ms period
            restart_policy={"Name": "no"},  # Managed by harness
            labels={"saber.component": "mcp-sidecar", "saber.managed": "true"},
            # Ensure Linux host is reachable using host-gateway alias
            extra_hosts={"host.docker.internal": "host-gateway"},
            # Docker logging configuration for reliable log collection
            log_config=docker.types.LogConfig(
                type=docker.types.LogConfig.types.JSON,
                config={
                    "max-size": "50m",
                    "max-file": "3",
                },
            ),
        )

        logger.info(f"📦 Container created: {container.id[:12]}")
        return container

    async def _wait_for_readiness(self) -> bool:
        """Wait for sidecar to become ready."""
        logger.info("⏳ Waiting for sidecar readiness...")

        start_time = time.time()
        while time.time() - start_time < self.config.health_check_timeout:
            try:
                async with aiohttp.ClientSession() as client:
                    # Use container name for network communication instead of localhost
                    sidecar_url = f"http://{self.config.container_name}:{self.config.port}/ready"
                    async with client.get(sidecar_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            logger.info("✅ Sidecar is ready")
                            return True
            except Exception as e:
                logger.debug(f"Readiness check failed: {e}")
                pass

            await asyncio.sleep(self.config.health_check_interval)

        logger.error(f"❌ Sidecar failed to become ready within {self.config.health_check_timeout}s")
        return False

    async def _cleanup_container(self, container: docker.models.containers.Container) -> None:
        """Clean up a container."""
        try:
            container.stop(timeout=10)
            container.remove()
            logger.debug(f"🧹 Cleaned up container: {container.id[:12]}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup container: {e}")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            if self.docker_client:
                self.docker_client.close()
        except Exception:
            pass  # Ignore cleanup errors
