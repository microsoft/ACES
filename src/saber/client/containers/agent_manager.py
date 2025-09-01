"""
Agent Container Manager

Manages individual agent containers for episodic execution.
Handles container lifecycle, resource limits, network configuration,
and termination monitoring for robust agent execution.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
import docker.errors
import docker.models.containers

logger = logging.getLogger(__name__)


@dataclass
class AgentContainerConfig:
    """Configuration for agent containers."""

    # Container settings
    base_image: str = "saber-agent-runner:latest"
    memory_limit: str = "1g"
    cpu_limit: float = 1.0
    network_name: str = "saber-network"

    # Security settings
    user: str = "agent:agent"  # Non-root user
    read_only_root: bool = True
    tmpfs_mounts: List[str] = field(default_factory=lambda: ["/tmp", "/var/tmp"])

    # Execution settings
    execution_timeout: int = 600  # 10 minutes
    termination_grace_period: int = 30  # 30 seconds

    # Environment
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentExecutionResult:
    """Result of agent container execution."""

    success: bool
    exit_code: int
    termination_reason: str
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    container_id: str = ""
    error: Optional[str] = None


class AgentManager:
    """
    Manages individual agent containers for episodic execution.

    Responsibilities:
    - Create agent containers with proper configuration
    - Monitor container execution and termination
    - Collect results and logs
    - Enforce resource limits and security policies
    - Handle graceful and forced termination
    """

    def __init__(self, config: Optional[AgentContainerConfig] = None):
        """Initialize agent manager with configuration."""
        self.config = config or AgentContainerConfig()
        self.docker_client = docker.DockerClient.from_env()  # type: ignore
        self.active_containers: Dict[str, Any] = {}

        logger.info("🤖 Agent container manager initialized")

    async def execute_agent(
        self,
        agent_code_path: Path,
        agent_id: str,
        sidecar_url: str,
        initial_prompt: str,
        task_id: str,
        episode_id: str,
        *,
        session_id: str,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> AgentExecutionResult:
        """
        Execute an agent in a container.

        Args:
            agent_code_path: Path to agent code
            agent_id: Unique agent identifier
            sidecar_url: URL of MCP sidecar service
            initial_prompt: Initial prompt for agent
            task_id: Task identifier
            episode_id: Episode identifier

        Returns:
            AgentExecutionResult with execution details
        """
        container_name = f"saber-agent-{agent_id}-{episode_id}"
        start_time = time.time()

        try:
            logger.info(f"🚀 Starting agent container: {container_name}")

            # Create and start container
            container = await self._create_agent_container(
                container_name=container_name,
                agent_code_path=agent_code_path,
                agent_id=agent_id,
                sidecar_url=sidecar_url,
                initial_prompt=initial_prompt,
                task_id=task_id,
                episode_id=episode_id,
                session_id=session_id,
                extra_env=extra_env or {},
            )

            if not container:
                return AgentExecutionResult(
                    success=False,
                    exit_code=-1,
                    termination_reason="container_creation_failed",
                    error="Failed to create agent container",
                )

            # Track active container
            self.active_containers[container_name] = container

            try:
                # Wait for container completion with timeout
                result = await self._wait_for_container_completion(container, container_name)

                # Collect logs
                stdout, stderr = await self._collect_container_logs(container)
                result.stdout = stdout
                result.stderr = stderr
                result.execution_time = time.time() - start_time
                result.container_id = container.id[:12]

                return result

            finally:
                # Cleanup container
                await self._cleanup_container(container, container_name)
                self.active_containers.pop(container_name, None)

        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")
            return AgentExecutionResult(
                success=False,
                exit_code=-1,
                termination_reason="execution_error",
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def terminate_agent(self, agent_id: str, episode_id: str) -> bool:
        """
        Terminate a running agent container.

        Args:
            agent_id: Agent identifier
            episode_id: Episode identifier

        Returns:
            bool: True if termination successful, False otherwise
        """
        container_name = f"saber-agent-{agent_id}-{episode_id}"

        if container_name not in self.active_containers:
            logger.warning(f"⚠️ Container not found for termination: {container_name}")
            return False

        container = self.active_containers[container_name]

        try:
            logger.info(f"🛑 Terminating agent container: {container_name}")

            # Graceful termination
            container.stop(timeout=self.config.termination_grace_period)

            logger.info(f"✅ Agent container terminated: {container_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to terminate container {container_name}: {e}")

            # Force termination as last resort
            try:
                container.kill()
                logger.warning(f"⚠️ Force killed container: {container_name}")
                return True
            except Exception as kill_error:
                logger.error(f"❌ Failed to force kill container: {kill_error}")
                return False

    async def cleanup_all_containers(self) -> None:
        """Clean up all active agent containers."""
        logger.info("🧹 Cleaning up all agent containers...")

        cleanup_tasks = []
        for container_name, container in list(self.active_containers.items()):
            task = self._cleanup_container(container, container_name)
            cleanup_tasks.append(task)

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self.active_containers.clear()
        logger.info("✅ All agent containers cleaned up")

    async def _create_agent_container(
        self,
        container_name: str,
        agent_code_path: Path,
        agent_id: str,
        sidecar_url: str,
        initial_prompt: str,
        task_id: str,
        episode_id: str,
        *,
        session_id: str,
        extra_env: Dict[str, str],
    ) -> Optional[Any]:
        """Create and start an agent container."""
        try:
            # Environment variables for agent execution (match AgentExecutor contract)
            environment = {
                "AGENT_ID": agent_id,
                "MCP_SIDECAR_URL": sidecar_url,
                "INITIAL_PROMPT": initial_prompt,
                "TASK_ID": task_id,
                # SESSION_ID is required by AgentExecutor; pass explicitly from harness/executor
                "SESSION_ID": session_id,
                "EPISODE_ID": episode_id,
                "MCP_CLIENT_TYPE": "standard",  # Use standard MCP libraries
                **self.config.environment,
                **(extra_env or {}),
            }

            # Volume mounts for agent code
            volumes = {str(agent_code_path.absolute()): {"bind": "/app/agent", "mode": "ro"}}  # Read-only

            # TMPFS mounts for temporary files
            tmpfs = {mount: "" for mount in self.config.tmpfs_mounts}

            # Resource limits
            mem_limit = self.config.memory_limit
            cpu_quota = int(self.config.cpu_limit * 100000)

            logger.debug(f"📦 Creating container {container_name} with image {self.config.base_image}")

            container = self.docker_client.containers.run(
                image=self.config.base_image,
                name=container_name,
                environment=environment,
                volumes=volumes,
                tmpfs=tmpfs,
                network=self.config.network_name,
                user=self.config.user,
                read_only=self.config.read_only_root,
                detach=True,
                remove=False,  # Keep for log collection
                mem_limit=mem_limit,
                cpu_quota=cpu_quota,
                cpu_period=100000,
                restart_policy={"Name": "no"},
                labels={
                    "saber.component": "agent",
                    "saber.agent_id": agent_id,
                    "saber.task_id": task_id,
                    "saber.episode_id": episode_id,
                    "saber.managed": "true",
                },
                # Security: Drop all capabilities
                cap_drop=["ALL"],
                # Security: No new privileges
                security_opt=["no-new-privileges"],
                # Command runs the packaged launcher which invokes AgentExecutor
                command=["python", "/app/agent/launch.py"],
            )

            logger.info(f"📦 Agent container created: {container.id[:12]}")
            return container

        except Exception as e:
            logger.error(f"❌ Failed to create agent container: {e}")
            return None

    async def _wait_for_container_completion(self, container: Any, container_name: str) -> AgentExecutionResult:
        """Wait for container to complete execution."""
        start_time = time.time()

        try:
            # Wait for container with timeout
            exit_code = container.wait(timeout=self.config.execution_timeout)["StatusCode"]
            execution_time = time.time() - start_time

            # Determine termination reason
            if exit_code == 0:
                termination_reason = "completed"
                success = True
            elif exit_code == 130:  # SIGINT
                termination_reason = "interrupted"
                success = False
            elif exit_code == 137:  # SIGKILL
                termination_reason = "killed"
                success = False
            else:
                termination_reason = "error"
                success = False

            logger.info(f"🏁 Container {container_name} completed with exit code {exit_code}")

            return AgentExecutionResult(
                success=success,
                exit_code=exit_code,
                termination_reason=termination_reason,
                execution_time=execution_time,
            )

        except Exception as e:
            if "timeout" in str(e).lower():
                logger.warning(f"⏰ Container {container_name} execution timeout")

                # Attempt graceful termination
                try:
                    container.stop(timeout=self.config.termination_grace_period)
                except Exception:
                    container.kill()  # Force kill if graceful fails

                return AgentExecutionResult(
                    success=False, exit_code=-1, termination_reason="timeout", execution_time=time.time() - start_time
                )
            else:
                raise

    async def _collect_container_logs(self, container: Any) -> tuple[str, str]:
        """Collect stdout and stderr from container."""
        try:
            logs = container.logs(stdout=True, stderr=True, stream=False, timestamps=False)

            # Docker combines stdout/stderr, so we'll return the combined output
            # In a more sophisticated implementation, we could separate them
            logs_str = logs.decode("utf-8", errors="replace")

            return logs_str, ""  # Return combined as stdout for now

        except Exception as e:
            logger.warning(f"⚠️ Failed to collect container logs: {e}")
            return "", f"Log collection failed: {e}"

    async def _cleanup_container(self, container: Any, container_name: str) -> None:
        """Clean up a container."""
        try:
            # Stop if still running
            container.reload()
            if container.status == "running":
                container.stop(timeout=10)

            # Remove container
            container.remove()
            logger.debug(f"🧹 Cleaned up container: {container_name}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup container {container_name}: {e}")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            if self.docker_client:
                self.docker_client.close()
        except Exception:
            pass  # Ignore cleanup errors
