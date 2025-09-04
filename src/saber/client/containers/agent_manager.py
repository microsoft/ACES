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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import docker.errors
import docker.models.containers
import docker.types

import docker

if TYPE_CHECKING:
    from ..logging import ContainerLogManager

logger = logging.getLogger(__name__)


@dataclass
class AgentContainerConfig:
    """Configuration for agent containers."""

    # Container settings
    base_image: str = "saber/agent-runner:latest"
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

    # Debug settings
    debug_mode: bool = False  # If True, skip container cleanup for debugging

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

    def __init__(
        self, config: Optional[AgentContainerConfig] = None, log_manager: Optional["ContainerLogManager"] = None
    ):
        """Initialize agent manager with configuration."""
        self.config = config or AgentContainerConfig()
        self.log_manager = log_manager
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

            # Start logging for agent container (prepare log file metadata only)
            if self.log_manager:
                # Log container lifecycle event
                self.log_manager.log_container_lifecycle_event(
                    event_type="agent_container_start",
                    container_info={
                        "name": container_name,
                        "container_id": container.id[:12],
                        "image": self.config.base_image,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "agent_id": agent_id,
                    },
                    additional_data={"session_id": session_id, "sidecar_url": sidecar_url},
                )

            # Track active container
            self.active_containers[container_name] = container

            try:
                # Wait for container completion with timeout
                result = await self._wait_for_container_completion(container, container_name)

                # Log completion event
                if self.log_manager:
                    self.log_manager.log_container_lifecycle_event(
                        event_type="agent_container_complete",
                        container_info={
                            "name": container_name,
                            "container_id": container.id[:12],
                            "task_id": task_id,
                            "episode_id": episode_id,
                        },
                        additional_data={
                            "success": result.success,
                            "exit_code": result.exit_code,
                            "termination_reason": result.termination_reason,
                            "execution_time": time.time() - start_time,
                        },
                    )

                # Collect logs
                stdout, stderr = await self._collect_container_logs(container)
                result.stdout = stdout
                result.stderr = stderr
                result.execution_time = time.time() - start_time
                result.container_id = container.id[:12]

                # Save logs to file if log manager is available
                if self.log_manager and stdout:
                    try:
                        await self._save_container_logs_to_file(container, stdout, stderr, task_id, episode_id)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save container logs to file: {e}")

                return result

            finally:
                # Stop logging for container
                if self.log_manager:
                    await self.log_manager.stop_logging_container(container.id[:12])

                # Cleanup container
                await self._cleanup_container(container, container_name)
                self.active_containers.pop(container_name, None)

        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")

            # Log error event
            if self.log_manager:
                self.log_manager.log_container_lifecycle_event(
                    event_type="agent_execution_error",
                    container_info={"name": container_name, "task_id": task_id, "episode_id": episode_id},
                    additional_data={"error": str(e), "execution_time": time.time() - start_time},
                )

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
            # Read metadata to get agent file path
            metadata_file = agent_code_path.parent / "metadata.json"
            agent_file_path = None
            if metadata_file.exists():
                import json

                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        agent_file_path = metadata.get("packaged_agent_file")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to read metadata.json: {e}")

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

            # Set agent discovery environment variable if available
            if agent_file_path:
                environment["AGENT_FILE"] = agent_file_path
                logger.debug(f"🔍 Set AGENT_FILE environment variable: {agent_file_path}")
            else:
                logger.warning("⚠️ No agent file path found in metadata - agent discovery may fail")

            # Instead of volume mounting (which fails in Docker-in-Docker),
            # we'll copy the agent files after container creation
            # Remove volume mounts - we'll copy files manually
            volumes: Dict[str, str] = {}  # No volume mounts

            # TMPFS mounts for temporary files
            tmpfs = {mount: "" for mount in self.config.tmpfs_mounts}

            # Resource limits
            mem_limit = self.config.memory_limit
            cpu_quota = int(self.config.cpu_limit * 100000)

            logger.debug(f"📦 Creating container {container_name} with image {self.config.base_image}")

            # Create container (but don't start it yet)
            container = self.docker_client.containers.create(
                image=self.config.base_image,
                name=container_name,
                environment=environment,
                volumes=volumes,
                tmpfs=tmpfs,
                network=self.config.network_name,
                user=self.config.user,
                read_only=False,  # Need write access to copy files
                mem_limit=mem_limit,
                cpu_quota=cpu_quota,
                cpu_period=100000,
                restart_policy={"Name": "no"},
                # Docker logging configuration for reliable log collection
                log_config=docker.types.LogConfig(
                    type=docker.types.LogConfig.types.JSON,
                    config={
                        "max-size": "50m",
                        "max-file": "3",
                    },
                ),
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

            # Copy agent files into the container
            await self._copy_agent_files_to_container(container, agent_code_path)

            # Now start the container
            container.start()
            logger.info(f"🚀 Agent container started: {container.id[:12]}")

            # Register container for log collection if log manager is available
            if self.log_manager:
                base_name = f"{container.name or container.id[:12]}.log"
                self.log_manager.register_container(
                    container=container, log_file_name=base_name, component_type="agent"
                )
                logger.debug(f"📝 Registered agent container for log collection: {container.id[:12]}")

            return container

        except Exception as e:
            logger.error(f"❌ Failed to create agent container: {e}")
            return None

    async def _copy_agent_files_to_container(self, container: Any, agent_code_path: Path) -> None:
        """Copy agent files from local filesystem into the container."""
        try:
            logger.info(f"📁 Copying agent files from {agent_code_path} to container {container.id[:12]}")

            # Copy all files from agent_code_path to /app/agent in the container
            import io
            import os
            import tarfile

            # Create tar archive of agent files
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                for root, dirs, files in os.walk(agent_code_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate relative path from agent_code_path
                        rel_path = os.path.relpath(file_path, agent_code_path)
                        tar.add(file_path, arcname=rel_path)

            tar_buffer.seek(0)

            # Extract tar archive to /app/agent in container (works on stopped containers)
            success = container.put_archive("/app/agent", tar_buffer.getvalue())
            if not success:
                raise RuntimeError("Failed to copy agent files to container")

            logger.info(f"✅ Successfully copied agent files to container {container.id[:12]}")

        except Exception as e:
            logger.error(f"❌ Failed to copy agent files to container: {e}")
            raise

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

    async def _save_container_logs_to_file(
        self, container: Any, stdout: str, stderr: str, task_id: str, episode_id: str
    ) -> None:
        """Save collected container logs to log file."""
        try:
            from datetime import datetime

            # Early return if log_manager is not available
            if not self.log_manager:
                logger.warning("Log manager not available for saving container logs")
                return

            # Create timestamped log file name matching container log manager pattern
            timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
            container_name = container.name or container.id[:12]

            # Build log file name matching the pattern used in start_logging_container
            base_name = f"{task_id}_{episode_id}_attempt_{int(time.time())}"
            log_filename = f"{timestamp}_agent_{base_name}_{container_name}.log"

            # Get the session log directory from log manager
            log_file_path = self.log_manager.session_log_dir / "container-logs" / "agent-containers" / log_filename

            # Ensure directory exists
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write logs to file with metadata header
            with open(log_file_path, "w", encoding="utf-8") as f:
                # Write session metadata header
                header_timestamp = datetime.utcnow().isoformat() + "Z"
                f.write(f"# Container Logs for {container_name}\n")
                f.write("# Component Type: agent\n")
                f.write(f"# Session ID: {self.log_manager.session_context.session_id}\n")
                f.write(f"# Task ID: {task_id}\n")
                f.write(f"# Episode ID: {episode_id}\n")
                f.write(f"# Container ID: {container.id}\n")
                f.write(f"# Log Collected: {header_timestamp}\n")
                f.write("# " + "=" * 60 + "\n\n")

                # Write the actual logs
                if stdout:
                    f.write(stdout)
                if stderr:
                    f.write("\n# STDERR:\n")
                    f.write(stderr)

            logger.info(f"📝 Saved agent container logs to: {log_file_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save container logs to file: {e}")

    async def _cleanup_container(self, container: Any, container_name: str) -> None:
        """Clean up a container."""
        if self.config.debug_mode:
            logger.info(f"🔍 Debug mode enabled - skipping cleanup for container: {container_name}")
            logger.info(f"🔍 To manually clean up: docker stop {container_name} && docker rm {container_name}")
            return

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
