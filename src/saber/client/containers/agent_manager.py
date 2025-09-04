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

import docker
import docker.errors
import docker.models.containers
import docker.types

if TYPE_CHECKING:
    from ..harness_models import DockerCommand
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

    # Docker commands to execute during container setup
    docker_commands: List["DockerCommand"] = field(default_factory=list)


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

        logger.info("🤖 Agent container manager initialized with custom image support")

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
            # Mount Docker socket for container self-termination
            volumes: Dict[str, dict[str, str]] = {
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}
            }

            # TMPFS mounts for temporary files
            tmpfs = {mount: "" for mount in self.config.tmpfs_mounts}

            # Resource limits
            mem_limit = self.config.memory_limit
            cpu_quota = int(self.config.cpu_limit * 100000)

            # Use the configured base image (which could be a custom agent image)
            image_to_use = self.config.base_image

            logger.debug(f"📦 Creating container {container_name} with image {image_to_use}")

            # Create container with blocking command to keep it alive for setup
            container = self.docker_client.containers.create(
                image=image_to_use,
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
                # Security: Drop most capabilities but keep needed ones for setup
                cap_drop=["ALL"],
                cap_add=["CHOWN", "DAC_OVERRIDE"],  # Need CHOWN and DAC_OVERRIDE for file permission setup
                # Security: No new privileges
                security_opt=["no-new-privileges"],
                # Start with blocking command to keep container alive for setup
                command=["tail", "-f", "/dev/null"],
            )

            logger.info(f"📦 Agent container created: {container.id[:12]}")

            # Start the container to keep it alive for setup operations
            container.start()
            logger.info(f"🚀 Agent container started for setup: {container.id[:12]}")

            # Copy agent files into the running container
            await self._copy_agent_files_to_container(container, agent_code_path)

            # Execute any additional docker commands while container is running
            await self._execute_docker_commands(container)

            # Now execute the actual launch.py command to start the agent
            logger.info(f"🚀 Starting agent execution in container {container.id[:12]}")
            logger.info("🔍 About to execute: python /app/agent/launch.py (non-blocking)")

            # Use low-level Docker API for proper non-blocking execution with monitoring
            # Redirect stdout/stderr to container's main process so logs are captured
            exec_id = self.docker_client.api.exec_create(
                container.id, ["bash", "-c", "python /app/agent/launch.py 2>&1 | tee /proc/1/fd/1"], user="agent"
            )["Id"]

            # Start execution in detached mode
            self.docker_client.api.exec_start(exec_id, detach=True)

            logger.info(f"🔍 Agent execution started in background, exec_id: {exec_id}")
            logger.info(f"✅ Agent execution launched successfully in container {container.id[:12]}")

            # Store exec_id for potential monitoring (can be used by caller if needed)
            container.exec_id = exec_id

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

    async def _execute_docker_commands(self, container: Any) -> None:
        """Execute additional Docker commands on the container."""
        if not self.config.docker_commands:
            logger.debug("No docker commands configured for agent container")
            return

        logger.info(f"🔧 Executing {len(self.config.docker_commands)} docker commands on container {container.id[:12]}")

        for i, cmd in enumerate(self.config.docker_commands, 1):
            try:
                if cmd.description:
                    logger.info(f"🔧 [{i}/{len(self.config.docker_commands)}] {cmd.description}")
                else:
                    logger.info(f"🔧 [{i}/{len(self.config.docker_commands)}] Executing {cmd.type} command")

                if cmd.type == "copy":
                    await self._execute_docker_copy_command(container, cmd)
                elif cmd.type == "exec":
                    await self._execute_docker_exec_command(container, cmd)
                else:
                    raise ValueError(f"Unsupported docker command type: {cmd.type}")

                logger.debug(f"✅ Docker command {i} completed successfully")

            except Exception as e:
                error_msg = f"Docker command {i} failed: {e}"
                if cmd.description:
                    error_msg = f"Docker command {i} ({cmd.description}) failed: {e}"
                logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e

        logger.info(f"✅ All docker commands completed successfully for container {container.id[:12]}")

    async def _execute_docker_copy_command(self, container: Any, cmd: "DockerCommand") -> None:
        """Execute a docker copy command."""
        import io
        import os
        import tarfile
        from pathlib import Path

        source_path = Path(cmd.source) if cmd.source is not None else Path("")
        destination_path = cmd.destination if cmd.destination is not None else ""

        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        logger.debug(f"Copying from {source_path} to container:{destination_path}")

        # Create tar archive of source files/directories
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            if source_path.is_file():
                # Copy single file
                tar.add(str(source_path), arcname=source_path.name)
            else:
                # Copy directory contents
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate relative path from source_path
                        rel_path = os.path.relpath(file_path, source_path)
                        tar.add(file_path, arcname=rel_path)

        tar_buffer.seek(0)

        # Extract to destination in container
        success = container.put_archive(destination_path, tar_buffer.getvalue())
        if not success:
            raise RuntimeError(f"Failed to copy {source_path} to container:{destination_path}")

    async def _execute_docker_exec_command(self, container: Any, cmd: "DockerCommand") -> None:
        """Execute a command inside the container."""
        if not cmd.command:
            raise ValueError("Exec command requires 'command' parameter")

        # Use specified user or default to root for setup commands
        exec_user = cmd.user if cmd.user else "root"

        logger.debug(f"Executing command in container as user '{exec_user}': {' '.join(cmd.command)}")

        # Execute command in container
        exit_code, output = container.exec_run(
            cmd=cmd.command,
            user=exec_user,
            workdir="/",
        )

        if exit_code != 0:
            error_output = output.decode("utf-8", errors="replace") if output else "No output"
            raise RuntimeError(f"Command failed with exit code {exit_code}: {error_output}")

        logger.debug(f"Command completed successfully: {' '.join(cmd.command)}")

    async def _wait_for_container_completion(self, container: Any, container_name: str) -> AgentExecutionResult:
        """Wait for container to complete execution by monitoring the exec process."""
        start_time = time.time()

        try:
            # Get the exec_id we stored earlier
            exec_id = getattr(container, "exec_id", None)
            if not exec_id:
                raise RuntimeError("No exec_id found - agent execution may not have started properly")

            logger.info(f"🔍 Monitoring exec process {exec_id} for completion")

            # Monitor the exec process until completion
            while True:
                try:
                    # Check if we've exceeded timeout
                    if time.time() - start_time > self.config.execution_timeout:
                        logger.warning(f"⏰ Exec process {exec_id} execution timeout")

                        # Kill the container to stop execution
                        container.stop(timeout=self.config.termination_grace_period)

                        return AgentExecutionResult(
                            success=False,
                            exit_code=-1,
                            termination_reason="timeout",
                            execution_time=time.time() - start_time,
                        )

                    # Inspect the exec process
                    exec_info = self.docker_client.api.exec_inspect(exec_id)

                    if not exec_info["Running"]:
                        # Exec process has completed
                        exit_code = exec_info.get("ExitCode", 0)
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

                        logger.info(f"🏁 Exec process {exec_id} completed with exit code {exit_code}")

                        # Stop the container since the agent execution is done
                        try:
                            container.stop(timeout=5)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to stop container after exec completion: {e}")

                        return AgentExecutionResult(
                            success=success,
                            exit_code=exit_code,
                            termination_reason=termination_reason,
                            execution_time=execution_time,
                        )

                    # Process still running, wait a bit before checking again
                    await asyncio.sleep(1)

                except docker.errors.NotFound:
                    # Exec process not found - probably completed or container stopped
                    logger.warning(f"⚠️ Exec process {exec_id} not found - container may have stopped")
                    return AgentExecutionResult(
                        success=False,
                        exit_code=-1,
                        termination_reason="exec_not_found",
                        execution_time=time.time() - start_time,
                    )

        except Exception as e:
            logger.error(f"❌ Error monitoring exec process: {e}")
            return AgentExecutionResult(
                success=False,
                exit_code=-1,
                termination_reason="monitoring_error",
                execution_time=time.time() - start_time,
                error=str(e),
            )

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
