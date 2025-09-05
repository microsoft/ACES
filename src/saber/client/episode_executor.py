#!/usr/bin/env python3
"""
Container-based Episode Executor

Replaces the embedded MCP approach with containerized agent execution.
Handles agent packaging, container orchestration, and multi-channel termination monitoring.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .api import SABERRestClient
from .containers import AgentManager, SidecarManager
from .containers.agent_manager import AgentContainerConfig, AgentExecutionResult
from .containers.sidecar_manager import SidecarConfig
from .harness_models import EpisodeResult, SABERHarnessConfig
from .logging import ClientLoggingConfig, ContainerLogManager, SessionLoggingContext

# Import MessageType for UI notifications
from .ui.models import MessageType

logger = logging.getLogger(__name__)


class ContainerEpisodeExecutor:
    """
    Container-based episode executor for SABER agents.

    Replaces embedded MCP client execution with containerized agent execution:
    - Packages agent code for container execution
    - Manages shared MCP sidecar service
    - Executes agents in isolated containers
    - Monitors multiple termination channels
    - Collects results from container output
    """

    def __init__(
        self,
        rest_client: SABERRestClient,
        session_id: str,
        parallelism: int = 1,
        saber_server_url: Optional[str] = None,
        saber_mcp_url: Optional[str] = None,
        harness_config: Optional[SABERHarnessConfig] = None,
        ui_manager: Optional[Any] = None,  # Type hint as Any to avoid circular imports
        session_log_dir: Optional[Path] = None,  # Shared log directory for all session logs
        harness: Optional[Any] = None,  # Reference to harness for agent file path access
    ):
        """Initialize container episode executor."""
        self.rest_client = rest_client
        self.session_id = session_id
        self.parallelism = parallelism
        self.harness_config = harness_config
        self.ui_manager = ui_manager
        self.session_log_dir = session_log_dir
        self.harness = harness  # Store harness reference

        # Initialize container logging if enabled
        self.log_manager: Optional[ContainerLogManager] = None
        if harness_config and harness_config.enable_container_logging:
            self._setup_container_logging()

        # Configure sidecar with proper server URLs
        sidecar_config = SidecarConfig()
        if saber_server_url:
            sidecar_config.saber_server_url = saber_server_url
        if saber_mcp_url:
            sidecar_config.saber_mcp_url = saber_mcp_url

        # Configure harness URL for progress reporting
        if harness and hasattr(harness, "progress_server") and harness.progress_server:
            # Use the actual progress server host/port
            sidecar_config.harness_url = f"http://host.docker.internal:{harness.progress_server.port}"

        # Detect current container's network for sidecar communication
        # This ensures the sidecar and client are on the same network for DNS resolution
        try:
            import socket

            import docker

            docker_client = docker.from_env()  # type: ignore[attr-defined]

            # Get the current container (client container)
            hostname = socket.gethostname()
            current_container = docker_client.containers.get(hostname)

            # Get the first network (there should typically be only one)
            networks = list(current_container.attrs["NetworkSettings"]["Networks"].keys())
            if networks:
                sidecar_config.network_name = networks[0]
                logger.info(f"🔍 Detected network for sidecar: {sidecar_config.network_name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not detect container network, using default: {e}")
            # Fall back to default network

        # Container managers
        # Configure debug mode from environment variable
        debug_mode = os.getenv("SABER_DEBUG_MODE", "").lower() in ("true", "1", "yes")
        self.sidecar_manager = SidecarManager(sidecar_config, debug_mode=debug_mode)

        # Configure agent manager to use the same network as sidecar
        agent_config = AgentContainerConfig()
        agent_config.network_name = sidecar_config.network_name

        # Configure custom agent image if specified in harness config
        if self.harness_config and self.harness_config.agent_config:
            agent_image = self.harness_config.agent_config.get("image")
            if agent_image:
                agent_config.base_image = agent_image
                logger.info(f"🐳 Using custom agent image: {agent_image}")

        # Configure docker commands if specified in harness config
        if self.harness_config and self.harness_config.docker_commands:
            agent_config.docker_commands = self.harness_config.docker_commands
            logger.info(f"📋 Configured {len(self.harness_config.docker_commands)} docker commands for agent setup")

        # Configure debug mode for agent containers
        agent_config.debug_mode = debug_mode

        if debug_mode:
            logger.info("🔍 Debug mode enabled - containers will not be cleaned up automatically")
            logger.info("🔍 Manual cleanup will be required: docker stop <container> && docker rm <container>")

        self.agent_manager = AgentManager(config=agent_config, log_manager=self.log_manager)

        # State
        self.sidecar_started = False
        self.active_episodes: Dict[str, asyncio.Task[None]] = {}
        self._harness_agent_id: Optional[str] = None

    def _setup_container_logging(self) -> None:
        """Set up container logging configuration."""
        try:
            # Use shared session log directory if provided, otherwise fall back to old behavior
            if self.session_log_dir:
                base_log_dir = self.session_log_dir
            else:
                client_log_dir = self.harness_config.client_log_dir if self.harness_config else None
                base_log_dir = client_log_dir if client_log_dir else Path("./client/logs")

            # Create logging configuration from harness config
            log_config = ClientLoggingConfig(
                base_log_dir=base_log_dir,
                enable_container_logging=self.harness_config.enable_container_logging if self.harness_config else True,
                retention_days=self.harness_config.log_retention_days if self.harness_config else 7,
                max_log_size_mb=self.harness_config.max_log_size_mb if self.harness_config else 100,
                compress_old_logs=self.harness_config.compress_old_logs if self.harness_config else True,
                log_level=self.harness_config.log_level if self.harness_config else "INFO",
            )

            # Create session context
            client_id = self.harness_config.client_id if self.harness_config else "saber-client"
            session_context = SessionLoggingContext(session_id=self.session_id, client_id=client_id)

            # Initialize log manager
            self.log_manager = ContainerLogManager(log_config, session_context, self.session_log_dir)
            logger.info(f"📝 Container logging enabled: {log_config.base_log_dir}")

        except Exception as e:
            logger.error(f"❌ Failed to setup container logging: {e}")
            self.log_manager = None

    async def initialize(self) -> None:
        """Initialize container infrastructure."""
        logger.info("🔧 Initializing container episode executor")

        # Start shared MCP sidecar
        await self.sidecar_manager.start_sidecar()
        self.sidecar_started = True

        # Start logging for sidecar container if logging is enabled
        if self.log_manager and self.sidecar_manager.container:
            await self.log_manager.start_logging_container(
                container=self.sidecar_manager.container, log_file_name="mcp_sidecar.log", component_type="sidecar"
            )

        # Register session with sidecar
        # Register harness agent session with sidecar for monitoring (optional)
        # NOTE: Harness registration uses episode_id=None since it's not episode-specific
        self._harness_agent_id = await self.sidecar_manager.register_agent_session(
            session_id=self.session_id, episode_id=None, task_id=None, agent_id="saber-harness"
        )

        logger.info("✅ Container infrastructure initialized")

    async def _start_episode_progress_stream(self, episode_id: str) -> None:
        """Start episode-specific SSE progress stream for tool call updates."""
        if not self.rest_client:
            logger.warning("No REST client available for progress stream")
            return

        # Import UI progress adapter and server API models
        from saber.api_models import ToolCallEventComplete, ToolCallEventStart

        from .ui.interfaces import UIProgressAdapter

        # Create UI progress adapter
        self.progress_adapter = UIProgressAdapter(ui_manager=self.ui_manager)

        def progress_callback(data: Dict[str, Any]) -> None:
            """Handle progress updates from episode-specific SSE stream."""
            try:
                logger.info(f"📡 Episode {episode_id} progress update: {data}")
                logger.info(f"🚨 EPISODE SSE DEBUG: Received event type: {data.get('type', 'UNKNOWN')}")

                # Parse SSE data into tool call events and schedule async operations
                event_type = data.get("type")

                if event_type == "connection":
                    logger.info(f"🚨 EPISODE SSE DEBUG: Connection event received for episode {episode_id}")
                    return

                elif event_type == "heartbeat":
                    logger.info(f"🚨 EPISODE SSE DEBUG: Heartbeat event received for episode {episode_id}")
                    return

                elif event_type == "tool_call_start":
                    logger.info(
                        f"🚨 EPISODE SSE DEBUG: Tool call start event received for "
                        f"{data.get('tool_name', 'unknown')} in episode {episode_id}"
                    )
                    # Create ToolCallEventStart object directly from SSE data (which already has the right structure)
                    tool_event_start = ToolCallEventStart(**data)
                    # Schedule the async call
                    logger.info("🚨 EPISODE SSE DEBUG: Creating async task for tool_call_start")
                    asyncio.create_task(self.progress_adapter.tool_call_start(tool_event_start))

                elif event_type == "tool_call_complete":
                    logger.info(
                        f"🚨 EPISODE SSE DEBUG: Tool call complete event received for "
                        f"{data.get('tool_name', 'unknown')} in episode {episode_id}"
                    )
                    # Create ToolCallEventComplete object directly from SSE data (which already has the right structure)
                    tool_event_complete = ToolCallEventComplete(**data)
                    # Schedule the async call
                    logger.info("🚨 EPISODE SSE DEBUG: Creating async task for tool_call_complete")
                    asyncio.create_task(self.progress_adapter.tool_call_complete(tool_event_complete))

                elif event_type in ["connection", "heartbeat"]:
                    # Ignore connection/heartbeat events
                    pass
                else:
                    logger.info(f"🚨 EPISODE SSE DEBUG: Unknown event type: {event_type}")
                    logger.debug(f"🔍 Unknown SSE event type: {event_type}")

            except Exception as e:
                logger.warning(f"⚠️ Error handling progress update for episode {episode_id}: {e}")

        # Start the episode-specific SSE stream with required episode_id
        success = await self.rest_client.start_progress_stream(
            progress_callback=progress_callback, episode_id=episode_id, agent_id="saber-harness"
        )

        if success:
            logger.info(f"✅ Episode {episode_id} SSE progress stream started")
        else:
            logger.warning(f"⚠️ Failed to start SSE progress stream for episode {episode_id}")

    async def _stop_episode_progress_stream(self) -> None:
        """Stop the current episode's progress stream."""
        try:
            await self.rest_client.stop_progress_stream()
            logger.info("✅ Episode progress stream stopped")
        except Exception as e:
            logger.warning(f"⚠️ Failed to stop episode progress stream: {e}")

    async def execute_episodes(
        self, episodes: List[Tuple[str, int]], agent: Any  # [(task_id, attempt), ...]
    ) -> List[EpisodeResult]:
        """
        Execute episodes using containerized agents.

        Args:
            episodes: List of (task_id, attempt) tuples
            agent: Agent object to execute

        Returns:
            List of episode results
        """
        if not self.sidecar_started:
            raise RuntimeError("Container infrastructure not initialized")

        logger.info(f"🎬 Executing {len(episodes)} episodes with parallelism {self.parallelism}")

        # Package agent code for containers
        agent_package = await self._package_agent(agent)

        if self.parallelism == 1:
            # Sequential execution
            results = []
            for i, (task_id, attempt) in enumerate(episodes):
                logger.info(f"🎬 Episode {i+1}/{len(episodes)}: {task_id} (attempt {attempt})")
                result = await self._execute_single_episode(task_id, attempt, agent_package)
                results.append(result)
                logger.info(f"{'✅' if result.success else '❌'} Episode complete: {result.termination_reason}")
            return results
        else:
            # Parallel execution
            return await self._execute_episodes_parallel(episodes, agent_package)

    async def _execute_episodes_parallel(
        self, episodes: List[Tuple[str, int]], agent_package: Path
    ) -> List[EpisodeResult]:
        """Execute episodes in parallel using containers."""
        logger.info(f"🧵 Starting parallel execution with {self.parallelism} workers")

        # Create semaphore to limit concurrent containers
        semaphore = asyncio.Semaphore(self.parallelism)

        async def execute_with_semaphore(task_id: str, attempt: int) -> EpisodeResult:
            async with semaphore:
                return await self._execute_single_episode(task_id, attempt, agent_package)

        # Execute episodes concurrently
        tasks = [execute_with_semaphore(task_id, attempt) for task_id, attempt in episodes]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_id, attempt = episodes[i]
                logger.error(f"❌ Episode {task_id} failed: {result}")
                final_results.append(
                    EpisodeResult(
                        task_id=task_id,
                        episode_id="",
                        attempt=attempt,
                        success=False,
                        termination_reason="error",
                        error=str(result),
                    )
                )
            else:
                # mypy doesn't understand the isinstance guard above
                final_results.append(result)  # type: ignore[arg-type]

        return final_results

    async def _execute_single_episode(self, task_id: str, attempt: int, agent_package: Path) -> EpisodeResult:
        """Execute a single episode in a container."""
        logger.debug(f"🎬 Starting containerized episode {attempt} for task {task_id}")

        # UI Progress: Episode start
        if self.ui_manager:
            await self._notify_episode_start(task_id, attempt)

        try:
            # Step 1: Create episode on server (episode-first pattern)
            episode_id = await self.rest_client.create_episode(task_id=task_id)
            logger.debug(f"📍 Created episode: {episode_id}")

            # Step 1.5: Start episode-specific progress stream
            await self._start_episode_progress_stream(episode_id)

            # UI Progress: Episode started
            if self.ui_manager:
                await self._notify_episode_started(task_id, episode_id, attempt)

            # Step 2: Get episode-specific policy
            policy_data = await self.rest_client.get_policy_info(episode_id=episode_id, session_id=self.session_id)
            initial_prompt = str(policy_data.get("prompt", ""))
            if not initial_prompt:
                raise Exception("No prompt found in policy data")

            logger.debug(f"📜 Retrieved policy prompt for task {task_id}")

            # Step 3: Execute agent in container with termination monitoring
            container_result = await self._execute_agent_container(
                task_id=task_id, episode_id=episode_id, agent_package=agent_package, initial_prompt=initial_prompt
            )

            # Step 4: Analyze results
            success, flag, termination_reason = self._analyze_container_result(container_result)

            # Step 5: End episode on server with results
            episode_result_data = {
                "success": success,
                "flag": flag,
                "termination_reason": termination_reason,
                "iterations": self._extract_step_count(container_result),
            }

            # FAIL FAST: Always end episode on server, don't ignore failures
            end_success = await self.rest_client.end_episode(
                episode_id=episode_id, reason=termination_reason, result=episode_result_data
            )
            if not end_success:
                logger.error(f"❌ Failed to properly end episode {episode_id} on server!")
                # FAIL FAST: Don't silently continue with unended episodes
                raise Exception(f"Failed to end episode {episode_id} on server")

            # Step 6: Stop episode-specific progress stream
            await self._stop_episode_progress_stream()

            result = EpisodeResult(
                task_id=task_id,
                episode_id=episode_id,
                attempt=attempt,
                success=success,
                flag=flag,
                termination_reason=termination_reason,
                iterations=self._extract_step_count(container_result),
            )

            # UI Progress: Episode complete
            if self.ui_manager:
                await self._notify_episode_complete(result)

            logger.info(f"✅ Episode {episode_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Container episode execution failed: {e}")

            # If we have an episode_id, try to end it properly
            episode_id_for_result = ""
            try:
                # Check if episode_id was created before the error
                if "episode_id" in locals():
                    episode_id_for_result = episode_id
                    logger.info(f"🧹 Attempting to end failed episode: {episode_id}")
                    # FAIL FAST: End episode with error state, don't ignore failures
                    end_success = await self.rest_client.end_episode(
                        episode_id=episode_id, reason="error", result={"success": False, "error": str(e)}
                    )
                    if not end_success:
                        logger.error(f"❌ Failed to end failed episode {episode_id} - may cause server-side orphan")
                        # Don't raise here since we're already in error handling
            except Exception as cleanup_error:
                logger.error(f"❌ Episode cleanup failed: {cleanup_error}")
                # Don't raise here since we're already in error handling

            # Stop episode progress stream in error case too
            try:
                await self._stop_episode_progress_stream()
            except Exception as stream_cleanup_error:
                logger.error(f"❌ Failed to stop episode progress stream: {stream_cleanup_error}")

            result = EpisodeResult(
                task_id=task_id,
                episode_id=episode_id_for_result,
                attempt=attempt,
                success=False,
                termination_reason="error",
                error=str(e),
            )

            # UI Progress: Episode failed
            if self.ui_manager:
                await self._notify_episode_complete(result)

            return result

    async def _execute_agent_container(
        self, task_id: str, episode_id: str, agent_package: Path, initial_prompt: str
    ) -> AgentExecutionResult:
        """Execute agent in container with multi-channel termination monitoring."""
        # Update session context with current task/episode info for logging
        if self.log_manager:
            self.log_manager.session_context.task_id = task_id
            self.log_manager.session_context.episode_id = episode_id

            # Log container lifecycle event
            self.log_manager.log_container_lifecycle_event(
                event_type="agent_execution_start",
                container_info={"task_id": task_id, "episode_id": episode_id, "agent_package": str(agent_package)},
                additional_data={"initial_prompt_length": len(initial_prompt), "session_id": self.session_id},
            )

        # Get sidecar URL
        sidecar_url = self.sidecar_manager.get_sidecar_url()

        # Create unique agent ID for this execution
        agent_id = f"{task_id}-{episode_id}"

        # Execute agent container with monitoring
        # Build agent discovery env hints
        extra_env = self._build_agent_env_hints(agent_package)

        container_task = asyncio.create_task(
            self.agent_manager.execute_agent(
                agent_code_path=agent_package,
                agent_id=agent_id,
                sidecar_url=sidecar_url,
                initial_prompt=initial_prompt,
                task_id=task_id,
                episode_id=episode_id,
                session_id=self.session_id,
                extra_env=extra_env,
            )
        )

        # Monitor episode termination via REST API
        monitor_task = asyncio.create_task(self._monitor_episode_termination(episode_id, container_task))

        try:
            # Wait for either container completion or episode termination
            done, pending = await asyncio.wait([container_task, monitor_task], return_when=asyncio.FIRST_COMPLETED)

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Get result from completed task
            if container_task in done:
                result = await container_task
                logger.debug(f"📊 Container execution completed: {result.termination_reason}")
                return result
            else:
                # Episode was terminated by server
                logger.info("🛑 Episode terminated by server")
                # Container should have been stopped by monitor
                try:
                    result = await container_task
                except asyncio.CancelledError:
                    result = AgentExecutionResult(
                        success=False,
                        exit_code=-1,
                        termination_reason="server_terminated",
                        stdout="",
                        stderr="",
                        execution_time=0.0,
                        container_id="",
                    )
                return result

        except Exception as e:
            logger.error(f"Container execution error: {e}")
            container_task.cancel()
            monitor_task.cancel()
            raise

    async def _monitor_episode_termination(self, episode_id: str, container_task: asyncio.Task[Any]) -> None:
        """Monitor episode termination via REST API and stop container if needed."""
        try:
            while not container_task.done():
                # Check episode status on server
                try:
                    # This would require an endpoint to check episode status
                    # For now, we'll rely on container completion
                    pass
                except Exception:
                    # If we can't check status, assume episode is still active
                    pass

                await asyncio.sleep(1.0)  # Check every second

        except asyncio.CancelledError:
            # Cancel container if episode is terminated
            logger.info("🛑 Cancelling container due to episode termination")
            container_task.cancel()
            raise

    async def _package_agent(self, agent: Any) -> Path:
        """Package agent code for container execution.

        Now uses direct file path approach instead of dynamic loading inspection.
        """
        logger.info("📦 Preparing agent launcher for container execution")

        temp_dir = Path(tempfile.mkdtemp(prefix="saber-agent-"))

        # Create agent subdirectory to match container mount structure
        agent_dir = temp_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal launcher script that invokes the AgentExecutor
        launcher_file = agent_dir / "launch.py"
        launcher_code = (
            "import sys, asyncio, os, signal, subprocess\n"
            "sys.path.insert(0, '/agent_runtime')\n"
            "from saber.client.agent_runtime import AgentExecutor\n"
            "\n"
            "def get_container_id():\n"
            '    """Get the current container\'s ID from /proc/self/cgroup"""\n'
            "    try:\n"
            "        with open('/proc/self/cgroup', 'r') as f:\n"
            "            for line in f:\n"
            "                if 'docker' in line:\n"
            "                    # Extract container ID from cgroup path\n"
            "                    parts = line.strip().split('/')\n"
            "                    for part in parts:\n"
            "                        if len(part) == 64 and part.isalnum():\n"
            "                            return part\n"
            "    except:\n"
            "        pass\n"
            "    return None\n"
            "\n"
            "def kill_container(container_id):\n"
            '    """Kill the container using docker command"""\n'
            "    try:\n"
            "        print(f'🛑 Killing container {container_id}...', flush=True)\n"
            "        subprocess.run(['docker', 'kill', container_id], check=False, capture_output=True)\n"
            "        print(f'✅ Container {container_id} killed successfully', flush=True)\n"
            "    except Exception as e:\n"
            "        print(f'⚠️ Failed to kill container: {e}', flush=True)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    container_id = get_container_id()\n"
            "    print(f'🐳 Running in container: {container_id}', flush=True)\n"
            "    \n"
            "    try:\n"
            "        print('🚀 Starting agent execution...', flush=True)\n"
            "        asyncio.run(AgentExecutor().run())\n"
            "        print('✅ Agent execution completed successfully!', flush=True)\n"
            "    except Exception as e:\n"
            "        print(f'❌ Agent execution failed: {e}', flush=True)\n"
            "        raise\n"
            "    finally:\n"
            "        if container_id:\n"
            "            kill_container(container_id)\n"
            "        else:\n"
            "            print('⚠️ Could not determine container ID, cannot self-terminate', flush=True)\n"
        )
        launcher_file.write_text(launcher_code)

        # MANDATORY: Get agent file path from harness - no fallbacks!
        if not (
            hasattr(self, "harness")
            and self.harness is not None
            and hasattr(self.harness, "agent_file_path")
            and self.harness.agent_file_path
        ):
            raise RuntimeError(
                "Agent packaging requires harness.agent_file_path to be set. " "No legacy fallbacks are supported."
            )

        agent_file_path = Path(self.harness.agent_file_path)
        if not agent_file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_file_path}")

        # Copy the agent file into agent_dir
        logger.info(f"📁 Copying agent file: {agent_file_path} -> {agent_dir}")
        copied = agent_dir / agent_file_path.name
        copied.write_text(agent_file_path.read_text())

        # Set the packaged agent path for the container
        packaged_agent_path = f"/app/agent/{agent_file_path.name}"

        # Set module and class information
        agent_module = agent_file_path.stem  # filename without extension
        agent_class = self.harness.agent_class_name if self.harness and self.harness.agent_class_name else "Agent"

        # Verify the file was actually copied
        if not copied.exists():
            raise RuntimeError(f"Failed to copy agent file to {copied}")

        logger.info(f"✅ Agent file copied successfully: {copied} ({copied.stat().st_size} bytes)")

        # Verify launch.py was created
        launch_file = agent_dir / "launch.py"
        if not launch_file.exists():
            raise RuntimeError(f"Launch script not found at {launch_file}")

        logger.info(f"✅ Launch script created: {launch_file} ({launch_file.stat().st_size} bytes)")

        # Set metadata
        packaged_agent_path = f"/app/agent/{agent_file_path.name}"
        agent_module = agent_file_path.stem  # filename without extension
        agent_class = self.harness.agent_class_name if self.harness and self.harness.agent_class_name else "Agent"

        # Create metadata
        metadata = {
            "launcher": "/app/agent/launch.py",
            "packaged_agent_file": packaged_agent_path,
            "agent_module": agent_module,
            "agent_class": agent_class,
            "agent_function": None,
        }

        metadata_file = temp_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        logger.info(f"✅ Metadata created: {metadata_file}")

        logger.info(f"✅ Agent launcher prepared at: {agent_dir}")
        return agent_dir

    def _build_agent_env_hints(self, agent_package: Path) -> Dict[str, str]:
        """Build default environment hints for agent runtime.

        We default to using the launcher; actual agent discovery variables should be
        provided by the caller (AGENT_MODULE/CLASS or AGENT_FILE). This function
        sets no agent discovery vars by default to avoid the previous recursion bug.
        """
        env: Dict[str, str] = {}
        try:
            meta_path = agent_package / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                # Prefer direct file if we successfully packaged the agent source
                if meta.get("packaged_agent_file"):
                    env["AGENT_FILE"] = meta["packaged_agent_file"]
                else:
                    # Fall back to module-based hints
                    if meta.get("agent_module"):
                        env["AGENT_MODULE"] = meta["agent_module"]
                    if meta.get("agent_class"):
                        env["AGENT_CLASS"] = meta["agent_class"]
                    if meta.get("agent_function"):
                        env["AGENT_FUNCTION"] = meta["agent_function"]
        except Exception as e:
            logger.debug(f"Failed to load agent env hints: {e}")
        return env

    def _analyze_container_result(self, result: AgentExecutionResult) -> Tuple[bool, Optional[str], str]:
        """Analyze container execution result."""
        success = result.success
        flag = None
        termination_reason = result.termination_reason

        # Try to extract flag from stdout
        if result.stdout:
            try:
                # Look for JSON result in stdout
                if "AGENT_RESULT_START" in result.stdout:
                    start_idx = result.stdout.find("AGENT_RESULT_START")
                    end_idx = result.stdout.find("AGENT_RESULT_END")
                    if start_idx != -1 and end_idx != -1:
                        json_str = result.stdout[start_idx + len("AGENT_RESULT_START") : end_idx].strip()
                        agent_result = json.loads(json_str)

                        if isinstance(agent_result, dict):
                            success = agent_result.get("success", success)
                            if "result" in agent_result and isinstance(agent_result["result"], dict):
                                flag = agent_result["result"].get("flag")
            except Exception as e:
                logger.warning(f"Failed to parse agent result: {e}")

        return success, flag, termination_reason

    def _extract_step_count(self, result: AgentExecutionResult) -> int:
        """Extract step count from container result."""
        # Try to extract from stdout or use default
        try:
            if "step_count" in result.stdout:
                # Try to extract step count from logs
                pass
        except Exception:
            pass

        return 0  # Default if not found

    # UI Progress Methods - Updated for new UIManager
    async def _notify_episode_start(self, task_id: str, attempt: int) -> None:
        """Notify UI manager that episode is starting."""
        try:
            if self.ui_manager:
                task_info = {
                    "task_id": task_id,
                    "name": f"Episode {attempt}",
                    "target": "saber_domain",
                    "description": f"Starting episode {attempt} for task {task_id}",
                    "attempt": attempt,
                }
                await self.ui_manager.task_start(task_info)
        except Exception as e:
            logger.warning(f"UI notification failed (episode_start): {e}")

    async def _notify_episode_started(self, task_id: str, episode_id: str, attempt: int) -> None:
        """Notify UI manager that episode has started successfully."""
        try:
            if self.ui_manager:
                self.ui_manager.update_task_progress(task_id, 0.1)  # 10% progress for started
        except Exception as e:
            logger.warning(f"UI notification failed (episode_started): {e}")

    async def _notify_episode_complete(self, result: EpisodeResult) -> None:
        """Notify UI manager that episode has completed."""
        try:
            if self.ui_manager:
                task_result = {
                    "task_id": result.task_id,
                    "success": result.success,
                    "flag": result.flag,
                    "method": "Container execution",
                    "vulnerability": f"Task {result.task_id}",
                    "execution_time": 0.0,  # Could be enhanced to track actual time
                    "tools_used": [],
                    "attempt": result.attempt,
                    "episode_id": result.episode_id,
                    "termination_reason": result.termination_reason,
                    "iterations": result.iterations,
                    "error": result.error,
                }
                await self.ui_manager.task_complete(task_result)

                if result.success and result.flag:
                    self.ui_manager.display_message(f"🚩 Flag captured: {result.flag}", MessageType.SUCCESS)
                elif result.success:
                    self.ui_manager.display_message(
                        f"✅ Task {result.task_id} completed successfully", MessageType.SUCCESS
                    )
                else:
                    self.ui_manager.display_message(
                        f"❌ Task {result.task_id} failed: {result.termination_reason}", MessageType.ERROR
                    )
        except Exception as e:
            logger.warning(f"UI notification failed (episode_complete): {e}")

    async def cleanup(self) -> None:
        """Clean up container infrastructure."""
        logger.info("🧹 Cleaning up container infrastructure")

        try:
            # NOTE: Episode-specific progress streams are cleaned up per-episode
            # No global stream cleanup needed

            # Finalize container logging session
            if self.log_manager:
                await self.log_manager.cleanup_and_finalize()

            # Unregister session
            if self.sidecar_started:
                if self._harness_agent_id:
                    await self.sidecar_manager.unregister_agent_session(self._harness_agent_id)

            # Stop sidecar (only if we're the last session)
            await self.sidecar_manager.stop_sidecar()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        logger.info("✅ Container infrastructure cleaned up")
