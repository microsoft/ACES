#!/usr/bin/env python3
"""
Container-based Episode Executor

Replaces the embedded MCP approach with containerized agent execution.
Handles agent packaging, container orchestration, and multi-channel termination monitoring.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .api import SABERRestClient
from .containers import AgentManager, ContainerFactory, SidecarManager
from .containers.agent_manager import AgentExecutionResult
from .harness_models import EpisodeResult

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

    def __init__(self, rest_client: SABERRestClient, session_id: str, parallelism: int = 1):
        """Initialize container episode executor."""
        self.rest_client = rest_client
        self.session_id = session_id
        self.parallelism = parallelism

        # Container managers
        self.sidecar_manager = SidecarManager()
        self.agent_manager = AgentManager()
        self.container_factory = ContainerFactory()

        # State
        self.sidecar_started = False
        self.active_episodes: Dict[str, asyncio.Task] = {}
        self._harness_agent_id: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize container infrastructure."""
        logger.info("🔧 Initializing container episode executor")

        # Start shared MCP sidecar
        await self.sidecar_manager.start_sidecar()
        self.sidecar_started = True

        # Register session with sidecar
        # Register harness agent session with sidecar for monitoring (optional)
        self._harness_agent_id = await self.sidecar_manager.register_agent_session(
            session_id=self.session_id, task_id=None, agent_id="saber-harness"
        )

        logger.info("✅ Container infrastructure initialized")

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

        try:
            # Step 1: Start episode on server
            episode_id = await self.rest_client.start_episode(task_id)
            logger.debug(f"📍 Started episode: {episode_id}")

            # Step 2: Get task-specific policy
            policy_data = await self.rest_client.get_policy_info(session_id=self.session_id, task_id=task_id)
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

            return EpisodeResult(
                task_id=task_id,
                episode_id=episode_id,
                attempt=attempt,
                success=success,
                flag=flag,
                termination_reason=termination_reason,
                iterations=self._extract_step_count(container_result),
            )

        except Exception as e:
            logger.error(f"Container episode execution failed: {e}")
            return EpisodeResult(
                task_id=task_id, episode_id="", attempt=attempt, success=False, termination_reason="error", error=str(e)
            )

    async def _execute_agent_container(
        self, task_id: str, episode_id: str, agent_package: Path, initial_prompt: str
    ) -> AgentExecutionResult:
        """Execute agent in container with multi-channel termination monitoring."""
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

    async def _monitor_episode_termination(self, episode_id: str, container_task: asyncio.Task) -> None:
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

        Strategy: create a tiny launcher that simply runs AgentExecutor; the actual
        agent code should be mounted separately or specified via env hints. This
        avoids embedding a recursive wrapper as the agent itself.
        """
        logger.info("📦 Preparing agent launcher for container execution")

        temp_dir = Path(tempfile.mkdtemp(prefix="saber-agent-"))

        # Create a minimal launcher script that invokes the AgentExecutor
        launcher_file = temp_dir / "launch.py"
        launcher_code = (
            "import sys, asyncio\n"
            "sys.path.insert(0, '/agent_runtime')\n"
            "from saber.client.agent_runtime import AgentExecutor\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    asyncio.run(AgentExecutor().run())\n"
        )
        launcher_file.write_text(launcher_code)

        # Attempt to package the agent's own source file for file-based loading
        packaged_agent_path: Optional[str] = None
        agent_module: Optional[str] = None
        agent_class: Optional[str] = None
        agent_function: Optional[str] = None

        try:
            import inspect

            # Support function-based agents
            if callable(agent) and hasattr(agent, "__name__") and hasattr(agent, "__module__"):
                src = inspect.getsourcefile(agent) or inspect.getfile(agent)
                agent_module = getattr(agent, "__module__", None)
                agent_function = getattr(agent, "__name__", None)
            else:
                # Likely a class instance
                acls = getattr(agent, "__class__", None)
                if acls is not None:
                    src = inspect.getsourcefile(acls) or inspect.getfile(acls)
                    agent_module = getattr(acls, "__module__", None)
                    agent_class = getattr(acls, "__name__", None)
                else:
                    src = None

            if src and Path(src).exists():
                # Copy source file into package
                src_path = Path(src)
                pkg_dir = temp_dir / "user_agent"
                pkg_dir.mkdir(parents=True, exist_ok=True)
                copied = pkg_dir / src_path.name
                copied.write_text(Path(src).read_text())
                packaged_agent_path = f"/app/agent/user_agent/{src_path.name}"
        except Exception as e:
            logger.debug(f"Agent source packaging skipped: {e}")

        # Metadata to help build env hints
        metadata = {
            "launcher": "/app/agent/launch.py",
            "packaged_agent_file": packaged_agent_path,
            "agent_module": agent_module,
            "agent_class": agent_class,
            "agent_function": agent_function,
        }
        (temp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info(f"✅ Agent launcher prepared at: {temp_dir}")
        return temp_dir

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

    async def cleanup(self) -> None:
        """Clean up container infrastructure."""
        logger.info("🧹 Cleaning up container infrastructure")

        try:
            # Unregister session
            if self.sidecar_started:
                if self._harness_agent_id:
                    await self.sidecar_manager.unregister_agent_session(self._harness_agent_id)

            # Stop sidecar (only if we're the last session)
            await self.sidecar_manager.stop_sidecar()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        logger.info("✅ Container infrastructure cleaned up")
