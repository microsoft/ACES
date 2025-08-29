#!/usr/bin/env python3
"""
SABER Harness - Implementation Plan

Clean harness implementation that orchestrates episodes client-side
for better control and parallelization support.

Implements the requirements from the implementation plan:
- R1: Universal agent adapter without code changes
- R2/R2b: Task selection (supplied vs auto-fetch all)
- R3: Episode initialization with task-specific policy
- R4: Native MCP client exposure
- R5/R6: Step counting and termination detection
- R7: Thread-based concurrency
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_wrapper import AgentWrapper
from .api import MCPClient, SABERMCPClient, SABERRestClient
from .harness_models import EpisodeResult, HarnessRunResult, SABERHarnessConfig
from .llm import create_llm_client
from .llm.termination_aware_client import TerminationAwareLLMClient

logger = logging.getLogger(__name__)


class SABERHarness:
    """
    Client-orchestrated SABER harness implementation.

    Features:
    - Universal agent adapter (R1)
    - Task selection (supplied vs auto-fetch) (R2/R2b)
    - Episode-level policy prompts (R3)
    - Native MCP client exposure (R4)
    - Termination detection and step counting (R5/R6)
    - Thread-based parallelism (R7)
    """

    def __init__(self, config: Optional[SABERHarnessConfig] = None):
        """Initialize harness with configuration."""
        self.config = config or SABERHarnessConfig()

        # Core state
        self.session_id: Optional[str] = None
        self.rest_client: Optional[SABERRestClient] = None
        self.llm_client: Optional[Any] = None
        self.agent: Optional[Any] = None

        # Configure logging
        logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper()))

    async def initialize(self, agent: Any, env_file: Optional[Path] = None) -> None:
        """Initialize the harness with an agent and optional environment file."""
        logger.info("🚀 Initializing SABER harness")

        self.agent = agent

        # Initialize REST client
        self.rest_client = SABERRestClient(
            base_url=self.config.server_url,
            client_id=self.config.client_id,
            request_timeout=self.config.request_timeout,
        )

        # Initialize optional LLM client
        self.llm_client = create_llm_client(
            provider=self.config.llm_provider, env_file=env_file, **self.config.llm_config
        )

        if self.llm_client:
            logger.info("✅ LLM client initialized")
        else:
            logger.info("ℹ️ No LLM client configuration found")

        logger.info("✅ SABER harness initialized successfully")

    async def run(self) -> HarnessRunResult:
        """Run complete test with client-orchestrated episodes."""
        try:
            logger.info("🚀 Starting SABER benchmark")
            logger.info(f"🎯 Parallelism: {self.config.parallelism}")

            if self.config.task_ids:
                logger.info(f"🎯 Specific tasks: {self.config.task_ids}")
            else:
                logger.info("🎯 All available tasks")

            logger.info("=" * 60)

            if not self.rest_client:
                raise RuntimeError("REST client not initialized")

            # Step 1: Create session
            self.session_id = await self.rest_client.create_session()
            logger.info(f"📍 Created session: {self.session_id}")

            # Step 2: Determine tasks
            tasks = await self._resolve_tasks()
            logger.info(f"📋 Resolved {len(tasks)} tasks for execution")

            # Step 3: Build episode queue (one attempt per task per requirement)
            episodes = [(task["task_id"], 1) for task in tasks]
            logger.info(f"🎬 Prepared {len(episodes)} episodes")

            # Step 4: Execute episodes with parallelism
            all_results = await self._execute_episodes(episodes)

            # Step 5: Cleanup session
            await self._cleanup_session()

            # Step 6: Return results
            successful_episodes = [r for r in all_results if r.success]
            return HarnessRunResult(
                session_id=self.session_id,
                total_episodes=len(all_results),
                successful_episodes=len(successful_episodes),
                episode_results=all_results,
                success=len(successful_episodes) > 0,
            )

        except Exception as e:
            logger.error(f"❌ SABER harness failed: {e}")
            await self._cleanup_session()
            raise

    async def _resolve_tasks(self) -> List[Dict[str, Any]]:
        """Resolve tasks according to R2/R2b."""
        assert self.rest_client is not None  # Ensured by run() method
        if self.config.task_ids:
            # R2: Use supplied tasks
            all_tasks = await self.rest_client.list_tasks()
            filtered_tasks = [task for task in all_tasks if task["task_id"] in self.config.task_ids]

            if not filtered_tasks:
                raise ValueError(f"No matching tasks found for: {self.config.task_ids}")

            missing_tasks = set(self.config.task_ids) - {t["task_id"] for t in filtered_tasks}
            if missing_tasks:
                logger.warning(f"⚠️ Tasks not found: {missing_tasks}")

            return filtered_tasks
        else:
            # R2b: Fetch and run all tasks (non-interactive default)
            tasks = await self.rest_client.list_tasks()
            if not tasks:
                logger.warning("⚠️ No tasks available")
            return tasks

    async def _execute_episodes(self, episodes: List[tuple[str, int]]) -> List[EpisodeResult]:
        """Execute episodes with thread-based parallelism."""
        if self.config.parallelism == 1:
            # Sequential execution
            results = []
            for i, (task_id, attempt) in enumerate(episodes):
                logger.info(f"\n🎬 Episode {i+1}/{len(episodes)}: {task_id} (attempt {attempt})")
                result = await self._run_single_episode(task_id, attempt)
                results.append(result)
                logger.info(f"{'✅' if result.success else '❌'} Episode complete: {result.termination_reason}")
            return results
        else:
            # Parallel execution using threads
            return await self._run_episodes_parallel(episodes)

    async def _run_episodes_parallel(self, episodes: List[tuple[str, int]]) -> List[EpisodeResult]:
        """Run episodes in parallel using thread pool."""
        logger.info(f"🧵 Starting parallel execution with {self.config.parallelism} workers")

        results = []

        def run_episode_sync(task_id: str, attempt: int) -> EpisodeResult:
            """Synchronous wrapper for episode execution."""
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._run_single_episode(task_id, attempt))
            finally:
                loop.close()

        # Execute episodes using thread pool
        with ThreadPoolExecutor(max_workers=self.config.parallelism) as executor:
            # Submit all episodes
            future_to_episode = {
                executor.submit(run_episode_sync, task_id, attempt): (task_id, attempt) for task_id, attempt in episodes
            }

            # Collect results as they complete
            for future in as_completed(future_to_episode):
                task_id, attempt = future_to_episode[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"{'✅' if result.success else '❌'} Episode {task_id} complete: {result.termination_reason}"
                    )
                except Exception as e:
                    logger.error(f"❌ Episode {task_id} failed: {e}")
                    results.append(
                        EpisodeResult(
                            task_id=task_id,
                            episode_id="",
                            attempt=attempt,
                            success=False,
                            termination_reason="error",
                            error=str(e),
                        )
                    )

        return results

    async def _run_single_episode(self, task_id: str, attempt: int) -> EpisodeResult:
        """Run a single episode with clean termination monitoring."""
        assert self.rest_client is not None  # Ensured by run() method
        logger.debug(f"🎬 Starting episode {attempt} for task {task_id}")

        try:
            # Step 3a: Start episode
            episode_id = await self.rest_client.start_episode(task_id)
            logger.debug(f"📍 Started episode: {episode_id}")

            # Step 3b: Get task-specific policy
            policy_data = await self.rest_client.get_policy_info(session_id=self.session_id, task_id=task_id)
            initial_prompt = str(policy_data.get("prompt", ""))
            if not initial_prompt:
                raise Exception("No prompt found in policy data")

            logger.debug(f"📜 Retrieved policy prompt for task {task_id}")

            # Step 3c: Connect MCP client for this episode
            mcp_client = await self._create_mcp_client(task_id)

            try:
                # Step 3d: Run agent with proper termination monitoring
                result = await self._execute_agent_with_termination(mcp_client, initial_prompt, task_id, episode_id)

                # Step 3e: Analyze results
                success, flag, termination_reason = self._analyze_episode_result(result, mcp_client)

                return EpisodeResult(
                    task_id=task_id,
                    episode_id=episode_id,
                    attempt=attempt,
                    success=success,
                    flag=flag,
                    termination_reason=termination_reason,
                    iterations=getattr(mcp_client, "_step_count", 0),
                )

            finally:
                # Step 3f: Cleanup MCP connection
                await mcp_client.disconnect()

        except Exception as e:
            logger.error(f"Episode execution failed: {e}")
            return EpisodeResult(
                task_id=task_id, episode_id="", attempt=attempt, success=False, termination_reason="error", error=str(e)
            )

    async def _create_mcp_client(self, task_id: str) -> SABERMCPClient:
        """Create and connect MCP client for episode."""
        # Create raw MCP client
        raw_mcp_client = MCPClient(
            base_url=self.config.mcp_url,
            session_id=self.session_id,
            task_id=task_id,
            client_id=self.config.client_id,
        )

        # Wrap with SABER monitoring
        saber_mcp_client = SABERMCPClient(raw_mcp_client)

        # Connect
        await saber_mcp_client.connect()
        logger.debug(f"🔗 Connected to MCP server for task {task_id}")

        # Get task configuration to determine max steps for proactive termination
        assert self.rest_client is not None  # Ensured by run() method
        try:
            # Try to get task info from server to determine step limits
            tasks = await self.rest_client.list_tasks()
            task_info = next((t for t in tasks if t["task_id"] == task_id), None)

            if task_info:
                # Use task-specific step limit if available, otherwise use default
                max_steps = task_info.get("max_steps", 3)  # Default from server logs shows 3
                saber_mcp_client.set_max_steps(max_steps)
                logger.info(f"🎯 Configured proactive termination for task {task_id}: {max_steps} steps")
            else:
                # Fallback to reasonable default
                saber_mcp_client.set_max_steps(3)
                logger.warning(f"⚠️ No task info found for {task_id}, using default 3 steps")

        except Exception as e:
            logger.warning(f"⚠️ Failed to get task step limits: {e}, using default 3 steps")
            saber_mcp_client.set_max_steps(3)

        return saber_mcp_client

    async def _execute_agent_with_termination(
        self, mcp_client: SABERMCPClient, initial_prompt: str, task_id: str, episode_id: str
    ) -> Any:
        """Execute agent with proactive termination monitoring."""
        # Wrap LLM client with termination checking
        termination_aware_llm = None
        if self.llm_client:
            termination_aware_llm = TerminationAwareLLMClient(
                wrapped_llm_client=self.llm_client, termination_check=lambda: mcp_client.episode_terminated
            )
            logger.info("🧠 LLM client wrapped with termination checking")

        # Create agent wrapper
        agent_wrapper = AgentWrapper(
            agent=self.agent, mcp_client=mcp_client, llm_client=termination_aware_llm or self.llm_client
        )

        logger.debug(f"🤖 Starting agent execution for episode {episode_id}")

        # Simplified monitoring - MCP client handles proactive termination
        async def monitor_safety_limits() -> None:
            """Monitor only for client safety limits (last resort)."""
            while True:
                # Check client safety cap (last resort, should never be reached)
                step_count = mcp_client.step_count
                if step_count >= self.config.max_steps_client_safety:
                    logger.warning(f"⚠️ Client safety limit reached: {step_count}")
                    agent_wrapper.set_shutdown()
                    break

                # Check if agent task completed or was cancelled
                if agent_wrapper._agent_task and agent_wrapper._agent_task.done():
                    break

                await asyncio.sleep(0.1)

        # Run agent and monitor concurrently
        monitor_task = asyncio.create_task(monitor_safety_limits())

        try:
            # Run agent - MCP client will handle proactive termination
            agent_result = await agent_wrapper.run(initial_prompt)

            # Cancel monitoring
            monitor_task.cancel()

            return agent_result

        except asyncio.CancelledError:
            # Agent was cancelled due to episode termination (expected behavior)
            logger.info("🛑 Agent cancelled due to episode termination")
            if termination_aware_llm:
                logger.info(f"📊 LLM calls attempted before termination: {termination_aware_llm.call_count}")
            monitor_task.cancel()
            return {"success": False, "flag": None, "reason": "episode_terminated", "terminated": True}
        except Exception:
            monitor_task.cancel()
            raise
        finally:
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    def _analyze_episode_result(self, agent_result: Any, mcp_client: SABERMCPClient) -> tuple[bool, Optional[str], str]:
        """Analyze episode result per R5/R6."""
        success = False
        flag = None

        # Determine termination reason with priority
        if mcp_client.episode_terminated:
            termination_reason = mcp_client.termination_reason or "server_terminated"
        elif getattr(mcp_client, "_step_count", 0) >= self.config.max_steps_client_safety:
            termination_reason = "max_steps_client_safety"
        elif agent_result and isinstance(agent_result, dict) and agent_result.get("terminated"):
            # Agent was cancelled due to framework termination
            termination_reason = agent_result.get("reason", "framework_cancelled")
        else:
            termination_reason = "completed"

        # Extract success and flag from agent result
        if agent_result and isinstance(agent_result, dict):
            success = agent_result.get("success", False)
            flag = agent_result.get("flag")

        return success, flag, termination_reason

    async def _cleanup_session(self) -> None:
        """Clean up SABER session."""
        try:
            if self.rest_client and self.session_id:
                await self.rest_client.terminate_session(self.session_id)
                logger.info(f"🧹 Cleaned up session: {self.session_id}")
            self.session_id = None
        except Exception as e:
            logger.error(f"❌ Session cleanup failed: {e}")
