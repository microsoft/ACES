#!/usr/bin/env python3
"""
SABER Harness - Enhanced

Enhanced harness that supports both autonomous agents with MCP integration
and traditional step-by-step agents. Provides comprehensive session management,
episode monitoring, and flexible agent execution patterns.
"""

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .api import SABERMCPClient, SABERRestClient
from .llm import create_llm_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set specific loggers to reduce clutter
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class SABERHarnessConfig:
    """Configuration for SABER harness."""

    server_url: str = "http://localhost:8000"
    mcp_url: str = "http://localhost:8001"
    max_steps: int = 100
    request_timeout: float = 300.0  # Increased to 5 minutes for agent execution
    log_level: str = "INFO"
    client_id: str = "saber-client"
    # Benchmark configuration
    task_ids: Optional[List[str]] = None  # Specific tasks to run (None = all available)
    episode_attempts: int = 1  # Number of episodes per task
    llm_provider: Optional[str] = None  # Auto-detect if None
    llm_config: Dict[str, Any] = field(default_factory=dict)
    agent_config: Dict[str, Any] = field(default_factory=dict)


class SABERHarness:
    """
    Enhanced SABER harness with autonomous agent support.

    Supports two execution patterns:
    1. Autonomous agents: Agent handles MCP tool execution autonomously
    2. Traditional agents: Harness manages step-by-step execution

    Features:
    - Session and episode management via REST API
    - MCP integration for secure tool execution
    - LLM client configuration and injection with factory pattern
    - Comprehensive logging and error handling
    - Flexible agent adaptation
    """

    def __init__(self, config: Optional[SABERHarnessConfig] = None):
        """Initialize harness with configuration."""
        self.config = config or SABERHarnessConfig()

        # Parse URLs to extract host and port
        rest_parsed = urlparse(self.config.server_url)
        mcp_parsed = urlparse(self.config.mcp_url)

        self.rest_host = rest_parsed.hostname or "localhost"
        self.rest_port = rest_parsed.port or 8000
        self.mcp_host = mcp_parsed.hostname or "localhost"
        self.mcp_port = mcp_parsed.port or 8001

        # SABER framework state
        self.session_id: Optional[str] = None
        self.episode_id: Optional[str] = None

        # API Clients
        self.rest_client: Optional[SABERRestClient] = None
        self.mcp_client: Optional[SABERMCPClient] = None
        self.llm_client: Optional[Any] = None
        self.agent: Optional[Any] = None

        # Configure logging
        logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper()))

    async def initialize(self, agent: Any, env_file: Optional[Path] = None) -> None:
        """
        Initialize the harness with an agent and optional environment file.

        Args:
            agent: The agent to test (any callable with proper interface)
            env_file: Optional path to .env file for LLM configuration
        """
        logger.info("🚀 Initializing SABER harness")

        self.agent = agent

        self.rest_client = SABERRestClient(
            base_url=self.config.server_url,
            client_id=self.config.client_id,
            request_timeout=self.config.request_timeout,
        )

        self.llm_client = create_llm_client(
            provider=self.config.llm_provider, env_file=env_file, **self.config.llm_config
        )

        if self.llm_client:
            logger.info("✅ LLM client initialized")
        else:
            logger.warning("⚠️ No LLM client configuration found")

        logger.info("✅ SABER harness initialized successfully")

    async def run_test(self) -> Dict[str, Any]:
        """
        Run a complete test of the agent against SABER framework in benchmark mode.

        Returns:
            Test results and metrics
        """
        try:
            # Always run in benchmark mode - no legacy modes
            logger.info("🚀 Starting SABER Benchmark Mode")
            if self.config.task_ids:
                logger.info(f"🎯 Specific tasks: {self.config.task_ids}")
            else:
                logger.info("🎯 All available tasks")
            logger.info(f"🔄 Episodes per task: {self.config.episode_attempts}")
            logger.info("=" * 60)

            if not self.rest_client:
                raise RuntimeError("REST client not initialized")

            self.session_id = await self.rest_client.create_session()

            # Start benchmark with all available tasks or specified tasks
            benchmark_data = await self.rest_client.start_benchmark(
                task_ids=self.config.task_ids, episode_attempts=self.config.episode_attempts
            )

            # Debug: Log the benchmark response structure
            logger.debug(f"🔍 Benchmark response: {benchmark_data}")

            # Extract the initial episode ID from benchmark response
            self.episode_id = benchmark_data.get("current_episode_id")

            # If not found at top level, check other possible locations
            if not self.episode_id:
                # Try nested in benchmark_session
                benchmark_session = benchmark_data.get("benchmark_session", {})
                self.episode_id = benchmark_session.get("current_episode_id")

            # Log benchmark details
            benchmark_session = benchmark_data.get("benchmark_session", {})
            tasks = benchmark_session.get("tasks", [])
            logger.info(f"📋 Benchmark Tasks: {len(tasks)}")
            for task in tasks:
                logger.info(f"   - {task['task_id']}: {task['title']}")
            current_task = benchmark_session.get("current_task_id")
            if current_task:
                logger.info(f"🚀 Starting with task: {current_task}")
                task_id = current_task  # Use the current benchmark task for MCP client

            if self.episode_id:
                logger.info(f"📝 Initial episode ID: {self.episode_id}")
            else:
                logger.warning("⚠️ No episode ID returned from benchmark start")
                logger.debug(f"🔍 Available keys in response: {list(benchmark_data.keys())}")
                if benchmark_session:
                    logger.debug(f"🔍 Available keys in benchmark_session: {list(benchmark_session.keys())}")

            self.mcp_client = SABERMCPClient(
                base_url=f"http://{self.mcp_host}:{self.mcp_port}",
                session_id=self.session_id,
                task_id=task_id,
                client_id=self.config.client_id,
            )

            # Connect to MCP server
            await self.mcp_client.connect()

            if hasattr(self.agent, "__call__") and self._is_autonomous_agent():
                return await self._run_autonomous_agent()
            else:
                return await self._run_traditional_agent()

        except Exception as e:
            logger.error(f"❌ SABER harness failed: {e}")
            await self._cleanup_session()
            return {"success": False, "error": str(e), "terminated_by_server": False}

    def _is_autonomous_agent(self) -> bool:
        """Check if agent supports autonomous execution pattern."""
        # Check if agent accepts MCP client and LLM client in constructor or has callable interface
        if hasattr(self.agent, "__call__") and callable(self.agent):
            import inspect

            sig = inspect.signature(self.agent.__call__)
            params = list(sig.parameters.keys())
            # Remove 'self' if present
            if params and params[0] == "self":
                params = params[1:]
            # Autonomous agents typically accept a prompt string
            return len(params) >= 1
        return False

    async def _run_autonomous_agent(self) -> Dict[str, Any]:
        """Run autonomous agent execution pattern with multi-task benchmark support."""
        logger.info("🤖 Running autonomous agent execution pattern")

        # Track benchmark state
        benchmark_complete = False
        all_results = []
        task_count = 0

        while not benchmark_complete:
            task_count += 1
            logger.info(f"🎯 Starting task {task_count} - Episode ID: {self.episode_id}")

            # Create fresh agent instance for each task if it's a class
            if inspect.isclass(self.agent):
                if self.mcp_client and self.llm_client:
                    agent_instance = self.agent(self.mcp_client, self.llm_client)
                else:
                    agent_instance = self.agent()
            else:
                agent_instance = self.agent

            # Build initial prompt for this task
            initial_prompt = await self._build_initial_prompt()
            logger.info(f"✅ Built initial prompt for task {task_count}")

            # Execute agent
            logger.info(f"🤖 Starting agent execution for task {task_count}...")

            # Run agent
            if not callable(agent_instance):
                raise RuntimeError("Agent instance is not callable")

            result = await agent_instance(initial_prompt)
            logger.info(f"🎯 Agent completed execution for task {task_count}")

            # Store task result
            task_result = {
                "task_number": task_count,
                "success": result.get("success", False),
                "flag": result.get("flag"),
                "iterations": result.get("iterations", 0),
                "max_iterations": result.get("max_iterations", 20),
                "episode_id": self.episode_id,
                "terminated_by_server": result.get("terminated_by_server", False),
                "termination_reason": result.get("termination_reason"),
            }
            all_results.append(task_result)

            # Check if benchmark continues by polling server for next task
            try:
                # Get current task info to check if benchmark continues
                if self.rest_client:
                    current_task = await self.rest_client.get_task_info(self.session_id)
                    if current_task and current_task.get("task_id") != task_result.get("task_id"):
                        # Server has assigned a new task, continue benchmark
                        logger.info(f"🔄 Advancing to next task - Task ID: {current_task.get('task_id')}")
                        if current_task.get("episode_id"):
                            self.episode_id = current_task["episode_id"]
                    else:
                        # No new task, benchmark is complete
                        benchmark_complete = True
                        logger.info(f"🏁 Benchmark completed after {task_count} tasks")
                else:
                    # No REST client, stop benchmark
                    benchmark_complete = True
                    logger.warning("⚠️ No REST client available to check benchmark continuation")
            except Exception as e:
                logger.warning(f"⚠️ Failed to check benchmark continuation: {e}")
                benchmark_complete = True

        # Cleanup
        await self._cleanup_session()

        # Return comprehensive results
        final_result = {
            "success": any(task["success"] for task in all_results),
            "tasks_completed": len(all_results),
            "all_task_results": all_results,
            "session_id": self.session_id,
            "benchmark_complete": True,
        }

        # Include flag from successful tasks
        successful_flags = [task["flag"] for task in all_results if task["success"] and task["flag"]]
        if successful_flags:
            final_result["flag"] = successful_flags[-1]  # Use last successful flag
            final_result["all_flags"] = successful_flags

        return final_result

    async def _run_traditional_agent(self) -> Dict[str, Any]:
        """Run traditional step-by-step agent execution pattern."""
        logger.info("🔄 Running traditional step-by-step agent execution pattern")

        # TODO: Implement traditional agent execution pattern
        # This would follow the original SABER client pattern with:
        # - Step-by-step prompt building
        # - Agent response processing
        # - Server step execution
        # - Response handling and next prompt building

        raise NotImplementedError("Traditional agent execution pattern not yet implemented")

    async def _build_initial_prompt(self) -> str:
        """Get initial prompt from server policy endpoint."""
        if not self.session_id:
            raise Exception("No active session")

        if not self.rest_client:
            raise RuntimeError("REST client not initialized")

        # Get policy data from server which contains the prompt
        policy_data = await self.rest_client.get_policy_info(self.session_id)

        # The policy now contains just the prompt
        prompt_value = policy_data.get("prompt", "")
        if not prompt_value:
            raise Exception("No prompt found in policy data")

        return str(prompt_value)

    async def _cleanup_session(self) -> None:
        """Clean up SABER session and clients."""
        try:
            # Disconnect MCP client
            if self.mcp_client:
                await self.mcp_client.disconnect()
                self.mcp_client = None

            # Terminate session via REST client
            if self.rest_client and self.session_id:
                await self.rest_client.terminate_session(self.session_id)

            self.session_id = None
            self.episode_id = None

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
