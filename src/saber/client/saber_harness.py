#!/usr/bin/env python3
"""
SABER Harness - Enhanced

Enhanced harness that supports both autonomous agents with MCP integration
and traditional step-by-step agents. Provides comprehensive session management,
episode monitoring, and flexible agent execution patterns.
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from ..base import SSEEventType
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
    task_id: str = "default_task"  # Deprecated: use task_ids instead
    client_id: str = "saber-client"
    # Unified benchmark configuration
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
    - Episode status monitoring via SSE
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

        # Episode termination control
        self.episode_terminated = asyncio.Event()
        self.termination_reason: Optional[str] = None

        # SSE Event Handler Registry
        self._event_handlers = {
            SSEEventType.ENVIRONMENT_RESET: self._handle_environment_reset,
            SSEEventType.MAX_STEPS_REACHED: self._handle_termination_event,
            SSEEventType.EPISODE_TIMEOUT: self._handle_termination_event,
            SSEEventType.MANUAL_TERMINATION: self._handle_termination_event,
            SSEEventType.EPISODE_COMPLETED: self._handle_termination_event,
            SSEEventType.EPISODE_FAILED: self._handle_termination_event,
            SSEEventType.EPISODE_STARTED: self._handle_progress_event,
            SSEEventType.EPISODE_PROGRESS: self._handle_progress_event,
        }

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

    def register_event_handler(self, event_type: SSEEventType, handler: Callable) -> None:
        """Register a custom handler for a specific SSE event type."""
        self._event_handlers[event_type] = handler
        logger.debug(f"📝 Registered custom handler for {event_type}")

    def unregister_event_handler(self, event_type: SSEEventType) -> None:
        """Remove a custom handler for a specific SSE event type."""
        if event_type in self._event_handlers:
            del self._event_handlers[event_type]
            logger.debug(f"🗑️ Unregistered handler for {event_type}")

    async def run_test(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete test of the agent against SABER framework.

        Args:
            task_id: Optional task ID override

        Returns:
            Test results and metrics
        """
        try:
            # Determine execution mode based on configuration
            if self.config.task_ids or self.config.episode_attempts > 1:
                # Unified benchmark mode (custom tasks or multiple episodes)
                logger.info("🚀 Starting SABER Unified Benchmark Mode")
                if self.config.task_ids:
                    logger.info(f"🎯 Specific tasks: {self.config.task_ids}")
                else:
                    logger.info("🎯 All available tasks")
                logger.info(f"🔄 Episodes per task: {self.config.episode_attempts}")
            else:
                # Legacy single task mode (backward compatibility)
                task_id = task_id or self.config.task_id
                logger.info(f"🚀 Starting SABER Single Task Mode - Task: {task_id}")
                # Convert to benchmark format for unified handling
                self.config.task_ids = [task_id]
                self.config.episode_attempts = 1

            logger.info("=" * 60)

            if not self.rest_client:
                raise RuntimeError("REST client not initialized")

            self.session_id = await self.rest_client.create_session()

            # Always use benchmark mode now (unified approach)
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
        """Run autonomous agent execution pattern."""
        logger.info("🤖 Running autonomous agent execution pattern")

        # Create agent instance with clients if it's a class
        if inspect.isclass(self.agent):
            if self.mcp_client and self.llm_client:
                agent_instance = self.agent(self.mcp_client, self.llm_client)
            else:
                agent_instance = self.agent()
        else:
            agent_instance = self.agent

        # Build initial prompt
        initial_prompt = await self._build_initial_prompt()
        logger.info("✅ Built initial prompt with policy and constraints")

        # Execute agent with episode monitoring
        logger.info("🤖 Starting autonomous agent execution with episode monitoring...")

        # Run agent and episode monitor concurrently
        if not callable(agent_instance):
            raise RuntimeError("Agent instance is not callable")
        agent_task = asyncio.create_task(agent_instance(initial_prompt))
        monitor_task = asyncio.create_task(self._monitor_episode_status())

        # Wait for either agent completion or episode termination
        done, pending = await asyncio.wait([agent_task, monitor_task], return_when=asyncio.FIRST_COMPLETED)

        # Cancel any remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Get results
        if agent_task in done:
            result = await agent_task
            logger.info("🎯 Agent completed execution")
        else:
            result = {"success": False, "terminated_by_server": True, "termination_reason": self.termination_reason}
            logger.info(f"⏹️ Episode terminated by server: {self.termination_reason}")

        # Cleanup
        await self._cleanup_session()

        # Return comprehensive results
        final_result = {
            "success": result.get("success", False),
            "flag": result.get("flag"),
            "iterations": result.get("iterations", 0),
            "max_iterations": result.get("max_iterations", 20),
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "terminated_by_server": result.get("terminated_by_server", False),
            "termination_reason": result.get("termination_reason"),
        }

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

    async def _monitor_episode_status(self) -> None:
        """Monitor episode status via Server-Sent Events for termination signals."""
        if not self.session_id:
            logger.warning("⚠️ Cannot monitor episode: missing session ID")
            return

        if not self.rest_client:
            logger.warning("⚠️ Cannot monitor episode: REST client not initialized")
            return

        if not self.episode_id:
            logger.warning("⚠️ Cannot monitor episode: missing episode ID")
            return

        # Monitor the current episode (works for both regular and benchmark modes)
        await self.rest_client.monitor_episode_events(
            episode_id=self.episode_id, session_id=self.session_id, event_handler=self._handle_episode_event
        )

    async def _handle_episode_event(self, event_data: Dict[str, Any]) -> None:
        """Handle incoming episode events from the server using event handler registry."""
        event_type = event_data.get("type")

        # Look up handler in registry
        if event_type is not None:
            handler = self._event_handlers.get(event_type)

            if handler:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(f"❌ Error handling {event_type} event: {e}")
            else:
                # Unknown event type - just log it
                logger.debug(f"📡 Unknown episode event: {event_type} - {event_data.get('message', '')}")

    async def _handle_termination_event(self, event_data: Dict[str, Any]) -> None:
        """Handle episode termination events."""
        event_type = event_data.get("type")
        self.termination_reason = event_data.get("reason", f"Episode terminated: {event_type}")
        logger.info(f"🛑 Episode termination signal received: {self.termination_reason}")
        self.episode_terminated.set()

    async def _handle_progress_event(self, event_data: Dict[str, Any]) -> None:
        """Handle episode progress/informational events."""
        event_type = event_data.get("type")
        message = event_data.get("message", "")
        logger.debug(f"📡 Episode event: {event_type} - {message}")

    async def _handle_environment_reset(self, event_data: Dict[str, Any]) -> None:
        """Handle environment reset events from server."""
        data = event_data.get("data", {})
        context_hint = data.get("context_hint", "Environment has changed")

        logger.info(f"🔄 {context_hint}")

        # Reset agent state if the agent supports it
        if self.agent is not None and hasattr(self.agent, "reset_for_environment_change"):
            try:
                await self.agent.reset_for_environment_change(data)
                logger.info("✅ Agent state reset for environment change")
            except Exception as e:
                logger.warning(f"⚠️ Failed to reset agent state: {e}")
        elif self.agent is not None and hasattr(self.agent, "reset_conversation_state"):
            try:
                self.agent.reset_conversation_state()
                logger.info("✅ Agent conversation state reset")
            except Exception as e:
                logger.warning(f"⚠️ Failed to reset agent conversation: {e}")
        else:
            logger.info("ℹ️ Agent does not support automatic state reset")

        # Optionally refresh task context
        try:
            if self.rest_client and self.session_id:
                # Get fresh policy/context for new environment
                await self.rest_client.get_policy_info(self.session_id)
                logger.debug("🔄 Refreshed policy context for new environment")
        except Exception as e:
            logger.warning(f"⚠️ Failed to refresh policy context: {e}")

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
