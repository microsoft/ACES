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
from typing import Any, Dict, Optional
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
    request_timeout: float = 30.0
    log_level: str = "INFO"
    task_id: str = "default_task"
    client_id: str = "saber-client"
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

    async def run_test(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete test of the agent against SABER framework.

        Args:
            task_id: Optional task ID override

        Returns:
            Test results and metrics
        """
        try:
            task_id = task_id or self.config.task_id
            logger.info(f"🚀 Starting SABER Agent Test - Task: {task_id}")
            logger.info("=" * 60)

            if not self.rest_client:
                raise RuntimeError("REST client not initialized")

            self.session_id = await self.rest_client.create_session()
            self.episode_id = await self.rest_client.start_episode(task_id)

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
        if not self.session_id or not self.episode_id:
            logger.warning("⚠️ Cannot monitor episode: missing session/episode ID")
            return

        if not self.rest_client:
            logger.warning("⚠️ Cannot monitor episode: REST client not initialized")
            return

        # Use REST client for episode monitoring
        await self.rest_client.monitor_episode_events(
            episode_id=self.episode_id, session_id=self.session_id, event_handler=self._handle_episode_event
        )

    async def _handle_episode_event(self, event_data: Dict[str, Any]) -> None:
        """Handle incoming episode events from the server."""
        event_type = event_data.get("type")

        if self._should_terminate_episode(event_data):
            self.termination_reason = event_data.get("reason", f"Episode terminated: {event_type}")
            logger.info(f"🛑 Episode termination signal received: {self.termination_reason}")
            self.episode_terminated.set()
        else:
            logger.debug(f"📡 Episode event: {event_type} - {event_data.get('message', '')}")

    def _should_terminate_episode(self, event_data: Dict[str, Any]) -> bool:
        """Determine if episode should be terminated based on server event."""
        termination_events = {
            "max_steps_reached",
            "episode_timeout",
            "manual_termination",
            "episode_completed",
            "episode_failed",
        }
        return event_data.get("type") in termination_events

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
