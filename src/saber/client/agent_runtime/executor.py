#!/usr/bin/env python3
"""
Agent Executor

Main executor that runs inside agent containers. Handles agent discovery,
loading, MCP client initialization, and execution with result collection.

This executor runs as the main entry point inside agent containers and
provides a standard interface for any agent implementation.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters import AgentAdapter, create_agent_adapter
from .mcp_factory import MCPClientFactory

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Main agent executor for container runtime.

    Responsible for:
    - Agent discovery and loading from environment
    - MCP client initialization using standard libraries
    - Agent execution with proper error handling
    - Result serialization and output
    - Graceful shutdown handling
    """

    def __init__(self) -> None:
        """Initialize agent executor."""
        self.agent: Any = None
        self.mcp_client: Any = None
        self.agent_adapter: Optional[AgentAdapter] = None
        self.shutdown_requested = False

    async def run(self) -> Dict[str, Any]:
        """
        Main execution entry point.

        Returns:
            Dict containing execution results
        """
        try:
            logger.info("🚀 Starting agent executor")

            # Step 1: Discover and load agent
            self.agent = await self._discover_and_load_agent()
            logger.info("✅ Agent loaded successfully")

            # Step 2: Initialize MCP client
            self.mcp_client = await self._initialize_mcp_client()
            logger.info("✅ MCP client connected")

            # Step 3: Create agent adapter
            self.agent_adapter = create_agent_adapter(agent=self.agent, mcp_client=self.mcp_client)
            logger.info("✅ Agent adapter created")

            # Step 4: Get initial prompt from environment
            initial_prompt = self._get_initial_prompt()
            logger.info("📜 Retrieved initial prompt")

            # Step 5: Execute agent
            result = await self._execute_agent(initial_prompt)
            logger.info("✅ Agent execution completed")

            return {"success": True, "result": result, "error": None}

        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "result": None, "error": str(e), "traceback": traceback.format_exc()}
        finally:
            await self._cleanup()

    async def _discover_and_load_agent(self) -> Any:
        """
        Discover and load agent from environment variables.

        Supports multiple discovery patterns:
        - AGENT_MODULE: Python module import path
        - AGENT_CLASS: Class name within module
        - AGENT_FUNCTION: Function name within module
        - AGENT_FILE: Direct file path to agent code
        """
        agent_module = os.getenv("AGENT_MODULE")
        agent_class = os.getenv("AGENT_CLASS")
        agent_function = os.getenv("AGENT_FUNCTION")
        agent_file = os.getenv("AGENT_FILE")

        if agent_file:
            return await self._load_agent_from_file(agent_file, agent_class, agent_function)
        elif agent_module:
            return await self._load_agent_from_module(agent_module, agent_class, agent_function)
        else:
            raise ValueError("No agent specification found. Set AGENT_MODULE or AGENT_FILE environment variable.")

    async def _load_agent_from_file(
        self, file_path: str, class_name: Optional[str] = None, function_name: Optional[str] = None
    ) -> Any:
        """Load agent from Python file."""
        import importlib.util

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Agent file not found: {file_path}")

        # Load module from file
        spec = importlib.util.spec_from_file_location("agent_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extract agent from module
        return self._extract_agent_from_module(module, class_name, function_name)

    async def _load_agent_from_module(
        self, module_path: str, class_name: Optional[str] = None, function_name: Optional[str] = None
    ) -> Any:
        """Load agent from Python module."""
        import importlib

        try:
            module = importlib.import_module(module_path)
            return self._extract_agent_from_module(module, class_name, function_name)
        except ImportError as e:
            raise ImportError(f"Could not import module {module_path}: {e}")

    def _extract_agent_from_module(
        self, module: Any, class_name: Optional[str] = None, function_name: Optional[str] = None
    ) -> Any:
        """Extract agent object from loaded module."""
        if function_name:
            # Return function directly
            if not hasattr(module, function_name):
                raise AttributeError(f"Function {function_name} not found in module")
            return getattr(module, function_name)

        elif class_name:
            # Instantiate class
            if not hasattr(module, class_name):
                raise AttributeError(f"Class {class_name} not found in module")
            agent_class = getattr(module, class_name)
            return agent_class()

        else:
            # Auto-discover: look for common patterns
            return self._auto_discover_agent(module)

    def _auto_discover_agent(self, module: Any) -> Any:
        """Auto-discover agent from module using common patterns."""
        # Look for common class names
        for class_name in ["Agent", "MyAgent", "ChatAgent", "LLMAgent"]:
            if hasattr(module, class_name):
                agent_class = getattr(module, class_name)
                logger.info(f"Auto-discovered agent class: {class_name}")
                return agent_class()

        # Look for common function names
        for func_name in ["run", "process", "chat", "respond", "generate"]:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    logger.info(f"Auto-discovered agent function: {func_name}")
                    return func

        # Look for any callable class
        for attr_name in dir(module):
            if not attr_name.startswith("_"):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, "__call__"):
                    logger.info(f"Auto-discovered callable class: {attr_name}")
                    return attr()

        raise ValueError("Could not auto-discover agent in module. Specify AGENT_CLASS or AGENT_FUNCTION.")

    async def _initialize_mcp_client(self) -> Any:
        """Initialize MCP client using standard libraries."""
        # Get MCP configuration from environment
        sidecar_url = os.getenv("MCP_SIDECAR_URL", "http://sidecar:8080")
        session_id = os.getenv("SESSION_ID")
        task_id = os.getenv("TASK_ID")
        client_id = os.getenv("CLIENT_ID", "agent-container")

        if not session_id:
            raise ValueError("SESSION_ID environment variable required")

        # Create MCP client using factory
        factory = MCPClientFactory()
        mcp_client = await factory.create_client(
            sidecar_url=sidecar_url, session_id=session_id, task_id=task_id, client_id=client_id
        )

        return mcp_client

    def _get_initial_prompt(self) -> str:
        """Get initial prompt from environment."""
        prompt = os.getenv("INITIAL_PROMPT")
        if not prompt:
            raise ValueError("INITIAL_PROMPT environment variable required")
        return prompt

    async def _execute_agent(self, initial_prompt: str) -> Any:
        """Execute agent with proper error handling and monitoring."""
        if not self.agent_adapter:
            raise RuntimeError("Agent adapter not initialized")

        logger.info("🤖 Starting agent execution")

        # Execute agent with shutdown monitoring
        result = await self.agent_adapter.run(initial_prompt)

        logger.info("✅ Agent execution completed successfully")
        return result

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.mcp_client:
                await self.mcp_client.disconnect()
                logger.info("🧹 MCP client disconnected")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        return

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self.shutdown_requested = True
        logger.info("🛑 Shutdown requested")


async def main() -> None:
    """Main entry point for container execution."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    executor = AgentExecutor()

    try:
        # Run agent
        result = await executor.run()

        # Output result as JSON to stdout for collection
        print("AGENT_RESULT_START")
        print(json.dumps(result, indent=2))
        print("AGENT_RESULT_END")

        # Exit with appropriate code
        exit_code = 0 if result["success"] else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error in agent executor: {e}")

        # Output error result
        error_result = {"success": False, "result": None, "error": str(e), "traceback": traceback.format_exc()}

        print("AGENT_RESULT_START")
        print(json.dumps(error_result, indent=2))
        print("AGENT_RESULT_END")

        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
