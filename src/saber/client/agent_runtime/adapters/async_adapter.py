#!/usr/bin/env python3
"""
Async Agent Adapter

Specialized adapter for agents with async methods that require special handling
for async/await patterns and concurrent execution.
"""

import asyncio
import logging
from typing import Any

from .class_adapter import ClassBasedAdapter

logger = logging.getLogger(__name__)


class AsyncAdapter(ClassBasedAdapter):
    """
    Specialized adapter for async agents.

    Extends ClassBasedAdapter with async-specific handling and monitoring.
    """

    def __init__(self, agent: Any, mcp_client: Any):
        """Initialize async agent adapter."""
        super().__init__(agent, mcp_client)

    async def run(self, initial_prompt: str) -> Any:
        """Execute async agent with enhanced monitoring."""
        logger.info("🤖 Starting async agent execution")

        try:
            # Create a task for the agent execution
            agent_task = asyncio.create_task(self._run_agent_with_monitoring(initial_prompt))

            # Monitor for shutdown requests
            monitor_task = asyncio.create_task(self._monitor_shutdown(agent_task))

            try:
                # Wait for either agent completion or shutdown
                done, pending = await asyncio.wait([agent_task, monitor_task], return_when=asyncio.FIRST_COMPLETED)

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Get result from completed task
                if agent_task in done:
                    result = await agent_task
                    logger.info("✅ Async agent execution completed")
                    return result
                else:
                    # Shutdown was requested
                    logger.info("🛑 Async agent execution cancelled due to shutdown")
                    return {"success": False, "reason": "shutdown_requested"}

            except asyncio.CancelledError:
                # Handle cancellation gracefully
                logger.info("🛑 Async agent execution cancelled")
                agent_task.cancel()
                monitor_task.cancel()
                return {"success": False, "reason": "cancelled"}

        except Exception as e:
            logger.error(f"❌ Async agent execution failed: {e}")
            raise

    async def _run_agent_with_monitoring(self, initial_prompt: str) -> Any:
        """Run agent with async monitoring capabilities."""
        # Use parent class implementation but with async enhancements
        return await super().run(initial_prompt)

    async def _monitor_shutdown(self, agent_task: asyncio.Task) -> None:
        """Monitor for shutdown requests and cancel agent if needed."""
        while not agent_task.done():
            if self.shutdown_requested:
                logger.info("🛑 Shutdown requested, cancelling agent task")
                agent_task.cancel()
                break

            await asyncio.sleep(0.1)  # Check every 100ms
