"""
ToolExecutionManager handles tool execution coordination.

This component manages the execution of tools with concurrency control,
timeout management, and execution context handling.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from ..base import ToolExecutionError, ToolResult
from .registrar import ToolRegistrar

logger = logging.getLogger(__name__)


class ToolExecutionManager:
    """
    Manages tool execution with concurrency control and timeout handling.

    Provides coordinated execution of security tools with proper resource
    management and execution context handling.
    """

    def __init__(self, registrar: ToolRegistrar, default_timeout: float = 300.0, max_concurrent: int = 10):
        """
        Initialize the ToolExecutionManager.

        Args:
            registrar: ToolRegistrar instance for tool retrieval
            default_timeout: Default execution timeout in seconds
            max_concurrent: Maximum concurrent executions allowed
        """
        self._registrar = registrar
        self._default_timeout = default_timeout
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"ToolExecutionManager initialized with timeout={default_timeout}s, max_concurrent={max_concurrent}"
        )

    async def execute_tool(
        self, name: str, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute a tool with the given parameters.

        Args:
            name: Tool name
            parameters: Tool parameters
            context: Optional execution context

        Returns:
            ToolResult with execution results

        Raises:
            ToolNotFoundError: If tool is not found
            ToolExecutionError: If execution fails
        """
        tool = self._registrar.get_tool(name)

        # Use semaphore to limit concurrent executions
        async with self._semaphore:
            try:
                # Set up execution context
                exec_context = context or {}
                exec_context.setdefault("tool_name", name)
                exec_context.setdefault("domain", tool.domain)

                # Execute with timeout
                try:
                    timeout = tool.executor.get_timeout() or self._default_timeout
                except Exception as e:
                    logger.warning(f"Failed to get timeout from executor, using default: {e}")
                    timeout = self._default_timeout

                result = await asyncio.wait_for(tool.execute(parameters, exec_context), timeout=timeout)

                return result

            except asyncio.TimeoutError:
                raise ToolExecutionError(f"Tool '{name}' execution timed out after {timeout}s")
            except Exception as e:
                logger.error(f"Tool execution error for '{name}': {e}")
                raise ToolExecutionError(f"Tool '{name}' execution failed: {str(e)}")

    def update_execution_settings(
        self, default_timeout: Optional[float] = None, max_concurrent: Optional[int] = None
    ) -> None:
        """
        Update execution settings.

        Args:
            default_timeout: New default timeout in seconds
            max_concurrent: New maximum concurrent executions
        """
        if default_timeout is not None:
            self._default_timeout = default_timeout
            logger.info(f"Updated default timeout to {default_timeout}s")

        if max_concurrent is not None:
            self._max_concurrent = max_concurrent
            self._semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(f"Updated max concurrent executions to {max_concurrent}")

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "default_timeout": self._default_timeout,
            "max_concurrent": self._max_concurrent,
            "available_permits": self._semaphore._value,
        }

    def __repr__(self) -> str:
        """String representation of the execution manager."""
        return f"ToolExecutionManager(timeout={self._default_timeout}s, max_concurrent={self._max_concurrent})"
