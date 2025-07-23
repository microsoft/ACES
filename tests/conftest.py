"""
Test configuration and utilities for SABER test suite.

This module provides common test fixtures, mock objects, and utilities
that can be shared across multiple test modules.
"""

import asyncio
from typing import Any, Dict, Optional
import logging

from saber.server.tools.base import ToolResult
from saber.server.tools.executors.base_executors import BaseToolExecutor

logger = logging.getLogger(__name__)


class MockExecutor(BaseToolExecutor):
    """
    Mock executor for testing and development.

    Returns predefined responses for testing purposes.
    """

    def __init__(self, mock_response: Any = None, mock_error: Optional[str] = None,
                 delay: float = 0.0, timeout: Optional[float] = None):
        """
        Initialize mock executor.

        Args:
            mock_response: Response to return on success
            mock_error: Error message to return on failure
            delay: Artificial delay in seconds
            timeout: Execution timeout in seconds
        """
        super().__init__(timeout)
        self._mock_response = mock_response
        self._mock_error = mock_error
        self._delay = delay

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Execute mock tool.

        Args:
            parameters: Tool parameters (logged but not used)
            context: Execution context (logged but not used)

        Returns:
            ToolResult with mock response
        """
        # Add artificial delay if specified
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        # Log execution for debugging
        logger.debug(f"Mock tool executed with parameters: {parameters}")

        # Return error or success based on configuration
        if self._mock_error:
            return ToolResult.error_result(self._mock_error)
        else:
            return ToolResult.success_result(self._mock_response or {"status": "success", "parameters": parameters})
