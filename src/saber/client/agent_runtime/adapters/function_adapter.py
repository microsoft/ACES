#!/usr/bin/env python3
"""
Function-Based Agent Adapter

Adapter for function-based agents that are simple callable functions.
Automatically detects function signature and provides MCP client access.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List

from .base import AgentAdapter

logger = logging.getLogger(__name__)


class FunctionBasedAdapter(AgentAdapter):
    """
    Adapter for function-based agents.

    Handles both sync and async functions with automatic parameter detection.
    """

    def __init__(self, agent: Callable, tool_injector: Any):
        """Initialize function-based agent adapter."""
        super().__init__(agent, tool_injector)
        self.function_params = self._analyze_function_signature()

    def _analyze_function_signature(self) -> Dict[str, Any]:
        """Analyze function signature to understand parameters."""
        sig = inspect.signature(self.agent)
        params: Dict[str, Any] = {}

        param_names = list(sig.parameters.keys())

        # Detect parameter purposes
        for param_name in param_names:
            param_name_lower = param_name.lower()

            if any(keyword in param_name_lower for keyword in ["prompt", "input", "message", "query", "text"]):
                params["prompt_param"] = param_name
            elif any(keyword in param_name_lower for keyword in ["tool", "mcp", "client", "context"]):
                params["context_param"] = param_name

        # Use first parameter as prompt if not detected
        if "prompt_param" not in params and param_names:
            params["prompt_param"] = param_names[0]

        params["is_async"] = inspect.iscoroutinefunction(self.agent)
        params["param_count"] = len(param_names)

        return params

    async def run(self, initial_prompt: str) -> Any:
        """Execute function-based agent."""
        logger.info("🤖 Starting function-based agent execution")

        try:
            # Prepare function arguments based on signature
            if self.function_params["param_count"] == 0:
                # No parameters - just call function
                args: List[Any] = []
                kwargs: Dict[str, Any] = {}
            elif self.function_params["param_count"] == 1:
                # Single parameter - assume it's the prompt
                prompt_param = self.function_params.get("prompt_param")
                if prompt_param:
                    kwargs = {prompt_param: initial_prompt}
                    args = []
                else:
                    args = [initial_prompt]
                    kwargs = {}
            else:
                # Multiple parameters - use named parameters
                kwargs = {}
                args = []

                prompt_param = self.function_params.get("prompt_param")
                if prompt_param:
                    kwargs[prompt_param] = initial_prompt

                context_param = self.function_params.get("context_param")
                if context_param:
                    kwargs[context_param] = self._prepare_agent_context()

                # If no prompt param detected, use positional
                if not prompt_param:
                    args = [initial_prompt]

            # Execute function
            if self.function_params.get("is_async", False):
                result = await self.agent(*args, **kwargs)
            else:
                result = self.agent(*args, **kwargs)

            logger.info("✅ Function-based agent execution completed")
            return result

        except Exception as e:
            logger.error(f"❌ Function-based agent execution failed: {e}")
            raise
