#!/usr/bin/env python3
"""
Class-Based Agent Adapter

Adapter for class-based agents with methods like run(), process(), chat(), respond().
Automatically detects the appropriate method and parameter names.
"""

import inspect
import logging
from typing import Any, Callable, Dict, Optional

from .base import AgentAdapter

logger = logging.getLogger(__name__)


class ClassBasedAdapter(AgentAdapter):
    """
    Adapter for class-based agents.

    Automatically detects agent methods and parameters:
    - Methods: run, process, chat, respond, generate, execute
    - Parameters: prompt, input, message, query, text
    """

    def __init__(self, agent: Any, mcp_client: Any):
        """Initialize class-based agent adapter."""
        super().__init__(agent, mcp_client)
        self.agent_method: Optional[Callable] = None
        self.method_params: Optional[Dict[str, Any]] = None
        self._detect_agent_interface()

    def _detect_agent_interface(self) -> None:
        """Detect agent method and parameter names."""
        # Common method names to try
        method_names = ["run", "process", "chat", "respond", "generate", "execute", "__call__"]

        for method_name in method_names:
            if hasattr(self.agent, method_name):
                method = getattr(self.agent, method_name)
                if callable(method):
                    self.agent_method = method
                    self.method_params = self._analyze_method_signature(method)
                    logger.info(f"✅ Detected agent method: {method_name}")
                    logger.info(f"📋 Method parameters: {self.method_params}")
                    break

        if not self.agent_method:
            raise ValueError(f"No suitable method found in agent. Tried: {method_names}")

    def _analyze_method_signature(self, method: Callable) -> Dict[str, Any]:
        """Analyze method signature to understand parameters."""
        sig = inspect.signature(method)
        params: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Detect parameter purpose based on name
            param_name_lower = param_name.lower()

            if any(keyword in param_name_lower for keyword in ["prompt", "input", "message", "query", "text"]):
                params["prompt_param"] = param_name
            elif any(keyword in param_name_lower for keyword in ["tool", "mcp", "client", "context"]):
                params["context_param"] = param_name
            elif param.default == inspect.Parameter.empty and not params.get("prompt_param"):
                # First required parameter is likely the prompt
                params["prompt_param"] = param_name

        # Default prompt parameter if not detected
        if "prompt_param" not in params:
            params["prompt_param"] = "prompt"

        params["is_async"] = inspect.iscoroutinefunction(method)

        return params

    async def run(self, initial_prompt: str) -> Any:
        """Execute class-based agent."""
        if not self.agent_method or not self.method_params:
            raise RuntimeError("Agent method not detected")

        logger.info("🤖 Starting class-based agent execution")

        try:
            # Prepare method arguments
            kwargs: Dict[str, Any] = {}

            # Add prompt
            prompt_param = self.method_params.get("prompt_param", "prompt")
            kwargs[prompt_param] = initial_prompt

            # Add context if method accepts it
            context_param = self.method_params.get("context_param")
            if context_param:
                kwargs[context_param] = self._prepare_agent_context()

            # Check if we need to inject MCP client in other ways
            if hasattr(self.agent, "mcp_client"):
                self.agent.mcp_client = self.mcp_client
            if hasattr(self.agent, "tools"):
                self.agent.tools = self.mcp_client

            # Execute method
            if self.method_params.get("is_async", False):
                result = await self.agent_method(**kwargs)
            else:
                result = self.agent_method(**kwargs)

            logger.info("✅ Class-based agent execution completed")
            return result

        except Exception as e:
            logger.error(f"❌ Class-based agent execution failed: {e}")
            raise
