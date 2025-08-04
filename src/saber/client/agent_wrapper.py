"""
SABER Client - Agent wrapper that adapts arbitrary agents to SABER interface.

This module provides automatic agent wrapping so customers don't need to
implement any specific interface. It detects agent capabilities and adapts
them to work with the SABER test harness.
"""

import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Optional


class AgentWrapper:
    """
    Wrapper that adapts arbitrary agent implementations to work with SABER.

    Automatically detects agent capabilities and calling conventions:
    - Sync vs async methods
    - Different method names (process, generate, respond, etc.)
    - Different parameter names (prompt, input, message, etc.)
    - Class vs function-based agents

    This wrapper provides the minimal interface needed by SABER's TestHarness:
    - async process_prompt(prompt: str) -> str
    - async reset() -> None
    """

    def __init__(self, agent_instance: Any):
        """
        Initialize wrapper with any agent instance.

        Args:
            agent_instance: The customer's agent (class instance, function, etc.)
        """
        self.agent = agent_instance
        self.process_method = self._detect_process_method()
        self.reset_method = self._detect_reset_method()
        self.is_async = self._is_async_method(self.process_method)

    def _detect_process_method(self) -> Callable:
        """Detect the agent's main processing method."""
        # Common method names customers might use
        method_names = [
            "process_prompt",
            "process",
            "generate",
            "respond",
            "answer",
            "call",
            "__call__",
            "run",
            "execute",
            "predict",
            "inference",
        ]

        for method_name in method_names:
            if hasattr(self.agent, method_name):
                method = getattr(self.agent, method_name)
                if callable(method):
                    return method  # type: ignore[no-any-return]

        # If it's a callable itself (function-based agent)
        if callable(self.agent):
            return self.agent  # type: ignore[no-any-return]

        raise ValueError(
            f"Could not detect processing method in agent {type(self.agent)}. " f"Expected one of: {method_names}"
        )

    def _detect_reset_method(self) -> Optional[Callable]:
        """Detect optional reset method."""
        reset_names = ["reset", "clear", "initialize", "restart", "new_session"]

        for method_name in reset_names:
            if hasattr(self.agent, method_name):
                method = getattr(self.agent, method_name)
                if callable(method):
                    return method  # type: ignore[no-any-return]
        return None

    def _is_async_method(self, method: Callable) -> bool:
        """Check if method is async."""
        return inspect.iscoroutinefunction(method)

    def _detect_parameter_name(self, method: Callable) -> str:
        """Detect what parameter name the agent expects."""
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        # Remove 'self' if present
        if params and params[0] == "self":
            params = params[1:]

        if not params:
            raise ValueError("Agent method must accept at least one parameter")

        # Use the first parameter (most agents will have prompt/input/message as first param)
        return params[0]

    async def process_prompt(self, prompt: str) -> str:
        """Process prompt using the wrapped agent."""
        param_name = self._detect_parameter_name(self.process_method)

        # Call the agent with the correct parameter name
        if self.is_async:
            result = await self.process_method(**{param_name: prompt})
        else:
            result = self.process_method(**{param_name: prompt})

        # Ensure result is a string
        if not isinstance(result, str):
            return str(result)

        return result

    async def reset(self) -> None:
        """Reset the wrapped agent if it supports reset."""
        if self.reset_method:
            if inspect.iscoroutinefunction(self.reset_method):
                await self.reset_method()
            else:
                self.reset_method()


class AgentLoader:
    """
    Loads agents from various sources and wraps them for SABER.

    Supports:
    - Python files with agent classes or functions
    - Modules with agent factories
    - Direct imports from packages
    """

    @staticmethod
    def load_from_path(agent_path: str, agent_class: Optional[str] = None) -> "AgentWrapper":
        """
        Load agent from file path.

        Args:
            agent_path: Path to Python file containing agent
            agent_class: Optional specific class name to load

        Returns:
            AgentWrapper: Wrapped agent ready for SABER
        """
        agent_path_obj = Path(agent_path)

        if not agent_path_obj.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_path}")

        # Load the module
        spec = importlib.util.spec_from_file_location("customer_agent", agent_path_obj)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {agent_path_obj}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the agent
        agent_instance = AgentLoader._extract_agent_from_module(module, agent_class)

        # Wrap it
        return AgentWrapper(agent_instance)

    @staticmethod
    def load_from_module(module_name: str, agent_class: Optional[str] = None) -> "AgentWrapper":
        """
        Load agent from importable module.

        Args:
            module_name: Module name (e.g., 'my_package.agent')
            agent_class: Optional specific class name to load

        Returns:
            AgentWrapper: Wrapped agent ready for SABER
        """
        module = importlib.import_module(module_name)
        agent_instance = AgentLoader._extract_agent_from_module(module, agent_class)
        return AgentWrapper(agent_instance)

    @staticmethod
    def _extract_agent_from_module(module: Any, agent_class: Optional[str] = None) -> Any:
        """Extract agent instance from loaded module."""

        if agent_class:
            # User specified exact class name
            if not hasattr(module, agent_class):
                raise AttributeError(f"Class '{agent_class}' not found in module")

            agent_cls = getattr(module, agent_class)
            return agent_cls()  # Instantiate

        # Auto-detect agent
        candidates = []

        # Look for classes that might be agents
        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            # Check if it's a class
            if inspect.isclass(obj):
                # Look for agent-like method names
                methods = [method for method in dir(obj) if not method.startswith("_")]
                agent_methods = ["process", "generate", "respond", "answer", "predict"]

                if any(method in methods for method in agent_methods):
                    candidates.append((name, obj))

            # Check if it's a callable function
            elif callable(obj) and not inspect.isbuiltin(obj):
                candidates.append((name, obj))

        if not candidates:
            raise ValueError(
                "No agent found in module. Expected class with process/generate/respond method or callable function"
            )

        # Select the best candidate
        selected_agent = None

        if len(candidates) == 1:
            # Single candidate - use it
            selected_agent = candidates[0][1]
        else:
            # Multiple candidates - try to pick the best one
            for name, agent in candidates:
                if "agent" in name.lower():
                    selected_agent = agent
                    break

            # If no preferred candidate found, use the first one
            if selected_agent is None:
                selected_agent = candidates[0][1]

        # Instantiate if it's a class
        assert selected_agent is not None
        if inspect.isclass(selected_agent):
            return selected_agent()
        return selected_agent  # type: ignore[unreachable]
