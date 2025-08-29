#!/usr/bin/env python3
"""
SABER Agent Wrapper

Universal adapter for arbitrary agent implementations to work with SABER.
Provides automatic detection and normalization of agent interfaces.
"""

import asyncio
import inspect
import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class AgentWrapper:
    """
    Universal agent adapter that normalizes arbitrary agent implementations.

    Automatically detects agent interface and provides consistent async interface
    regardless of underlying agent implementation.
    """

    def __init__(self, agent: Any, mcp_client: Any, llm_client: Optional[Any] = None):
        """
        Initialize agent wrapper.

        Args:
            agent: The user-supplied agent (class, instance, or callable)
            mcp_client: MCP client for tool access
            llm_client: Optional LLM client
        """
        self.user_agent = agent
        self.mcp_client = mcp_client
        self.llm_client = llm_client
        self.agent_instance: Optional[Any] = None
        self.agent_callable: Optional[Callable] = None

        # Thread-safe termination flag for cross-thread communication
        self._shutdown_flag = threading.Event()
        # Task reference for forceful cancellation
        self._agent_task: Optional[asyncio.Task] = None

        # Detect and prepare agent interface
        self._detect_agent_interface()

    def _detect_agent_interface(self) -> None:
        """Detect agent interface and prepare for invocation."""
        logger.debug(f"Detecting interface for agent: {type(self.user_agent)}")

        # If it's a class, instantiate it
        if inspect.isclass(self.user_agent):
            self.agent_instance = self._instantiate_agent_class()
            agent_obj = self.agent_instance
        else:
            # It's already an instance or function
            agent_obj = self.user_agent
            self.agent_instance = agent_obj

        # Find the callable method using precedence order from plan
        callable_candidates = [
            ("run", getattr(agent_obj, "run", None)),
            ("__call__", agent_obj if callable(agent_obj) else None),
        ]

        for method_name, method in callable_candidates:
            if method and callable(method):
                logger.debug(f"Found callable method: {method_name}")
                self.agent_callable = method
                break

        if not self.agent_callable:
            raise ValueError(
                f"Agent {type(self.user_agent)} has no callable interface (run, __call__, or direct callable)"
            )

        logger.info(f"✅ Agent interface detected: {self.agent_callable}")

    def _instantiate_agent_class(self) -> Any:
        """Instantiate agent class with appropriate parameters."""
        # Get constructor signature
        sig = inspect.signature(self.user_agent.__init__)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'

        # Build constructor arguments
        kwargs = {}

        # Common parameter patterns
        if "mcp_client" in params:
            kwargs["mcp_client"] = self.mcp_client
        elif any("mcp" in p.lower() for p in params):
            # Positional MCP client
            kwargs[next(p for p in params if "mcp" in p.lower())] = self.mcp_client

        if "llm_client" in params and self.llm_client:
            kwargs["llm_client"] = self.llm_client

        if "model_name" in params and self.llm_client:
            model_name = getattr(self.llm_client, "model_name", "gpt-4")
            kwargs["model_name"] = model_name

        logger.debug(f"Instantiating agent with kwargs: {list(kwargs.keys())}")

        try:
            if kwargs:
                return self.user_agent(**kwargs)
            else:
                # Try with just MCP client as positional arg
                if params:
                    return self.user_agent(self.mcp_client)
                else:
                    return self.user_agent()
        except Exception as e:
            logger.error(f"Failed to instantiate agent {self.user_agent.__name__}: {e}")
            raise

    def shutdown_check(self) -> bool:
        """Check if agent should shutdown (thread-safe)."""
        return self._shutdown_flag.is_set()

    def set_shutdown(self) -> None:
        """Signal agent to shutdown and cancel task if running."""
        logger.info("🛑 Setting shutdown flag and cancelling agent task")
        self._shutdown_flag.set()

        # Forcefully cancel the agent task if it's running
        if self._agent_task and not self._agent_task.done():
            logger.info("🔥 Forcefully cancelling agent task")
            self._agent_task.cancel()

    async def run(self, initial_prompt: str) -> Any:
        """
        Run agent with initial prompt using detected interface.

        Supports both sync and async agents with flexible parameter patterns.
        Implements forceful termination via task cancellation.
        """
        if not self.agent_callable:
            raise RuntimeError("No callable agent interface detected")

        # Get method signature to determine parameters
        sig = inspect.signature(self.agent_callable)
        params = list(sig.parameters.keys())

        # Filter out 'self' if present (for bound methods)
        if params and params[0] == "self":
            params = params[1:]

        logger.debug(f"Agent signature parameters: {params}")

        # Build arguments using precedence from plan
        kwargs: Dict[str, Any] = {}

        # Always provide initial_prompt as first positional or named
        if params:
            # Common prompt parameter names
            prompt_param = None
            for param_name in ["initial_prompt", "prompt", "input", "message"]:
                if param_name in params:
                    prompt_param = param_name
                    break

            if prompt_param:
                kwargs[prompt_param] = initial_prompt
            else:
                # Use first parameter as prompt
                kwargs[params[0]] = initial_prompt

        # Add other parameters based on signature (only if they exist)
        if "mcp_client" in params:
            kwargs["mcp_client"] = self.mcp_client
        if "shutdown_check" in params:
            kwargs["shutdown_check"] = self.shutdown_check
        if "llm_client" in params and self.llm_client:
            kwargs["llm_client"] = self.llm_client

        logger.info(f"🤖 Calling agent with parameters: {list(kwargs.keys())}")

        try:
            # Wrap agent execution in a cancellable task
            async def execute_agent() -> Any:
                if inspect.iscoroutinefunction(self.agent_callable):
                    assert self.agent_callable is not None  # mypy hint
                    return await self.agent_callable(**kwargs)
                else:
                    # Run sync method in thread to avoid blocking
                    assert self.agent_callable is not None  # mypy hint
                    agent_func = self.agent_callable  # Capture for lambda
                    return await asyncio.get_event_loop().run_in_executor(None, lambda: agent_func(**kwargs))

            # Create the agent task and store reference for cancellation
            self._agent_task = asyncio.create_task(execute_agent())

            # Wait for completion or cancellation
            result = await self._agent_task

            logger.info("✅ Agent execution completed successfully")
            return result

        except asyncio.CancelledError:
            logger.warning("⚠️ Agent execution was cancelled (episode terminated)")
            return {"success": False, "flag": None, "reason": "cancelled_by_framework", "terminated": True}
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")
            raise
        finally:
            # Clear task reference
            self._agent_task = None
