#!/usr/bin/env python3
"""
Termination-Aware LLM Client Wrapper

Wraps any LLM client to add episode termination checking, preventing
unnecessary API calls when episodes are already terminated.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TerminationAwareLLMClient:
    """
    Wrapper around LLM clients that checks for episode termination before making API calls.

    Prevents expensive LLM API calls when the episode has already been terminated
    by the SABER framework.
    """

    def __init__(self, wrapped_llm_client: Any, termination_check: Optional[Callable[[], bool]] = None):
        """
        Initialize termination-aware LLM client wrapper.

        Args:
            wrapped_llm_client: The underlying LLM client to wrap
            termination_check: Function that returns True if episode is terminated
        """
        self.wrapped_client = wrapped_llm_client
        self.termination_check = termination_check
        self._call_count = 0

    def set_termination_check(self, termination_check: Callable[[], bool]) -> None:
        """Set or update the termination check function."""
        self.termination_check = termination_check

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any:
        """
        Create completion with termination checking.

        Checks if episode is terminated before making expensive LLM API call.
        """
        self._call_count += 1

        # Check for termination before making expensive API call
        if self.termination_check and self.termination_check():
            logger.warning(f"🚫 LLM call #{self._call_count} blocked - episode already terminated")
            raise asyncio.CancelledError("Episode terminated - LLM call blocked")

        logger.debug(f"🧠 LLM call #{self._call_count} proceeding - episode active")

        # Delegate to wrapped client
        return self.wrapped_client.create_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def complete(self, prompt: str) -> str:
        """Async completion method with termination checking."""
        self._call_count += 1

        if self.termination_check and self.termination_check():
            logger.warning(f"🚫 LLM call #{self._call_count} blocked - episode already terminated")
            raise asyncio.CancelledError("Episode terminated - LLM call blocked")

        logger.debug(f"🧠 LLM call #{self._call_count} proceeding - episode active")

        if hasattr(self.wrapped_client, "complete"):
            result = await self.wrapped_client.complete(prompt)
            return str(result)
        else:
            raise AttributeError("Wrapped client does not support 'complete' method")

    async def __call__(self, prompt: str) -> str:
        """Callable interface with termination checking."""
        self._call_count += 1

        if self.termination_check and self.termination_check():
            logger.warning(f"🚫 LLM call #{self._call_count} blocked - episode already terminated")
            raise asyncio.CancelledError("Episode terminated - LLM call blocked")

        logger.debug(f"🧠 LLM call #{self._call_count} proceeding - episode active")

        if callable(self.wrapped_client):
            result = await self.wrapped_client(prompt)
            return str(result)
        else:
            raise AttributeError("Wrapped client is not callable")

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attributes to the wrapped client."""
        return getattr(self.wrapped_client, name)

    @property
    def call_count(self) -> int:
        """Get the number of LLM calls attempted."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter for new episodes."""
        self._call_count = 0
