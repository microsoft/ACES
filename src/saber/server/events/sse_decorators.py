"""
SSE Event Broadcasting Decorators for SABER SessionManager.

Provides decorators to automatically send SSE events for environment transitions
and other session state changes without cluttering business logic.
"""

import functools
import logging
from datetime import datetime
from typing import Any, Callable, Optional

from ...base import SSEEventType
from ..episodes.constants import EpisodeResponseKeys

logger = logging.getLogger(__name__)


def broadcast_environment_transition(
    old_task_key: str = EpisodeResponseKeys.PREVIOUS_TASK_ID,
    new_task_key: str = EpisodeResponseKeys.NEXT_TASK_ID,
    session_key: str = "session_id",
    condition_key: Optional[str] = EpisodeResponseKeys.ENVIRONMENT_CHANGED,
) -> Callable:
    """
    Decorator to automatically broadcast SSE environment transition events.

    Args:
        old_task_key: Key in function result containing old task ID
        new_task_key: Key in function result containing new task ID
        session_key: Key in function args/kwargs containing session ID
        condition_key: Optional key in result that must be True to broadcast

    Usage:
        @broadcast_environment_transition()
        async def end_episode(self, session_id: str, ...) -> Dict[str, Any]:
            # Business logic here
            return {
                EpisodeResponseKeys.PREVIOUS_TASK_ID: old_id,
                EpisodeResponseKeys.NEXT_TASK_ID: new_id,
                EpisodeResponseKeys.ENVIRONMENT_CHANGED: True
            }
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Execute the original function
            result = await func(self, *args, **kwargs)

            try:
                # Extract session_id from function arguments
                session_id = None
                if args and session_key == "session_id":
                    session_id = args[0]  # Assume first arg is session_id
                elif session_key in kwargs:
                    session_id = kwargs[session_key]

                if not session_id:
                    logger.warning(f"Could not extract session_id for SSE broadcast in {func.__name__}")
                    return result

                # Check if we should broadcast (if condition_key is specified)
                if condition_key and isinstance(result, dict):
                    if not result.get(condition_key, False):
                        return result

                # Extract task IDs from result and send SSE event
                if isinstance(result, dict):
                    old_task_id = result.get(old_task_key)
                    new_task_id = result.get(new_task_key)

                    if old_task_id and new_task_id and hasattr(self, "benchmark_manager"):
                        # Get task information for context
                        old_task = self.benchmark_manager.get_task(old_task_id)
                        new_task = self.benchmark_manager.get_task(new_task_id)

                        # Create environment transition event
                        event_data = {
                            "type": SSEEventType.ENVIRONMENT_RESET,
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {
                                "reason": "task_transition",
                                "old_task": {"task_id": old_task_id, "title": old_task.title},
                                "new_task": {"task_id": new_task_id, "title": new_task.title},
                                "reset_recommended": True,
                                "context_hint": (
                                    f"Environment changed from {old_task.title} to {new_task.title}. "
                                    "Clear your conversation history and start fresh."
                                ),
                            },
                        }

                        # TODO: Send via existing SSE infrastructure
                        # This would integrate with the SSE event queue/broadcast system
                        logger.info(
                            f"SSE environment transition event prepared for session {session_id}: "
                            f"{old_task_id} -> {new_task_id}"
                        )
                        logger.debug(f"Event data: {event_data}")

            except Exception as e:
                logger.warning(f"Failed to broadcast SSE event in {func.__name__}: {e}")
                # Don't let SSE failures break the main function

            return result

        return wrapper

    return decorator


def broadcast_session_event(event_type: str, data_extractor: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
    """
    Generic decorator for broadcasting session-level SSE events.

    Args:
        event_type: Type of SSE event to broadcast
        data_extractor: Optional function to extract event data from result

    Usage:
        @broadcast_session_event("session_terminated")
        async def terminate_session(self, session_id: str) -> None:
            # Business logic here
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Execute the original function
            result = await func(self, *args, **kwargs)

            try:
                # Extract session_id (assume first arg)
                session_id = args[0] if args else None

                if session_id:
                    # Create event data
                    event_data = {
                        "type": event_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"session_id": session_id, "function": func.__name__},
                    }

                    # Add custom data if extractor provided
                    if data_extractor and callable(data_extractor):
                        custom_data = data_extractor(result, *args, **kwargs)
                        if custom_data and isinstance(custom_data, dict):
                            data_dict = event_data["data"]
                            if isinstance(data_dict, dict):
                                data_dict.update(custom_data)

                    # Broadcast via SSE (would need to implement _send_session_event)
                    if hasattr(self, "_send_session_event"):
                        await self._send_session_event(session_id, event_data)
                        logger.debug(f"SSE session event broadcasted for {func.__name__}: {event_type}")

            except Exception as e:
                logger.warning(f"Failed to broadcast session SSE event in {func.__name__}: {e}")

            return result

        return wrapper

    return decorator


def broadcast_benchmark_event(event_type: str) -> Callable[..., Any]:
    """
    Decorator for broadcasting benchmark-specific SSE events.

    Args:
        event_type: Type of benchmark event (e.g., "benchmark_started", "benchmark_completed")

    Usage:
        @broadcast_benchmark_event("benchmark_completed")
        async def end_benchmark(self, session_id: str) -> Dict[str, Any]:
            # Business logic here
            return {"benchmark_complete": True}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = await func(self, *args, **kwargs)

            try:
                session_id = args[0] if args else None

                if session_id and isinstance(result, dict):
                    event_data = {
                        "type": event_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"session_id": session_id, **result},  # Include all result data
                    }

                    if hasattr(self, "_send_benchmark_event"):
                        await self._send_benchmark_event(session_id, event_data)
                        logger.debug(f"SSE benchmark event broadcasted for {func.__name__}: {event_type}")

            except Exception as e:
                logger.warning(f"Failed to broadcast benchmark SSE event in {func.__name__}: {e}")

            return result

        return wrapper

    return decorator
