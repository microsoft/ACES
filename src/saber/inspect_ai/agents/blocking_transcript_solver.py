"""
Blocking transcript solver for AI red team testing.

This solver implements a turn-based coordination system where the blue team
agent blocks after each generate() iteration, waiting for the red team to
modify its transcript before proceeding.

Architecture:
1. Blue team executes generate() and pushes transcript to server
2. This solver blocks and polls for transcript modifications
3. Red team modifies transcript and sets TRANSCRIPT_LAST_MODIFIED_AT timestamp
4. Blue team detects change via timestamp comparison and pulls modified transcript
5. Blue team continues with modified context

Logging category: ``LogCategory.AGENT``.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from inspect_ai.model import ChatMessage
from inspect_ai.solver import Solver, TaskState, solver

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys

logger = get_saber_logger(LogCategory.AGENT, __name__)


@solver
def blocking_transcript_solver(
    rest_url: str,
    session_id: str,
    episode_id: str,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: Optional[float] = 300.0,
    max_iterations: Optional[int] = None,
) -> Solver:
    """
    Post-generate blocking solver that waits for red team transcript modifications.
    
    This solver runs AFTER blue team's generate() completes:
    1. Blue team has already executed (generate + tool calls)
    2. Transcript has been pushed to server (TranscriptSyncingModelWrapper)
    3. NOW block and wait for red team to modify transcript
    4. When TRANSCRIPT_LAST_MODIFIED_AT timestamp changes, pull modified transcript
    5. Replace TaskState.messages and continue to next iteration
    
    Flow per iteration:
    - Blue team generate() executes → transcript pushed
    - This solver blocks waiting for timestamp change
    - Red team sets timestamp after modifying transcript
    - Pull modified transcript, update tracking, continue
    
    Args:
        rest_url: SABER server REST API URL
        session_id: Session identifier
        episode_id: Blue team episode identifier
        poll_interval_seconds: How often to poll for modifications (default 2s)
        timeout_seconds: Max wait time before proceeding (default 5min, None = infinite)
        max_iterations: Max number of blocking cycles (None = unlimited)
    
    Returns:
        Solver that blocks after generate() until transcript is modified
    
    Example:
        ```python
        from inspect_ai import Task
        from inspect_ai.solver import chain, generate
        from saber.inspect_ai.agents import blocking_transcript_solver
        
        task = Task(
            dataset=[...],
            plan=chain(
                generate(),  # Blue team executes first
                blocking_transcript_solver(  # Then blocks waiting for red team
                    rest_url="http://localhost:8000",
                    session_id="session-123",
                    episode_id="ep-blue-456",
                    poll_interval_seconds=2.0,
                    timeout_seconds=300.0
                ),
                # Chain repeats: generate → block → generate → block ...
            )
        )
        ```
    """
    
    iteration_count = 0
    
    async def solve(state: TaskState, generate) -> TaskState:
        """Block after generate, waiting for red team transcript modification."""
        
        nonlocal iteration_count
        iteration_count += 1
        
        # Check if we've hit max iterations
        if max_iterations and iteration_count > max_iterations:
            logger.info(
                f"Max iterations ({max_iterations}) reached, skipping blocking",
                extra={"episode_id": episode_id, "iteration": iteration_count}
            )
            return state
        
        logger.info(
            f"Blue team blocking (iteration {iteration_count}), waiting for red team modifications",
            extra={
                "episode_id": episode_id,
                "current_message_count": len(state.messages),
                "iteration": iteration_count
            }
        )
        
        # Poll for timestamp changes
        start_time = asyncio.get_event_loop().time()
        last_pull_timestamp = datetime.now(timezone.utc)  # Client-side tracking
        modification_detected = False
        
        while True:
            # Check timeout
            if timeout_seconds and (asyncio.get_event_loop().time() - start_time) > timeout_seconds:
                logger.warning(
                    "Blue team timeout waiting for red team modifications",
                    extra={
                        "episode_id": episode_id,
                        "timeout": timeout_seconds,
                        "iteration": iteration_count
                    }
                )
                break
            
            # Poll for timestamp metadata
            try:
                metadata = await _get_transcript_metadata(rest_url, session_id, episode_id)
                
                # Check if last_modified > last_pull (change detection)
                last_modified_str = metadata.get("last_modified_at")
                if last_modified_str:
                    last_modified = datetime.fromisoformat(last_modified_str.replace('Z', '+00:00'))
                    
                    # Ensure last_modified is timezone-aware
                    if last_modified.tzinfo is None:
                        last_modified = last_modified.replace(tzinfo=timezone.utc)
                    
                    if last_modified > last_pull_timestamp:
                        logger.info(
                            "Transcript modification detected via timestamp comparison",
                            extra={
                                "episode_id": episode_id,
                                "modification_count": metadata.get("modification_count", 0),
                                "last_modified_at": last_modified_str,
                                "wait_time_seconds": asyncio.get_event_loop().time() - start_time,
                                "iteration": iteration_count
                            }
                        )
                        modification_detected = True
                        break
            except Exception as e:
                logger.error(
                    "Error polling transcript metadata",
                    extra={
                        "episode_id": episode_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "iteration": iteration_count
                    }
                )
            
            # Wait before next poll
            await asyncio.sleep(poll_interval_seconds)
        
        # Pull modified transcript if timestamp indicated changes
        if modification_detected:
            try:
                modified_transcript = await _pull_modified_transcript(
                    rest_url, session_id, episode_id
                )
                
                # Replace entire message history with red team's modified version
                state.messages = modified_transcript["messages"]
                
                # Update client-side tracking
                last_pull_timestamp = datetime.now(timezone.utc)
                
                logger.info(
                    "Blue team transcript updated from red team modifications",
                    extra={
                        "episode_id": episode_id,
                        "new_message_count": len(state.messages),
                        "modification_count": modified_transcript.get("modification_count", 0),
                        "iteration": iteration_count
                    }
                )
            except Exception as e:
                logger.error(
                    "Error pulling modified transcript",
                    extra={
                        "episode_id": episode_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "iteration": iteration_count
                    }
                )
        
        return state
    
    return solve


async def _get_transcript_metadata(
    rest_url: str, session_id: str, episode_id: str
) -> dict:
    """Get transcript metadata including timestamps.
    
    Args:
        rest_url: SABER server REST API URL
        session_id: Session identifier
        episode_id: Episode identifier
    
    Returns:
        {
            "last_pushed_at": str | None,
            "last_modified_at": str | None,
            "modification_count": int
        }
    """
    url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/transcript/metadata"
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def _pull_modified_transcript(
    rest_url: str, session_id: str, episode_id: str
) -> dict:
    """Pull the modified transcript from server.
    
    Args:
        rest_url: SABER server REST API URL
        session_id: Session identifier
        episode_id: Episode identifier
    
    Returns:
        {
            "messages": list[ChatMessage],
            "modification_count": int
        }
    """
    url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/transcript"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Deserialize messages
        from ..integration.transcript_sync import _deserialize_message
        messages = [_deserialize_message(msg) for msg in data.get("messages", [])]
        
        return {
            "messages": messages,
            "modification_count": data.get("modification_count", 0)
        }
