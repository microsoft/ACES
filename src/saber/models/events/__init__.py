"""
SABER Event Models - Real-time event streaming models.

These models define the structure for real-time events sent via Server-Sent Events (SSE).
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class ToolCallEventStart(BaseModel):
    """Tool call started event model."""

    call_id: str
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: Optional[str] = None
    current_step: Optional[int] = None
    max_steps: Optional[int] = None


class ToolCallEventProgress(BaseModel):
    """Tool call progress event model."""

    call_id: str
    tool_name: str
    progress_info: Optional[str] = None
    progress: Optional[float] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: Optional[str] = None


class ToolCallEventComplete(BaseModel):
    """Tool call completed event model."""

    call_id: str
    tool_name: str
    success: bool
    arguments: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    output: Optional[str] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: Optional[str] = None
    current_step: Optional[int] = None
    max_steps: Optional[int] = None
