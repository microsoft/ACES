"""
SABER API Models - Shared data models for client-server communication.

These models define the API contract between SABER clients and servers.
They should be used by both client and server implementations to ensure
consistent data structures.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepResponse(BaseModel):
    """Response from server step execution."""

    success: bool = Field(description="Command succeeded")
    output: str = Field(description="Command output to show agent")
    error: Optional[str] = Field(None, description="Error message if failed")
    done: bool = Field(description="Episode complete")
    task_completed: bool = Field(default=False, description="Current task completed")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional server info")


class TaskInfo(BaseModel):
    """Information about the current task."""

    task_id: str
    title: str
    description: str
    completed: bool = Field(default=False, description="Task completion status")
    episode_id: Optional[str] = None
    state: Optional[str] = None
    step_count: Optional[int] = None
    duration: Optional[float] = None
    subtasks: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed subtask information")


class PolicyInfo(BaseModel):
    """Domain policy information."""

    domain: Optional[str] = None
    available_commands: List[str]
    guidelines: str
    constraints: List[str]


class EpisodeInfo(BaseModel):
    """Episode information."""

    episode_id: str
    task_id: str
    message: Optional[str] = None


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str
    message: str


class ToolCallEventStart(BaseModel):
    """Tool call started event model."""

    call_id: str
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
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
    task_id: Optional[str] = None
    timestamp: Optional[str] = None
    current_step: Optional[int] = None
    max_steps: Optional[int] = None
