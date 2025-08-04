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
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional server info")


class TaskInfo(BaseModel):
    """Information about the current task."""

    task_id: str
    title: str
    description: str
    current_subtask: Optional[str] = None
    episode_id: Optional[str] = None
    completed_subtasks: Optional[List[str]] = None
    in_progress_subtasks: Optional[List[str]] = None
    not_visited_subtasks: Optional[List[str]] = None


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
