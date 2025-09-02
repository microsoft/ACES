#!/usr/bin/env python3
"""
Null UI Adapter

Fallback adapter when UI is disabled or unavailable.
Provides basic console output for debugging.
"""

from datetime import datetime
from typing import Any

from ..interfaces import MCPToolCall, SABERUISessionInfo, SABERUISessionSummary, SABERUITaskInfo, SABERUITaskResult


class NullUIAdapter:
    """No-op adapter for when UI is disabled or unavailable."""

    def __init__(self) -> None:
        """Initialize null adapter."""
        self.start_time = datetime.now()

    async def session_start(self, session_info: SABERUISessionInfo) -> None:
        """Initialize session with basic logging."""
        print("🚀 SABER Session Started")
        print(f"   Session ID: {session_info.session_id}")
        print(f"   Total Tasks: {session_info.total_tasks}")
        print(f"   Parallel Episodes: {session_info.parallel_episodes}")
        print("")

    async def session_update(self, session_info: SABERUISessionInfo) -> None:
        """Update session information."""
        print(f"📊 Session Updated - Total Tasks: {session_info.total_tasks}")

    async def task_start(self, task_info: SABERUITaskInfo) -> None:
        """Start task with basic logging."""
        print(f"📋 Starting Task: {task_info.task_id} (attempt {task_info.attempt})")
        print(f"   State: {task_info.state.value}")
        if task_info.progress_message:
            print(f"   Progress: {task_info.progress_message}")
        print("")

    async def task_update(self, task_info: SABERUITaskInfo) -> None:
        """Update task progress."""
        print(f"🔄 Task Update: {task_info.task_id}")
        print(f"   State: {task_info.state.value}")
        if task_info.progress_message:
            print(f"   Progress: {task_info.progress_message}")

    async def tool_call_start(self, tool_call: MCPToolCall) -> None:
        """Log tool call start."""
        print(f"🔧 Tool Call Started: {tool_call.tool_name} (ID: {tool_call.call_id})")
        if tool_call.input_args:
            print(f"   Args: {tool_call.input_args}")

    async def tool_call_complete(self, tool_call: MCPToolCall) -> None:
        """Log tool call completion."""
        status_emoji = "✅" if tool_call.status.value == "completed" else "❌"
        print(f"{status_emoji} Tool Call Complete: {tool_call.tool_name} (ID: {tool_call.call_id})")
        if tool_call.output:
            print(f"   Output: {tool_call.output[:100]}...")
        if tool_call.error:
            print(f"   Error: {tool_call.error}")

    async def task_complete(self, result: SABERUITaskResult) -> None:
        """Complete task with result summary."""
        status_emoji = "✅" if result.success else "❌"
        print(f"{status_emoji} Task Complete: {result.task_id}")
        print(f"   Episode ID: {result.episode_id}")
        print(f"   Success: {result.success}")
        print(f"   Iterations: {result.iterations}")
        print(f"   Termination: {result.termination_reason}")
        if result.flag:
            print(f"   Flag: {result.flag}")
        if result.error_message:
            print(f"   Error: {result.error_message}")
        print("")

    async def session_complete(self, summary: SABERUISessionSummary) -> None:
        """Complete session with summary."""
        success_rate = (summary.successful_episodes / summary.total_episodes * 100) if summary.total_episodes > 0 else 0
        print(f"🎉 Session Complete: {summary.session_id}")
        print(f"   Total Episodes: {summary.total_episodes}")
        print(f"   Successful: {summary.successful_episodes}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Final Success: {summary.final_success}")
        print("")

    async def session_cleanup(self) -> None:
        """Clean up session resources."""
        print("🧹 Session cleanup complete")

    async def __aenter__(self) -> "NullUIAdapter":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async context manager exit."""
        return False
