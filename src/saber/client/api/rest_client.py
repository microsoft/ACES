#!/usr/bin/env python3
"""
SABER REST Client

Handles all REST API communication with SABER server including:
- Session management
- Episode lifecycle
- Task and policy information retrieval
"""

import logging
from typing import Any, Dict, List, Optional, cast

import aiohttp

logger = logging.getLogger(__name__)


class SABERRestClient:
    """
    REST client for SABER server communication.

    Provides clean interface for all REST operations including:
    - Session lifecycle management
    - Episode management
    - Task and policy data retrieval
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        client_id: str = "saber-client",
        request_timeout: float = 30.0,
    ):
        """
        Initialize REST client.

        Args:
            base_url: Base URL of SABER REST server
            client_id: Client identifier for sessions
            request_timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.request_timeout = request_timeout
        self.session_id: Optional[str] = None

    async def create_session(self) -> str:
        """Create a new SABER session and return session ID."""
        url = f"{self.base_url}/session"
        params = {"client_id": self.client_id}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    self.session_id = cast(str, data["session_id"])
                    logger.info(f"✅ Created SABER session: {self.session_id}")
                    return self.session_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create session: {response.status} - {error_text}")

    async def get_benchmark(self) -> Dict[str, Any]:
        """Get complete benchmark task list for client-orchestrated execution."""
        url = f"{self.base_url}/get-benchmark"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(
                        f"✅ Retrieved benchmark: {data['total_episodes']} episodes for {data['total_tasks']} tasks"
                    )
                    return cast(Dict[str, Any], data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get benchmark: {response.status} - {error_text}")

    async def start_episode(self, task_id: str, session_id: Optional[str] = None) -> str:
        """Start episode for the given task and return episode ID."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")

        url = f"{self.base_url}/session/{session_id}/start-episode"
        params = {"task_id": task_id}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    episode_id = cast(str, data["episode_id"])
                    logger.info(f"✅ Started episode: {episode_id} for task: {task_id}")
                    return episode_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to start episode: {response.status} - {error_text}")

    async def start_benchmark(
        self, session_id: Optional[str] = None, task_ids: Optional[List[str]] = None, episode_attempts: int = 1
    ) -> Dict[str, Any]:
        """Start benchmark mode with specified tasks and episode attempts."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")

        url = f"{self.base_url}/session/{session_id}/start-benchmark"

        # Build benchmark configuration
        benchmark_config: Dict[str, Any] = {"episode_attempts": episode_attempts}
        if task_ids:
            benchmark_config["task_ids"] = task_ids

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=benchmark_config, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    task_desc = f"tasks: {task_ids}" if task_ids else "all available tasks"
                    logger.info(
                        f"✅ Started benchmark mode for session: {session_id} "
                        f"({task_desc}, {episode_attempts} episodes each)"
                    )
                    return cast(Dict[str, Any], data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to start benchmark: {response.status} - {error_text}")

    async def get_task_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current task information."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")

        url = f"{self.base_url}/session/{session_id}/current-task"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    return cast(Dict[str, Any], await response.json())
                else:
                    raise Exception(f"Failed to get task info: {response.status}")

    async def get_policy_info(self, session_id: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get policy information for a specific task."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")
        if not task_id:
            raise Exception("Task ID required for policy retrieval")

        url = f"{self.base_url}/session/{session_id}/policy"
        params = {"task_id": task_id}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=self.request_timeout) as response:
                if response.status == 200:
                    return cast(Dict[str, Any], await response.json())
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get policy info: {response.status} - {error_text}")

    async def list_tasks(self) -> List[Dict[str, Any]]:
        """List all available tasks."""
        url = f"{self.base_url}/benchmark"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    tasks = cast(List[Dict[str, Any]], data.get("tasks", []))
                    logger.info(f"✅ Retrieved {len(tasks)} available tasks")
                    return tasks
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list tasks: {response.status} - {error_text}")

    async def terminate_session(self, session_id: Optional[str] = None) -> bool:
        """Terminate the session."""
        session_id = session_id or self.session_id
        if not session_id:
            return True  # Already terminated

        url = f"{self.base_url}/session/{session_id}"

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    logger.info(f"✅ Terminated session: {session_id}")
                    if session_id == self.session_id:
                        self.session_id = None
                    return True
                else:
                    logger.warning(f"⚠️ Failed to terminate session: {response.status}")
                    return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check against the server."""
        url = f"{self.base_url}/health"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    return cast(Dict[str, Any], await response.json())
                else:
                    raise Exception(f"Health check failed: {response.status}")
