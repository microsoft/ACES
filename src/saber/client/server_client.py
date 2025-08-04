"""
SABER Client - Server communication client.

Provides simple REST API client for communicating with the SABER server.
No complex logic - just HTTP requests to server endpoints.
"""

from typing import Any, Dict, Optional

import httpx

from ..api_models import EpisodeInfo, PolicyInfo, StepResponse, TaskInfo


class ServerClient:
    """
    Simple REST API client for SABER server.

    Handles HTTP communication with the server without complex logic.
    All session state is maintained on the server side.
    """

    def __init__(self, server_url: str, timeout: float = 30.0):
        """
        Initialize server client.

        Args:
            server_url: Base URL of the SABER server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._client = httpx.AsyncClient(timeout=timeout)

    async def create_session(self, client_id: str = "saber_client") -> str:
        """
        Create a new session with the server.

        Args:
            client_id: Client identifier for the session

        Returns:
            session_id: Unique session identifier
        """
        response = await self._client.post(f"{self.server_url}/session", params={"client_id": client_id})
        response.raise_for_status()

        data = response.json()
        self.session_id = data["session_id"]
        return self.session_id

    async def start_episode(self, task_id: str = "default_task") -> EpisodeInfo:
        """
        Start a new episode for the current session.

        Args:
            task_id: ID of the task to start (defaults to "default_task")

        Returns:
            EpisodeInfo: Episode details
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        response = await self._client.post(
            f"{self.server_url}/session/{self.session_id}/start-episode", params={"task_id": task_id}
        )
        response.raise_for_status()

        data = response.json()
        return EpisodeInfo(**data)

    async def execute_step(self, agent_response: str) -> StepResponse:
        """
        Execute a step with the agent's response/command.

        Args:
            agent_response: The command or response from the agent

        Returns:
            StepResponse: Execution results and next state info
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        response = await self._client.post(
            f"{self.server_url}/session/{self.session_id}/step", json={"command": agent_response}
        )
        response.raise_for_status()

        data = response.json()

        # Extract step response from server response
        step_data = {
            "success": data.get("success", False),
            "output": data.get("data", {}).get("output", ""),
            "error": data.get("error"),
            "done": data.get("step", {}).get("done", False),
            "info": data.get("step", {}),
        }

        return StepResponse(**step_data)

    async def get_current_task(self) -> TaskInfo:
        """
        Get information about the current task.

        Returns:
            TaskInfo: Current task details
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        response = await self._client.get(f"{self.server_url}/session/{self.session_id}/current-task")
        response.raise_for_status()

        data = response.json()
        return TaskInfo(**data)

    async def get_policy(self) -> PolicyInfo:
        """
        Get domain policy information.

        Returns:
            PolicyInfo: Available commands, guidelines, and constraints
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        response = await self._client.get(f"{self.server_url}/session/{self.session_id}/policy")
        response.raise_for_status()

        data = response.json()
        return PolicyInfo(**data)

    async def close_session(self) -> None:
        """Close the current session and cleanup."""
        if not self.session_id:
            return

        try:
            response = await self._client.delete(f"{self.server_url}/session/{self.session_id}")
            response.raise_for_status()
        finally:
            self.session_id = None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check server health status.

        Returns:
            Dict: Server health information
        """
        response = await self._client.get(f"{self.server_url}/health")
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def __aenter__(self) -> "ServerClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        await self.close_session()
        await self._client.aclose()
