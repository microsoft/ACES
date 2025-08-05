"""
SABER Client - Server communication client.

Provides simple REST API client for communicating with the SABER server.
No complex logic - just HTTP requests to server endpoints.
Includes rich console UI for better user experience.
"""

from typing import Any, Dict, List, Optional

import httpx

from ..api_models import EpisodeInfo, PolicyInfo, StepResponse, TaskInfo
from .ui.console import SABERConsole, get_console
from .ui.panels import PanelFormatter, get_panel_formatter
from .ui.progress import ProgressManager, get_progress_manager


class ServerClient:
    """
    Simple REST API client for SABER server.

    Handles HTTP communication with the server without complex logic.
    All session state is maintained on the server side.
    Provides rich console UI for better user experience.
    """

    # Type annotations for instance attributes
    console: Optional[SABERConsole]
    progress: Optional[ProgressManager]
    panels: Optional[PanelFormatter]
    session_id: Optional[str]

    def __init__(self, server_url: str, timeout: float = 30.0, ui_enabled: bool = True):
        """
        Initialize server client.

        Args:
            server_url: Base URL of the SABER server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            ui_enabled: Enable rich console UI output
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.ui_enabled = ui_enabled
        self._client = httpx.AsyncClient(timeout=timeout)
        self.session_id = None

        # UI components
        if ui_enabled:
            self.console = get_console()
            self.progress = get_progress_manager()
            self.panels = get_panel_formatter()
        else:
            self.console = None
            self.progress = None
            self.panels = None

    async def create_session(self, client_id: str = "saber_client") -> str:
        """
        Create a new session with the server.

        Args:
            client_id: Client identifier for the session

        Returns:
            session_id: Unique session identifier
        """
        if self.ui_enabled and self.console:
            self.console.info(f"Creating session with client ID: {client_id}")

        try:
            if self.ui_enabled and self.progress:
                async with self.progress.async_spinner("Connecting to server"):
                    response = await self._client.post(f"{self.server_url}/session", params={"client_id": client_id})
                    response.raise_for_status()
            else:
                response = await self._client.post(f"{self.server_url}/session", params={"client_id": client_id})
                response.raise_for_status()

            data = response.json()
            session_id: str = data["session_id"]
            self.session_id = session_id

            if self.ui_enabled and self.panels:
                self.panels.connection_panel(
                    server_url=self.server_url, session_id=self.session_id, client_id=client_id, status="Connected"
                )

            return session_id

        except Exception as e:
            if self.ui_enabled and self.console:
                self.console.error(f"Failed to create session: {e}")
            raise

    async def list_tasks(self) -> List[TaskInfo]:
        """
        Get list of available tasks.

        Returns:
            List[TaskInfo]: Available tasks
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        try:
            response = await self._client.get(f"{self.server_url}/session/{self.session_id}/tasks")
            response.raise_for_status()

            data = response.json()
            tasks = [TaskInfo(**task) for task in data.get("tasks", [])]

            if self.ui_enabled and self.console:
                self.console.server_status(f"Found {len(tasks)} available tasks")

            return tasks

        except Exception as e:
            if self.ui_enabled and self.console:
                self.console.error(f"Failed to list tasks: {e}")
            raise

    async def start_task(self, task_id: str) -> TaskInfo:
        """
        Start a specific task.

        Args:
            task_id: ID of the task to start

        Returns:
            TaskInfo: Started task details
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        try:
            if self.ui_enabled and self.console:
                self.console.task_status(f"Starting task: {task_id}")

            if self.ui_enabled and self.progress:
                async with self.progress.async_spinner(f"Starting task {task_id}"):
                    response = await self._client.post(
                        f"{self.server_url}/session/{self.session_id}/start-task", params={"task_id": task_id}
                    )
                    response.raise_for_status()
            else:
                response = await self._client.post(
                    f"{self.server_url}/session/{self.session_id}/start-task", params={"task_id": task_id}
                )
                response.raise_for_status()

            data = response.json()
            task_info = TaskInfo(**data)

            if self.ui_enabled and self.panels:
                progress_str = None
                if task_info.completed_subtasks is not None and hasattr(task_info, "subtasks"):
                    # Note: We'd need to add subtasks to TaskInfo model or get it differently
                    total_subtasks = (
                        len(task_info.completed_subtasks)
                        + len(task_info.in_progress_subtasks or [])
                        + len(task_info.not_visited_subtasks or [])
                    )
                    progress_str = f"{len(task_info.completed_subtasks)}/{total_subtasks}"

                self.panels.task_panel(
                    task_title=task_info.title,
                    task_description=task_info.description,
                    progress=progress_str,
                    status="active",
                )

            return task_info

        except Exception as e:
            if self.ui_enabled and self.console:
                self.console.error(f"Failed to start task: {e}")
            raise

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

    async def execute_step(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> StepResponse:
        """
        Execute a step with the agent's command.

        Args:
            command: The command to execute
            parameters: Optional parameters for the command

        Returns:
            StepResponse: Execution results and next state info
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        if parameters is None:
            parameters = {}

        payload = {"command": command, "parameters": parameters}

        try:
            if self.ui_enabled and self.console:
                self.console.step_status(f"Executing: {command}")

            if self.ui_enabled and self.progress:
                async with self.progress.async_spinner(f"Running {command}"):
                    response = await self._client.post(
                        f"{self.server_url}/session/{self.session_id}/step", json=payload
                    )
                    response.raise_for_status()
            else:
                response = await self._client.post(f"{self.server_url}/session/{self.session_id}/step", json=payload)
                response.raise_for_status()

            data = response.json()

            # Extract step response from server response
            step_data = {
                "success": data.get("success", False),
                "output": data.get("data", {}).get("output", ""),
                "error": data.get("error"),
                "done": data.get("step", {}).get("done", False),
                "info": data.get("step", {}),
                "task_completed": data.get("task_completed", False),  # Add task completion flag
            }

            step_response = StepResponse(**step_data)

            if self.ui_enabled and self.console:
                if step_response.success:
                    self.console.success(f"Command '{command}' completed successfully")
                    if step_response.output:
                        # Show truncated output
                        output_preview = (
                            step_response.output[:200] + "..."
                            if len(step_response.output) > 200
                            else step_response.output
                        )
                        self.console.dim(f"Output: {output_preview}")
                else:
                    self.console.error(f"Command '{command}' failed: {step_response.error or 'Unknown error'}")

                if step_response.done:
                    self.console.success("🎉 Task completed!")

            return step_response

        except Exception as e:
            if self.ui_enabled and self.console:
                self.console.error(f"Failed to execute step: {e}")
            raise

    async def get_current_task(self) -> TaskInfo:
        """
        Get information about the current task.

        Returns:
            TaskInfo: Current task details
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        try:
            response = await self._client.get(f"{self.server_url}/session/{self.session_id}/current-task")
            response.raise_for_status()

            data = response.json()
            task_info = TaskInfo(**data)

            if self.ui_enabled and self.console:
                status = "completed" if task_info.completed else "active"
                self.console.task_status(f"Current task: {task_info.title} ({status})")

            return task_info

        except Exception as e:
            if self.ui_enabled and self.console:
                self.console.error(f"Failed to get current task: {e}")
            raise

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

            if self.ui_enabled and self.console:
                self.console.success("Disconnected from server")

        finally:
            self.session_id = None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check server health status.

        Returns:
            Dict: Server health information
        """
        try:
            response = await self._client.get(f"{self.server_url}/health")
            response.raise_for_status()

            health_data = response.json()

            if self.ui_enabled and self.console:
                status = health_data.get("status", "unknown")
                self.console.server_status(f"Server health: {status}")

            return health_data  # type: ignore[no-any-return]

        except Exception as e:
            if self.ui_enabled and self.console:
                self.console.error(f"Health check failed: {e}")
            raise

    async def wait_for_server(self, max_retries: int = 30, delay: float = 2.0) -> bool:
        """
        Wait for the server to be ready with UI feedback.

        Args:
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds

        Returns:
            bool: True if server is ready, False if timeout
        """
        if self.ui_enabled and self.console:
            self.console.info("Waiting for server to be ready...")

        for attempt in range(max_retries):
            try:
                health = await self.health_check()
                if health.get("status") == "healthy":
                    if self.ui_enabled and self.console:
                        self.console.success("Server is ready!")
                    return True
            except Exception:
                pass

            if attempt < max_retries - 1:
                if self.ui_enabled and self.console and attempt % 5 == 0:
                    remaining = max_retries - attempt - 1
                    self.console.info(f"Still waiting... ({remaining} attempts remaining)")

                import asyncio

                await asyncio.sleep(delay)

        if self.ui_enabled and self.console:
            self.console.error(f"Server not ready after {max_retries * delay} seconds")
        return False

    async def __aenter__(self) -> "ServerClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        await self.close_session()
        await self._client.aclose()
