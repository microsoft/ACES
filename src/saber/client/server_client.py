"""
SABER Client - Server communication client.

Provides REST API client for session management and MCP client for tool execution.
REST API handles sessions, episodes, policy, status, and events.
MCP handles tool discovery and tool execution.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..api_models import EpisodeInfo, PolicyInfo, StepResponse, TaskInfo
from .ui.console import SABERConsole, get_console
from .ui.panels import PanelFormatter, get_panel_formatter
from .ui.progress import ProgressManager, get_progress_manager

logger = logging.getLogger(__name__)


class ServerClient:
    """
    SABER server client with dual protocol support.

    Handles REST API for session management and MCP for tool execution.
    REST API: sessions, episodes, policy, status, events
    MCP: tool discovery and execution
    """

    # Type annotations for instance attributes
    console: Optional[SABERConsole]
    progress: Optional[ProgressManager]
    panels: Optional[PanelFormatter]
    session_id: Optional[str]
    mcp_client: Optional[Any]  # Will be MCPClient when implemented

    def __init__(
        self,
        rest_api_url_or_server_url: str,
        mcp_api_url: Optional[str] = None,
        timeout: float = 30.0,
        ui_enabled: bool = True,
    ):
        """
        Initialize server client with dual protocol support.

        Args:
            rest_api_url_or_server_url: Base URL of the SABER REST API (e.g., "http://localhost:8000")
                                       For backward compatibility, also accepts server_url
            mcp_api_url: Base URL of the SABER MCP API (e.g., "http://localhost:3001")
            timeout: Request timeout in seconds
            ui_enabled: Enable rich console UI output
        """
        self.rest_api_url = rest_api_url_or_server_url.rstrip("/")
        # For backward compatibility
        self.server_url = self.rest_api_url
        self.mcp_api_url = mcp_api_url.rstrip("/") if mcp_api_url else None
        self.timeout = timeout
        self.ui_enabled = ui_enabled
        self._client = httpx.AsyncClient(timeout=timeout)
        self.session_id = None
        self.mcp_client = None

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
                    response = await self._client.post(f"{self.rest_api_url}/session", params={"client_id": client_id})
                    response.raise_for_status()
            else:
                response = await self._client.post(f"{self.rest_api_url}/session", params={"client_id": client_id})
                response.raise_for_status()

            data = response.json()
            session_id: str = data["session_id"]
            self.session_id = session_id

            if self.ui_enabled and self.panels:
                self.panels.connection_panel(
                    server_url=self.rest_api_url, session_id=self.session_id, client_id=client_id, status="Connected"
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
            response = await self._client.get(f"{self.rest_api_url}/session/{self.session_id}/tasks")
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
                        f"{self.rest_api_url}/session/{self.session_id}/start-task", params={"task_id": task_id}
                    )
                    response.raise_for_status()
            else:
                response = await self._client.post(
                    f"{self.rest_api_url}/session/{self.session_id}/start-task", params={"task_id": task_id}
                )
                response.raise_for_status()

            data = response.json()
            task_info = TaskInfo(**data)

            if self.ui_enabled and self.panels:
                progress_str = None
                if task_info.step_count is not None:
                    progress_str = f"Step {task_info.step_count}"

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
            f"{self.rest_api_url}/session/{self.session_id}/start-episode", params={"task_id": task_id}
        )
        response.raise_for_status()

        data = response.json()
        return EpisodeInfo(**data)

    async def get_current_task(self) -> TaskInfo:
        """
        Get information about the current task.

        Returns:
            TaskInfo: Current task details
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        try:
            response = await self._client.get(f"{self.rest_api_url}/session/{self.session_id}/current-task")
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

        response = await self._client.get(f"{self.rest_api_url}/session/{self.session_id}/policy")
        response.raise_for_status()

        data = response.json()
        return PolicyInfo(**data)

    async def close_session(self) -> None:
        """Close the current session and cleanup."""
        if not self.session_id:
            return

        try:
            response = await self._client.delete(f"{self.rest_api_url}/session/{self.session_id}")
            response.raise_for_status()

            if self.ui_enabled and self.console:
                self.console.success("Disconnected from server")

        finally:
            self.session_id = None

    async def execute_step(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> StepResponse:
        """
        DEPRECATED: Execute a step with the agent's command.

        This method is provided for backward compatibility only.
        New code should use MCP tool execution via call_tool().

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

        if self.ui_enabled and self.console:
            self.console.step_status(f"Executing (deprecated): {command}")

        # Make REST API call to step endpoint for backward compatibility
        try:
            response = await self._client.post(
                f"{self.rest_api_url}/session/{self.session_id}/step",
                json={"command": command, "parameters": parameters},
            )
            response.raise_for_status()
            response_data = response.json()

            # Parse the step info properly
            step_data = response_data.get("step", {})
            step_info = {k: v for k, v in step_data.items() if k != "done"}

            return StepResponse(
                success=response_data.get("success", False),
                output=response_data.get("data", {}).get("output", ""),
                error=response_data.get("error"),
                done=step_data.get("done", False),
                info=step_info,
                task_completed=step_data.get("done", False),
            )

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return StepResponse(
                success=False,
                output="",
                error=f"Step execution failed: {str(e)}",
                done=True,
                info={},
                task_completed=False,
            )

    async def connect_mcp(self) -> None:
        """
        Connect to MCP server for tool execution.

        Raises:
            ValueError: If no MCP URL configured or no active session
        """
        if not self.mcp_api_url:
            raise ValueError("No MCP API URL configured")

        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")

        if self.ui_enabled and self.console:
            self.console.info("Connecting to MCP server...")

        # TODO: Implement MCPClient connection
        # self.mcp_client = MCPClient(self.mcp_api_url, {"session_id": self.session_id})
        # await self.mcp_client.connect()

        if self.ui_enabled and self.console:
            self.console.success("Connected to MCP server")

    async def disconnect_mcp(self) -> None:
        """Disconnect from MCP server."""
        if self.mcp_client:
            # TODO: Implement disconnect
            # await self.mcp_client.disconnect()
            self.mcp_client = None

            if self.ui_enabled and self.console:
                self.console.info("Disconnected from MCP server")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available MCP tools.

        Returns:
            List of MCP tool definitions

        Raises:
            ValueError: If MCP not connected
        """
        if not self.mcp_client:
            raise ValueError("MCP not connected. Call connect_mcp() first.")

        # TODO: Implement tool listing
        # return await self.mcp_client.list_tools()
        return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute MCP tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            MCP tool execution result

        Raises:
            ValueError: If MCP not connected
        """
        if not self.mcp_client:
            raise ValueError("MCP not connected. Call connect_mcp() first.")

        # Add session context to arguments
        arguments = arguments.copy()
        arguments["session_id"] = self.session_id

        if self.ui_enabled and self.console:
            self.console.step_status(f"Executing tool: {name}")

        # TODO: Implement tool execution
        # result = await self.mcp_client.call_tool(name, arguments)
        #
        # if self.ui_enabled and self.console:
        #     if not result.get("isError", False):
        #         self.console.success(f"Tool '{name}' completed successfully")
        #     else:
        #         self.console.error(f"Tool '{name}' failed")
        #
        # return result

        return {"content": [{"type": "text", "text": "MCP client not implemented yet"}], "isError": False}

    async def health_check(self) -> Dict[str, Any]:
        """
        Check server health status.

        Returns:
            Dict: Server health information
        """
        try:
            response = await self._client.get(f"{self.rest_api_url}/health")
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
