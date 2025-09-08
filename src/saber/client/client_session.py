"""
Client Session Manager for SABER client-server communication.

Manages sessions, episodes, policy interactions, and MCP tool access from the client side.
"""

import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp

from ..models import (  # Use shared api models directly
    BenchmarkInfo,
    EpisodeCreateResponse,
    PolicyResponse,
    SessionCreateResponse,
    TaskInfo,
)
from ..models.mcp import MCPTool, MCPToolCallRequest, MCPToolCallResponse, SessionContext
from .api.mcp_client import MCPClient
from .models import SessionManagerConfig

logger = logging.getLogger(__name__)


class ClientSessionManager:
    """
    Manages SABER sessions and episodes for client-server communication.

    Handles:
    - Session creation and management via REST API
    - Episode lifecycle via REST API
    - Policy endpoint integration
    - MCP tool discovery and execution
    """

    def __init__(self, config: SessionManagerConfig):
        """
        Initialize session manager with unified configuration.

        Args:
            config: Unified SessionManagerConfig
        """
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.client_id = config.client_id
        self.timeout = config.rest_timeout
        self.mcp_config = config.to_mcp_config()
        self._current_session_id: Optional[str] = None

        # Episode-scoped MCP client pool - each episode gets its own client
        self.mcp_client_pool: Dict[str, MCPClient] = {}

        logger.debug(f"Initialized ClientSessionManager for: {self.base_url}")
        logger.debug(f"MCP endpoint: {self.mcp_config.base_url}")

    async def create_session(self) -> str:
        """
        Create a new SABER session via REST API.

        Returns:
            Session ID

        Raises:
            Exception: If session creation fails
        """
        logger.info(f"Creating new SABER session for client: {self.client_id}")

        url = f"{self.base_url}/session"
        params = {"client_id": self.client_id}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    session_response = SessionCreateResponse(**data)
                    self._current_session_id = session_response.session_id

                    logger.info(f"Created session: {session_response.session_id}")
                    return session_response.session_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create session: {response.status} - {error_text}")

    async def create_episode(self, session_id: str, task_id: str) -> EpisodeCreateResponse:
        """
        Create a new episode within a session via REST API.

        Args:
            session_id: Session ID
            task_id: Task ID for the episode

        Returns:
            EpisodeCreateResponse object from server

        Raises:
            Exception: If episode creation fails
        """
        logger.info(f"Creating episode for session {session_id}, task {task_id}")

        url = f"{self.base_url}/session/{session_id}/episodes"
        params = {"task_id": task_id}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    episode_response = EpisodeCreateResponse(**data)

                    logger.info(f"Created episode: {episode_response.episode_id}")
                    return episode_response
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create episode: {response.status} - {error_text}")

    async def get_policy_response(self, session_id: str, episode_id: str) -> Optional[PolicyResponse]:
        """
        Get policy response for dynamic prompt generation via REST API.

        Args:
            session_id: Session ID
            episode_id: Episode ID

        Returns:
            Policy response with prompt, or None if not available

        Raises:
            Exception: If policy request fails
        """
        logger.debug(f"Getting policy response for episode {episode_id}")

        url = f"{self.base_url}/session/{session_id}/episodes/{episode_id}/policy"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        policy_response = PolicyResponse(prompt=data.get("prompt", ""), domain=data.get("domain"))

                        logger.debug(f"Retrieved policy response for episode {episode_id}")
                        return policy_response
                    elif response.status == 404:
                        logger.debug(f"No policy available for episode {episode_id}")
                        return None
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get policy: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            logger.warning(f"Policy request timeout for episode {episode_id}")
            return None
        except Exception as e:
            logger.error(f"Policy request failed for episode {episode_id}: {e}")
            raise

    async def get_available_tasks(self) -> List[TaskInfo]:
        """
        Get all available tasks from SABER server.

        Returns:
            List of TaskInfo objects from server

        Raises:
            Exception: If task discovery fails
        """
        logger.info("Discovering all available tasks via REST API")

        url = f"{self.base_url}/benchmark"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Parse the BenchmarkInfo response
                    benchmark_info = BenchmarkInfo(**data)
                    logger.info(f"Discovered {len(benchmark_info.tasks)} available tasks")
                    return benchmark_info.tasks
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get available tasks: {response.status} - {error_text}")

    async def get_tasks(self, task_ids: List[str]) -> List[TaskInfo]:
        """
        Get full task data for specific task IDs.

        Args:
            task_ids: List of task IDs to retrieve

        Returns:
            List of TaskInfo objects

        Raises:
            Exception: If task retrieval fails
        """
        logger.info(f"Retrieving {len(task_ids)} tasks via REST API")

        # Get all available tasks as TaskInfo objects
        all_tasks = await self.get_available_tasks()

        # Filter to requested task IDs
        requested_tasks = [task for task in all_tasks if task.task_id in task_ids]

        # Check if all requested tasks were found
        found_ids = {task.task_id for task in requested_tasks}
        missing_ids = set(task_ids) - found_ids

        if missing_ids:
            raise Exception(f"Tasks not found: {missing_ids}")

        logger.info(f"Retrieved {len(requested_tasks)} tasks successfully as TaskInfo objects")
        return requested_tasks

    async def terminate_session(self, session_id: str) -> None:
        """
        Terminate a session via REST API.

        Args:
            session_id: Session ID to terminate

        Raises:
            Exception: If session termination fails
        """
        logger.info(f"Terminating session: {session_id}")

        url = f"{self.base_url}/session/{session_id}"

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, timeout=self.timeout) as response:
                if response.status == 200:
                    logger.info(f"Session {session_id} terminated successfully")
                    if self._current_session_id == session_id:
                        self._current_session_id = None
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to terminate session {session_id}: {response.status} - {error_text}")
                    # Don't raise - termination failures shouldn't break cleanup

    async def end_episode(
        self, session_id: str, episode_id: str, reason: str = "completed", result: Optional[str] = None
    ) -> None:
        """
        End an episode via REST API and cleanup its MCP client.

        Args:
            session_id: Session ID
            episode_id: Episode ID
            reason: Completion reason
            result: Optional result data
        """
        logger.debug(f"Ending episode {episode_id} with reason: {reason}")

        # Cleanup episode MCP client first
        await self.cleanup_episode_mcp_client(episode_id)

        url = f"{self.base_url}/session/{session_id}/episodes/{episode_id}"
        params = {"reason": reason}
        if result:
            params["result"] = result

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        logger.debug(f"Episode {episode_id} ended successfully")
                    else:
                        error_text = await response.text()
                        logger.warning(f"Failed to end episode {episode_id}: {response.status} - {error_text}")
        except Exception as e:
            logger.warning(f"Episode end request failed for {episode_id}: {e}")
            # Don't raise - episode end failures shouldn't break cleanup

    async def update_episode_status(self, episode_id: str, status: str) -> None:
        """
        Update episode status.

        Args:
            episode_id: Episode ID
            status: New status
        """
        logger.debug(f"Updating episode {episode_id} status to: {status}")

        # TODO: Implement if server supports status updates
        await asyncio.sleep(0.01)

    def get_current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session_id

    def get_or_create_session_id(self) -> str:
        """Get current session ID or raise if none exists."""
        if not self._current_session_id:
            raise RuntimeError("No active session - call create_session() first")
        return self._current_session_id

    # ========================================
    # Enhanced MCP Tool Integration Methods
    # ========================================

    def _build_episode_context(self, episode_id: str, task_id: Optional[str] = None) -> SessionContext:
        """Build session context for episode-specific MCP requests."""
        if not self._current_session_id:
            raise RuntimeError("No active session for MCP operations")

        return SessionContext(
            session_id=self._current_session_id,
            episode_id=episode_id,
            task_id=task_id,
            client_id=self.mcp_config.client_id,
        )

    async def ensure_episode_mcp_client(self, episode_id: str, task_id: Optional[str] = None) -> None:
        """
        Ensure an MCP client exists for the given episode.

        Creates a new episode-specific MCP client if one doesn't exist.
        Each client is connected with episode context in headers.

        Args:
            episode_id: Episode ID (required for all MCP operations)
            task_id: Optional task ID for request context

        Raises:
            RuntimeError: No active session or MCP connection fails
        """
        if episode_id in self.mcp_client_pool:
            # Client already exists for this episode
            return

        if not self._current_session_id:
            raise RuntimeError("No active session - call create_session() first")

        # Create new MCP client for this episode
        episode_mcp_client = MCPClient(self.mcp_config)
        episode_context = self._build_episode_context(episode_id, task_id)

        try:
            await episode_mcp_client.connect(episode_context)
            self.mcp_client_pool[episode_id] = episode_mcp_client
            logger.info(f"Created MCP client for episode: {episode_id}")
        except Exception as e:
            logger.error(f"Failed to create MCP client for episode {episode_id}: {e}")
            # Fail fast - episode fails if MCP client cannot be created
            raise RuntimeError(f"Episode {episode_id} failed: MCP client connection failed: {e}") from e

    def _get_episode_mcp_client(self, episode_id: str) -> MCPClient:
        """
        Get the MCP client for a specific episode.

        Args:
            episode_id: Episode ID

        Returns:
            MCPClient for the episode

        Raises:
            RuntimeError: No MCP client exists for episode
        """
        if episode_id not in self.mcp_client_pool:
            raise RuntimeError(f"No MCP client for episode {episode_id} - call ensure_episode_mcp_client() first")

        return self.mcp_client_pool[episode_id]

    async def cleanup_episode_mcp_client(self, episode_id: str) -> None:
        """
        Cleanup MCP client for a specific episode.

        Args:
            episode_id: Episode ID to cleanup
        """
        if episode_id in self.mcp_client_pool:
            mcp_client = self.mcp_client_pool[episode_id]
            try:
                await mcp_client.disconnect()
                logger.info(f"Disconnected MCP client for episode: {episode_id}")
            except Exception as e:
                logger.warning(f"Error disconnecting MCP client for episode {episode_id}: {e}")
            finally:
                del self.mcp_client_pool[episode_id]

    async def cleanup_all_mcp_clients(self) -> None:
        """Cleanup all episode MCP clients."""
        episode_ids = list(self.mcp_client_pool.keys())
        for episode_id in episode_ids:
            await self.cleanup_episode_mcp_client(episode_id)

    async def list_mcp_tools(self, episode_id: str, task_id: Optional[str] = None) -> List[MCPTool]:
        """
        List available MCP tools for a specific episode.

        Args:
            episode_id: Episode ID (required for all MCP operations)
            task_id: Optional task ID for request context

        Returns:
            List of available tools

        Raises:
            RuntimeError: No MCP client exists for episode
        """
        episode_mcp_client = self._get_episode_mcp_client(episode_id)
        return await episode_mcp_client.discover_tools(episode_id=episode_id, task_id=task_id)

    async def execute_mcp_tool(self, request: MCPToolCallRequest) -> MCPToolCallResponse:
        """
        Execute MCP tool using episode-specific client.

        Args:
            request: Typed tool execution request (must include episode_id)

        Returns:
            Typed tool execution response

        Raises:
            RuntimeError: No MCP client exists for episode
            ValueError: Missing episode_id in request
        """
        if not request.episode_id:
            raise ValueError("episode_id is required for all MCP tool executions")

        episode_mcp_client = self._get_episode_mcp_client(request.episode_id)
        return await episode_mcp_client.execute_tool(request)

    async def cleanup(self) -> None:
        """Cleanup session manager resources including all MCP connections."""
        await self.cleanup_all_mcp_clients()

        if self._current_session_id:
            await self.terminate_session(self._current_session_id)

        logger.info("ClientSessionManager cleanup completed")
