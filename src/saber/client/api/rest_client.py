#!/usr/bin/env python3
"""
SABER REST Client

Handles all REST API communication with SABER server and MCP sidecar including:
- Session management
- Episode lifecycle
- Task and policy information retrieval
- MCP sidecar session registration
- Tool call progress SSE streams
"""

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, cast

import aiohttp
import httpx

logger = logging.getLogger(__name__)


class SABERRestClient:
    """
    REST client for SABER server and MCP sidecar communication.

    Provides clean interface for all REST operations including:
    - Session lifecycle management
    - Episode management
    - Task and policy data retrieval
    - MCP sidecar session registration
    - Tool call progress SSE streams
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        client_id: str = "saber-client",
        request_timeout: float = 30.0,
        sidecar_url: Optional[str] = None,
    ):
        """Initialize REST client."""

        # Base settings
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.request_timeout = request_timeout
        self.session_id: Optional[str] = None
        self.sidecar_url = sidecar_url

        # SSE connection management
        self._sse_session: Optional[aiohttp.ClientSession] = None  # kept for other REST calls if needed
        self._sse_task: Optional[asyncio.Task] = None
        self._progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._sse_client: Optional[httpx.AsyncClient] = None
        # Dedicated thread-based SSE (to avoid event-loop starvation by blocking work)
        self._sse_thread: Optional[threading.Thread] = None
        self._sse_stop = threading.Event()

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

    # === MCP Sidecar Communication ===

    async def register_with_sidecar(
        self, agent_id: str, saber_session_id: Optional[str] = None, task_id: Optional[str] = None
    ) -> bool:
        """
        Register this harness session with the MCP sidecar.

        Args:
            agent_id: Identifier for this harness as an agent
            saber_session_id: SABER session ID (defaults to current session)
            task_id: Optional task ID for context

        Returns:
            bool: True if registration successful, False otherwise
        """
        if not self.sidecar_url:
            logger.warning("No sidecar URL configured - skipping sidecar registration")
            return False

        session_id = saber_session_id or self.session_id
        if not session_id:
            raise Exception("No SABER session ID available for sidecar registration")

        url = f"{self.sidecar_url}/admin/sessions"
        payload = {
            "agent_id": agent_id,
            "saber_session_id": session_id,
            "task_id": task_id,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=self.request_timeout) as response:
                    if response.status == 200:
                        await response.json()
                        logger.info(f"✅ Registered with sidecar: {agent_id} → {session_id}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Sidecar registration failed: {response.status} - {error_text}")
                        return False
        except Exception as e:
            logger.error(f"❌ Sidecar registration error: {e}")
            return False

    async def unregister_from_sidecar(self, agent_id: str) -> bool:
        """
        Unregister this harness session from the MCP sidecar.

        Args:
            agent_id: Identifier used during registration

        Returns:
            bool: True if unregistration successful, False otherwise
        """
        if not self.sidecar_url:
            return True  # Nothing to unregister

        url = f"{self.sidecar_url}/admin/sessions/{agent_id}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, timeout=self.request_timeout) as response:
                    if response.status == 200:
                        logger.info(f"✅ Unregistered from sidecar: {agent_id}")
                        return True
                    else:
                        logger.warning(f"⚠️ Sidecar unregistration failed: {response.status}")
                        return False
        except Exception as e:
            logger.warning(f"⚠️ Sidecar unregistration error: {e}")
            return False

    async def start_progress_stream(
        self, progress_callback: Callable[[Dict[str, Any]], None], agent_id: str = "saber-harness"
    ) -> bool:
        """
        Start SSE stream for tool call progress updates from sidecar.

        Args:
            progress_callback: Function to call with progress updates
            agent_id: Agent ID to listen for progress updates

        Returns:
            bool: True if stream started successfully, False otherwise
        """
        if not self.sidecar_url:
            logger.warning("No sidecar URL configured - cannot start progress stream")
            return False

        # Prevent duplicates
        if self._sse_thread and self._sse_thread.is_alive():
            logger.warning("Progress stream already running")
            return True

        self._progress_callback = progress_callback
        self._sse_stop.clear()
        loop = asyncio.get_running_loop()

        def _thread_target() -> None:
            try:
                self._run_progress_stream_sync(agent_id, loop)
            except Exception as e:
                logger.error(f"❌ Progress stream thread error: {e}")

        self._sse_thread = threading.Thread(target=_thread_target, name="SSEClientThread", daemon=True)
        self._sse_thread.start()
        logger.info(f"📡 Started progress stream for agent: {agent_id}")
        return True

    async def stop_progress_stream(self) -> None:
        """Stop the SSE progress stream."""
        # Signal stop and close clients
        self._sse_stop.set()
        if self._sse_client:
            try:
                await self._sse_client.aclose()
            except Exception:
                pass
            self._sse_client = None
        if self._sse_session:
            await self._sse_session.close()
            self._sse_session = None
        # Join thread
        if self._sse_thread and self._sse_thread.is_alive():
            self._sse_thread.join(timeout=2.0)
        self._sse_thread = None

        self._progress_callback = None
        logger.info("📡 Progress stream stopped")

    def _run_progress_stream_sync(self, agent_id: str, loop: asyncio.AbstractEventLoop) -> None:
        """Blocking SSE consumer running in a dedicated thread (httpx-sse sync).

        Schedules progress callbacks back onto the asyncio loop.
        """
        if not self.sidecar_url:
            return
        from httpx_sse import connect_sse

        url = f"{self.sidecar_url}/progress/stream"
        headers = {
            "X-Agent-ID": agent_id,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        # Keep read timeout None to stream indefinitely; use pool/write/connect timeouts
        timeout = httpx.Timeout(
            connect=self.request_timeout, read=None, write=self.request_timeout, pool=self.request_timeout
        )
        client = httpx.Client(timeout=timeout)
        logger.info("Connecting to progress SSE with httpx-sse…")
        try:
            with connect_sse(client, "GET", url, headers=headers) as event_source:
                logger.info("✅ Progress stream connected (httpx-sse)")
                for sse in event_source.iter_sse():
                    if self._sse_stop.is_set():
                        break
                    if sse is None:
                        continue
                    raw_data = getattr(sse, "data", None)
                    if not raw_data:
                        continue
                    try:
                        import json

                        data = json.loads(raw_data)
                    except Exception as e:  # json error or others
                        logger.warning(f"⚠️ Invalid progress payload: {e}; payload={raw_data!r}")
                        continue

                    if self._progress_callback:
                        loop.call_soon_threadsafe(self._progress_callback, data)
                    if self._sse_stop.is_set():
                        break
        except Exception as e:
            if not self._sse_stop.is_set():
                logger.error(f"❌ Progress stream error: {e}")
        finally:
            try:
                client.close()
            except Exception:
                pass
