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
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, cast

import aiohttp
import httpx

from ...base import MCPHeaders

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
        self._sse_task: Optional[asyncio.Task[None]] = None
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

    async def get_policy_info(self, episode_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get policy information for a specific episode."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")
        if not episode_id:
            raise Exception("Episode ID required for policy retrieval")

        url = f"{self.base_url}/session/{session_id}/episodes/{episode_id}/policy"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
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

    # === Episode-First Architecture Methods ===

    async def create_episode(self, session_id: Optional[str] = None, task_id: Optional[str] = None) -> str:
        """Create a new episode for the session and return episode ID."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")
        if not task_id:
            raise Exception("Task ID required for episode creation")

        url = f"{self.base_url}/session/{session_id}/episodes?task_id={task_id}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    episode_id = cast(str, result["episode_id"])
                    logger.info(f"✅ Created episode: {episode_id} for task: {task_id}")
                    return episode_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create episode: {response.status} - {error_text}")

    async def end_episode(
        self,
        episode_id: str,
        reason: str = "completed",
        result: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """End an episode with the specified reason and result."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")

        url = f"{self.base_url}/session/{session_id}/episodes/{episode_id}"
        data: Dict[str, Any] = {"reason": reason}
        if result:
            data["result"] = result

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, json=data, timeout=self.request_timeout) as response:
                if response.status == 200:
                    logger.info(f"✅ Ended episode: {episode_id} (reason: {reason})")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to end episode: {response.status} - {error_text}")
                    return False

    async def get_episode_task(self, episode_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get task information for a specific episode."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")

        url = f"{self.base_url}/session/{session_id}/episodes/{episode_id}/task"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    return cast(Dict[str, Any], await response.json())
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get episode task: {response.status} - {error_text}")

    async def start_progress_stream(
        self, progress_callback: Callable[[Dict[str, Any]], None], episode_id: str, agent_id: str = "saber-harness"
    ) -> bool:
        """Start SSE stream for tool call progress updates from server with episode filtering."""
        if not self.session_id:
            logger.warning("No session ID available - cannot start progress stream")
            return False

        if not episode_id:
            raise ValueError("episode_id is required for episode-first streaming")

        self._progress_callback = progress_callback
        self._sse_stop = threading.Event()

        # Start the SSE stream in a background thread
        loop = asyncio.get_event_loop()
        self._sse_thread = threading.Thread(
            target=self._run_progress_stream_sync, args=(agent_id, episode_id, loop), daemon=True
        )
        self._sse_thread.start()
        return True

    def _run_progress_stream_sync(self, agent_id: str, episode_id: str, loop: asyncio.AbstractEventLoop) -> None:
        """Run the SSE stream synchronously in a background thread."""
        url = f"{self.base_url}/tool-events/stream"
        headers = {
            MCPHeaders.SESSION_ID: self.session_id,  # Use session ID for filtering
            MCPHeaders.EPISODE_ID: episode_id,  # Required episode filtering
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        # Keep read timeout None to stream indefinitely; use pool/write/connect timeouts
        timeout = httpx.Timeout(
            connect=self.request_timeout, read=None, write=self.request_timeout, pool=self.request_timeout
        )
        client = httpx.Client(timeout=timeout)
        logger.info("Connecting to server tool events SSE with httpx-sse…")

        try:
            from httpx_sse import connect_sse

            with connect_sse(client, "GET", url, headers=headers) as event_source:
                logger.info("✅ Tool events stream connected to server (httpx-sse)")
                for sse in event_source.iter_sse():
                    if self._sse_stop.is_set():
                        break
                    if sse is None:
                        continue
                    raw_data = getattr(sse, "data", None)
                    if not raw_data:
                        continue
                    try:
                        data = json.loads(raw_data)
                    except Exception as e:  # json error or others
                        logger.warning(f"⚠️ Invalid tool event payload: {e}; payload={raw_data!r}")
                        continue

                    if self._progress_callback:
                        loop.call_soon_threadsafe(self._progress_callback, data)
                    if self._sse_stop.is_set():
                        break
        except ImportError:
            logger.warning("httpx-sse not available - progress stream disabled")
        except Exception as e:
            if not self._sse_stop.is_set():
                logger.error(f"❌ Tool events stream error: {e}")
        finally:
            try:
                client.close()
            except Exception:
                pass

    async def stop_progress_stream(self) -> None:
        """Stop the SSE stream."""
        if hasattr(self, "_sse_stop"):
            self._sse_stop.set()

        if hasattr(self, "_sse_thread") and self._sse_thread:
            # Wait for thread to complete
            self._sse_thread.join(timeout=1.0)
            self._sse_thread = None
