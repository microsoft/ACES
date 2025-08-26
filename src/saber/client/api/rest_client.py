#!/usr/bin/env python3
"""
SABER REST Client

Handles all REST API communication with SABER server including:
- Session management
- Episode lifecycle
- Task and policy information retrieval
- Event monitoring via SSE
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, cast

import aiohttp

logger = logging.getLogger(__name__)


class SABERRestClient:
    """
    REST client for SABER server communication.

    Provides clean interface for all REST operations including:
    - Session lifecycle management
    - Episode management
    - Task and policy data retrieval
    - Server-Sent Events monitoring
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

    async def get_policy_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get policy information."""
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")

        url = f"{self.base_url}/session/{session_id}/policy"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    return cast(Dict[str, Any], await response.json())
                else:
                    raise Exception(f"Failed to get policy info: {response.status}")

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

    async def monitor_episode_events(
        self,
        episode_id: str,
        session_id: Optional[str] = None,
        event_handler: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        """
        Monitor episode status via Server-Sent Events.

        Args:
            episode_id: Episode ID to monitor
            session_id: Session ID (uses instance session_id if not provided)
            event_handler: Optional callback for handling events
        """
        session_id = session_id or self.session_id
        if not session_id:
            raise Exception("No active session")

        sse_url = f"{self.base_url}/session/{session_id}/episodes/{episode_id}/events"
        logger.info(f"📡 Starting episode monitoring via SSE: {sse_url}")

        try:
            async with aiohttp.ClientSession() as session:
                logger.info(f"🔗 DEBUG: Opening SSE connection to {sse_url}")
                async with session.get(sse_url, headers={"Accept": "text/event-stream"}) as response:
                    logger.info(f"🔗 DEBUG: SSE Response status: {response.status}, headers: {dict(response.headers)}")
                    if response.status != 200:
                        logger.error(f"❌ Failed to connect to episode events: {response.status}")
                        return

                    logger.info("✅ Connected to episode event stream")

                    current_event_type = None
                    line_count = 0
                    async for line in response.content:
                        line_count += 1
                        line_str = line.decode("utf-8").strip()

                        if line_count % 10 == 0:  # Log every 10 lines to track activity
                            logger.debug(f"🔗 DEBUG: SSE line {line_count}: {line_str[:100]}...")

                        if line_str.startswith("event:"):
                            current_event_type = line_str[6:].strip()
                            logger.debug(f"🔗 DEBUG: SSE event type: {current_event_type}")
                        elif line_str.startswith("data:") and current_event_type:
                            try:
                                data_str = line_str[5:].strip()
                                if data_str:
                                    event_data = json.loads(data_str)
                                    logger.debug(f"🔗 DEBUG: SSE event data received for {current_event_type}")
                                    if event_handler:
                                        await event_handler(event_data)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse event data: {line_str}")

                    logger.warning(f"🔗 DEBUG: SSE stream ended normally after {line_count} lines")

        except asyncio.TimeoutError as e:
            logger.error(f"❌ Episode monitoring failed due to timeout: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"❌ Episode monitoring failed due to client error: {e}")
        except Exception as e:
            logger.error(f"❌ Episode monitoring failed with unexpected error: {type(e).__name__}: {e}")
            import traceback

            logger.error(f"❌ Episode monitoring traceback: {traceback.format_exc()}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check against the server."""
        url = f"{self.base_url}/health"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    return cast(Dict[str, Any], await response.json())
                else:
                    raise Exception(f"Health check failed: {response.status}")
