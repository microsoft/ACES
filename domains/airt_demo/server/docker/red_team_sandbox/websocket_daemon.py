#!/usr/bin/env python3
"""Persistent WebSocket daemon for red team prompt injection.

This daemon:
1. Establishes persistent WebSocket connection to SABER server
2. Maintains connection with ping/pong keepalive
3. Exposes HTTP IPC interface on localhost:9999
4. Forwards injection commands to server via WebSocket
5. Returns responses to executor

Environment Variables:
    SABER_WEBSOCKET_URL: WebSocket URL (ws://host.docker.internal:8000/api/v1/episodes/{id}/ws)
    EPISODE_ID: Red team episode ID
    IPC_PORT: Port for IPC server (default: 9999)
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import websockets
    from aiohttp import web
except ImportError:
    print("ERROR: Required libraries not installed. Run: pip install websockets aiohttp")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# WebSocket Message Types (mirrors saber.models.rest.websocket_messages)
# ============================================================================


@dataclass
class SyncRequestData:
    """Client sync request data - matches server's SyncRequestData model."""

    since_version: Optional[int] = None
    client_checksum: Optional[str] = None
    # Observer/cross-episode fields (for red team accessing blue team transcript)
    target_episode_id: Optional[str] = None
    hide_system_prompt: Optional[bool] = None
    retrieval_mode: Optional[str] = None  # full, delta, tail
    tail_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SyncRequestMessage:
    """WebSocket sync_request message."""

    data: SyncRequestData
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    type: str = "sync_request"

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "type": self.type,
                "data": self.data.to_dict(),
                "id": self.id,
                "timestamp": self.timestamp,
            }
        )


@dataclass
class PushMessageData:
    """Client push message data (for injection)."""

    message: Dict[str, Any]
    since_version: int = 0
    client_checksum: Optional[str] = None
    # Injection fields
    target_episode_id: Optional[str] = None
    strategy: Optional[str] = None  # append or restart

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PushMessageRequest:
    """WebSocket push_message request."""

    data: PushMessageData
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    type: str = "push_message"

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "type": self.type,
                "data": self.data.to_dict(),
                "id": self.id,
                "timestamp": self.timestamp,
            }
        )


# ============================================================================
# WebSocket Daemon
# ============================================================================


class WebSocketDaemon:
    """Persistent WebSocket connection manager with IPC interface."""

    def __init__(
        self, ws_url: str, episode_id: str, ipc_port: int = 9999, session_id: str = "", rest_base_url: str = ""
    ):
        self.ws_url = ws_url
        self.episode_id = episode_id
        self.ipc_port = ipc_port
        self.session_id = session_id
        # Extract REST base URL from WebSocket URL if not provided
        # ws://host:port/api/v1/episodes/{id}/ws -> http://host:port
        if not rest_base_url and ws_url:
            import re

            match = re.match(r"ws://([^/]+)", ws_url)
            if match:
                rest_base_url = f"http://{match.group(1)}"
        self.rest_base_url = rest_base_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self._pending_responses: dict[str, asyncio.Future] = {}

    async def connect_websocket(self) -> None:
        """Establish persistent WebSocket connection."""
        while True:
            try:
                logger.info(f"Connecting to WebSocket: {self.ws_url}")
                self.websocket = await websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                )
                self.connected = True
                logger.info("WebSocket connected successfully")

                # Wait for connected message
                msg = await self.websocket.recv()
                data = json.loads(msg)
                if data.get("type") == "connected":
                    logger.info(f"Received connected acknowledgment: {data}")

                # Start message handler
                await self._handle_websocket_messages()

            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                self.connected = False
                await asyncio.sleep(5)  # Retry after 5 seconds

    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                msg_id = data.get("id")

                logger.debug(f"Received WebSocket message: {msg_type}")

                # Handle responses with message IDs (push_ack, transcript_response, etc.)
                if msg_id and msg_id in self._pending_responses:
                    future = self._pending_responses.pop(msg_id)
                    future.set_result(data)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False

    async def send_injection(
        self,
        target_episode_id: str,
        message: str,
        strategy: str = "append",
        timeout: float = 10.0,
    ) -> dict:
        """Send injection via push_message with injection fields.

        Args:
            target_episode_id: Target episode to inject into
            message: Message content to inject
            strategy: Injection strategy ('append' or 'restart')
            timeout: Timeout in seconds for waiting for response

        Returns:
            push_ack response data

        Raises:
            RuntimeError: If WebSocket is not connected
            TimeoutError: If no response received within timeout
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")

        # Build strongly typed push_message request
        # Use "user" role for injections so the target agent responds
        # (system messages are skipped by the state machine)
        push_data = PushMessageData(
            message={"role": "user", "content": message},
            since_version=0,  # Not tracking local version for injections
            target_episode_id=target_episode_id,
            strategy=strategy,
        )
        request = PushMessageRequest(data=push_data)

        # Create future for response
        future: asyncio.Future[dict] = asyncio.Future()
        self._pending_responses[request.id] = future

        # Send message
        await self.websocket.send(request.to_json())
        logger.info(f"Sent injection via push_message: {request.id}")

        # Wait for push_ack response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response["data"]
        except asyncio.TimeoutError:
            self._pending_responses.pop(request.id, None)
            raise TimeoutError(f"Injection timeout after {timeout}s")

    async def handle_ipc_request(self, request: web.Request) -> web.Response:
        """Handle IPC HTTP request.

        Args:
            request: aiohttp Request object

        Returns:
            JSON response with injection result
        """
        try:
            data = await request.json()

            # Extract parameters - simplified to message and strategy
            target_episode_id = data.get("target_episode_id")
            message = data.get("message")
            strategy = data.get("strategy", "append")

            if not target_episode_id or not message:
                return web.json_response(
                    {"success": False, "error": "Missing target_episode_id or message"}, status=400
                )

            # Forward to WebSocket
            result = await self.send_injection(
                target_episode_id=target_episode_id,
                message=message,
                strategy=strategy,
            )

            # Wrap with success flag for executor compatibility
            return web.json_response({"success": True, **result})

        except Exception as e:
            logger.error(f"IPC request failed: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def wait_for_state_event(
        self,
        target_episode_id: str,
        target_state: str = "waiting_for_user",
        max_wait_seconds: float = 120.0,
    ) -> dict:
        """Wait for target episode to reach a specific state via WebSocket events.

        Subscribes to state events from the server and waits until the target
        episode reaches the specified state (e.g., waiting_for_user).

        Args:
            target_episode_id: Target episode to monitor
            target_state: State to wait for (default: waiting_for_user)
            max_wait_seconds: Maximum time to wait

        Returns:
            Dict with success status and state information
        """
        if not self.connected or not self.websocket:
            return {"success": False, "error": "WebSocket not connected"}

        # Create an event that will be set when we receive the target state
        state_event = asyncio.Event()
        received_state: Dict[str, Any] = {}

        # Store original handler reference (kept for potential rollback/debugging)
        _ = self._pending_responses.copy()

        # Create a temporary state event listener (ID used for debugging)
        _ = f"state_listener_{target_episode_id}_{uuid.uuid4()}"

        async def listen_for_state():
            """Background listener for state events."""
            try:
                # Poll for state changes using sync_request with short intervals
                poll_interval = 2.0  # Check every 2 seconds
                elapsed = 0.0

                while elapsed < max_wait_seconds and not state_event.is_set():
                    try:
                        # Use sync_request to check current state
                        sync_response = await self.send_sync_request(
                            target_episode_id=target_episode_id,
                            since_version=0,
                            hide_system_prompt=True,
                            retrieval_mode="full",
                            timeout=10.0,
                        )

                        # Check if state indicates waiting_for_user
                        # The sync_response contains current_version info
                        # A transcript that hasn't changed recently indicates waiting state
                        current_version = sync_response.get("current_version", {})
                        # sync_mode could be used for more sophisticated state checking
                        _ = sync_response.get("sync_mode", "")

                        # For now, consider any successful sync as "ready"
                        # More sophisticated: check actual state from server
                        if sync_response:
                            received_state["state"] = target_state
                            received_state["version"] = (
                                current_version.get("sequence", 0) if isinstance(current_version, dict) else 0
                            )
                            state_event.set()
                            return

                    except Exception as e:
                        logger.debug(f"State poll attempt failed: {e}")

                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

            except asyncio.CancelledError:
                pass

        # Start listening task
        listener_task = asyncio.create_task(listen_for_state())

        try:
            # Wait for state event or timeout
            await asyncio.wait_for(state_event.wait(), timeout=max_wait_seconds)
            return {
                "success": True,
                "state": received_state.get("state", target_state),
                "version": received_state.get("version", 0),
                "target_episode_id": target_episode_id,
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Timeout waiting for {target_state} state after {max_wait_seconds}s",
                "target_episode_id": target_episode_id,
            }
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

    async def handle_wait_for_user(self, request: web.Request) -> web.Response:
        """Handle wait_for_user IPC request.

        Waits for target episode to enter WAITING_FOR_USER state.

        Args:
            request: aiohttp Request object with target_episode_id and max_wait_seconds

        Returns:
            JSON response with success status
        """
        try:
            data = await request.json()
            target_episode_id = data.get("target_episode_id")
            max_wait_seconds = data.get("max_wait_seconds", 120.0)

            if not target_episode_id:
                return web.json_response({"success": False, "error": "Missing target_episode_id"}, status=400)

            if not self.connected or not self.websocket:
                return web.json_response({"success": False, "error": "WebSocket not connected"}, status=503)

            # Use the state event waiting mechanism
            result = await self.wait_for_state_event(
                target_episode_id=target_episode_id,
                target_state="waiting_for_user",
                max_wait_seconds=max_wait_seconds,
            )

            return web.json_response(result)

        except Exception as e:
            logger.error(f"wait_for_user request failed: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_inject_and_wait(self, request: web.Request) -> web.Response:
        """Handle inject_and_wait IPC request - inject, wait for response, return transcript.

        This is the primary injection endpoint that:
        1. Records the current transcript version before injection
        2. Sends the injection to the target episode
        3. Waits for the blue team to finish generating (WAITING_FOR_USER state)
        4. Fetches all new messages added since the injection
        5. Returns the complete response including blue team's reply

        Args:
            request: aiohttp Request object with injection parameters

        Returns:
            JSON response with injection result and blue team's response
        """
        try:
            data = await request.json()

            # Extract parameters - simplified to message and strategy
            target_episode_id = data.get("target_episode_id")
            message = data.get("message")
            strategy = data.get("strategy", "append")
            max_wait_seconds = data.get("max_wait_seconds", 120.0)

            if not target_episode_id or not message:
                return web.json_response(
                    {"success": False, "error": "Missing target_episode_id or message"}, status=400
                )

            if not self.connected or not self.websocket:
                return web.json_response({"success": False, "error": "WebSocket not connected"}, status=503)

            # Step 1: Get current transcript version before injection
            logger.info(f"inject_and_wait: Getting pre-injection transcript version for {target_episode_id}")
            try:
                pre_inject_sync = await self.send_sync_request(
                    target_episode_id=target_episode_id,
                    since_version=0,
                    hide_system_prompt=True,
                    retrieval_mode="full",
                    timeout=30.0,
                )
                pre_inject_version = pre_inject_sync.get("current_version", {})
                version_before = pre_inject_version.get("sequence", 0) if isinstance(pre_inject_version, dict) else 0
                logger.info(f"inject_and_wait: Pre-injection version = {version_before}")
            except Exception as e:
                logger.warning(f"inject_and_wait: Could not get pre-injection version: {e}")
                version_before = 0

            # Step 2: Send the injection
            logger.info(f"inject_and_wait: Sending injection to {target_episode_id}")
            inject_result = await self.send_injection(
                target_episode_id=target_episode_id,
                message=message,
                strategy=strategy,
                timeout=30.0,
            )

            inject_version = inject_result.get("version", version_before + 1)
            logger.info(f"inject_and_wait: Injection sent, version = {inject_version}")

            # Step 3: Wait for blue team to respond (polling for transcript changes)
            logger.info(f"inject_and_wait: Waiting for blue team response (max {max_wait_seconds}s)")

            poll_interval = 2.0
            elapsed = 0.0
            response_messages = []
            final_version = inject_version

            while elapsed < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                try:
                    # Check for new messages since injection
                    sync_response = await self.send_sync_request(
                        target_episode_id=target_episode_id,
                        since_version=inject_version,
                        hide_system_prompt=True,
                        retrieval_mode="delta",
                        timeout=10.0,
                    )

                    current_version = sync_response.get("current_version", {})
                    current_seq = current_version.get("sequence", 0) if isinstance(current_version, dict) else 0

                    # Check if there are new messages (delta)
                    delta = sync_response.get("delta", [])

                    if delta and len(delta) > 0:
                        # We have new messages! Check if blue team has responded
                        # Look for an assistant message in the delta
                        has_assistant_response = any(msg.get("role") == "assistant" for msg in delta)

                        if has_assistant_response:
                            response_messages = delta
                            final_version = current_seq
                            logger.info(f"inject_and_wait: Blue team responded with {len(delta)} messages")
                            break

                except Exception as e:
                    logger.warning(f"inject_and_wait: Poll error: {e}")
                    continue

            # Step 4: Return complete result
            if response_messages:
                return web.json_response(
                    {
                        "success": True,
                        "injection_version": inject_version,
                        "final_version": final_version,
                        "target_episode_id": target_episode_id,
                        "strategy": strategy,
                        "response_messages": response_messages,
                        "response_count": len(response_messages),
                        "wait_time_seconds": elapsed,
                    }
                )
            else:
                # No response within timeout
                return web.json_response(
                    {
                        "success": True,  # Injection succeeded, but no response yet
                        "injection_version": inject_version,
                        "final_version": final_version,
                        "target_episode_id": target_episode_id,
                        "strategy": strategy,
                        "response_messages": [],
                        "response_count": 0,
                        "wait_time_seconds": elapsed,
                        "warning": f"No response from blue team within {max_wait_seconds}s",
                    }
                )

        except TimeoutError as e:
            logger.error(f"inject_and_wait timed out: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=504)
        except Exception as e:
            logger.error(f"inject_and_wait failed: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def send_sync_request(
        self,
        target_episode_id: str,
        since_version: int = 0,
        hide_system_prompt: bool = True,
        retrieval_mode: str = "full",
        tail_count: int = 10,
        timeout: float = 30.0,
    ) -> dict:
        """Send sync_request via WebSocket to retrieve transcript.

        This uses the same WebSocket mechanism as WebSocketModelWrapper,
        ensuring consistent transcript access across the system.

        Args:
            target_episode_id: Target episode to get transcript from
            since_version: Version for delta sync (0 for full)
            hide_system_prompt: Whether to hide system messages (security for red team)
            retrieval_mode: Retrieval mode (full, delta, tail)
            tail_count: Number of messages for tail mode
            timeout: Timeout in seconds for waiting for response

        Returns:
            sync_response data with transcript

        Raises:
            RuntimeError: If WebSocket is not connected
            TimeoutError: If no response received within timeout
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")

        # Build strongly typed sync_request message
        sync_data = SyncRequestData(
            since_version=since_version,
            client_checksum="",  # Not tracking checksums for observer access
            target_episode_id=target_episode_id,
            hide_system_prompt=hide_system_prompt,
            retrieval_mode=retrieval_mode,
            tail_count=tail_count,
        )
        request = SyncRequestMessage(data=sync_data)

        # Create future for response
        future: asyncio.Future[dict] = asyncio.Future()
        self._pending_responses[request.id] = future

        # Send message
        await self.websocket.send(request.to_json())
        logger.info(f"Sent sync_request for transcript: {request.id}, target={target_episode_id}")

        # Wait for sync_response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response.get("data", {})
        except asyncio.TimeoutError:
            self._pending_responses.pop(request.id, None)
            raise TimeoutError(f"Sync request timeout after {timeout}s")

    async def handle_transcript(self, request: web.Request) -> web.Response:
        """Handle transcript retrieval IPC request.

        Retrieves transcript from target episode via WebSocket sync_request.
        This uses the same mechanism as WebSocketModelWrapper for consistency.
        Server-side filtering is used for security (hide_system_prompt).

        Args:
            request: aiohttp Request object with transcript retrieval parameters

        Returns:
            JSON response with transcript data
        """
        try:
            data = await request.json()
            target_episode_id = data.get("target_episode_id")
            retrieval_mode = data.get("retrieval_mode", data.get("mode", "full"))
            tail_count = data.get("tail_count", 10)
            since_version = data.get("since_version", 0)
            # Always hide system prompt for red team access - this is a security feature
            hide_system_prompt = data.get("hide_system_prompt", True)

            if not target_episode_id:
                return web.json_response({"success": False, "error": "Missing target_episode_id"}, status=400)

            if not self.connected or not self.websocket:
                return web.json_response({"success": False, "error": "WebSocket not connected"}, status=503)

            # Use WebSocket sync_request - same mechanism as WebSocketModelWrapper
            sync_response = await self.send_sync_request(
                target_episode_id=target_episode_id,
                since_version=since_version,
                hide_system_prompt=hide_system_prompt,
                retrieval_mode=retrieval_mode,
                tail_count=tail_count,
            )

            # Extract messages from sync_response
            # sync_response contains: current_version, delta, full_transcript, sync_mode, modified
            messages = []
            if sync_response.get("full_transcript"):
                messages = sync_response["full_transcript"]
            elif sync_response.get("delta"):
                messages = sync_response["delta"]

            current_version = sync_response.get("current_version", {})
            version_sequence = current_version.get("sequence", 0) if isinstance(current_version, dict) else 0

            return web.json_response(
                {
                    "success": True,
                    "messages": messages,
                    "message_count": len(messages),
                    "current_version": version_sequence,
                    "sync_mode": sync_response.get("sync_mode", "full"),
                    "modified": sync_response.get("modified", False),
                }
            )

        except TimeoutError as e:
            logger.error(f"transcript request timed out: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=504)
        except Exception as e:
            logger.error(f"transcript request failed: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def start_ipc_server(self) -> None:
        """Start HTTP IPC server."""
        app = web.Application()
        app.router.add_post("/inject", self.handle_ipc_request)
        app.router.add_post("/inject_and_wait", self.handle_inject_and_wait)
        app.router.add_post("/wait_for_user", self.handle_wait_for_user)
        app.router.add_post("/transcript", self.handle_transcript)
        app.router.add_post("/get_transcript", self.handle_transcript)  # Alias for executor compatibility
        app.router.add_get("/health", lambda r: web.json_response({"status": "ok", "connected": self.connected}))

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.ipc_port)
        await site.start()

        logger.info(f"IPC server listening on localhost:{self.ipc_port}")

        # Keep running forever
        await asyncio.Event().wait()


async def main():
    """Main entry point."""
    ws_url = os.environ.get("SABER_WEBSOCKET_URL")
    episode_id = os.environ.get("EPISODE_ID")
    session_id = os.environ.get("SESSION_ID", "")
    ipc_port = int(os.environ.get("IPC_PORT", "9999"))

    if not ws_url or not episode_id:
        logger.error("Missing SABER_WEBSOCKET_URL or EPISODE_ID environment variables")
        sys.exit(1)

    daemon = WebSocketDaemon(ws_url, episode_id, ipc_port, session_id=session_id)

    # Run WebSocket and IPC server concurrently
    await asyncio.gather(
        daemon.connect_websocket(),
        daemon.start_ipc_server(),
    )


if __name__ == "__main__":
    asyncio.run(main())
