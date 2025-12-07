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
from datetime import datetime
from typing import Optional

try:
    import websockets
    from aiohttp import web
except ImportError:
    print("ERROR: Required libraries not installed. Run: pip install websockets aiohttp")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WebSocketDaemon:
    """Persistent WebSocket connection manager with IPC interface."""

    def __init__(self, ws_url: str, episode_id: str, ipc_port: int = 9999):
        self.ws_url = ws_url
        self.episode_id = episode_id
        self.ipc_port = ipc_port
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

                if msg_type == "push_ack" and msg_id in self._pending_responses:
                    # Complete pending request (handles both normal and injection)
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
        rewind_count: Optional[int] = None,
        insert_position: Optional[int] = None,
        timeout: float = 10.0,
    ) -> dict:
        """Send injection via push_message with injection fields.

        Args:
            target_episode_id: Target episode to inject into
            message: Message content to inject
            strategy: Injection strategy (append, rewind, insert, replace)
            rewind_count: Number of messages to rewind (for rewind strategy)
            insert_position: Position to insert at (for insert strategy)
            timeout: Timeout in seconds for waiting for response

        Returns:
            push_ack response data

        Raises:
            RuntimeError: If WebSocket is not connected
            TimeoutError: If no response received within timeout
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")

        msg_id = str(uuid.uuid4())

        # Build push_message with injection fields
        payload = {
            "type": "push_message",  # UNIFIED: same type as normal push
            "data": {
                "message": {"role": "system", "content": message},
                "since_version": 0,  # Not tracking local version for injections
                # INJECTION FIELDS
                "target_episode_id": target_episode_id,
                "strategy": strategy,
            },
            "id": msg_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if rewind_count is not None:
            payload["data"]["rewind_count"] = rewind_count
        if insert_position is not None:
            payload["data"]["insert_position"] = insert_position

        # Create future for response
        future = asyncio.Future()
        self._pending_responses[msg_id] = future

        # Send message
        await self.websocket.send(json.dumps(payload))
        logger.info(f"Sent injection via push_message: {msg_id}")

        # Wait for push_ack response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response["data"]
        except asyncio.TimeoutError:
            self._pending_responses.pop(msg_id, None)
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

            # Extract parameters
            target_episode_id = data.get("target_episode_id")
            message = data.get("message")
            strategy = data.get("strategy", "append")
            rewind_count = data.get("rewind_count")
            insert_position = data.get("insert_position")

            if not target_episode_id or not message:
                return web.json_response(
                    {"success": False, "error": "Missing target_episode_id or message"}, status=400
                )

            # Forward to WebSocket
            result = await self.send_injection(
                target_episode_id=target_episode_id,
                message=message,
                strategy=strategy,
                rewind_count=rewind_count,
                insert_position=insert_position,
            )

            return web.json_response(result)

        except Exception as e:
            logger.error(f"IPC request failed: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def start_ipc_server(self) -> None:
        """Start HTTP IPC server."""
        app = web.Application()
        app.router.add_post("/inject", self.handle_ipc_request)
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
    ipc_port = int(os.environ.get("IPC_PORT", "9999"))

    if not ws_url or not episode_id:
        logger.error("Missing SABER_WEBSOCKET_URL or EPISODE_ID environment variables")
        sys.exit(1)

    daemon = WebSocketDaemon(ws_url, episode_id, ipc_port)

    # Run WebSocket and IPC server concurrently
    await asyncio.gather(
        daemon.connect_websocket(),
        daemon.start_ipc_server(),
    )


if __name__ == "__main__":
    asyncio.run(main())
