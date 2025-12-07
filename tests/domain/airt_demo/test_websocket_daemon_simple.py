"""Simplified tests for WebSocket daemon - no hanging async tasks."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web

# Import the daemon module from the red_team_sandbox Docker context
import sys
from pathlib import Path

daemon_path = Path(__file__).parent.parent.parent.parent / "domains" / "airt_demo" / "server" / "docker" / "red_team_sandbox"
sys.path.insert(0, str(daemon_path))

try:
    from websocket_daemon import WebSocketDaemon
except ImportError:
    WebSocketDaemon = None


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection."""
    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    return mock_ws


@pytest.fixture
def daemon_config():
    """Configuration for daemon testing."""
    return {
        "ws_url": "ws://localhost:8000/api/v1/episodes/test-red-ep/ws",
        "episode_id": "test-red-ep",
        "ipc_port": 9999
    }


@pytest.mark.asyncio
async def test_daemon_initialization(daemon_config):
    """Test daemon can be initialized."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)
    assert daemon.ws_url == daemon_config["ws_url"]
    assert daemon.episode_id == daemon_config["episode_id"]
    assert daemon.ipc_port == daemon_config["ipc_port"]
    assert daemon.connected is False


@pytest.mark.asyncio
async def test_daemon_sends_injection_message(mock_websocket, daemon_config):
    """Test daemon can send injection via WebSocket."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)
    daemon.websocket = mock_websocket
    daemon.connected = True

    sent_msg = None

    async def capture_send(message):
        nonlocal sent_msg
        sent_msg = json.loads(message)
        # Immediately resolve the pending response
        daemon._pending_responses[sent_msg["id"]].set_result({
            "type": "push_ack",
            "id": sent_msg["id"],
            "data": {"success": True, "version": 1, "message_count": 1}
        })

    mock_websocket.send.side_effect = capture_send

    result = await daemon.send_injection(
        target_episode_id="blue-123",
        message="Test injection",
        strategy="append"
    )

    assert sent_msg is not None
    assert sent_msg["type"] == "push_message"
    assert sent_msg["data"]["target_episode_id"] == "blue-123"
    assert sent_msg["data"]["message"]["content"] == "Test injection"
    assert sent_msg["data"]["strategy"] == "append"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_daemon_sends_rewind_injection(mock_websocket, daemon_config):
    """Test daemon sends rewind injection correctly."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)
    daemon.websocket = mock_websocket
    daemon.connected = True

    sent_msg = None

    async def capture_send(message):
        nonlocal sent_msg
        sent_msg = json.loads(message)
        daemon._pending_responses[sent_msg["id"]].set_result({
            "type": "push_ack",
            "id": sent_msg["id"],
            "data": {"success": True, "version": 3, "message_count": 3}
        })

    mock_websocket.send.side_effect = capture_send

    result = await daemon.send_injection(
        target_episode_id="blue-123",
        message="Rewind test",
        strategy="rewind",
        rewind_count=2
    )

    assert sent_msg["data"]["strategy"] == "rewind"
    assert sent_msg["data"]["rewind_count"] == 2


@pytest.mark.asyncio
async def test_daemon_sends_insert_injection(mock_websocket, daemon_config):
    """Test daemon sends insert injection correctly."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)
    daemon.websocket = mock_websocket
    daemon.connected = True

    sent_msg = None

    async def capture_send(message):
        nonlocal sent_msg
        sent_msg = json.loads(message)
        daemon._pending_responses[sent_msg["id"]].set_result({
            "type": "push_ack",
            "id": sent_msg["id"],
            "data": {"success": True, "version": 4, "message_count": 4}
        })

    mock_websocket.send.side_effect = capture_send

    result = await daemon.send_injection(
        target_episode_id="blue-123",
        message="Insert test",
        strategy="insert",
        insert_position=1
    )

    assert sent_msg["data"]["strategy"] == "insert"
    assert sent_msg["data"]["insert_position"] == 1


@pytest.mark.asyncio
async def test_daemon_timeout_on_no_response(mock_websocket, daemon_config):
    """Test daemon times out if no response received."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)
    daemon.websocket = mock_websocket
    daemon.connected = True

    # Don't resolve the future - just send
    mock_websocket.send = AsyncMock()

    with pytest.raises(TimeoutError):
        await daemon.send_injection(
            target_episode_id="blue-123",
            message="Timeout test",
            strategy="append",
            timeout=0.1
        )


@pytest.mark.asyncio
async def test_daemon_raises_error_when_not_connected(daemon_config):
    """Test daemon raises error when not connected."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)
    daemon.connected = False

    with pytest.raises(RuntimeError, match="WebSocket not connected"):
        await daemon.send_injection(
            target_episode_id="blue-123",
            message="Test",
            strategy="append"
        )


@pytest.mark.asyncio
async def test_daemon_ipc_request_handler(mock_websocket, daemon_config):
    """Test IPC request handler."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)
    daemon.websocket = mock_websocket
    daemon.connected = True

    async def capture_send(message):
        data = json.loads(message)
        daemon._pending_responses[data["id"]].set_result({
            "type": "push_ack",
            "id": data["id"],
            "data": {"success": True, "version": 1, "message_count": 1}
        })

    mock_websocket.send.side_effect = capture_send

    mock_request = Mock(spec=web.Request)
    mock_request.json = AsyncMock(return_value={
        "target_episode_id": "blue-456",
        "message": "IPC test",
        "strategy": "append"
    })

    response = await daemon.handle_ipc_request(mock_request)

    assert response.status == 200
    response_data = json.loads(response.text)
    assert response_data["success"] is True


@pytest.mark.asyncio
async def test_daemon_ipc_missing_params(daemon_config):
    """Test IPC returns error for missing parameters."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)

    mock_request = Mock(spec=web.Request)
    mock_request.json = AsyncMock(return_value={
        "target_episode_id": "blue-456"
        # Missing message
    })

    response = await daemon.handle_ipc_request(mock_request)

    assert response.status == 400
    response_data = json.loads(response.text)
    assert response_data["success"] is False


@pytest.mark.asyncio
async def test_daemon_handles_concurrent_requests(mock_websocket, daemon_config):
    """Test daemon handles multiple concurrent requests."""
    if WebSocketDaemon is None:
        pytest.skip("WebSocketDaemon not implemented yet")

    daemon = WebSocketDaemon(**daemon_config)
    daemon.websocket = mock_websocket
    daemon.connected = True

    sent_count = 0

    async def capture_send(message):
        nonlocal sent_count
        sent_count += 1
        data = json.loads(message)
        daemon._pending_responses[data["id"]].set_result({
            "type": "push_ack",
            "id": data["id"],
            "data": {"success": True, "version": 1, "message_count": 1}
        })

    mock_websocket.send.side_effect = capture_send

    results = await asyncio.gather(
        daemon.send_injection("blue-1", "Msg 1", "append"),
        daemon.send_injection("blue-2", "Msg 2", "append"),
        daemon.send_injection("blue-3", "Msg 3", "append"),
    )

    assert len(results) == 3
    assert all(r["success"] for r in results)
    assert sent_count == 3
