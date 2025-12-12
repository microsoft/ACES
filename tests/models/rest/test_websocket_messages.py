"""Tests for WebSocket message Pydantic models (injection extensions)."""
from datetime import datetime
import uuid
from saber.models.rest.websocket_messages import (
    PushMessageData,
    PushAckMessage,
    PushAckData,
    WebSocketMessageType,
)


def test_push_message_with_injection_fields():
    """Test push_message with optional injection fields for red team."""
    data = PushMessageData(
        message={"role": "system", "content": "Injected prompt"},
        since_version=10,
        client_checksum="abc123",
        # Injection fields
        target_episode_id="ep-blue-123",
        strategy="append",
    )

    assert data.target_episode_id == "ep-blue-123"
    assert data.strategy == "append"
    assert data.message["role"] == "system"


def test_push_message_with_restart_strategy():
    """Test push_message with restart injection strategy.

    Restart resets the transcript to initial state and appends a new message.
    """
    data = PushMessageData(
        message={"role": "user", "content": "Fresh start"},
        since_version=10,
        target_episode_id="ep-blue-123",
        strategy="restart",
    )

    assert data.strategy == "restart"
    assert data.message["content"] == "Fresh start"


def test_push_ack_with_injection_response():
    """Test push_ack response includes injection metadata."""
    ack_data = PushAckData(
        version=11,
        checksum="def456",
        # Injection response fields
        modification_count=3,
        target_episode_id="ep-blue-123",
    )

    assert ack_data.modification_count == 3
    assert ack_data.target_episode_id == "ep-blue-123"

    # Test full message wrapper
    msg = PushAckMessage(
        data=ack_data,
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
    )

    assert msg.type == WebSocketMessageType.PUSH_ACK.value
    assert msg.data.modification_count == 3


def test_push_message_normal_blue_team():
    """Test push_message without injection fields (blue team normal use)."""
    data = PushMessageData(
        message={"role": "assistant", "content": "Normal response"},
        since_version=5,
        client_checksum="xyz789",
        # No injection fields - normal blue team push
    )

    assert data.target_episode_id is None
    assert data.strategy is None
    assert data.message["role"] == "assistant"
