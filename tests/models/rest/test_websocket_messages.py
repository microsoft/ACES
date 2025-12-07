"""Tests for WebSocket message TypedDicts (injection extensions)."""
from datetime import datetime
import uuid
from saber.models.rest.websocket_messages import (
    PushMessage,
    PushAckMessage,
)


def test_push_message_with_injection_fields():
    """Test push_message with optional injection fields for red team."""
    msg: PushMessage = {
        "type": "push_message",
        "data": {
            "message": {"role": "system", "content": "Injected prompt"},
            "since_version": 10,
            "client_checksum": "abc123",
            # Injection fields
            "target_episode_id": "ep-blue-123",
            "strategy": "append",
        },
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
    }

    assert msg["type"] == "push_message"
    assert msg["data"]["target_episode_id"] == "ep-blue-123"
    assert msg["data"]["strategy"] == "append"


def test_push_message_with_rewind_strategy():
    """Test push_message with rewind injection strategy."""
    msg: PushMessage = {
        "type": "push_message",
        "data": {
            "message": {"role": "system", "content": "Test"},
            "since_version": 10,
            "target_episode_id": "ep-blue-123",
            "strategy": "rewind",
            "rewind_count": 5,
        },
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
    }

    assert msg["data"]["rewind_count"] == 5


def test_push_ack_with_injection_response():
    """Test push_ack response includes injection metadata."""
    msg: PushAckMessage = {
        "type": "push_ack",
        "data": {
            "version": 11,
            "checksum": "def456",
            # Injection response fields
            "modification_count": 3,
            "target_episode_id": "ep-blue-123",
        },
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
    }

    assert msg["data"]["modification_count"] == 3
    assert msg["data"]["target_episode_id"] == "ep-blue-123"


def test_push_message_normal_blue_team():
    """Test push_message without injection fields (blue team normal use)."""
    msg: PushMessage = {
        "type": "push_message",
        "data": {
            "message": {"role": "assistant", "content": "Normal response"},
            "since_version": 5,
            "client_checksum": "xyz789",
            # No injection fields - normal blue team push
        },
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
    }

    assert "target_episode_id" not in msg["data"]
    assert "strategy" not in msg["data"]
