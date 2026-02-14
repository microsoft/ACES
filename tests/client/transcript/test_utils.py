"""Tests for transcript utility functions.

Tests cover:
- is_websocket_closed for various WebSocket states
- Compatibility with different websocket library versions
"""

import pytest

from saber.client.transcript.utils import is_websocket_closed


class TestIsWebsocketClosed:
    """Tests for is_websocket_closed utility function."""

    def test_none_websocket_is_closed(self):
        """Test that None WebSocket is considered closed."""
        assert is_websocket_closed(None) is True

    def test_websocket_with_closed_attribute_true(self):
        """Test WebSocket with closed=True is closed (websockets <15.0)."""
        class MockWebSocket:
            closed = True

        assert is_websocket_closed(MockWebSocket()) is True

    def test_websocket_with_closed_attribute_false(self):
        """Test WebSocket with closed=False is not closed (websockets <15.0)."""
        class MockWebSocket:
            closed = False

        assert is_websocket_closed(MockWebSocket()) is False

    def test_unknown_websocket_is_closed(self):
        """Test that unknown WebSocket type defaults to closed."""
        class UnknownWebSocket:
            pass

        assert is_websocket_closed(UnknownWebSocket()) is True

    def test_websocket_with_state_attribute_closed(self):
        """Test WebSocket with state=CLOSED is closed (websockets >=15.0)."""
        try:
            from websockets.protocol import State as WebSocketState

            class MockWebSocket:
                state = WebSocketState.CLOSED

            assert is_websocket_closed(MockWebSocket()) is True
        except ImportError:
            pytest.skip("websockets >=15.0 not installed")

    def test_websocket_with_state_attribute_closing(self):
        """Test WebSocket with state=CLOSING is closed (websockets >=15.0)."""
        try:
            from websockets.protocol import State as WebSocketState

            class MockWebSocket:
                state = WebSocketState.CLOSING

            assert is_websocket_closed(MockWebSocket()) is True
        except ImportError:
            pytest.skip("websockets >=15.0 not installed")

    def test_websocket_with_state_attribute_open(self):
        """Test WebSocket with state=OPEN is not closed (websockets >=15.0)."""
        try:
            from websockets.protocol import State as WebSocketState

            class MockWebSocket:
                state = WebSocketState.OPEN

            assert is_websocket_closed(MockWebSocket()) is False
        except ImportError:
            pytest.skip("websockets >=15.0 not installed")

    def test_websocket_with_state_attribute_connecting(self):
        """Test WebSocket with state=CONNECTING is not closed (websockets >=15.0)."""
        try:
            from websockets.protocol import State as WebSocketState

            class MockWebSocket:
                state = WebSocketState.CONNECTING

            assert is_websocket_closed(MockWebSocket()) is False
        except ImportError:
            pytest.skip("websockets >=15.0 not installed")
