"""Utility functions for transcript synchronization.

This module provides common utilities used by the transcript sync components.
"""

from typing import Any


def is_websocket_closed(websocket: Any) -> bool:
    """Check if a WebSocket connection is closed.

    Compatible with both websockets <15.0 (.closed attribute)
    and websockets >=15.0 (.state attribute with State enum).

    Args:
        websocket: WebSocket connection object

    Returns:
        True if the connection is closed or closing, False if open
    """
    if websocket is None:
        return True

    # Try to import WebSocketState for websockets >=15.0
    try:
        from websockets.protocol import State as WebSocketState

        # websockets >=15.0 uses .state attribute with State enum
        if hasattr(websocket, "state"):
            return websocket.state in (WebSocketState.CLOSED, WebSocketState.CLOSING)
    except ImportError:
        pass

    # websockets <15.0 uses .closed attribute
    if hasattr(websocket, "closed"):
        return bool(websocket.closed)

    # Default to closed if we can't determine
    return True


__all__ = ["is_websocket_closed"]
