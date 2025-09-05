"""
SABER Client Events Package

Event-driven architecture for decoupling harness from UI with real-time updates.
Follows fail-fast principles - no silent failures, hard crashes on errors.
"""

from .bus import EventBus
from .types import EventType, SaberEvent

__all__ = ["EventType", "SaberEvent", "EventBus"]
