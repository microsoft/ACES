"""SABER UI Module - Clean UI integration with multiple backend support."""

from .inspect_ai_backend import InspectAIBackend  # noqa: F401
from .interfaces import ToolCallProgressReporter, UIProgressAdapter  # noqa: F401
from .manager import UIManager  # noqa: F401
from .models import MessageType, TaskInfo, TaskStatus, UIBackendType, UIConfig, UIMessage  # noqa: F401

# Conditional textual mode imports
try:
    from .textual_adapter import SABERProgressMapper, SABERSessionContext, SABERTaskProfile  # noqa: F401
    from .textual_session import SABERTextualSession, create_textual_session  # noqa: F401

    TEXTUAL_EXPORTS = [
        "SABERTextualSession",
        "create_textual_session",
        "SABERTaskProfile",
        "SABERProgressMapper",
        "SABERSessionContext",
    ]
except ImportError:
    TEXTUAL_EXPORTS = []

__all__ = [
    # Core classes
    "UIManager",
    "UIConfig",
    # Enums
    "UIBackendType",
    "MessageType",
    "TaskStatus",
    # Data models
    "UIMessage",
    "TaskInfo",
    # Tool call monitoring
    "ToolCallProgressReporter",
    "UIProgressAdapter",
    # Backend implementations
    "InspectAIBackend",
] + TEXTUAL_EXPORTS
