"""
Event Bus Subscribers Package

Contains event subscribers that convert event bus events to specific actions.
"""

from .ui_adapter import UIAdapterSubscriber

__all__ = ["UIAdapterSubscriber"]
