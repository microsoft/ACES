#!/usr/bin/env python3
"""
SABER UI Adapters

UI adapter implementations for different backends.
"""

from .console_adapter import ConsoleUIAdapter

# Make adapters available for import
from .null_adapter import NullUIAdapter

__all__ = ["NullUIAdapter", "ConsoleUIAdapter"]
