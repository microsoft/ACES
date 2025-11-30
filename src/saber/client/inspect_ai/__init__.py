"""SABER × Inspect AI Integration - Deprecated Import Location

This module is deprecated. Imports have been moved to saber.inspect_ai.
For backwards compatibility, we re-export from the new location.

Deprecated: Import from saber.inspect_ai instead of saber.client.inspect_ai

Note: Monkey-patch functions have been removed as context extraction now uses
transcript-based approach.
"""

# Re-export from new location for backwards compatibility
from saber.inspect_ai import create_saber_dataset, saber_scorer

__all__ = [
    "create_saber_dataset",
    "saber_scorer",
]
