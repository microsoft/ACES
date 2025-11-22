"""SABER × Inspect AI Integration - Deprecated Import Location

This module is deprecated. Imports have been moved to saber.inspect_ai.
For backwards compatibility, we re-export from the new location.

Deprecated: Import from saber.inspect_ai instead of saber.client.inspect_ai
"""

# Re-export from new location for backwards compatibility
from saber.inspect_ai import create_saber_dataset, saber_scorer
from saber.inspect_ai.context_injection import _get_context, saber_execute_tools, saber_tool_params

__all__ = [
    "create_saber_dataset",
    "saber_scorer",
    "_get_context",
    "saber_execute_tools",
    "saber_tool_params",
]
