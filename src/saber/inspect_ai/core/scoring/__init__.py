"""SABER Scoring System - Factory-based scoring methods.

This module provides a factory pattern for registering and using scoring methods,
allowing domains to implement their own custom scoring strategies.

Standard scoring methods (llm_judge, static, tool_call) are included by default.
Domains can register additional scoring methods via the registry.
"""

# Auto-register standard scorers
from . import register_standard  # noqa: F401
from .registry import (
    ScorerMetadata,
    ScoringRegistry,
    get_submission_scorer,
    get_subtask_scorer,
    get_subtask_scorer_metadata,
    register_submission_scorer,
    register_subtask_scorer,
)
from .standard_submission import score_submission_llm, score_submission_static
from .standard_subtask import (
    score_subtask_llm,
    score_subtask_skip,
    score_subtask_static,
    score_subtask_tool_call,
    score_subtask_tool_call_count,
)

__all__ = [
    # Registry
    "ScoringRegistry",
    "ScorerMetadata",
    "get_submission_scorer",
    "get_subtask_scorer",
    "get_subtask_scorer_metadata",
    "register_submission_scorer",
    "register_subtask_scorer",
    # Standard submission scorers
    "score_submission_static",
    "score_submission_llm",
    # Standard subtask scorers
    "score_subtask_static",
    "score_subtask_llm",
    "score_subtask_tool_call",
    "score_subtask_tool_call_count",
    "score_subtask_skip",
]
