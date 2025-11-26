"""Task filtering for SABER benchmark tasks.

This module provides polymorphic task filtering with support for:
- Exact task ID matching
- Glob pattern matching (*, ?)
- Comma-separated patterns with OR logic
- Both legacy TaskInfo and new BenchmarkTask types

The filtering logic uses polymorphic dispatch via the matches_filter()
method when available, falling back to legacy exact/glob matching.
"""

import fnmatch
from typing import Any, List, Union

from inspect_ai._util.error import PrerequisiteError

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


def apply_task_filter(
    tasks: List[Any],
    task_filter: Union[str, List[str]],
    domain_slug: str,
) -> List[Any]:
    """Apply task filter with exact and glob pattern matching using polymorphic dispatch.

    Supports both legacy TaskInfo and new polymorphic BenchmarkTask types.
    Uses the polymorphic matches_filter() method on each task for type-safe filtering.

    Supports:
    - Exact match: task_filter="labyrinth_linguist_task_hard"
    - Glob pattern: task_filter="labyrinth_*"
    - Glob pattern: task_filter="*_hard"
    - Multiple filters (OR logic): task_filter="xss_*,sql_*"
    - Multiple filters with exact: task_filter="xss_0_flag_capture,sql_*,cmd_injection_task"

    For OrchestratedTask: Matching ANY sub-task includes the entire orchestration.
    For SingleEpisodeTask: Matches against the single task_id.
    For TaskInfo: Legacy exact/glob matching against task_id.

    Multiple filters are separated by commas and matched with OR logic.
    Each filter can be an exact match or a glob pattern.

    Args:
        tasks: List of TaskInfo or BenchmarkTask objects from server
        task_filter: Filter pattern (exact, glob, or comma-separated patterns)
                    Can be a string or a list (Inspect AI may parse comma-separated values as lists)
        domain_slug: Domain slug for error messages

    Returns:
        Filtered list of tasks (deduplicated)

    Raises:
        PrerequisiteError: If no tasks match any of the filters
    """
    # Handle both string and list inputs (Inspect AI may parse "a,b" as ["a", "b"])
    if isinstance(task_filter, list):
        filter_patterns = [str(p).strip() for p in task_filter]
    else:
        # Split on comma to support multiple filters
        filter_patterns = [pattern.strip() for pattern in task_filter.split(",")]

    # Collect all matching tasks across all patterns
    # Use dict to deduplicate - key depends on task type
    matched_tasks: dict[str, Any] = {}

    for pattern in filter_patterns:
        if not pattern:  # Skip empty patterns
            continue

        # Use polymorphic matches_filter() if available (BenchmarkTask types)
        # Otherwise fall back to legacy exact/glob matching (TaskInfo)
        for task in tasks:
            if hasattr(task, "matches_filter"):
                # New polymorphic task types (SingleEpisodeTask, OrchestratedTask)
                if task.matches_filter(pattern):
                    # Use benchmark_task_id as key for deduplication
                    matched_tasks[task.benchmark_task_id] = task
            else:
                # Legacy TaskInfo - use exact/glob matching
                task_id = task.task_id
                if task_id == pattern or fnmatch.fnmatch(task_id, pattern):
                    matched_tasks[task_id] = task

    # Convert back to list and maintain consistent ordering
    if matched_tasks:
        # Sort by task ID for deterministic ordering
        def get_sort_key(t: Any) -> str:
            if hasattr(t, "benchmark_task_id"):
                return str(t.benchmark_task_id)
            else:
                return str(t.task_id)

        return sorted(matched_tasks.values(), key=get_sort_key)

    # No matches - provide helpful error
    available_ids = []
    for task in tasks:
        if hasattr(task, "benchmark_task_id"):
            available_ids.append(task.benchmark_task_id)
        else:
            available_ids.append(task.task_id)

    # Format task_filter for error message
    filter_display = task_filter if isinstance(task_filter, str) else ",".join(task_filter)
    raise PrerequisiteError(
        f"No tasks matched filter '{filter_display}' in domain '{domain_slug}'.\n\n"
        f"Available tasks ({len(available_ids)}):\n"
        + "\n".join(f"  - {task_id}" for task_id in sorted(available_ids)[:20])
        + (f"\n  ... and {len(available_ids) - 20} more" if len(available_ids) > 20 else "")
    )
