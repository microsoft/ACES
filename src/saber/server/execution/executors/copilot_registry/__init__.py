"""
Copilot executor registry containing GitHub Copilot CLI-compatible tool implementations.

This module contains executors that implement the tool interfaces expected by
the GitHub Copilot CLI agent (@github/copilot v0.0.384+).

These executors provide:
- view: Read file contents or list directory contents
- write: Write new files with specified content
- edit: Make string replacements in files (str_replace)
- grep: Fast code search using regex patterns
- glob: Fast file pattern matching using glob patterns

Importing this module will automatically register:
1. Individual Copilot executors (view, write, edit, grep, glob)
2. The "copilot" executor group containing all of the above

Usage in task configuration:
    execution_config:
      executors:
        copilot:  # Expands to: view, write, edit, grep, glob
          timeout: 30
"""

from ..executor_registry import register_executor_group
from .edit_executor import EditExecutor
from .glob_executor import GlobExecutor
from .grep_executor import GrepExecutor
from .view_executor import ViewExecutor
from .write_executor import WriteExecutor

# Define the list of Copilot executor types
COPILOT_EXECUTOR_TYPES = ["view", "write", "edit", "grep", "glob"]

# Register the "copilot" executor group
# This allows users to specify "copilot" in their config to get all Copilot executors
register_executor_group("copilot", COPILOT_EXECUTOR_TYPES, "copilot_registry")

__all__ = [
    "ViewExecutor",
    "WriteExecutor",
    "EditExecutor",
    "GrepExecutor",
    "GlobExecutor",
    "COPILOT_EXECUTOR_TYPES",
]
