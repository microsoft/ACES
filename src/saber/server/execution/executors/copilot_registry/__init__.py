"""
Copilot executor registry containing GitHub Copilot CLI-compatible tool implementations.

This module contains executors that implement the tool interfaces expected by
the GitHub Copilot CLI agent (@github/copilot v0.0.384+).

These executors provide:
- view: Read file contents or list directory contents
- create: Create new files with specified content
- edit: Make string replacements in files (str_replace)
- grep: Fast code search using regex patterns
- glob: Fast file pattern matching using glob patterns

Importing this module will automatically register the Copilot executors.
"""

from .create_executor import CreateExecutor
from .edit_executor import EditExecutor
from .glob_executor import GlobExecutor
from .grep_executor import GrepExecutor
from .view_executor import ViewExecutor

__all__ = [
    "ViewExecutor",
    "CreateExecutor",
    "EditExecutor",
    "GrepExecutor",
    "GlobExecutor",
]
