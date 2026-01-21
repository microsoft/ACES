"""
Standard executor registry containing core executor implementations.

This module contains the standard set of executors that are commonly used
across different security domains and benchmarking scenarios.

Importing this module will automatically register:
1. Individual standard executors (bash, python, etc.)
2. The "standard" executor group containing the core executors

Usage in task configuration:
    execution_config:
      executors:
        standard:  # Expands to: bash, python
          timeout: 30
"""

from ..executor_registry import register_executor_group

# Import executors - this will trigger their registration
from .bash_executor import BashExecutor
from .get_target_transcript_executor import GetTargetTranscriptExecutor
from .inject_prompt_executor import InjectPromptExecutor
from .python_executor import PythonExecutor

# TODO: Fix SQLExecutor _max_rows attribute issue
# from .sql_executor import SQLExecutor

# Define the list of standard executor types (core tools commonly needed)
STANDARD_EXECUTOR_TYPES = ["bash", "python"]

# Register the "standard" executor group
register_executor_group("standard", STANDARD_EXECUTOR_TYPES, "standard_registry")

__all__ = [
    "BashExecutor",
    "GetTargetTranscriptExecutor",
    "InjectPromptExecutor",
    "PythonExecutor",
    "STANDARD_EXECUTOR_TYPES",
]  # "SQLExecutor" temporarily removed
