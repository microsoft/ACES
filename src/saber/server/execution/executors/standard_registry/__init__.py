"""
Standard executor registry containing core executor implementations.

This module contains the standard set of executors that are commonly used
across different security domains and benchmarking scenarios.

Importing this module will automatically register the standard executors.
"""

# Import executors - this will trigger their registration
from .bash_executor import BashExecutor
from .inject_prompt_executor import InjectPromptExecutor
from .python_executor import PythonExecutor

# TODO: Fix SQLExecutor _max_rows attribute issue
# from .sql_executor import SQLExecutor

__all__ = ["BashExecutor", "InjectPromptExecutor", "PythonExecutor"]  # "SQLExecutor" temporarily removed
