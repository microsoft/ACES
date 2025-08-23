"""
Standard executor registry containing core executor implementations.

This module contains the standard set of executors that are commonly used
across different security domains and benchmarking scenarios.

Importing this module will automatically register the standard executors.
"""

# Import executors - this will trigger their registration
from .cli_executor import CLIExecutor
from .python_executor import PythonExecutor

__all__ = ["CLIExecutor", "PythonExecutor"]
