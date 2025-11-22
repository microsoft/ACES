"""SABER sandbox integration for Inspect AI.

This module provides integration with SABER (Security Assessment Backend for
Evaluation and Research) through a custom sandbox environment that delegates
to SABER's DomainOrchestrator for Docker Compose lifecycle management.

NOTE: To see SABER's informational logs during evaluation, set the Inspect AI
log level to 'info':

    inspect eval --log-level info ...

Or set the environment variable before running:

    export INSPECT_LOG_LEVEL=info
    inspect eval ...

Public API:
- SABERSandboxEnvironment: Sandbox environment class with lifecycle management
- SABERToolSource: ToolSource implementation for lazy tool provision
- saber_tools: Helper function to create a ToolSource for SABER tools
- create_domain_task: Factory function to create domain task callables
"""

from ..logging_config import LogCategory, get_saber_logger

# Import context injection - patches applied at import time for backward compatibility
# TODO: Migrate to scoped patching using saber_context_injection_patch() context manager
from .context_injection import saber_execute_tools, saber_tool_params
from .saber import SABERSandboxEnvironment, SandboxError
from .saber_dataset import create_saber_dataset
from .saber_scorer import saber_scorer
from .tasks import create_domain_task
from .tools import SABERToolSource, saber_tools

logger = get_saber_logger(LogCategory.AGENT, __name__)
logger.info("Initializing SABER sandbox integration for Inspect AI")

# Monkey-patch execute_tools AND tool_params for context injection
# Note: This is applied at import time for backward compatibility.
# For better test isolation and explicit scope control, use the
# saber_context_injection_patch() context manager instead.
import inspect_ai.agent._react
import inspect_ai.model._call_tools

inspect_ai.agent._react.execute_tools = saber_execute_tools
inspect_ai.model._call_tools.execute_tools = saber_execute_tools
inspect_ai.model._call_tools.tool_params = saber_tool_params

logger.info("Monkey-patched execute_tools + tool_params with SABER context injection")

__all__ = [
    "SABERSandboxEnvironment",
    "SandboxError",
    "SABERToolSource",
    "saber_tools",
    "create_domain_task",
    "create_saber_dataset",
    "saber_scorer",
]
