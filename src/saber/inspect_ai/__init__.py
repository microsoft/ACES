"""SABER sandbox integration for Inspect AI.

This module contains Inspect AI-specific integration code that builds upon
the generic harness-agnostic client code in saber.client. It provides sandbox
environments, tool sources, datasets, and scorers specifically for Inspect AI.

Dependencies:
- saber.client: Generic client code (SABERRestClient, SABERConfig, models)
- inspect_ai: The Inspect AI evaluation framework

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
from .core.saber_dataset import create_saber_dataset
from .core.saber_scorer import saber_scorer
from .core.tasks import create_domain_task
from .integration.tools import SABERToolSource, saber_tools
from .saber import SABERSandboxEnvironment, SandboxError

logger = get_saber_logger(LogCategory.AGENT, __name__)
logger.info("Initializing SABER sandbox integration for Inspect AI")

# Monkey-patching has been completely removed - context extraction now uses
# transcript-based approach via transcript_sync.py

logger.info("SABER transcript-based context extraction enabled")

__all__ = [
    "SABERSandboxEnvironment",
    "SandboxError",
    "SABERToolSource",
    "saber_tools",
    "create_domain_task",
    "create_saber_dataset",
    "saber_scorer",
]
