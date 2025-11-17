"""SABER × Inspect AI Integration

Low-level inspect_ai specific implementations and utilities.
This module should only be imported by SABER internals, not by end users.

For public SABER agent API, use:
    from saber.client import SABERAgentRegistry, SABERAgentFactory
"""

# Import SABER agent registrations to trigger decorator execution
# from . import saber_agent  # This triggers SABER agent registration decorators

# # Low-level inspect_ai implementation registry (internal use only)
# from .agent_implementations import InspectAIImplementationRegistry, register_inspect_ai_implementation

# # Context injection exports (monkey-patching happens automatically on import)
# from .context_injection import saber_execute_tools

# Dataset and evaluation exports
from .saber_dataset import create_saber_dataset

# Scorer exports
from .saber_scorer import saber_scorer

# # Main eval_async integration exports
# from .saber_eval_async import run_saber_eval_async


__all__ = [
    # Agent registrations (triggers decorators)
    # "saber_agent",
    # # Low-level implementation registry (internal)
    # "InspectAIImplementationRegistry",
    # "register_inspect_ai_implementation",
    # # Main eval_async integration
    # "run_saber_eval_async",
    # Dataset and evaluation
    "create_saber_dataset",
    # # Context injection (automatic via monkey-patch)
    # "saber_execute_tools",
    # # Scorer
    "saber_scorer",
]
