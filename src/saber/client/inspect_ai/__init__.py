"""SABER × Inspect AI Integration
Public exports for the Inspect AI integration layer with eval_async support.
This provides eval_async integration between SABER's new agent architecture
and inspect_ai's evaluation framework.
"""

# Import agent registrations to trigger decorator execution
from . import configurable_agent  # This triggers agent registration decorators

# Agent registry exports
from .agent_registry import InspectAIAgentFactory

# Dataset and evaluation exports
from .saber_dataset import create_saber_dataset

# Main eval_async integration exports
from .saber_eval_async import run_saber_eval_async

# Scorer exports
from .saber_scorer import SABERTaskScorer, saber_server_scorer, saber_task_scorer

__all__ = [
    # Agent registrations (triggers decorators)
    "configurable_agent",
    # Agent factory
    "InspectAIAgentFactory",
    # Main eval_async integration
    "run_saber_eval_async",
    # Dataset and evaluation
    "create_saber_dataset",
    # Scorer
    "saber_task_scorer",
    "saber_server_scorer",
    "SABERTaskScorer",
]
