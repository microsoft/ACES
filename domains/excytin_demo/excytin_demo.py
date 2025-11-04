"""Excytin Demo - Incident Response domain for Inspect AI.

This module exposes the Excytin Demo domain as an Inspect AI task that can be
evaluated with commands like:

    inspect eval external/saber/domains/excytin_demo --model openai/gpt-4
    inspect eval external/saber/domains/excytin_demo -T task_filter=incident_5_* --model anthropic/claude-3-opus
"""

from pathlib import Path

from inspect_ai import task

# Import SABER's task factory
from saber.inspect_ai import create_domain_task

# Get the domains root
# This file is at: external/saber/domains/excytin_demo/excytin_demo.py
# We need: external/saber/domains (the domains directory itself)
_domains_root = Path(__file__).resolve().parent.parent

# Create the task factory (returns a callable that Inspect AI will invoke)
_excytin_demo_factory = create_domain_task(
    domain_slug="excytin_demo",
    domains_root=_domains_root,
    default_agent="react",
)


# Wrap in @task decorator for Inspect AI discovery
@task
def excytin_demo(**kwargs):
    """Excytin Demo - Incident Response domain.

    Cybersecurity incident response benchmark with database forensics and SQL analysis.
    Dynamically loads tasks from the running SABER server.

    Args:
        rest_port: REST API port (default: 8000)
        mcp_port: MCP API port (default: 8001)
        task_filter: Optional task filter (exact match or glob pattern)
        agent: Agent implementation to use (default: "react")
        log_level: Logging level for domain services (default: "INFO")
        build: Build missing images before starting (default: False)
        rebuild: Remove and rebuild images matching this prefix (e.g., 'server')
        rebuild_all: Remove and rebuild all images (default: False)
        stop_saber_after: Stop SABER domain after task completes (default: False).
            If False (default), server stays running for faster re-runs.

    Returns:
        Inspect AI Task with SABER excytin_demo dataset loaded from server

    Examples:
        # Basic evaluation (server stays running after)
        inspect eval external/saber/domains/excytin_demo --model openai/gpt-4

        # Use custom agent
        inspect eval external/saber/domains/excytin_demo --model openai/gpt-4 -T agent=custom_example

        # Build missing images first
        inspect eval external/saber/domains/excytin_demo --model openai/gpt-4 -T build=true

        # Rebuild all images (clean slate)
        inspect eval external/saber/domains/excytin_demo --model openai/gpt-4 -T rebuild_all=true

        # Stop server after evaluation completes
        inspect eval external/saber/domains/excytin_demo --model openai/gpt-4 -T stop_saber_after=true

        # Filter to incident_5 tasks
        inspect eval external/saber/domains/excytin_demo --model openai/gpt-4 -T task_filter="incident_5_*"

        # Run a single task
        inspect eval external/saber/domains/excytin_demo --model openai/gpt-4 -T task_filter="incident_5_task_1"
    """
    return _excytin_demo_factory(**kwargs)
