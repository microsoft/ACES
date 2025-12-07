"""AI Red Team Testing Demo - SABER Domain

This module exposes the airt_demo domain as an Inspect AI task that can be
evaluated with commands like:

    inspect eval domains/airt_demo --model openai/gpt-4
    inspect eval domains/airt_demo -T task_filter="airt_demo_*" --model anthropic/claude-3-opus
"""

from pathlib import Path

from inspect_ai import task

# Import SABER's task factory
from saber.inspect_ai import create_domain_task

# Get the workspace root (parent of domains/)
# This file is at: domains/airt_demo/airt_demo.py
# We need: /path/to/workspace/domains (the domains directory itself)
_domains_root = Path(__file__).resolve().parent.parent

# Create the task factory (returns a callable that Inspect AI will invoke)
_airt_demo_factory = create_domain_task(
    domain_slug="airt_demo",
    domains_root=_domains_root,
    default_agent="react",
)


# Wrap in @task decorator for Inspect AI discovery
@task
def airt_demo(**kwargs):
    """AI Red Team Testing Demo - AI safety guardrail testing

    Dynamically loads tasks from the running SABER server.

    Args:
        rest_port: REST API port (default: 8000)
        mcp_port: MCP API port (default: 8001)
        task_filter: Optional task filter (exact match, glob pattern, or comma-separated)
        log_level: Logging level for domain services (default: "INFO")
        build: Build missing images before starting (default: False)
        rebuild: Remove and rebuild images matching this prefix (e.g., 'server')
        rebuild_all: Remove and rebuild all images (default: False)
        stop_saber_after: Stop SABER domain after task completes (default: False)
        roles_file: Path to roles configuration file for dual-agent setup

    Returns:
        Inspect AI Task with SABER airt_demo dataset loaded from server

    Examples:
        # Basic evaluation (server stays running after)
        inspect eval domains/airt_demo --model openai/gpt-4

        # Build missing images first
        inspect eval domains/airt_demo --model openai/gpt-4 -T build=true

        # Rebuild all images (clean slate)
        inspect eval domains/airt_demo --model openai/gpt-4 -T rebuild_all=true

        # Filter to specific tasks
        inspect eval domains/airt_demo --model openai/gpt-4 -T task_filter="airt_demo_blue_agent"

        # Use dual-role configuration
        inspect eval domains/airt_demo -T roles_file=airt_demo_roles.yaml
    """
    return _airt_demo_factory(**kwargs)
