"""AI Red Team Testing Demo - SABER Domain

This module exposes the airt_demo domain as an Inspect AI task that can be
evaluated with commands like:

    inspect eval domains/airt_demo -T roles_file=domains/airt_demo/ai_redteam_roles.yaml

NOTE: This domain REQUIRES a roles_file to configure different models and behaviors
for red and blue team agents. Using --model alone is not supported.
"""

from pathlib import Path

from inspect_ai import task
from inspect_ai._util.error import PrerequisiteError

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
def airt_demo(roles_file: str | None = None, roles: str | dict | None = None, **kwargs):
    """AI Red Team Testing Demo - AI safety guardrail testing

    This domain implements adversarial AI red teaming where a red team agent
    attempts to bypass the safety guardrails of a blue team agent through
    prompt injection attacks.

    REQUIRED: You must specify a roles_file to configure different models
    and behaviors for red and blue team agents.

    Args:
        roles_file: REQUIRED - Path to roles configuration YAML file
        roles: Alternative inline role configuration (dict or JSON string)
        rest_port: REST API port (default: 8000)
        mcp_port: MCP API port (default: 8001)
        task_filter: Optional task filter (exact match, glob pattern, or comma-separated)
        log_level: Logging level for domain services (default: "INFO")
        build: Build missing images before starting (default: False)
        rebuild: Remove and rebuild images matching this prefix (e.g., 'server')
        rebuild_all: Remove and rebuild all images (default: False)
        stop_saber_after: Stop SABER domain after task completes (default: False)

    Returns:
        Inspect AI Task with SABER airt_demo dataset loaded from server

    Examples:
        # Use the default dual-role configuration (RECOMMENDED)
        inspect eval domains/airt_demo -T roles_file=domains/airt_demo/ai_redteam_roles.yaml

        # With rebuild
        inspect eval domains/airt_demo -T roles_file=domains/airt_demo/ai_redteam_roles.yaml -T rebuild=server

        # Filter to specific role
        inspect eval domains/airt_demo -T roles_file=domains/airt_demo/ai_redteam_roles.yaml -T task_filter="*blue*"
    """
    # Validate that roles_file or roles is provided
    if roles_file is None and roles is None:
        raise PrerequisiteError(
            "The airt_demo domain REQUIRES a roles configuration file.\n\n"
            "This domain runs adversarial red/blue team agents that need different\n"
            "models and configurations (e.g., blue team has submit disabled).\n\n"
            "Usage:\n"
            "  inspect eval domains/airt_demo -T roles_file=domains/airt_demo/ai_redteam_roles.yaml\n\n"
            "You can customize the roles file or create your own. See:\n"
            "  domains/airt_demo/ai_redteam_roles.yaml\n\n"
            "NOTE: Do not use --model with this domain. Model selection is handled\n"
            "      per-role in the roles configuration file."
        )

    return _airt_demo_factory(roles_file=roles_file, roles=roles, **kwargs)
