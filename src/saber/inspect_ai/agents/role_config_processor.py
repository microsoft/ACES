"""Role Configuration Processing for SABER Tasks.

This module handles role-based configuration processing for SABER domains,
including extraction of role overrides from kwargs, loading and merging
configurations, and validation of role model assignments.
"""

from typing import Any

from inspect_ai._util.error import PrerequisiteError

from ...client.config_loader import RoleConfigLoader
from ...client.models import RoleBasedConfig
from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


def all_roles_have_models(role_config: RoleBasedConfig) -> bool:
    """Check if all roles in configuration have models defined.

    Args:
        role_config: Role-based configuration object

    Returns:
        True if all roles have models (or defaults provide a model), False otherwise
    """
    # Must have at least one role defined
    if not role_config.roles:
        return False

    # If defaults has a model, all roles will inherit it
    if role_config.defaults and role_config.defaults.model:
        return True

    # Check if all explicit roles have models
    for _role_name, role_conf in role_config.roles.items():
        if not role_conf.model:
            return False

    return True


def extract_role_overrides_from_kwargs(kwargs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract role-specific overrides from kwargs.

    Looks for kwargs with underscore-separated role names like:
    - red_model=gpt-4
    - blue_agent=custom_agent

    Args:
        kwargs: Keyword arguments dictionary (will be modified - matched keys removed)

    Returns:
        Dictionary mapping role names to their field overrides
        Example: {"red": {"model": "gpt-4"}, "blue": {"agent": "custom_agent"}}
    """
    role_overrides: dict[str, dict[str, Any]] = {}

    # Process in a list copy to allow modification during iteration
    for key, value in list(kwargs.items()):
        if "_" in key:
            parts = key.split("_", 1)
            if len(parts) == 2:
                # Potential role override like "red_model"
                role_name, field_name = parts
                if role_name not in role_overrides:
                    role_overrides[role_name] = {}
                role_overrides[role_name][field_name] = value
                # Remove from kwargs so it's not passed downstream
                del kwargs[key]

    return role_overrides


def load_and_merge_role_config(
    roles_source: Any,
    role_overrides: dict[str, dict[str, Any]],
    domain_slug: str,
) -> RoleBasedConfig:
    """Load role configuration from source and merge with overrides.

    Args:
        roles_source: Path to config file or config dict
        role_overrides: Role-specific overrides extracted from kwargs
        domain_slug: Domain slug for logging

    Returns:
        Merged RoleBasedConfig object

    Raises:
        PrerequisiteError: If loading or merging fails
    """
    try:
        # Load base configuration from file or dict
        base_config = RoleConfigLoader.load(roles_source)

        # Merge with overrides from kwargs
        merged_config = RoleConfigLoader.merge_with_overrides(base_config, role_overrides)

        logger.info(
            "Loaded role-based configuration",
            extra={
                "domain": domain_slug,
                "roles": list(merged_config.roles.keys()) if merged_config.roles else [],
                "has_defaults": merged_config.defaults is not None,
                "override_count": len(role_overrides),
            },
        )

        return merged_config

    except Exception as e:
        raise PrerequisiteError(f"Failed to load role configuration: {e}") from e


def process_role_configuration(
    roles_file: str | None,
    roles: Any | None,
    kwargs: dict[str, Any],
    domain_slug: str,
) -> RoleBasedConfig | None:
    """Process role-based configuration from multiple sources.

    Handles role configuration from:
    1. roles_file parameter (preferred for file-based configs)
    2. roles parameter (for inline dicts or JSON strings)
    3. kwargs role overrides (e.g., red_model=gpt-4)

    Args:
        roles_file: Path to YAML/JSON role configuration file
        roles: Inline role configuration (dict or JSON string)
        kwargs: Keyword arguments that may contain role overrides (will be modified)
        domain_slug: Domain slug for logging

    Returns:
        Processed RoleBasedConfig object, or None if no role config provided

    Raises:
        PrerequisiteError: If configuration processing fails
    """
    # Determine config source (prefer roles_file if both provided)
    roles_source = roles_file or roles

    if roles_source is None:
        return None

    # Extract role-specific overrides from kwargs (e.g., red_model=gpt-4)
    role_overrides = extract_role_overrides_from_kwargs(kwargs)

    # Load and merge configuration
    return load_and_merge_role_config(roles_source, role_overrides, domain_slug)
