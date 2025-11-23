"""
Simple SABER Configuration Loader

Replaces the old HarnessConfigLoader with a simple YAML loader that works
with the new SABERConfig format and eval_async integration.

Logging category: CONFIG

Following SABER's philosophy:
- Fail fast when configuration files are invalid or missing
- Clean validation of required fields
- No silent fallbacks that mask configuration issues
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from ..logging_config import (
    LogCategory,
    get_saber_logger,
)
from .exceptions import InvalidRoleOverrideError, RoleConfigLoadError
from .models import RoleAgentConfig, RoleBasedConfig

logger = get_saber_logger(LogCategory.CONFIG, __name__)


# =============================================================================
# Role-Based Configuration Loader
# =============================================================================


class RoleConfigLoader:
    """Load role-based configuration from multiple sources.

    Supports:
    - Inline JSON strings
    - JSON files
    - YAML files
    - Python dicts
    """

    @staticmethod
    def load(source: Union[str, Dict, Path]) -> RoleBasedConfig:
        """Load role configuration from various sources.

        Args:
            source: Can be:
                - String as file path → loads YAML/JSON file
                - String starting with '{' → inline JSON
                - Dict → direct Python dict
                - Path → file path

        Returns:
            Parsed RoleBasedConfig

        Raises:
            ValueError: If source format is invalid
            FileNotFoundError: If file not found
        """
        # Handle dict directly
        if isinstance(source, dict):
            return RoleBasedConfig(**source)

        # Handle Path objects
        if isinstance(source, Path):
            return RoleConfigLoader._load_from_file(source)

        # Handle string sources
        if isinstance(source, str):
            # Inline JSON (starts with {)
            if source.strip().startswith("{"):
                return RoleConfigLoader._load_from_json_string(source)

            # Try as file path (relative or absolute)
            file_path = Path(source).expanduser()  # Expand ~ in paths

            # If relative path, try multiple resolution strategies:
            # 1. From current working directory (for direct runs)
            # 2. Walk up to find workspace root with configs/ (for inspect eval)
            if not file_path.is_absolute():
                cwd_path = Path.cwd() / file_path

                if cwd_path.exists():
                    file_path = cwd_path
                else:
                    # Try to find workspace root by looking for configs/ directory
                    search_dir = Path.cwd()
                    found = False

                    # Walk up directory tree looking for configs/
                    for _ in range(5):  # Limit search depth
                        candidate = search_dir / file_path
                        if candidate.exists():
                            file_path = candidate
                            found = True
                            break

                        # Move up one directory
                        parent = search_dir.parent
                        if parent == search_dir:  # Reached filesystem root
                            break
                        search_dir = parent

                    if not found:
                        file_path = cwd_path  # Use original for error message

            if file_path.exists():
                return RoleConfigLoader._load_from_file(file_path)

            raise ValueError(
                f"Invalid role config source: {source}\n"
                f"Expected: file path (e.g., configs/roles.yaml), {{json}}, or dict\n"
                f"Resolved path: {file_path}\n"
                f"Current directory: {Path.cwd()}"
            )

        raise TypeError(f"Unsupported source type: {type(source)}")

    @staticmethod
    def _load_from_file(file_path: Path) -> RoleBasedConfig:
        """Load configuration from YAML or JSON file."""
        import json

        if not file_path.exists():
            raise RoleConfigLoadError(
                f"Role config file not found: {file_path}",
                details={"file_path": str(file_path.absolute())},
                suggestion="Use -T roles_file=path/to/config.yaml with correct path (relative or absolute)",
            )

        with open(file_path, "r") as f:
            content = f.read()

        # Detect format by extension
        if file_path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(content)
        elif file_path.suffix == ".json":
            data = json.loads(content)
        else:
            # Try YAML first, then JSON
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    raise RoleConfigLoadError(
                        f"Could not parse {file_path} as YAML or JSON",
                        details={"file_path": str(file_path), "error": str(e)},
                        suggestion="Check file syntax for YAML or JSON errors",
                    )

        return RoleBasedConfig(**data)

    @staticmethod
    def _load_from_json_string(json_str: str) -> RoleBasedConfig:
        """Load configuration from inline JSON string."""
        import json

        try:
            data = json.loads(json_str)
            return RoleBasedConfig(**data)
        except json.JSONDecodeError as e:
            raise RoleConfigLoadError(
                "Invalid JSON in role config",
                details={"error": str(e), "json_str": json_str[:100]},
                suggestion="Check for trailing commas, quotes, and braces",
            )

    @staticmethod
    def merge_with_overrides(
        base_config: RoleBasedConfig, overrides: Dict[str, Any], strict: bool = True
    ) -> RoleBasedConfig:
        """Merge base config with CLI overrides.

        Supports overrides like:
            blue_model=gpt-4.1  → roles.blue.model = gpt-4.1
            red_agent=custom    → roles.red.agent = custom

        Args:
            base_config: Base configuration from file/JSON
            overrides: Dict of override keys and values
            strict: If True, raise error on unknown fields (default: True).
                   If False, unknown fields are silently skipped.

        Returns:
            Merged RoleBasedConfig

        Raises:
            ValueError: If strict=True and unknown field in override

        Examples:
            >>> base = RoleBasedConfig(roles={"red": RoleAgentConfig(agent="react")})
            >>> overrides = {"red_model": "gpt-4", "blue_model": "claude-3"}
            >>> merged = RoleConfigLoader.merge_with_overrides(base, overrides)
            >>> merged.roles["red"].model
            'gpt-4'
        """
        # Clone base config
        merged_data = base_config.dict()

        # Get valid field names dynamically from RoleAgentConfig
        valid_fields = set(RoleAgentConfig.model_fields.keys())
        unknown_overrides = []

        for key, value in overrides.items():
            # Parse role_field=value patterns
            if "_" not in key:
                continue

            parts = key.split("_", 1)
            if len(parts) != 2:
                continue

            role, field = parts

            # Validate field name
            if field not in valid_fields:
                unknown_overrides.append(key)
                if strict:
                    raise InvalidRoleOverrideError(
                        f"Unknown field in override '{key}'",
                        details={"override_key": key, "valid_fields": sorted(valid_fields)},
                        suggestion=f"Use one of: {', '.join(sorted(valid_fields))}",
                    )
                continue

            # Warn if creating new role (possible typo)
            if role not in merged_data["roles"]:
                logger.warning(
                    f"Creating new role '{role}' from override. "
                    f"Existing roles: {list(merged_data['roles'].keys())}",
                    extra={
                        "event": "role_config_new_role",
                        "role": role,
                        "field": field,
                        "existing_roles": list(merged_data["roles"].keys()),
                    },
                )
                merged_data["roles"][role] = {}

            # Set the override
            merged_data["roles"][role][field] = value

        if unknown_overrides and not strict:
            logger.debug(
                f"Skipped unknown overrides (strict=False): {unknown_overrides}",
                extra={"event": "role_config_unknown_overrides", "unknown_overrides": unknown_overrides},
            )

        return RoleBasedConfig(**merged_data)
