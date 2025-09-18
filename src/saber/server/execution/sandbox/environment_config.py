"""
Environment configuration for Docker Compose orchestration.

This module provides strongly-typed configuration objects for managing
Docker Compose environment variables and settings.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ComposeEnvironmentConfig:
    """
    Configuration for Docker Compose environment variables and settings.

    This replaces the need for multiple method parameters and provides
    a clean way to pass environment-specific configuration to the orchestrator.
    """

    # Episode/project identification
    episode_id: Optional[str] = None
    project_name: Optional[str] = None

    # Configuration type (sandbox or permanent)
    config_type: str = "sandbox"

    # Network configuration
    permanent_network_prefix: Optional[str] = None
    target_episode_id: Optional[str] = None  # Episode to attach to for network sharing

    # Additional environment variables for compose substitution
    additional_variables: Optional[Dict[str, str]] = None

    def to_env_dict(self) -> Dict[str, str]:
        """
        Convert configuration to environment variables dictionary.

        Returns:
            Dictionary of environment variables for Docker Compose
        """
        env_vars = {}

        # Standard episode/project variables
        if self.episode_id:
            env_vars["EPISODE_ID"] = self.episode_id

        if self.project_name:
            env_vars["COMPOSE_PROJECT_NAME"] = self.project_name
        elif self.episode_id:
            # Auto-generate project name from episode_id
            env_vars["COMPOSE_PROJECT_NAME"] = f"saber-episode-{self.episode_id}"

        # Network configuration
        if self.permanent_network_prefix:
            env_vars["PERMANENT_NETWORK_PREFIX"] = self.permanent_network_prefix

        if self.target_episode_id:
            env_vars["TARGET_EPISODE_ID"] = self.target_episode_id

        # Additional custom variables
        if self.additional_variables:
            env_vars.update(self.additional_variables)

        return env_vars

    def get_project_name(self) -> Optional[str]:
        """
        Get the effective project name for Docker Compose.

        Returns:
            Project name (explicit or auto-generated from episode_id)
        """
        if self.project_name:
            return self.project_name
        elif self.episode_id:
            return f"saber-episode-{self.episode_id}"
        else:
            return None
