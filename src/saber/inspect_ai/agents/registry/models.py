"""Shared models and constants for SABER agent registry.

This module defines enums, constants, and typed dictionaries that are
shared across all agent implementations (React, Copilot, Claude Code).

Agent-specific models are in their respective submodules:
- copilot/models.py: Copilot SDK-specific models
- claude_code/: (currently inline, may be extracted later)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict


class TranscriptWebSocketConfig(TypedDict, total=False):
    """WebSocket configuration for transcript sync."""

    push: dict[str, bool]
    pull: dict[str, bool]


class TranscriptConfig(TypedDict, total=False):
    """Transcript configuration from task metadata."""

    websocket: TranscriptWebSocketConfig


class AgentPromptKwargs(TypedDict, total=False):
    """Kwargs passed to agent create_with_prompts factory."""

    instruction_prompt: str
    assistant_prompt: str
    submit_prompt: str
    continue_prompt: str
    transcript_config: TranscriptConfig | None
    submit: bool | None
    skill_directories: list[str] | None
    agent_persona: str | None


class AgentType(str, Enum):
    """Supported SABER agent types."""

    REACT = "react"
    COPILOT = "copilot"
    CLAUDE_CODE = "claude_code"

    @classmethod
    def supports_skill_directories(cls, agent_name: str) -> bool:
        """Check if an agent type supports skill_directories parameter."""
        return agent_name in (cls.COPILOT.value, cls.CLAUDE_CODE.value)

    @classmethod
    def supports_agent_persona(cls, agent_name: str) -> bool:
        """Check if an agent type supports agent_persona parameter.

        Currently only the Copilot agent supports custom agent personas
        via the custom_agents configuration.

        Args:
            agent_name: Name of the agent type to check.

        Returns:
            True if the agent supports agent_persona, False otherwise.
        """
        return agent_name == cls.COPILOT.value


class ModelPrefix(str, Enum):
    """Model name prefixes for provider detection."""

    AZURE_OPENAI = "openai/azure/"
    AZURE = "azure/"
    OPENAI = "openai/"
    ANTHROPIC = "anthropic/"


class ProviderType(str, Enum):
    """Supported LLM provider types for BYOK configuration.

    Used by agents that support Bring Your Own Key (BYOK) to
    specify which provider to use.
    """

    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EnvVar(str, Enum):
    """Environment variable names for provider configuration.

    Standard environment variable names used across agents for
    configuring LLM provider credentials and endpoints.
    """

    # Azure OpenAI
    AZUREAI_OPENAI_BASE_URL = "AZUREAI_OPENAI_BASE_URL"
    AZURE_OPENAI_BASE_URL = "AZURE_OPENAI_BASE_URL"
    AZUREAI_OPENAI_API_KEY = "AZUREAI_OPENAI_API_KEY"
    AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
    AZUREAI_OPENAI_API_VERSION = "AZUREAI_OPENAI_API_VERSION"
    OPENAI_API_VERSION = "OPENAI_API_VERSION"

    # OpenAI
    OPENAI_API_KEY = "OPENAI_API_KEY"

    # Anthropic
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"


class DefaultUrl(str, Enum):
    """Default API endpoint URLs for LLM providers."""

    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com"


# =============================================================================
# Provider Configuration Classes (Shared across agents)
# =============================================================================


@dataclass(kw_only=True)
class ProviderConfig(ABC):
    """Base class for LLM provider configurations.

    This abstract base class defines the common interface for all
    BYOK (Bring Your Own Key) provider configurations.

    Supports both API key and bearer token authentication:
    - api_key: Traditional API key authentication
    - bearer_token: Entra ID / OAuth token authentication (takes precedence over api_key)

    Note: Uses kw_only=True to allow derived classes to have fields
    with default values without violating dataclass field ordering rules.
    """

    base_url: str
    api_key: str | None = None
    bearer_token: str | None = None  # Entra ID token, takes precedence over api_key

    def __post_init__(self) -> None:
        """Validate that at least one auth method is provided."""
        if not self.api_key and not self.bearer_token:
            raise ValueError("Either api_key or bearer_token must be provided")

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type for this configuration."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for SDK configurations.

        Returns:
            Dictionary with provider configuration
        """
        result: dict[str, Any] = {
            "type": self.provider_type.value,
            "base_url": self.base_url,
        }
        # bearer_token takes precedence over api_key
        if self.bearer_token:
            result["bearer_token"] = self.bearer_token
        elif self.api_key:
            result["api_key"] = self.api_key
        return result


@dataclass(kw_only=True)
class OpenAIProviderConfig(ProviderConfig):
    """OpenAI provider configuration.

    Example:
        config = OpenAIProviderConfig(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
        )
    """

    base_url: str = DefaultUrl.OPENAI.value

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI


@dataclass(kw_only=True)
class AnthropicProviderConfig(ProviderConfig):
    """Anthropic provider configuration.

    Example:
        config = AnthropicProviderConfig(
            api_key="sk-ant-...",
            base_url="https://api.anthropic.com",
        )
    """

    base_url: str = DefaultUrl.ANTHROPIC.value

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC


@dataclass(kw_only=True)
class AzureProviderConfig(ProviderConfig):
    """Azure OpenAI provider configuration.

    Extends the base provider config with Azure-specific options
    like API version. Supports both API key and Entra ID (bearer token) auth.

    Note: The api_version default is set to a commonly used Azure API version.
    For Copilot SDK specifically, use copilot.models.DefaultValue.AZURE_API_VERSION.

    Example with API key:
        config = AzureProviderConfig(
            base_url="https://your-resource.openai.azure.com/openai/deployments/gpt-4o",
            api_key="your-api-key",
            api_version="2024-02-15-preview",
        )

    Example with Entra ID (bearer token):
        config = AzureProviderConfig(
            base_url="https://your-resource.openai.azure.com/openai/deployments/gpt-4o",
            bearer_token="eyJ0eXAi...",  # Token from DefaultAzureCredential
            api_version="2024-02-15-preview",
        )
    """

    api_version: str = "2025-03-01-preview"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.AZURE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for SDK configurations.

        Includes Azure-specific options in the output.

        Returns:
            Dictionary with Azure provider configuration
        """
        result = super().to_dict()
        result["azure"] = {"api_version": self.api_version}
        return result


__all__ = [
    # TypedDicts
    "TranscriptWebSocketConfig",
    "TranscriptConfig",
    "AgentPromptKwargs",
    # Enums
    "AgentType",
    "ModelPrefix",
    "ProviderType",
    "EnvVar",
    "DefaultUrl",
    # Provider configs
    "ProviderConfig",
    "OpenAIProviderConfig",
    "AnthropicProviderConfig",
    "AzureProviderConfig",
]
