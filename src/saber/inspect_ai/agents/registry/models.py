"""Models and constants for Copilot agent configuration.

This module defines enums, constants, and strongly-typed configuration
classes for provider types, environment variables, session configs,
and default values used by the Copilot agent integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..integration.copilot_tools import Tool


class ProviderType(str, Enum):
    """Supported LLM provider types for BYOK configuration."""

    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelPrefix(str, Enum):
    """Model name prefixes for provider detection."""

    AZURE_OPENAI = "openai/azure/"
    AZURE = "azure/"
    OPENAI = "openai/"
    ANTHROPIC = "anthropic/"


class EnvVar(str, Enum):
    """Environment variable names for provider configuration."""

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
    """Default API endpoint URLs."""

    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com"


class DefaultValue(str, Enum):
    """Default configuration values."""

    AZURE_API_VERSION = "2025-03-01-preview"
    MODEL = "gpt-4o"


# Copilot SDK supported models
class CopilotModel(str, Enum):
    """Models supported by the Copilot SDK."""

    GPT_4O = "gpt-4o"
    GPT_5 = "gpt-5"
    CLAUDE_SONNET_4 = "claude-sonnet-4"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4.5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4.5"


# =============================================================================
# Provider Configuration Classes
# =============================================================================


@dataclass(kw_only=True)
class ProviderConfig(ABC):
    """Base class for LLM provider configurations.

    This abstract base class defines the common interface for all
    BYOK (Bring Your Own Key) provider configurations.

    Note: Uses kw_only=True to allow derived classes to have fields
    with default values without violating dataclass field ordering rules.
    """

    base_url: str
    api_key: str

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type for this configuration."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by Copilot SDK.

        Returns:
            Dictionary with provider configuration
        """
        return {
            "type": self.provider_type.value,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }


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
    like API version.

    Example:
        config = AzureProviderConfig(
            base_url="https://your-resource.openai.azure.com/openai/deployments/gpt-4o",
            api_key="your-api-key",
            api_version="2024-02-15-preview",
        )
    """

    api_version: str = DefaultValue.AZURE_API_VERSION.value

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.AZURE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by Copilot SDK.

        Includes Azure-specific options in the output.

        Returns:
            Dictionary with Azure provider configuration
        """
        result = super().to_dict()
        result["azure"] = {"api_version": self.api_version}
        return result


# =============================================================================
# System Message Configuration
# =============================================================================


@dataclass
class SystemMessageConfig:
    """Configuration for the system message in a Copilot session.

    Attributes:
        content: The system message content
        mode: How to apply the message - 'append' adds to default,
              'replace' replaces the default system message
    """

    content: str
    mode: Literal["append", "replace"] = "append"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format expected by Copilot SDK.

        Returns:
            Dictionary with system message configuration
        """
        return {
            "mode": self.mode,
            "content": self.content,
        }


# =============================================================================
# Session Configuration
# =============================================================================


@dataclass
class CopilotSessionConfig:
    """Strongly-typed configuration for a Copilot SDK session.

    This class provides type safety and validation for session configuration,
    replacing the untyped dict[str, Any] approach.

    Attributes:
        model: The model to use (e.g., 'gpt-4o', 'claude-sonnet-4')
        tools: List of Tool objects available to the agent
        available_tools: List of tool names to restrict the agent to.
                        IMPORTANT: This prevents Copilot from accessing
                        built-in tools like filesystem access.
        system_message: System message configuration
        streaming: Whether to enable streaming responses
        provider: Optional BYOK provider configuration

    Example:
        config = CopilotSessionConfig(
            model="gpt-4o",
            tools=my_tools,
            available_tools=[t.name for t in my_tools],
            system_message=SystemMessageConfig(
                content="You are a security researcher.",
                mode="append",
            ),
            streaming=False,
            provider=AzureProviderConfig(
                base_url="https://my-resource.openai.azure.com/...",
                api_key="my-key",
            ),
        )
        session = await client.create_session(config.to_dict())
    """

    model: str
    tools: list[Tool]
    available_tools: list[str]
    system_message: SystemMessageConfig
    streaming: bool = False
    provider: ProviderConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by Copilot SDK.

        Returns:
            Dictionary suitable for passing to client.create_session()
        """
        result: dict[str, Any] = {
            "model": self.model,
            "tools": self.tools,
            "available_tools": self.available_tools,
            "system_message": self.system_message.to_dict(),
            "streaming": self.streaming,
        }

        if self.provider is not None:
            result["provider"] = self.provider.to_dict()

        return result

    @classmethod
    def create(
        cls,
        model: str,
        tools: list[Tool],
        system_content: str,
        *,
        streaming: bool = False,
        system_mode: Literal["append", "replace"] = "append",
        provider: ProviderConfig | None = None,
    ) -> CopilotSessionConfig:
        """Factory method for creating a session config.

        This is a convenience method that automatically extracts tool names
        and creates the system message config.

        Args:
            model: The model to use
            tools: List of Tool objects
            system_content: System message content
            streaming: Enable streaming responses
            system_mode: How to apply system message ('append' or 'replace')
            provider: Optional BYOK provider configuration

        Returns:
            Configured CopilotSessionConfig instance
        """
        return cls(
            model=model,
            tools=tools,
            available_tools=[t.name for t in tools],
            system_message=SystemMessageConfig(
                content=system_content,
                mode=system_mode,
            ),
            streaming=streaming,
            provider=provider,
        )


__all__ = [
    # Enums
    "ProviderType",
    "ModelPrefix",
    "EnvVar",
    "DefaultUrl",
    "DefaultValue",
    "CopilotModel",
    # Provider configs
    "ProviderConfig",
    "OpenAIProviderConfig",
    "AnthropicProviderConfig",
    "AzureProviderConfig",
    # Session configs
    "SystemMessageConfig",
    "CopilotSessionConfig",
]
