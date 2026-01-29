"""Copilot SDK-specific models and configuration.

This module defines strongly-typed configuration classes specific to the
GitHub Copilot SDK integration.

Shared models (ProviderType, ProviderConfig, etc.) are imported from the parent
registry/models.py module and re-exported for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

# Import shared models from parent - re-export for backwards compatibility
from ..models import (
    AnthropicProviderConfig,
    AzureProviderConfig,
    DefaultUrl,
    EnvVar,
    OpenAIProviderConfig,
    ProviderConfig,
    ProviderType,
)

if TYPE_CHECKING:
    from ..custom_agent import CustomAgentConfig
    from .tools import Tool


class DefaultValue(str, Enum):
    """Default configuration values for Copilot SDK."""

    AZURE_API_VERSION = "2025-03-01-preview"
    # Use gpt-5 as default since it provides better transcript data via report_intent tool
    MODEL = "gpt-5"


class CopilotBuiltInTools(str, Enum):
    """Built-in tools provided by the Copilot SDK.

    These tools are handled by the SDK itself and don't require
    custom tool handlers. They must be included in availableTools
    to be accessible to the agent.
    """

    REPORT_INTENT = "report_intent"
    """Allows GPT-5 to report its intent/reasoning for better transcripts."""

    SKILL = "skill"
    """Loads skill content from skillDirectories when called by the agent."""


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
                        built-in tools like filesystem access. We always
                        include 'report_intent' to capture GPT-5's intent
                        for better transcripts.
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
    skill_directories: list[str] | None = None
    custom_agents: list[CustomAgentConfig] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by Copilot SDK.

        Returns:
            Dictionary suitable for passing to client.create_session()
        """
        result: dict[str, Any] = {
            "model": self.model,
            "tools": self.tools,
            # Note: SDK expects snake_case 'available_tools'
            "available_tools": self.available_tools,
            # Note: SDK expects snake_case 'system_message'
            "system_message": self.system_message.to_dict(),
            "streaming": self.streaming,
        }

        if self.provider is not None:
            result["provider"] = self.provider.to_dict()

        if self.skill_directories is not None:
            # Note: SDK expects snake_case 'skill_directories'
            result["skill_directories"] = self.skill_directories

        if self.custom_agents is not None:
            # Note: SDK expects snake_case 'custom_agents'
            result["custom_agents"] = [agent.to_sdk_dict() for agent in self.custom_agents]

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
        skill_directories: list[str] | None = None,
        custom_agents: list[CustomAgentConfig] | None = None,
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
            skill_directories: Optional list of directories containing skill files
            custom_agents: Optional list of custom agent persona configurations

        Returns:
            Configured CopilotSessionConfig instance
        """
        # Build list of available tools
        # Include report_intent to allow GPT-5 to report its intent/reasoning
        # This is a Copilot built-in tool that doesn't need a handler
        available = [t.name for t in tools] + [CopilotBuiltInTools.REPORT_INTENT.value]

        # Include 'skill' tool when skill_directories is provided
        # The skill tool is a Copilot SDK built-in that loads skill content
        if skill_directories:
            available.append(CopilotBuiltInTools.SKILL.value)

        return cls(
            model=model,
            tools=tools,
            available_tools=available,
            system_message=SystemMessageConfig(
                content=system_content,
                mode=system_mode,
            ),
            streaming=streaming,
            provider=provider,
            skill_directories=skill_directories,
            custom_agents=custom_agents,
        )


__all__ = [
    # Enums
    "ProviderType",
    "EnvVar",
    "DefaultUrl",
    "DefaultValue",
    "CopilotBuiltInTools",
    # Provider configs
    "ProviderConfig",
    "OpenAIProviderConfig",
    "AnthropicProviderConfig",
    "AzureProviderConfig",
    # Session configs
    "SystemMessageConfig",
    "CopilotSessionConfig",
]
