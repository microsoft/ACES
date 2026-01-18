"""Models and constants for Copilot agent configuration.

This module defines enums and constants for provider types, environment
variables, and default values used by the Copilot agent integration.
"""

from enum import Enum


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


__all__ = [
    "ProviderType",
    "ModelPrefix",
    "EnvVar",
    "DefaultUrl",
    "DefaultValue",
    "CopilotModel",
]
