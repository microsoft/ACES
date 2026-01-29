"""Provider configuration utilities for BYOK (Bring Your Own Key) authentication.

This module provides functions for building provider configurations
for agents that support BYOK authentication (Copilot, Claude Code, etc.).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from inspect_ai.model._model import active_model

from ....logging_config import LogCategory, get_saber_logger
from .models import (
    AnthropicProviderConfig,
    AzureProviderConfig,
    EnvVar,
    ModelPrefix,
    OpenAIProviderConfig,
    ProviderConfig,
    ProviderType,
)

if TYPE_CHECKING:
    pass

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Default Azure API version used when not specified
DEFAULT_AZURE_API_VERSION = "2025-03-01-preview"


def get_provider_config_from_inspect() -> ProviderConfig | None:
    """Extract provider configuration from the active Inspect AI model.

    Supports both API key and Entra ID (bearer token) authentication.
    For Azure, if no API key is found but a token_provider exists (from
    DefaultAzureCredential via az login), it will use bearer token auth.

    Returns:
        ProviderConfig subclass instance, or None if using Copilot auth
    """
    model = active_model()
    if model is None:
        logger.debug("No active Inspect AI model, using default auth")
        return None

    api = model.api
    model_name = api.model_name

    # Check for token_provider (Entra ID auth from inspect_ai)
    token_provider = getattr(api, "token_provider", None)

    logger.info(
        "Checking active model for provider config",
        extra={
            "model_name": model_name,
            "api_class": api.__class__.__name__,
            "has_base_url": hasattr(api, "base_url"),
            "has_api_key": hasattr(api, "api_key"),
            "has_token_provider": token_provider is not None,
        },
    )

    provider_type: ProviderType | None = None
    base_url: str | None = None
    api_key: str | None = None
    bearer_token: str | None = None
    api_version: str | None = None

    # Azure OpenAI
    if model_name.startswith(ModelPrefix.AZURE_OPENAI) or model_name.startswith(ModelPrefix.AZURE):
        provider_type = ProviderType.AZURE
        base_url = getattr(api, "base_url", None) or getattr(api, "endpoint_url", None)
        if not base_url:
            base_url = os.environ.get(EnvVar.AZUREAI_OPENAI_BASE_URL) or os.environ.get(EnvVar.AZURE_OPENAI_BASE_URL)

        # Try API key first
        api_key = (
            getattr(api, "api_key", None)
            or os.environ.get(EnvVar.AZUREAI_OPENAI_API_KEY)
            or os.environ.get(EnvVar.AZURE_OPENAI_API_KEY)
        )

        # If no API key, try Entra ID token provider (from inspect_ai's DefaultAzureCredential)
        if not api_key and token_provider is not None:
            try:
                bearer_token = token_provider()
                logger.info(
                    "Using Entra ID bearer token from inspect_ai token_provider",
                    extra={"model_name": model_name},
                )
            except Exception as e:
                logger.warning(f"Failed to get bearer token from token_provider: {e}")

        api_version = os.environ.get(EnvVar.AZUREAI_OPENAI_API_VERSION) or os.environ.get(
            EnvVar.OPENAI_API_VERSION, DEFAULT_AZURE_API_VERSION
        )
        deployment_name = model_name.split("/")[-1]
        if base_url and "/openai/deployments/" not in base_url:
            base_url = base_url.rstrip("/") + f"/openai/deployments/{deployment_name}"

    # Standard OpenAI
    elif model_name.startswith(ModelPrefix.OPENAI) or (
        hasattr(api, "__class__") and "OpenAI" in api.__class__.__name__
    ):
        provider_type = ProviderType.OPENAI
        base_url = getattr(api, "base_url", None) or "https://api.openai.com/v1"
        api_key = getattr(api, "api_key", None) or os.environ.get(EnvVar.OPENAI_API_KEY)

    # Anthropic
    elif model_name.startswith(ModelPrefix.ANTHROPIC) or "claude" in model_name.lower():
        provider_type = ProviderType.ANTHROPIC
        base_url = getattr(api, "base_url", None) or "https://api.anthropic.com"
        api_key = getattr(api, "api_key", None) or os.environ.get(EnvVar.ANTHROPIC_API_KEY)

    # Build provider config (supports both api_key and bearer_token)
    if provider_type and base_url and (api_key or bearer_token):
        config: AzureProviderConfig | OpenAIProviderConfig | AnthropicProviderConfig
        if provider_type == ProviderType.AZURE:
            config = AzureProviderConfig(
                base_url=base_url,
                api_key=api_key,
                bearer_token=bearer_token,
                api_version=api_version or DEFAULT_AZURE_API_VERSION,
            )
        elif provider_type == ProviderType.OPENAI:
            config = OpenAIProviderConfig(base_url=base_url, api_key=api_key)
        else:  # provider_type == ProviderType.ANTHROPIC
            config = AnthropicProviderConfig(base_url=base_url, api_key=api_key)

        logger.info(
            "Derived provider config from Inspect AI model",
            extra={
                "provider_type": provider_type.value,
                "model_name": model_name,
                "auth_method": "bearer_token" if bearer_token else "api_key",
            },
        )
        return config

    return None


def build_provider_config(
    provider_type: str | None,
    provider_base_url: str | None,
    provider_api_key: str | None,
    provider_api_version: str | None,
) -> ProviderConfig | None:
    """Build provider config from explicit parameters or derive from Inspect AI.

    Args:
        provider_type: Provider type - 'openai', 'azure', or 'anthropic'
        provider_base_url: API endpoint URL
        provider_api_key: API key
        provider_api_version: Azure API version (optional)

    Returns:
        ProviderConfig subclass instance, or None if not configured
    """
    if provider_type and provider_base_url and provider_api_key:
        if provider_type == ProviderType.AZURE or provider_type == "azure":
            return AzureProviderConfig(
                base_url=provider_base_url,
                api_key=provider_api_key,
                api_version=provider_api_version or DEFAULT_AZURE_API_VERSION,
            )
        elif provider_type == ProviderType.OPENAI or provider_type == "openai":
            return OpenAIProviderConfig(
                base_url=provider_base_url,
                api_key=provider_api_key,
            )
        elif provider_type == ProviderType.ANTHROPIC or provider_type == "anthropic":
            return AnthropicProviderConfig(
                base_url=provider_base_url,
                api_key=provider_api_key,
            )

    # Fall back to deriving from Inspect AI
    return get_provider_config_from_inspect()


__all__ = [
    "build_provider_config",
    "get_provider_config_from_inspect",
]
