"""
LLM Client Factory

Factory for creating different types of LLM clients based on configuration.
Supports extensible LLM provider integration.
"""

import os
from pathlib import Path
from typing import Any, Optional

from .azure_openai_client import AzureOpenAIClient


class LLMClientFactory:
    """
    Factory for creating LLM clients.

    Supports different LLM providers and automatically detects
    available configurations from environment.
    """

    SUPPORTED_PROVIDERS = {
        "azureopenai": AzureOpenAIClient,
        "azure_openai": AzureOpenAIClient,  # Alias
    }

    @classmethod
    def create_client(
        cls, provider: Optional[str] = None, env_file: Optional[Path] = None, **kwargs: Any
    ) -> Optional[Any]:
        """
        Create LLM client based on provider or auto-detection.

        Args:
            provider: Specific provider name (e.g., "azureopenai")
            env_file: Optional path to .env file
            **kwargs: Additional configuration parameters

        Returns:
            LLM client instance or None if no configuration found
        """
        if provider:
            return cls._create_specific_client(provider, env_file, **kwargs)
        else:
            return cls._auto_detect_client(env_file, **kwargs)

    @classmethod
    def _create_specific_client(cls, provider: str, env_file: Optional[Path] = None, **kwargs: Any) -> Any:
        """Create client for specific provider."""
        provider_key = provider.lower()
        if provider_key not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported: {list(cls.SUPPORTED_PROVIDERS.keys())}")

        client_class = cls.SUPPORTED_PROVIDERS[provider_key]
        return client_class(env_file=env_file, **kwargs)

    @classmethod
    def _auto_detect_client(cls, env_file: Optional[Path] = None, **kwargs: Any) -> Optional[Any]:
        """Auto-detect available LLM provider from environment."""
        # Try Azure OpenAI first
        if cls._has_azure_openai_config(env_file):
            try:
                return AzureOpenAIClient(env_file=env_file, **kwargs)
            except Exception:
                pass

        # Add more providers here as they're implemented
        # if cls._has_openai_config(env_file):
        #     return OpenAIClient(env_file=env_file, **kwargs)

        return None

    @classmethod
    def _has_azure_openai_config(cls, env_file: Optional[Path] = None) -> bool:
        """Check if Azure OpenAI configuration is available."""
        # Load env file if provided
        if env_file and env_file.exists():
            from dotenv import load_dotenv

            load_dotenv(env_file, override=True)

        required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]

        return all(os.getenv(var) for var in required_vars)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported LLM providers."""
        return list(cls.SUPPORTED_PROVIDERS.keys())


def create_llm_client(provider: Optional[str] = None, env_file: Optional[Path] = None, **kwargs: Any) -> Optional[Any]:
    """
    Convenience function to create LLM client.

    Args:
        provider: Specific provider name or None for auto-detection
        env_file: Optional path to .env file
        **kwargs: Additional configuration parameters

    Returns:
        LLM client instance or None if no configuration found
    """
    return LLMClientFactory.create_client(provider, env_file, **kwargs)
