#!/usr/bin/env python3
"""
Azure OpenAI Client for SABER Agents

Provides a standardized interface for Azure OpenAI interactions that can be
injected into agents for configurable LLM access.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """
    Azure OpenAI client wrapper for SABER agents.

    Handles Azure OpenAI configuration and provides a standardized interface
    for LLM interactions with proper error handling and fallbacks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        env_file: Optional[Path] = None,
    ):
        """
        Initialize Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key (defaults to env var)
            endpoint: Azure OpenAI endpoint (defaults to env var)
            deployment: Azure deployment name (defaults to env var)
            api_version: API version to use
            env_file: Optional path to .env file for configuration
        """
        # Load environment file if provided
        if env_file and env_file.exists():
            load_dotenv(env_file, override=True)

        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")

        self._client: Any = None
        self._model_name = self.deployment
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        if not all([self.api_key, self.endpoint, self.deployment, self.api_version]):
            missing = []
            if not self.api_key:
                missing.append("AZURE_OPENAI_API_KEY")
            if not self.endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not self.deployment:
                missing.append("AZURE_OPENAI_DEPLOYMENT")
            if not self.api_version:
                missing.append("AZURE_OPENAI_API_VERSION")

            raise ValueError(f"Missing required Azure OpenAI configuration: {', '.join(missing)}")

        try:
            import openai

            if not all([self.api_key, self.endpoint, self.api_version]):
                raise ValueError("Missing required configuration parameters")

            # Type-safe assignment
            assert self.api_key is not None
            assert self.endpoint is not None
            assert self.api_version is not None

            # Create OpenAI client with aggressive retry configuration
            self._client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                max_retries=10,  # More retries at OpenAI level
                timeout=60.0,  # Shorter timeout for faster failure detection
            )
            logger.info(
                f"🤖 Azure OpenAI client initialized with deployment: {self.deployment} (10 retries, 60s timeout)"
            )
        except ImportError:
            raise ImportError("openai library not installed. Install with: uv add openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI client: {e}")

    @property
    def chat(self) -> Any:
        """Access to chat completions."""
        if self._client is None:
            raise RuntimeError("Azure OpenAI client not initialized")
        return self._client.chat

    @property
    def model_name(self) -> str:
        """Get the model/deployment name."""
        if self._model_name is None:
            raise RuntimeError("Model name not available")
        return self._model_name

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any:
        """
        Create a chat completion with standardized parameters and aggressive retry logic.

        Args:
            messages: Conversation messages
            tools: Available tools for function calling
            tool_choice: Tool choice strategy
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            OpenAI completion response
        """
        max_app_retries = 5  # Application-level retries on top of client retries
        base_delay = 1.0  # Start with 1 second delay

        for attempt in range(max_app_retries):
            try:
                completion_args = {
                    "model": self.deployment,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs,
                }

                if tools:
                    completion_args["tools"] = tools
                    completion_args["tool_choice"] = tool_choice

                if self._client is None:
                    raise RuntimeError("Azure OpenAI client not initialized")

                # Log retry attempt for visibility
                if attempt > 0:
                    logger.info(f"🔄 OpenAI completion retry attempt {attempt + 1}/{max_app_retries}")

                return self._client.chat.completions.create(**completion_args)

            except Exception as e:
                is_last_attempt = attempt == max_app_retries - 1

                if is_last_attempt:
                    logger.error(f"❌ Azure OpenAI completion failed after {max_app_retries} attempts: {e}")
                    raise
                else:
                    # Simple exponential backoff without jitter
                    delay = min(base_delay * (2**attempt), 8.0)  # Cap at 8 seconds

                    logger.warning(f"⚠️ OpenAI completion failed (attempt {attempt + 1}/{max_app_retries}): {e}")
                    logger.info(f"⏳ Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attributes to the underlying client."""
        if self._client is None:
            raise RuntimeError("Azure OpenAI client not initialized")
        return getattr(self._client, name)
