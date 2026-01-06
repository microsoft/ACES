"""Environment and configuration utilities for SABER Inspect AI sandbox.

This module provides:
- .env file loading for eval-retry support
- Inspect AI sandbox configuration interface methods
- Configuration validation and deserialization
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from saber.logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


def load_environment() -> None:
    """Load environment variables from .env file for eval-retry support.

    This ensures API keys and other config are available when retrying evals.
    Tries to find .env file in saber package directory first, then falls back
    to current working directory.
    """
    try:
        from dotenv import load_dotenv

        # Try to find .env file in saber package directory
        saber_dir = Path(__file__).parent.parent.parent
        env_file = saber_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.debug(f"Loaded environment from {env_file}", extra={"env_file": str(env_file)})
        else:
            # Try current working directory
            load_dotenv()
            logger.debug("Loaded environment from default locations")
    except ImportError:
        logger.debug("python-dotenv not available, skipping .env loading")
    except Exception as e:
        logger.debug(f"Failed to load .env file: {e}")


def get_config_files() -> list[str]:
    """Return list of config files for SABER sandbox.

    SABER doesn't use file-based config (all config is passed programmatically),
    so return empty list.

    Returns:
        Empty list (no config files needed)
    """
    return []


def deserialize_config(config: dict[str, Any]) -> BaseModel:
    """Deserialize SABER sandbox config from dict.

    Since SABER passes config as constructor kwargs (not a BaseModel),
    we create a simple BaseModel wrapper to satisfy the Inspect AI interface.

    Args:
        config: Configuration dictionary with SABER sandbox parameters

    Returns:
        BaseModel instance with config fields (frozen for hashability)
    """
    # Create a dynamic model with the config fields (frozen=True for hashability)
    SABERConfig = create_model(
        "SABERConfig",
        domain_slug=(str, ...),
        domains_root=(Path, ...),
        rest_port=(int, 8000),
        mcp_port=(int, 8001),
        compose_template_path=(Path | None, None),
        cleanup=(bool, False),  # Default to False - keep server running
        __config__=ConfigDict(frozen=True),
    )

    return SABERConfig(**config)


# Default concurrency for SABER sandbox (controls --max-samples default)
DEFAULT_SABER_CONCURRENCY = 8


def get_default_concurrency() -> int | None:
    """Default max_sandboxes for SABER provider.

    Returns a sensible default for concurrent sample execution. This value is
    used by Inspect AI's --max-samples when not explicitly specified.

    Concurrency is controlled at the Inspect AI level via --max-samples:
    - Each sample corresponds to one SABER episode
    - Orchestrated tasks (e.g., blue+red) run as separate samples that coordinate
    - Higher values allow more parallel evaluations but use more resources

    Users can override with: --max-samples N

    Returns:
        Default concurrency limit (8) for balanced resource usage
    """
    return DEFAULT_SABER_CONCURRENCY
