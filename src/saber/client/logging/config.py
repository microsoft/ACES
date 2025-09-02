"""
Client-Side Container Logging Configuration

Configuration classes and settings for the client-side container logging system.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class ClientLoggingConfig:
    """Configuration for client-side container logging."""

    # Base directory for storing logs
    base_log_dir: Union[Path, str] = Path("./client/logs")

    # Enable/disable container logging
    enable_container_logging: bool = True

    # Logging level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"

    # Maximum log file size in MB before rotation
    max_log_size_mb: int = 50

    # Number of days to retain logs
    retention_days: int = 30

    # Compress old logs to save space
    compress_old_logs: bool = True

    # Buffer size for log streaming (bytes)
    log_buffer_size: int = 8192

    # Timeout for log collection operations (seconds)
    log_collection_timeout: int = 30

    # Include container metadata in logs
    include_metadata: bool = True

    # Log format (json or text)
    log_format: str = "json"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Ensure base_log_dir is a Path object
        if isinstance(self.base_log_dir, str):
            self.base_log_dir = Path(self.base_log_dir)

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}")

        # Validate numeric values
        if self.max_log_size_mb <= 0:
            raise ValueError("max_log_size_mb must be positive")
        if self.retention_days <= 0:
            raise ValueError("retention_days must be positive")
        if self.log_buffer_size <= 0:
            raise ValueError("log_buffer_size must be positive")
        if self.log_collection_timeout <= 0:
            raise ValueError("log_collection_timeout must be positive")

        # Validate log format
        valid_formats = ["json", "text"]
        if self.log_format.lower() not in valid_formats:
            raise ValueError(f"Invalid log_format: {self.log_format}. Must be one of {valid_formats}")


@dataclass
class SessionLoggingContext:
    """Context information for a logging session."""

    session_id: str
    client_id: Optional[str] = None
    task_id: Optional[str] = None
    episode_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate session context."""
        if not self.session_id:
            raise ValueError("session_id is required")
