"""
SABER Standardized Logging System

This module provides standardized logging utilities with emoji-based categorization
for improved readability and module identification in SABER logs.
"""

import logging
from enum import Enum
from typing import Any, Optional


class LogCategory(Enum):
    """
    Standardized logging categories with associated emojis for SABER modules.
    Each category represents a specific functional area of the system.
    """

    # Server Core Components
    SESSION_MANAGER = ("🏗️", "SessionManager - Central orchestration and server lifecycle")
    REST_API = ("🌐", "SessionRestAPI - HTTP endpoints and REST protocol")
    MCP_API = ("🔌", "SessionMCPAPI - MCP protocol and tool execution")
    EVALUATION = ("📊", "EvaluationManager - Performance tracking and metrics")

    # Execution Framework
    EXECUTION = ("⚙️", "ExecutionManager - Command execution coordination")
    DOCKER = ("🐳", "Docker Operations - Container management and sandbox")
    SECURITY = ("🔒", "Security & Validation - Security checks and filtering")
    TASK_EXEC = ("🏃", "Task Execution - Individual command/task execution")

    # Task & Episode Management
    TASK_MANAGER = ("📋", "TaskManager - Task definitions and configuration")
    EPISODE = ("🎬", "EpisodeManager - Episode lifecycle and RL workflows")
    POLICY = ("📝", "PolicyManager - Policy documents and domain rules")

    # Client Side Components
    AGENT = ("🤖", "Agent Operations - Agent execution and reasoning")
    HARNESS = ("🚀", "TestHarness - Test orchestration and harness ops")
    COMMUNICATION = ("💬", "Communication - Server-client protocols")
    LLM = ("🧠", "LLM Operations - Language model interactions")

    # System Operations
    CONFIG = ("🔧", "Configuration - System setup and environment loading")
    CLEANUP = ("🧹", "Cleanup - Resource cleanup and session termination")
    WARNING = ("⚠️", "Warnings - Non-critical issues and health checks")
    ERROR = ("❌", "Errors - Failures and exceptions")

    # Specialized Operations
    NETWORK = ("🌐", "Network Operations - Network setup and connectivity")
    FILE_OPS = ("📁", "File Operations - File transfers and management")
    HEALTH = ("💚", "Health Checks - Service health monitoring")

    def __init__(self, emoji: str, description: str) -> None:
        self.emoji = emoji
        self.description = description


class SaberLogger:
    """
    Enhanced logger wrapper that provides standardized emoji-based logging
    for SABER system components.
    """

    def __init__(self, name: str, category: LogCategory) -> None:
        """
        Initialize SABER logger with module name and category.

        Args:
            name: Logger name (typically __name__)
            category: LogCategory enum value for this module
        """
        self.logger = logging.getLogger(name)
        self.category = category
        self.emoji = category.emoji

    def _format_message(self, message: str, session_id: Optional[str] = None) -> str:
        """
        Format log message with emoji and optional session ID.

        Args:
            message: Log message content
            session_id: Optional session identifier

        Returns:
            Formatted message string
        """
        if session_id:
            return f"{self.emoji} [{session_id}] {message}"
        return f"{self.emoji} {message}"

    def debug(self, message: str, session_id: Optional[str] = None) -> None:
        """Log debug message with emoji prefix."""
        self.logger.debug(self._format_message(message, session_id))

    def info(self, message: str, session_id: Optional[str] = None) -> None:
        """Log info message with emoji prefix."""
        self.logger.info(self._format_message(message, session_id))

    def warning(self, message: str, session_id: Optional[str] = None) -> None:
        """Log warning message with emoji prefix."""
        self.logger.warning(self._format_message(message, session_id))

    def error(self, message: str, session_id: Optional[str] = None) -> None:
        """Log error message with emoji prefix."""
        self.logger.error(self._format_message(message, session_id))

    def critical(self, message: str, session_id: Optional[str] = None) -> None:
        """Log critical message with emoji prefix."""
        self.logger.critical(self._format_message(message, session_id))


def get_logger(name: str, category: LogCategory) -> SaberLogger:
    """
    Factory function to create standardized SABER loggers.

    Args:
        name: Logger name (typically __name__)
        category: LogCategory enum value

    Returns:
        SaberLogger instance
    """
    return SaberLogger(name, category)


# Convenience functions for common logging patterns
def log_session_start(logger: SaberLogger, session_id: str, details: str = "") -> None:
    """Log session start with standardized format."""
    msg = f"Session started{': ' + details if details else ''}"
    logger.info(msg, session_id)


def log_session_end(logger: SaberLogger, session_id: str, reason: str = "") -> None:
    """Log session end with standardized format."""
    msg = f"Session ended{': ' + reason if reason else ''}"
    logger.info(msg, session_id)


def log_operation_start(logger: SaberLogger, operation: str, session_id: Optional[str] = None, **kwargs: Any) -> None:
    """Log operation start with optional parameters."""
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    msg = f"{operation} started{': ' + params if params else ''}"
    logger.info(msg, session_id)


def log_operation_success(logger: SaberLogger, operation: str, session_id: Optional[str] = None, **kwargs: Any) -> None:
    """Log successful operation completion."""
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    msg = f"{operation} completed successfully{': ' + params if params else ''}"
    logger.info(msg, session_id)


def log_operation_failure(logger: SaberLogger, operation: str, error: str, session_id: Optional[str] = None) -> None:
    """Log operation failure with error details."""
    msg = f"{operation} failed: {error}"
    logger.error(msg, session_id)


def log_timeout(
    logger: SaberLogger, operation: str, timeout: int, session_id: Optional[str] = None, **kwargs: Any
) -> None:
    """Log operation timeout with standardized format."""
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    msg = f"{operation} timed out after {timeout}s{': ' + params if params else ''}"
    logger.warning(msg, session_id)


# Module-specific logger factory functions
def get_session_manager_logger(name: str) -> SaberLogger:
    """Get logger for SessionManager components."""
    return get_logger(name, LogCategory.SESSION_MANAGER)


def get_docker_logger(name: str) -> SaberLogger:
    """Get logger for Docker operations."""
    return get_logger(name, LogCategory.DOCKER)


def get_execution_logger(name: str) -> SaberLogger:
    """Get logger for execution components."""
    return get_logger(name, LogCategory.EXECUTION)


def get_api_logger(name: str, is_mcp: bool = False) -> SaberLogger:
    """Get logger for API components."""
    category = LogCategory.MCP_API if is_mcp else LogCategory.REST_API
    return get_logger(name, category)


def get_agent_logger(name: str) -> SaberLogger:
    """Get logger for agent components."""
    return get_logger(name, LogCategory.AGENT)


def get_cleanup_logger(name: str) -> SaberLogger:
    """Get logger for cleanup operations."""
    return get_logger(name, LogCategory.CLEANUP)
