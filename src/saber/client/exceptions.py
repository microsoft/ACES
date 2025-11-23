"""
SABER Evaluation Exceptions

Custom exception classes for SABER evaluation operations with detailed error context.
Following fail-fast philosophy with actionable error messages.
"""

from typing import Any, Dict, Optional


class SABEREvaluationError(Exception):
    """Base exception for SABER evaluation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, suggestion: Optional[str] = None):
        """
        Initialize SABER evaluation error with context.

        Args:
            message: Error message
            details: Additional error context
            suggestion: Actionable suggestion for fixing the error
        """
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion

        # Build comprehensive error message
        full_message = message
        if self.suggestion:
            full_message += f"\nSuggestion: {self.suggestion}"
        if self.details:
            full_message += f"\nDetails: {self.details}"

        super().__init__(full_message)


class ConfigurationValidationError(SABEREvaluationError):
    """Configuration validation failed during upfront validation."""

    pass


class ServerConnectivityError(SABEREvaluationError):
    """SABER server is unreachable or returning errors."""

    pass


class AgentInitializationError(SABEREvaluationError):
    """Agent initialization failed."""

    pass


class AgentConfigurationError(SABEREvaluationError):
    """Agent configuration validation failed."""

    pass


class DatasetCreationError(SABEREvaluationError):
    """Dataset creation or task discovery failed."""

    pass


class EvaluationExecutionError(SABEREvaluationError):
    """Error during eval_async execution."""

    pass


class ResourceCleanupError(SABEREvaluationError):
    """Error during resource cleanup - typically non-fatal."""

    pass


# Evaluation Retrieval Exceptions - Added for Phase 2 Client Integration


class EvaluationRetrievalError(SABEREvaluationError):
    """Base exception for evaluation retrieval errors."""

    pass


class EvaluationNotFoundError(EvaluationRetrievalError):
    """Evaluation result not found (404)."""

    pass


class SessionEvaluationError(EvaluationRetrievalError):
    """Session-level evaluation access error (500)."""

    pass


class InvalidEvaluationRequestError(EvaluationRetrievalError):
    """Invalid evaluation request parameters (422)."""

    pass


# Role-Based Configuration Exceptions


class RoleConfigurationError(SABEREvaluationError):
    """Base exception for role-based configuration errors."""

    pass


class RoleNotFoundError(RoleConfigurationError):
    """Requested role not found in configuration."""

    pass


class RoleConfigLoadError(RoleConfigurationError):
    """Failed to load role configuration from file/string."""

    pass


class InvalidRoleOverrideError(RoleConfigurationError):
    """Invalid role override in CLI parameters."""

    pass
