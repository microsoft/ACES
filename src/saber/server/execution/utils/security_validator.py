"""
Security validation utilities for command execution.

This module provides utilities for validating command security,
including pattern detection, argument validation, and path safety checks.
"""

import logging
import re
import shlex
from pathlib import Path
from typing import List, Optional

from ..base import ValidationResult
from .security_constants import (
    ALLOWED_CONTROL_CHARS,
    BLOCKED_COMMANDS,
    DANGEROUS_PATTERNS,
    DANGEROUS_SEMICOLON_PATTERNS,
    DEFAULT_SECURITY_SETTINGS,
    MIN_CONTROL_CHAR_CODE,
    NULL_BYTE,
    SENSITIVE_DIRECTORIES,
    SHELL_METACHARACTERS_PATTERN,
    SUSPICIOUS_EXTENSIONS,
)

logger = logging.getLogger(__name__)


class SecurityValidator:
    """
    Utility class for validating command security.

    Provides comprehensive security validation for command-line execution
    including pattern detection, argument validation, and path safety checks.
    """

    def __init__(self, allowed_commands: Optional[List[str]] = None):
        """
        Initialize security validator.

        Args:
            allowed_commands: Commands that this specific tool is allowed to use
                             (typically passed by the tool itself, not from config)
        """
        self._allowed_commands = set(allowed_commands) if allowed_commands else set()

    def validate_base_command(self, command: str) -> None:
        """
        Validate the base command for security.

        Args:
            command: Base command to validate

        Raises:
            ValueError: If command is not allowed for security reasons
        """
        command_parts = shlex.split(command)
        if not command_parts:
            raise ValueError("Empty command not allowed")

        base_cmd = Path(command_parts[0]).name

        # Check against blocked commands first
        if base_cmd in BLOCKED_COMMANDS:
            # If command is blocked, check if it's explicitly allowed by whitelist
            if not self._allowed_commands or base_cmd not in self._allowed_commands:
                raise ValueError(f"Command '{base_cmd}' is not allowed for security reasons")

        # If command is not in blocked list, it's allowed regardless of whitelist

    def validate_command_string(self, command_str: str, allow_semicolons: bool = False) -> ValidationResult:
        """
        Validate a command string for malicious patterns.

        Args:
            command_str: Command string to validate
            allow_semicolons: Whether to allow semicolons (for command chaining)

        Returns:
            ValidationResult indicating if command is safe
        """
        result = ValidationResult.success()

        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command_str, re.IGNORECASE):
                result.add_error(f"Dangerous pattern detected: {pattern}")

        # Check for dangerous semicolon patterns only if semicolons are not explicitly allowed
        if not allow_semicolons:
            for pattern in DANGEROUS_SEMICOLON_PATTERNS:
                if re.search(pattern, command_str, re.IGNORECASE):
                    result.add_error(f"Dangerous semicolon pattern detected: {pattern}")

            # Also check for any semicolon if not in allow mode
            if ";" in command_str:
                result.add_error("Semicolons not allowed (use chain_commands=true for command chaining)")

        # Check for null bytes (can bypass filters)
        if NULL_BYTE in command_str:
            result.add_error("Null bytes not allowed in commands")

        # Check for excessive length (potential buffer overflow)
        if len(command_str) > DEFAULT_SECURITY_SETTINGS["max_command_length"]:
            result.add_error(
                f"Command string too long (max {DEFAULT_SECURITY_SETTINGS['max_command_length']} characters)"
            )

        # Check for unicode control characters
        control_chars = [c for c in command_str if ord(c) < MIN_CONTROL_CHAR_CODE and c not in ALLOWED_CONTROL_CHARS]
        if control_chars:
            result.add_error("Control characters not allowed in commands")

        return result

    def validate_arguments(self, args: List[str]) -> ValidationResult:
        """
        Validate command arguments for security issues.

        Args:
            args: List of command arguments

        Returns:
            ValidationResult indicating if arguments are safe
        """
        result = ValidationResult.success()

        for i, arg in enumerate(args):
            # Skip the base command
            if i == 0:
                continue

            # Check for path traversal
            if ".." in arg or arg.startswith("/"):
                if not self.is_safe_path(arg):
                    result.add_error(f"Unsafe path in argument: {arg}")

            # Check for shell metacharacters in arguments
            if re.search(SHELL_METACHARACTERS_PATTERN, arg):
                result.add_error(f"Shell metacharacters in argument: {arg}")

            # Check for suspicious file extensions
            if any(arg.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS):
                result.add_warning(f"Executable file extension in argument: {arg}")

        return result

    def is_safe_path(self, path_str: str) -> bool:
        """
        Check if a path is safe.

        Args:
            path_str: Path string to check

        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Block any path with path traversal attempts
            if ".." in path_str:
                return False

            path = Path(path_str).resolve()

            # Block access to sensitive directories
            for sensitive in SENSITIVE_DIRECTORIES:
                if str(path).startswith(sensitive):
                    return False

            # Block absolute paths to system directories
            if path.is_absolute():
                # Allow only certain safe absolute paths
                safe_prefixes = ["/tmp", "/var/tmp", "/home"]
                if not any(str(path).startswith(prefix) for prefix in safe_prefixes):
                    return False

            return True

        except (OSError, ValueError):
            return False

    def validate_full_command(self, command_args: List[str]) -> ValidationResult:
        """
        Perform complete security validation on command and arguments.

        Args:
            command_args: Complete command with arguments

        Returns:
            ValidationResult with all validation results combined
        """
        if not command_args:
            result = ValidationResult.success()
            result.add_error("Empty command arguments")
            return result

        # Build full command string for validation
        full_command = " ".join(shlex.quote(arg) for arg in command_args)

        # Validate command string
        command_validation = self.validate_command_string(full_command)

        # Validate arguments
        args_validation = self.validate_arguments(command_args)

        # Combine results
        combined_result = ValidationResult.success()
        combined_result.errors.extend(command_validation.errors)
        combined_result.errors.extend(args_validation.errors)
        combined_result.warnings.extend(command_validation.warnings)
        combined_result.warnings.extend(args_validation.warnings)

        # Update valid flag based on errors
        if combined_result.errors:
            combined_result.valid = False

        return combined_result
