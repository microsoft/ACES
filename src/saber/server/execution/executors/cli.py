"""
Docker-based Command Line Interface executor for executing validated shell commands.

This module provides a secure CLI executor that accepts command strings over the MCP protocol
and executes them in Docker containers. Security validation is performed at the executor level
before execution.
"""

import logging
import shlex
from typing import Any, Dict, List, Optional

from ..base import CommandResult, Parameter, ParameterType, ValidationResult
from ..exceptions import SandboxExecutionError
from ..sandbox.sandbox_manager import SandboxManager
from ..utils.security_validator import SecurityValidator
from .docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


class CLIExecutor(DockerExecutor):
    """
    Docker-based CLI executor for secure command execution in containers.

    This executor provides a unified interface for executing shell commands
    in isolated Docker containers with comprehensive security validation.

    Execution Features:
    - Docker container isolation
    - Timeout enforcement
    - Shell/direct execution modes
    - Working directory specification
    - Session-based container management
    """

    _security_command_metadata = {
        "domain": "general",
        "name": "docker_cli",
        "description": "Execute validated shell commands in Docker containers",
        "author": "SABER Team",
        "security_level": "high",
        "requires_validation": True,
    }

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        cli_config: Optional[Dict[str, Any]] = None,
        allowed_commands: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Docker CLI executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            cli_config: Optional CLI configuration dictionary
            allowed_commands: Optional list of allowed commands for security validation
            **kwargs: Additional arguments passed to parent

        Raises:
            SandboxExecutionError: If sandbox_manager is None or invalid
        """
        super().__init__(sandbox_manager=sandbox_manager, docker_config=cli_config, **kwargs)

        cli_config = cli_config or {}

        # Initialize security validator for this executor
        self._security_validator = SecurityValidator(allowed_commands=allowed_commands)

        # Add parameter for the command string
        self.add_parameter(
            Parameter(
                name="command",
                type=ParameterType.STRING,
                description="Command string to execute in Docker container (will be validated for security)",
                required=True,
            )
        )

        # Add parameter for shell interpretation mode
        # Use default from CLI config if provided
        default_shell_mode = cli_config.get("default_shell_mode", False)
        self.add_parameter(
            Parameter(
                name="shell",
                type=ParameterType.BOOLEAN,
                description="Whether to execute command through shell (enables pipes, redirections, etc.)",
                required=False,
                default=default_shell_mode,
            )
        )

    def build_command(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """
        Build the command arguments from the provided command string.

        This method parses the command string and prepares it for execution.

        Args:
            parameters: Tool parameters including the command string
            context: Execution context

        Returns:
            List of command arguments ready for Docker execution

        Raises:
            ValueError: If command string is invalid or empty
        """
        command_str = parameters["command"].strip()

        if not command_str:
            raise ValueError("Command string cannot be empty")

        # Parse the command string into arguments
        try:
            if parameters.get("shell", False):
                # If shell mode is requested, we'll execute via shell
                # This is more dangerous but sometimes necessary for complex commands
                return ["/bin/sh", "-c", command_str]
            else:
                # Parse command into individual arguments (safer)
                # This prevents shell injection but limits shell features
                parsed_args = shlex.split(command_str)
                if not parsed_args:
                    raise ValueError("Parsed command resulted in empty argument list")
                return parsed_args

        except ValueError as e:
            raise ValueError(f"Failed to parse command string: {e}")

    def parse_output(self, stdout: str, stderr: str, return_code: int) -> CommandResult:
        """
        Parse command output into a structured result.

        Args:
            stdout: Standard output from the command
            stderr: Standard error from the command
            return_code: Process exit code

        Returns:
            CommandResult with structured output data
        """
        # Determine if command was successful
        success = return_code == 0

        # Create result data structure
        result_data = {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "success": success,
            "output": stdout if success else stderr,  # Primary output
        }

        # Add metadata about execution
        metadata = {
            "command_type": "docker_cli",
            "execution_environment": "docker_container",
            "exit_code": return_code,
            "has_stdout": bool(stdout.strip()),
            "has_stderr": bool(stderr.strip()),
            "output_length": len(stdout) + len(stderr),
        }

        if success:
            return CommandResult.success_result(data=result_data, metadata=metadata)
        else:
            # For failed commands, include both stdout and stderr in error message
            error_msg = f"Command failed with exit code {return_code}"
            if stderr.strip():
                error_msg += f": {stderr.strip()}"
            elif stdout.strip():
                error_msg += f". Output: {stdout.strip()}"

            return CommandResult.error_result(error=error_msg, metadata={**metadata, "raw_data": result_data})

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters including security validation for CLI commands.

        Args:
            parameters: Tool parameters to validate

        Returns:
            ValidationResult with validation details
        """
        # First do basic parameter validation from parent
        basic_validation = super().validate_parameters(parameters)
        if not basic_validation.valid:
            return basic_validation

        # Extract command for security validation
        command = parameters.get("command", "")
        if not command or not isinstance(command, str):
            return ValidationResult.failure(["Command parameter is required and must be a string"])

        # Perform security validation on the command
        try:
            command_args = shlex.split(command.strip())
            if not command_args:
                return ValidationResult.failure(["Command cannot be empty"])

            security_validation = self._security_validator.validate_full_command(command_args)
            if not security_validation.valid:
                return ValidationResult.failure(
                    [f"Command security validation failed: {', '.join(security_validation.errors)}"]
                )

            # Add any security warnings to the validation result
            if security_validation.warnings:
                for warning in security_validation.warnings:
                    basic_validation.add_warning(warning)

            return basic_validation

        except ValueError as e:
            return ValidationResult.failure([f"Failed to parse command: {str(e)}"])

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute the command-line tool in Docker container.

        Args:
            parameters: Tool parameters
            context: Execution context including session_id

        Returns:
            CommandResult with execution results

        Note:
            Security validation is performed at this executor level before execution.
            All execution happens in Docker containers.
        """
        try:
            # Validate parameters including security validation
            validation_result = self.validate_parameters(parameters)
            if not validation_result.valid:
                return CommandResult.error_result(
                    error=f"Parameter validation failed: {', '.join(validation_result.errors)}"
                )

            # Log any security warnings
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"CLI security warning: {warning}")

            # Extract session ID from context
            session_id = context.get("session_id")
            if not session_id:
                raise SandboxExecutionError("session_id required in context for Docker execution")

            # Get Docker environment for session (inherited from DockerExecutor)
            environment = self.get_session_environment(session_id)

            # Build command
            command_args = self.build_command(parameters, context)

            # Execute command in Docker container
            result = environment.execute_command(command=command_args)

            # Parse output
            tool_result = self.parse_output(result.stdout, result.stderr, result.exit_code)

            # Add execution metadata
            container = environment.get_execution_container()
            container_id = container.id[:12] if container else "unknown"

            tool_result.metadata.update(
                {
                    "container_id": container_id,
                    "session_id": session_id,
                    "execution_time": result.execution_time,
                }
            )

            return tool_result

        except Exception as e:
            logger.error(f"Docker command execution error: {e}")
            return CommandResult.error_result(f"Docker command execution failed: {str(e)}")

    def get_security_info(self) -> Dict[str, Any]:
        """
        Get information about security settings including Docker info.

        Returns:
            Dictionary with security configuration
        """
        # Get Docker-specific information from parent class
        return self.get_docker_info()
