"""
Base executor implementations for common tool patterns.

This module provides concrete base classes that implement common patterns
for security tool executors, reducing boilerplate code for tool developers.
"""

import asyncio
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Parameter, ToolExecutor, ToolResult, ValidationResult
from ..security_constants import DEFAULT_SANDBOX_CWD, DEFAULT_SECURITY_LIMITS, RESTRICTED_ENVIRONMENT
from ..utils.security_validator import SecurityValidator

logger = logging.getLogger(__name__)


class BaseToolExecutor(ToolExecutor):
    """
    Base implementation with common functionality for tool executors.
    """

    def __init__(self, timeout: Optional[float] = None):
        """
        Initialize base executor.

        Args:
            timeout: Execution timeout in seconds
        """
        self._timeout = timeout
        self._parameters: Dict[str, Parameter] = {}

    def get_timeout(self) -> Optional[float]:
        """Get the execution timeout."""
        return self._timeout

    def get_parameters(self) -> Dict[str, Parameter]:
        """Get the tool parameters."""
        return self._parameters.copy()

    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the tool."""
        self._parameters[parameter.name] = parameter

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters against the tool's parameter definitions.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult.success()

        # Check for required parameters
        for param_name, param_def in self._parameters.items():
            if param_def.required and param_name not in parameters:
                result.add_error(f"Required parameter '{param_name}' is missing")
                continue

            if param_name in parameters:
                is_valid, error_msg = param_def.validate_value(parameters[param_name])
                if not is_valid and error_msg:
                    result.add_error(error_msg)

        # Check for unknown parameters
        for param_name in parameters:
            if param_name not in self._parameters:
                result.add_warning(f"Unknown parameter '{param_name}' will be ignored")

        return result


class CommandLineToolExecutor(BaseToolExecutor):
    """
    Executor for command-line tools.

    Provides a framework for executing external command-line tools
    with parameter mapping and output parsing. Includes comprehensive
    security validation to prevent command injection attacks.
    """

    def __init__(
        self,
        command: str,
        timeout: Optional[float] = None,
        allowed_commands: Optional[List[str]] = None,
        sandbox_path: Optional[str] = None,
    ):
        """
        Initialize command-line executor.

        Args:
            command: Base command to execute
            timeout: Execution timeout in seconds
            allowed_commands: List of allowed base commands (whitelist)
            sandbox_path: Optional sandbox directory to restrict operations
        """
        super().__init__(timeout)
        self._command = command
        self._sandbox_path = Path(sandbox_path) if sandbox_path else None

        # Initialize security validator
        self._security_validator = SecurityValidator(allowed_commands=allowed_commands, sandbox_path=sandbox_path)

        # Validate the base command
        self._security_validator.validate_base_command(command)

    @abstractmethod
    def build_command(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """
        Build the command line from parameters.

        Args:
            parameters: Tool parameters
            context: Execution context

        Returns:
            List of command arguments

        Note:
            This method should return properly escaped arguments.
            The security validation will be performed after this method.
        """
        pass

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str, return_code: int) -> ToolResult:
        """
        Parse command output into a ToolResult.

        Args:
            stdout: Standard output
            stderr: Standard error
            return_code: Process return code

        Returns:
            ToolResult with parsed output
        """
        pass

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Execute the command-line tool with comprehensive security validation.

        Args:
            parameters: Tool parameters
            context: Execution context

        Returns:
            ToolResult with execution results
        """
        try:
            # Build command
            command_args = self.build_command(parameters, context)

            # Security validation using the validator utility
            validation_result = self._security_validator.validate_full_command(command_args)
            if not validation_result.valid:
                return ToolResult.error_result(
                    f"Command security validation failed: {', '.join(validation_result.errors)}"
                )

            # Log security warnings
            if validation_result.warnings:
                logger.warning(f"Command security warnings: {', '.join(validation_result.warnings)}")

            # Additional runtime safety measures
            env = RESTRICTED_ENVIRONMENT.copy()

            # Execute command with restrictions
            process = await asyncio.create_subprocess_exec(
                *command_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self._sandbox_path) if self._sandbox_path else DEFAULT_SANDBOX_CWD,
                preexec_fn=self._setup_security_restrictions,
                limit=int(DEFAULT_SECURITY_LIMITS["subprocess_output_limit"]),
            )

            # Wait with timeout
            timeout = self.get_timeout() or DEFAULT_SECURITY_LIMITS["timeout"]
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.terminate()
                await asyncio.sleep(DEFAULT_SECURITY_LIMITS["process_terminate_wait"])
                if process.returncode is None:
                    process.kill()
                return ToolResult.error_result(f"Command timed out after {timeout} seconds")

            # Parse output with size limits
            max_output = int(DEFAULT_SECURITY_LIMITS["max_output_size"])
            stdout_str = stdout.decode("utf-8", errors="replace")[:max_output]
            stderr_str = stderr.decode("utf-8", errors="replace")[:max_output]

            # Parse output
            result = self.parse_output(stdout_str, stderr_str, process.returncode or -1)

            # Add security metadata
            result.metadata.update(
                {
                    "security_validated": True,
                    "command": command_args[0],
                    "sandbox_path": str(self._sandbox_path) if self._sandbox_path else None,
                    "return_code": process.returncode or -1,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return ToolResult.error_result(f"Command execution failed: {str(e)}")

    def _setup_security_restrictions(self) -> None:
        """Set up additional security restrictions for the subprocess."""
        import os
        import resource

        try:
            # Set resource limits using security constants
            limits = DEFAULT_SECURITY_LIMITS
            resource.setrlimit(resource.RLIMIT_CPU, (int(limits["max_cpu_time"]), int(limits["max_cpu_time"])))
            resource.setrlimit(resource.RLIMIT_AS, (int(limits["max_memory"]), int(limits["max_memory"])))
            resource.setrlimit(resource.RLIMIT_FSIZE, (int(limits["max_file_size"]), int(limits["max_file_size"])))
            resource.setrlimit(
                resource.RLIMIT_NOFILE, (int(limits["max_file_descriptors"]), int(limits["max_file_descriptors"]))
            )
            resource.setrlimit(resource.RLIMIT_NPROC, (int(limits["max_processes"]), int(limits["max_processes"])))

            # Drop to nobody user if running as root (requires appropriate setup)
            if os.getuid() == 0:
                import pwd

                nobody = pwd.getpwnam("nobody")
                os.setgid(nobody.pw_gid)
                os.setuid(nobody.pw_uid)

        except (ImportError, OSError, KeyError) as e:
            # Log but don't fail - restrictions are best effort
            logger.warning(f"Could not set all security restrictions: {e}")

    def get_security_info(self) -> Dict[str, Any]:
        """
        Get information about security settings.

        Returns:
            Dictionary with security configuration
        """
        security_info = self._security_validator.get_security_info()
        security_info.update({"base_command": self._command, "timeout": self.get_timeout()})
        return security_info
