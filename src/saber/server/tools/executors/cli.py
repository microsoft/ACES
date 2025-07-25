"""
Command Line Interface tool for executing validated shell commands.

This module provides a secure CLI tool that accepts command strings over the MCP protocol
and executes them in a controlled environment. Security validation is performed at the
ExecutionManager level before execution reaches this class.
"""

import asyncio
import logging
import shlex
from typing import Any, Dict, List, Optional

from ..base import Parameter, ParameterType, ToolResult
from ..utils.security_constants import DEFAULT_SANDBOX_CWD, DEFAULT_SECURITY_LIMITS, RESTRICTED_ENVIRONMENT
from .base_executors import ToolExecutor

logger = logging.getLogger(__name__)


class CLIExecutor(ToolExecutor):
    """
    Consolidated CLI tool for flexible command execution.

    This tool provides a unified interface for executing shell commands
    with comprehensive security validation and configurable parameters.

    Security Features (handled at ExecutionManager level):
    - Command validation before execution
    - Allowed commands checking
    - Shell injection prevention
    - Argument sanitization

    Execution Features:
    - Timeout enforcement
    - Shell/direct execution modes
    - Working directory specification
    - Environment variable control
    """

    _security_tool_metadata = {
        "domain": "general",
        "name": "cli",
        "description": "Execute validated shell commands in a secure environment",
        "author": "SABER Team",
        "security_level": "high",
        "requires_validation": True,
    }

    def __init__(self, cli_config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._command = "sh"  # Base command for CLI execution

        cli_config = cli_config or {}

        # Add parameter for the command string
        self.add_parameter(
            Parameter(
                name="command",
                type=ParameterType.STRING,
                description="Command string to execute (will be validated for security)",
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
            List of command arguments ready for subprocess execution

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

    def parse_output(self, stdout: str, stderr: str, return_code: int) -> ToolResult:
        """
        Parse command output into a structured result.

        Args:
            stdout: Standard output from the command
            stderr: Standard error from the command
            return_code: Process exit code

        Returns:
            ToolResult with structured output data
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
            "command_type": "cli",
            "exit_code": return_code,
            "has_stdout": bool(stdout.strip()),
            "has_stderr": bool(stderr.strip()),
            "output_length": len(stdout) + len(stderr),
        }

        if success:
            return ToolResult.success_result(data=result_data, metadata=metadata)
        else:
            # For failed commands, include both stdout and stderr in error message
            error_msg = f"Command failed with exit code {return_code}"
            if stderr.strip():
                error_msg += f": {stderr.strip()}"
            elif stdout.strip():
                error_msg += f". Output: {stdout.strip()}"

            return ToolResult.error_result(error=error_msg, metadata={**metadata, "raw_data": result_data})

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Execute the command-line tool.

        Args:
            parameters: Tool parameters
            context: Execution context

        Returns:
            ToolResult with execution results

        Note:
            Security validation is now performed at the ExecutionManager level
            before this method is called.
        """
        try:
            # Build command
            command_args = self.build_command(parameters, context)

            # Additional runtime safety measures
            env = RESTRICTED_ENVIRONMENT.copy()

            # Execute command with restrictions
            process = await asyncio.create_subprocess_exec(
                *command_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=DEFAULT_SANDBOX_CWD,
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
            result = self.parse_output(
                stdout_str, stderr_str, process.returncode if process.returncode is not None else -1
            )

            # Add execution metadata
            result.metadata.update(
                {
                    "command": command_args[0],
                    "return_code": process.returncode if process.returncode is not None else -1,
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
        return {
            "base_command": self._command,
            "timeout": self.get_timeout(),
        }
