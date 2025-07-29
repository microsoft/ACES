"""
Docker-based Command Line Interface executor for executing validated shell commands.

This module provides a secure CLI executor that accepts command strings over the MCP protocol
and executes them in Docker containers. Security validation is performed at the
ExecutionManager level before execution reaches this class.
"""

import logging
import shlex
from typing import Any, Dict, List, Optional

from ..base import CommandResult, Parameter, ParameterType
from ..exceptions import SandboxExecutionError
from ..sandbox.sandbox_manager import SandboxManager
from .base_executors import CommandExecutor

logger = logging.getLogger(__name__)


class DockerCLIExecutor(CommandExecutor):
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
        self, sandbox_manager: SandboxManager, cli_config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize Docker CLI executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            cli_config: Optional CLI configuration dictionary
            **kwargs: Additional arguments passed to parent

        Raises:
            SandboxExecutionError: If sandbox_manager is None or invalid
        """
        super().__init__(**kwargs)

        # Require sandbox manager - no fallback to local execution
        if sandbox_manager is None:
            raise SandboxExecutionError("sandbox_manager is required for Docker execution")

        self._sandbox_manager = sandbox_manager
        cli_config = cli_config or {}

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

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute the command-line tool in Docker container.

        Args:
            parameters: Tool parameters
            context: Execution context including session_id

        Returns:
            CommandResult with execution results

        Note:
            Security validation is performed at the ExecutionManager level
            before this method is called. All execution happens in Docker containers.
        """
        try:
            # Extract session ID from context
            session_id = context.get("session_id")
            if not session_id:
                raise SandboxExecutionError("session_id required in context for Docker execution")

            # Get or create Docker environment for session
            environment = self._sandbox_manager.get_session_environment(session_id)
            if not environment:
                environment = self._sandbox_manager.create_session_environment(session_id)

            # Build command
            command_args = self.build_command(parameters, context)

            # Execute command in Docker container
            result = await environment.execute_command(command=command_args, working_dir="/workspace")

            # Parse output
            tool_result = self.parse_output(result.stdout, result.stderr, result.exit_code)

            # Add execution metadata
            tool_result.metadata.update(
                {
                    "container_id": environment.get_container_id()[:12],
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
        Get information about security settings including sandbox info.

        Returns:
            Dictionary with security configuration
        """
        base_info: Dict[str, Any] = {
            "execution_environment": "docker_container",
            "sandbox_manager": "enabled",
            "timeout": self.get_timeout(),
        }

        try:
            sandbox_config = self._sandbox_manager.get_sandbox_config()
            sandbox_info: Dict[str, Any] = {}

            if sandbox_config.get("image"):
                sandbox_info["image"] = str(sandbox_config["image"])
            if sandbox_config.get("network_mode"):
                sandbox_info["network_mode"] = str(sandbox_config["network_mode"])
            if sandbox_config.get("read_only_root") is not None:
                sandbox_info["read_only_root"] = bool(sandbox_config["read_only_root"])
            if sandbox_config.get("user"):
                sandbox_info["user"] = str(sandbox_config["user"])
            if sandbox_config.get("resource_limits"):
                sandbox_info["resource_limits"] = dict(sandbox_config["resource_limits"])

            base_info["sandbox_config"] = sandbox_info
        except Exception as e:
            logger.warning(f"Could not retrieve sandbox config: {e}")

        return base_info
