"""
ExecutionManager implementation for command execution.

This version provides access to CLI commands with comprehensive
security validation capabilities.
"""

import asyncio
import logging
import shlex
from typing import Any, Dict, List, Optional

import yaml

from .base import CommandResult, ValidationResult
from .exceptions import ExecutionManagerError
from .executors.cli import DockerCLIExecutor
from .sandbox.sandbox_manager import SandboxManager
from .utils.security_validator import SecurityValidator

logger = logging.getLogger(__name__)


class ExecutionConfiguration:
    """
    Configuration management for execution manager.

    Handles YAML configuration files and provides structured access to
    execution, security, and sandbox settings. Renamed from CLIConfiguration
    to reflect broader scope beyond just CLI tools.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """
        Initialize the CLIConfiguration.

        Args:
            config: Optional configuration dictionary
            config_file: Optional path to YAML configuration file
        """
        self._config: Dict[str, Any] = {}

        if config_file:
            try:
                self.load_configuration(config_file)
            except Exception as e:
                logger.error(f"Failed to load configuration file '{config_file}': {e}")
                self._config = {}
        else:
            self._config = config or {}

    def load_configuration(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise

    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration."""
        config = self._config.get("execution", {})
        return dict(config) if config else {}

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        config = self._config.get("security", {})
        return dict(config) if config else {}

    def get_cli_config(self) -> Dict[str, Any]:
        """Get CLI-specific configuration."""
        config = self._config.get("cli", {})
        return dict(config) if config else {}

    def get_execution_timeout(self) -> float:
        """Get default execution timeout."""
        timeout = self.get_execution_config().get("timeout", 300.0)
        return float(timeout)

    def get_max_concurrent(self) -> int:
        """Get maximum concurrent executions."""
        max_concurrent = self.get_execution_config().get("max_concurrent", 10)
        return int(max_concurrent)

    def get_allowed_commands(self) -> List[str]:
        """Get allowed commands override list."""
        commands = self.get_security_config().get("allowed_commands", [])
        return list(commands) if commands else []

    def get_max_command_length(self) -> int:
        """Get maximum command length."""
        length = self.get_security_config().get("max_command_length", 10000)
        return int(length)

    def get_sandbox_config(self) -> Dict[str, Any]:
        """Get sandbox configuration."""
        config = self._config.get("sandbox", {})
        return dict(config) if config else {}


class ExecutionManager:
    """
    Execution manager containing only a CLI executor with security validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """
        Initialize the execution manager.

        Args:
            config: Optional configuration dictionary
            config_file: Optional path to YAML configuration file
        """
        self._configuration = ExecutionConfiguration(config, config_file)

        allowed_commands = self._configuration.get_allowed_commands()
        self._security_validator = SecurityValidator(
            allowed_commands=allowed_commands if allowed_commands else None,
        )

        # Initialize sandbox manager (required for Docker execution)
        sandbox_config = self._configuration.get_sandbox_config()
        if not sandbox_config.get("enabled", False):
            raise ExecutionManagerError("Sandbox execution is required but not enabled in configuration")

        self._sandbox_manager = SandboxManager(sandbox_config)

        cli_config = self._configuration.get_cli_config()
        self._cli_tool = DockerCLIExecutor(
            sandbox_manager=self._sandbox_manager,
            cli_config=cli_config,
            timeout=self._configuration.get_execution_timeout(),
        )

        # Concurrency control
        max_concurrent = self._configuration.get_max_concurrent()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"ExecutionManager initialized with max_concurrent={max_concurrent}")

    async def step(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> CommandResult:
        """
        Execute the CLI command step with the given parameters.

        Args:
            parameters: CLI command parameters (must include 'command')
            context: Optional execution context

        Returns:
            CommandResult with execution results
        """

        async with self._semaphore:
            try:
                # Validate parameters first
                validation_result = self._cli_tool.validate_parameters(parameters)
                if not validation_result.valid:
                    return CommandResult.error_result(
                        error=f"Parameter validation failed: {', '.join(validation_result.errors)}"
                    )

                # Security validation before execution
                # Build the command that would be executed to validate it
                command = parameters.get("command", "")
                if command:
                    # Parse command string to get command arguments for full validation
                    try:
                        command_args = shlex.split(command)
                        if command_args:
                            # Validate the full command using security validator
                            security_validation = self._security_validator.validate_full_command(command_args)
                            if not security_validation.valid:
                                return CommandResult.error_result(
                                    error=f"Command security validation failed: {', '.join(security_validation.errors)}"
                                )

                            # Log security warnings if any
                            if security_validation.warnings:
                                logger.warning(f"Command security warnings: {', '.join(security_validation.warnings)}")
                    except ValueError as e:
                        return CommandResult.error_result(error=f"Failed to parse command: {str(e)}")

                # Execute the CLI command
                return await self._cli_tool.execute(parameters, context or {})

            except Exception as e:
                logger.error(f"CLI execution failed: {e}")
                return CommandResult.error_result(error=str(e))

    def validate_command(self, command: str) -> ValidationResult:
        """
        Validate a command string using the security validator.

        Args:
            command: Command string to validate

        Returns:
            ValidationResult with validation details
        """
        return self._security_validator.validate_command_string(command)

    def get_security_info(self) -> Dict[str, Any]:
        """
        Get security configuration information.

        Returns:
            Dictionary with security settings
        """
        return self._security_validator.get_security_info()

    def to_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Convert the CLI command to MCP format.

        Returns:
            List containing single MCP tool definition
        """
        # Get command metadata and schema from the CLI executor itself
        metadata = self._cli_tool._security_command_metadata

        return [
            {
                "name": metadata["name"],
                "description": metadata["description"],
                "inputSchema": self._cli_tool.to_mcp_schema(),
            }
        ]

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "max_concurrent": self._configuration.get_max_concurrent(),
            "current_available": self._semaphore._value,
            "timeout": self._configuration.get_execution_timeout(),
            "security_config": self.get_security_info(),
        }

    def load_configuration(self, config_path: str) -> None:
        """
        Load configuration from YAML file and update components.

        Args:
            config_path: Path to configuration file
        """
        self._configuration.load_configuration(config_path)

        allowed_commands = self._configuration.get_allowed_commands()
        self._security_validator = SecurityValidator(
            allowed_commands=allowed_commands if allowed_commands else None,
        )

        # Recreate sandbox manager with new configuration
        sandbox_config = self._configuration.get_sandbox_config()
        if not sandbox_config.get("enabled", False):
            raise ExecutionManagerError("Sandbox execution is required but not enabled in configuration")

        self._sandbox_manager = SandboxManager(sandbox_config)

        cli_config = self._configuration.get_cli_config()
        self._cli_tool = DockerCLIExecutor(
            sandbox_manager=self._sandbox_manager,
            cli_config=cli_config,
            timeout=self._configuration.get_execution_timeout(),
        )

        max_concurrent = self._configuration.get_max_concurrent()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info("Configuration updated successfully")

    def get_configuration(self) -> ExecutionConfiguration:
        """
        Get the configuration manager.

        Returns:
            ExecutionConfiguration instance
        """
        return self._configuration
