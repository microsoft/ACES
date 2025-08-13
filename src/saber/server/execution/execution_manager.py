"""
ExecutionManager implementation for command execution.

This version provides access to CLI commands with comprehensive
security validation capabilities.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import yaml

from ..tasks.base import Action
from .base import CommandResult, ValidationResult
from .executors.docker_executor import DockerExecutor
from .executors.factory import ExecutorFactory
from .sandbox.environment_spec import EnvironmentSpec
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

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a configuration section by name.

        Args:
            section_name: Name of the configuration section

        Returns:
            Configuration section dictionary (empty dict if not found)
        """
        config = self._config.get(section_name, {})
        return dict(config) if config else {}

    def get_execution_timeout(self) -> float:
        """Get default execution timeout."""
        timeout = self.get_section("execution").get("timeout", 300.0)
        return float(timeout)

    def get_max_concurrent(self) -> int:
        """Get maximum concurrent executions."""
        max_concurrent = self.get_section("execution").get("max_concurrent", 10)
        return int(max_concurrent)

    def get_allowed_commands(self) -> List[str]:
        """Get allowed commands override list."""
        commands = self.get_section("security").get("allowed_commands", [])
        return list(commands) if commands else []

    def get_max_command_length(self) -> int:
        """Get maximum command length."""
        length = self.get_section("security").get("max_command_length", 10000)
        return int(length)

    def get_sandbox_config(self) -> Dict[str, Any]:
        """Get sandbox configuration."""
        return self.get_section("sandbox")


class ExecutionManager:
    """
    Execution manager supporting multiple executor types with factory pattern.

    Provides unified interface for executing different types of commands (CLI, Python, etc.)
    with security validation and Docker isolation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """
        Initialize the execution manager.

        Args:
            config: Optional configuration dictionary
            config_file: Optional path to YAML configuration file
        """
        self._configuration = ExecutionConfiguration(config, config_file)

        # Initialize security validator
        allowed_commands = self._configuration.get_allowed_commands()
        self._security_validator = SecurityValidator(
            allowed_commands=allowed_commands if allowed_commands else None,
        )

        # Initialize sandbox manager
        sandbox_config = self._configuration.get_sandbox_config()
        self._sandbox_manager = SandboxManager(sandbox_config)

        # Initialize executor factory with full configuration
        # Factory will extract relevant sections for each executor type
        self._executor_factory = ExecutorFactory(
            sandbox_manager=self._sandbox_manager, configuration=self._configuration
        )

        # Concurrency control
        max_concurrent = self._configuration.get_max_concurrent()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"ExecutionManager initialized with max_concurrent={max_concurrent}")
        logger.info(f"Available executor types: {self._executor_factory.get_available_executors()}")

    async def step(self, action: Action, context: Optional[Dict[str, Any]] = None) -> CommandResult:
        """
        Execute the action with the appropriate executor.

        Args:
            action: Action object containing command and parameters
            context: Optional execution context

        Returns:
            CommandResult with execution results
        """

        async with self._semaphore:
            try:
                # Determine executor type from action or command content
                executor_type = self._determine_executor_type(action)
                executor = self._executor_factory.get_executor(executor_type)

                # Extract parameters from action
                parameters = {"command": action.command}
                parameters.update(action.parameters)

                # Validate parameters first
                validation_result = executor.validate_parameters(parameters)
                if not validation_result.valid:
                    return CommandResult.error_result(
                        error=f"Parameter validation failed: {', '.join(validation_result.errors)}"
                    )

                # Execute using the appropriate executor
                return await executor.execute(parameters, context or {})

            except Exception as e:
                logger.error(f"Execution failed: {e}")
                return CommandResult.error_result(error=str(e))

    def _determine_executor_type(self, action: Action) -> str:
        """
        Determine the appropriate executor type for the given action.

        Args:
            action: Action to analyze

        Returns:
            Executor type string
        """
        # Check if action has explicit executor type (for testing or future extensibility)
        if hasattr(action, "executor_type") and action.executor_type:
            return str(action.executor_type)

        # Check action parameters for hints
        if "code" in action.parameters:
            return "python"

        # Use factory's command analysis
        command = action.command or ""
        return self._executor_factory._analyze_command(command)

    def get_executor(self, executor_type: str) -> DockerExecutor:
        """
        Get a specific executor instance.

        Args:
            executor_type: Type of executor to retrieve

        Returns:
            Executor instance
        """
        return self._executor_factory.get_executor(executor_type)

    def create_environment(self, session_id: str, environment_spec: EnvironmentSpec) -> None:
        """
        Create a sandbox environment for a session with the given specification.

        Args:
            session_id: Session identifier
            environment_spec: Environment specification for multi-container orchestration
        """
        try:
            self._sandbox_manager.create_session_environment(session_id, environment_spec)
            logger.info(f"Created sandbox environment for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to create environment for session {session_id}: {e}")
            raise

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up execution resources for a session.

        Args:
            session_id: Session identifier to clean up
        """
        try:
            self._sandbox_manager.cleanup_session(session_id)
            logger.info(f"Cleaned up execution resources for session {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up execution resources for session {session_id}: {e}")

    def cleanup_all_sessions(self) -> None:
        """Clean up all execution resources."""
        try:
            self._sandbox_manager.cleanup_all_sessions()
            self._executor_factory.cleanup_all_executors()
            logger.info("Cleaned up all execution resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

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
        Convert all available executors to MCP format.

        Returns:
            List containing MCP tool definitions for all executors
        """
        return self._executor_factory.get_all_mcp_tools()

    def list_commands(self) -> List[Dict[str, Any]]:
        """
        List all available commands from all executors.

        Returns:
            List of command definitions
        """
        commands = []

        for executor_type in self._executor_factory.get_available_executors():
            try:
                executor = self._executor_factory.get_executor(executor_type)
                metadata = getattr(executor, "_security_command_metadata", {})

                command_info = {
                    "executor_type": executor_type,
                    "name": metadata.get("name", f"{executor_type}_command"),
                    "description": metadata.get("description", f"{executor_type.title()} executor"),
                    "domain": metadata.get("domain", "general"),
                    "security_level": metadata.get("security_level", "medium"),
                    "parameters": list(executor.get_parameters().keys()),
                }

                commands.append(command_info)

            except Exception as e:
                logger.error(f"Failed to get info for executor {executor_type}: {e}")

        return commands

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
            "executor_info": self._executor_factory.get_executor_info(),
        }

    def load_configuration(self, config_path: str) -> None:
        """
        Load configuration from YAML file and update components.

        Args:
            config_path: Path to configuration file
        """
        self._configuration.load_configuration(config_path)

        # Update security validator
        allowed_commands = self._configuration.get_allowed_commands()
        self._security_validator = SecurityValidator(
            allowed_commands=allowed_commands if allowed_commands else None,
        )

        # Recreate sandbox manager with new configuration
        sandbox_config = self._configuration.get_sandbox_config()
        self._sandbox_manager = SandboxManager(sandbox_config)

        # Recreate executor factory with new configuration
        self._executor_factory = ExecutorFactory(
            sandbox_manager=self._sandbox_manager, configuration=self._configuration
        )

        # Update concurrency control
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
