"""
RegistryConfiguration handles configuration management for the tool registry.

This component manages loading, validation, and application of configuration
settings from YAML files or dictionaries.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class RegistryConfiguration:
    """
    Manages configuration loading and validation for the tool registry.

    Handles YAML configuration files and provides structured access to
    tool, execution, security, and domain-specific settings.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """
        Initialize the RegistryConfiguration.

        Args:
            config: Optional configuration dictionary
            config_file: Optional path to YAML configuration file
        """
        self._config: Dict[str, Any] = {}

        # Load configuration from file if provided, otherwise use provided config or empty dict
        if config_file:
            try:
                self._config = self._load_config_from_file(config_file)
            except Exception as e:
                logger.error(f"Failed to load configuration file '{config_file}': {e}")
                self._config = {}
        else:
            self._config = config or {}

        logger.info("RegistryConfiguration initialized")

    def load_configuration(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        try:
            config = self._load_config_from_file(config_path)
            if config:
                self._config.update(config)
            else:
                logger.warning(f"Configuration file '{config_path}' is empty or invalid")

            logger.info(f"Loaded configuration from '{config_path}'")

        except Exception as e:
            logger.error(f"Failed to load configuration from '{config_path}': {e}")

    def get_tools_config(self) -> Dict[str, Any]:
        """
        Get tools configuration section.

        Returns:
            Dictionary with tools configuration including domains, specific_tools, and execution settings
        """
        tools_config = self._config.get("tools", {})
        assert isinstance(tools_config, dict)
        return tools_config

    def get_execution_config(self) -> Dict[str, Any]:
        """
        Get execution configuration settings.

        Returns:
            Dictionary with execution settings like timeout and max_concurrent
        """
        tools_config = self.get_tools_config()
        execution_config = tools_config.get("execution", {})
        assert isinstance(execution_config, dict)
        return execution_config

    def get_security_config(self) -> Dict[str, Any]:
        """
        Get security configuration settings.

        Returns:
            Dictionary with security settings like sandbox_path and max_command_length
        """
        security_config = self._config.get("security", {})
        assert isinstance(security_config, dict)
        return security_config

    def get_domain_settings(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get domain-specific configuration settings.

        Args:
            domain: Specific domain name, or None for all domain settings

        Returns:
            Dictionary with domain-specific settings
        """
        domain_settings = self._config.get("domain_settings", {})
        assert isinstance(domain_settings, dict)
        if domain:
            domain_specific = domain_settings.get(domain, {})
            assert isinstance(domain_specific, dict)
            return domain_specific
        return domain_settings

    def get_test_settings(self) -> Dict[str, Any]:
        """
        Get test-specific configuration settings.

        Returns:
            Dictionary with test settings like mock_execution and validate_only
        """
        test_settings = self._config.get("test_settings", {})
        assert isinstance(test_settings, dict)
        return test_settings

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration settings.

        Returns:
            Dictionary with logging configuration
        """
        logging_config = self._config.get("logging", {})
        assert isinstance(logging_config, dict)
        return logging_config

    def get_domains_to_load(self) -> List[str]:
        """
        Get list of domains to auto-discover.

        Returns:
            List of domain names to load
        """
        tools_config = self.get_tools_config()
        domains = tools_config.get("domains", [])
        assert isinstance(domains, list)
        return domains

    def get_specific_tools_to_load(self) -> List[str]:
        """
        Get list of specific tools to auto-discover.

        Returns:
            List of tool specifications to load (e.g., "malware.static_analysis.File")
        """
        tools_config = self.get_tools_config()
        specific_tools = tools_config.get("specific_tools", [])
        assert isinstance(specific_tools, list)
        return specific_tools

    def get_execution_timeout(self) -> float:
        """
        Get execution timeout setting.

        Returns:
            Timeout in seconds, defaults to 300.0 if not configured
        """
        execution_config = self.get_execution_config()
        timeout = execution_config.get("timeout", 300.0)
        assert isinstance(timeout, (int, float))
        return float(timeout)

    def get_max_concurrent(self) -> int:
        """
        Get maximum concurrent executions setting.

        Returns:
            Maximum concurrent executions, defaults to 10 if not configured
        """
        execution_config = self.get_execution_config()
        max_concurrent = execution_config.get("max_concurrent", 10)
        assert isinstance(max_concurrent, int)
        return max_concurrent

    def get_sandbox_path(self) -> Optional[str]:
        """
        Get sandbox path for secure execution.

        Returns:
            Sandbox path string or None if not configured
        """
        security_config = self.get_security_config()
        return security_config.get("sandbox_path")

    def get_max_command_length(self) -> int:
        """
        Get maximum command length limit.

        Returns:
            Maximum command length, defaults to 10000 if not configured
        """
        security_config = self.get_security_config()
        max_length = security_config.get("max_command_length", 10000)
        assert isinstance(max_length, int)
        return max_length

    def is_mock_execution_enabled(self) -> bool:
        """
        Check if mock execution is enabled for testing.

        Returns:
            True if mock execution is enabled, False otherwise
        """
        test_settings = self.get_test_settings()
        mock_execution = test_settings.get("mock_execution", False)
        assert isinstance(mock_execution, bool)
        return mock_execution

    def is_validate_only_mode(self) -> bool:
        """
        Check if validate-only mode is enabled.

        Returns:
            True if validate-only mode is enabled, False otherwise
        """
        test_settings = self.get_test_settings()
        validate_only = test_settings.get("validate_only", False)
        assert isinstance(validate_only, bool)
        return validate_only

    def get_full_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            config: Configuration dictionary to merge
        """
        self._config.update(config)
        logger.info("Configuration updated")

    def _load_config_from_file(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        config_path = Path(config_file)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_file}")
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {config_file}: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"RegistryConfiguration(sections={list(self._config.keys())})"
