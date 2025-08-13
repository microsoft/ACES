"""
Tests for modular configuration system.

This module tests the new generic configuration delegation system where each
executor extracts its own configuration from the global configuration instead
of having executor-specific parsing methods in ExecutionConfiguration.
"""

import pytest
from unittest.mock import MagicMock

from saber.server.execution.execution_manager import ExecutionConfiguration, ExecutionManager
from saber.server.execution.executors.factory import ExecutorFactory
from saber.server.execution.executors.cli import CLIExecutor
from saber.server.execution.executors.python_executor import PythonExecutor
from saber.server.execution.sandbox.sandbox_manager import SandboxManager


class TestModularConfiguration:
    """Test cases for the modular configuration system."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/base-sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser"
        }
        return manager

    @pytest.fixture
    def comprehensive_config(self):
        """Create a comprehensive configuration for testing."""
        return {
            # Manager-level configuration
            "execution": {
                "timeout": 300.0,
                "max_concurrent": 5
            },
            "security": {
                "allowed_commands": ["file", "strings", "cat", "echo"],
                "max_command_length": 10000
            },
            "sandbox": {
                "image": "saber/base-sandbox:latest",
                "network_mode": "none"
            },

            # Executor-specific configurations
            "cli": {
                "default_shell_mode": True,
                "timeout": 120.0  # Override global timeout
            },
            "python": {
                "allowed_modules": ["requests", "json", "os"],
                "virtual_env": "/opt/venv",
                "timeout": 600.0  # Different timeout for Python
            },

            # Future executor configurations
            "database": {
                "connection_timeout": 30.0,
                "max_connections": 10
            },
            "file_ops": {
                "allowed_extensions": [".txt", ".json", ".yaml"],
                "max_file_size": 1000000
            }
        }

    def test_generic_get_section(self, comprehensive_config):
        """Test the generic get_section method for any configuration section."""
        config = ExecutionConfiguration(config=comprehensive_config)

        # Test existing sections
        exec_config = config.get_section("execution")
        assert exec_config == {"timeout": 300.0, "max_concurrent": 5}

        cli_config = config.get_section("cli")
        assert cli_config == {"default_shell_mode": True, "timeout": 120.0}

        python_config = config.get_section("python")
        assert python_config == {
            "allowed_modules": ["requests", "json", "os"],
            "virtual_env": "/opt/venv",
            "timeout": 600.0
        }

        # Test future executor configurations
        db_config = config.get_section("database")
        assert db_config == {"connection_timeout": 30.0, "max_connections": 10}

        file_config = config.get_section("file_ops")
        assert file_config == {
            "allowed_extensions": [".txt", ".json", ".yaml"],
            "max_file_size": 1000000
        }

        # Test non-existent section
        missing_config = config.get_section("nonexistent")
        assert missing_config == {}

    def test_executor_factory_with_modular_config(self, mock_sandbox_manager, comprehensive_config):
        """Test that ExecutorFactory correctly uses the modular configuration system."""
        config = ExecutionConfiguration(config=comprehensive_config)
        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=config)

        # Verify factory can extract configuration
        assert factory._configuration == config

        # Test that configuration sections can be extracted
        cli_config = config.get_section("cli")
        assert cli_config["default_shell_mode"] is True
        assert cli_config["timeout"] == 120.0

        python_config = config.get_section("python")
        assert python_config["allowed_modules"] == ["requests", "json", "os"]
        assert python_config["timeout"] == 600.0

    def test_executor_creation_with_extracted_config(self, mock_sandbox_manager, comprehensive_config):
        """Test that executors are created correctly with extracted configuration."""
        config = ExecutionConfiguration(config=comprehensive_config)
        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=config)

        # Create CLI executor
        cli_executor = factory.get_executor("cli")
        assert isinstance(cli_executor, CLIExecutor)

        # Create Python executor
        python_executor = factory.get_executor("python")
        assert isinstance(python_executor, PythonExecutor)

        # Verify different instances
        assert cli_executor is not python_executor

    def test_manager_level_config_methods_still_work(self, comprehensive_config):
        """Test that manager-level configuration methods still work."""
        config = ExecutionConfiguration(config=comprehensive_config)

        # These methods should still work for manager-level config
        assert config.get_execution_timeout() == 300.0
        assert config.get_max_concurrent() == 5
        assert config.get_allowed_commands() == ["file", "strings", "cat", "echo"]
        assert config.get_max_command_length() == 10000

    def test_execution_manager_with_modular_config(self, comprehensive_config):
        """Test that ExecutionManager works with the modular configuration system."""
        # Create execution manager with configuration
        manager = ExecutionManager(config=comprehensive_config)

        # Verify manager has configuration
        assert manager._configuration._config == comprehensive_config

        # Verify factory was created with configuration
        assert manager._executor_factory._configuration == manager._configuration

    def test_configuration_isolation(self, comprehensive_config):
        """Test that different executors get isolated configuration sections."""
        config = ExecutionConfiguration(config=comprehensive_config)

        # Get different sections
        cli_config = config.get_section("cli")
        python_config = config.get_section("python")
        security_config = config.get_section("security")

        # Verify isolation - changes to one section don't affect others
        cli_config["new_setting"] = "test"

        # Re-fetch to verify isolation
        fresh_python_config = config.get_section("python")
        fresh_security_config = config.get_section("security")

        assert "new_setting" not in fresh_python_config
        assert "new_setting" not in fresh_security_config

    def test_empty_configuration_defaults(self):
        """Test behavior with empty configuration."""
        config = ExecutionConfiguration()

        # All sections should return empty dict
        assert config.get_section("cli") == {}
        assert config.get_section("python") == {}
        assert config.get_section("nonexistent") == {}

        # Manager-level defaults should still work
        assert config.get_execution_timeout() == 300.0
        assert config.get_max_concurrent() == 10
        assert config.get_allowed_commands() == []

    def test_partial_configuration(self):
        """Test behavior with partial configuration."""
        partial_config = {
            "cli": {"default_shell_mode": True},
            "security": {"allowed_commands": ["echo"]}
        }
        config = ExecutionConfiguration(config=partial_config)

        # Existing sections should work
        assert config.get_section("cli") == {"default_shell_mode": True}
        assert config.get_section("security") == {"allowed_commands": ["echo"]}

        # Missing sections should return empty dict
        assert config.get_section("python") == {}
        assert config.get_section("execution") == {}

    def test_configuration_scalability(self):
        """Test that the configuration system scales to many executor types."""
        many_executors_config = {}

        # Create configurations for 20 different executor types
        for i in range(20):
            executor_name = f"executor_{i:02d}"
            many_executors_config[executor_name] = {
                "timeout": 100 + i * 10,
                "setting_a": f"value_{i}",
                "setting_b": i * 2,
                "enabled": i % 2 == 0
            }

        config = ExecutionConfiguration(config=many_executors_config)

        # Verify all configurations can be retrieved
        for i in range(20):
            executor_name = f"executor_{i:02d}"
            executor_config = config.get_section(executor_name)

            assert executor_config["timeout"] == 100 + i * 10
            assert executor_config["setting_a"] == f"value_{i}"
            assert executor_config["setting_b"] == i * 2
            assert executor_config["enabled"] == (i % 2 == 0)

        # Verify non-existent executor returns empty
        assert config.get_section("executor_99") == {}
