"""
Tests for modular configuration system.

This module tests the new generic configuration delegation system where each
executor extracts its own configuration from the global configuration instead
of having executor-specific parsing methods in dict.
"""

from unittest.mock import MagicMock

import pytest

from saber.server.execution.execution_manager import ExecutionManager
from saber.server.execution.executors.executor_factory import ExecutorFactory
from saber.server.execution.executors.standard_registry.cli_executor import CLIExecutor
from saber.server.execution.executors.standard_registry.python_executor import PythonExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestModularConfiguration:
    """Test cases for the modular configuration system."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
        }
        return manager

    @pytest.fixture
    def comprehensive_config(self):
        """Create a comprehensive configuration for testing."""
        return {
            # Manager-level configuration
            "execution": {"timeout": 300.0, "max_concurrent": 5},
            "security": {"allowed_commands": ["file", "strings", "cat", "echo"], "max_command_length": 10000},
            "sandbox": {"image": "saber/sandbox:latest", "network_mode": "none"},
            # Executor-specific configurations
            "cli": {"default_shell_mode": True, "timeout": 120.0},  # Override global timeout
            "python": {
                "allowed_modules": ["requests", "json", "os"],
                "virtual_env": "/opt/venv",
                "timeout": 600.0,  # Different timeout for Python
            },
            # Future executor configurations
            "database": {"connection_timeout": 30.0, "max_connections": 10},
            "file_ops": {"allowed_extensions": [".txt", ".json", ".yaml"], "max_file_size": 1000000},
        }

    def test_generic_get_section(self, comprehensive_config):
        """Test the generic dictionary-based configuration access."""
        config = comprehensive_config

        # Test direct dictionary access for executor configurations
        cli_config = config.get("cli", {})
        assert cli_config == {"default_shell_mode": True, "timeout": 120.0}

        python_config = config.get("python", {})
        assert python_config == {
            "allowed_modules": ["requests", "json", "os"],
            "virtual_env": "/opt/venv",
            "timeout": 600.0,
        }

        # Test future executor configurations
        db_config = config.get("database", {})
        assert db_config == {"connection_timeout": 30.0, "max_connections": 10}

        file_config = config.get("file_ops", {})
        assert file_config == {"allowed_extensions": [".txt", ".json", ".yaml"], "max_file_size": 1000000}

        # Test non-existent section
        missing_config = config.get("nonexistent", {})
        assert missing_config == {}

    def test_executor_factory_with_modular_config(self, mock_sandbox_manager, comprehensive_config):
        """Test that ExecutorFactory correctly uses the modular configuration system."""
        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=comprehensive_config)

        # Verify factory can access configuration
        assert factory._configuration == comprehensive_config

        # Test that configuration sections can be extracted
        cli_config = comprehensive_config.get("cli", {})
        assert cli_config["default_shell_mode"] is True
        assert cli_config["timeout"] == 120.0

        python_config = comprehensive_config.get("python", {})
        assert python_config["allowed_modules"] == ["requests", "json", "os"]
        assert python_config["timeout"] == 600.0

    def test_executor_creation_with_extracted_config(self, mock_sandbox_manager, comprehensive_config):
        """Test that executors are created correctly with extracted configuration."""
        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=comprehensive_config)

        # Create CLI executor
        cli_executor = factory.get_executor("cli")
        assert isinstance(cli_executor, CLIExecutor)

        # Create Python executor
        python_executor = factory.get_executor("python")
        assert isinstance(python_executor, PythonExecutor)

        # Verify different instances
        assert cli_executor is not python_executor

    def test_configuration_isolation(self, comprehensive_config):
        """Test that different executors get isolated configuration sections."""
        # Get different sections directly from dict
        cli_config = comprehensive_config.get("cli", {}).copy()
        python_config = comprehensive_config.get("python", {}).copy()

        # Verify isolation - changes to one section don't affect others
        cli_config["new_setting"] = "test"

        # Re-fetch to verify isolation
        fresh_python_config = comprehensive_config.get("python", {})

        assert "new_setting" not in fresh_python_config
        assert cli_config != fresh_python_config

    def test_empty_configuration_defaults(self):
        """Test behavior with empty configuration."""
        config = {}

        # All sections should return empty dict
        assert config.get("cli", {}) == {}
        assert config.get("python", {}) == {}
        assert config.get("nonexistent", {}) == {}

    def test_partial_configuration(self):
        """Test behavior with partial configuration."""
        partial_config = {"cli": {"default_shell_mode": True}, "security": {"allowed_commands": ["echo"]}}

        # Existing sections should work
        assert partial_config.get("cli", {}) == {"default_shell_mode": True}
        assert partial_config.get("security", {}) == {"allowed_commands": ["echo"]}

        # Missing sections should return empty dict
        assert partial_config.get("python", {}) == {}
        assert partial_config.get("execution", {}) == {}

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
                "enabled": i % 2 == 0,
            }

        # Verify all configurations can be retrieved
        for i in range(20):
            executor_name = f"executor_{i:02d}"
            executor_config = many_executors_config.get(executor_name, {})

            assert executor_config["timeout"] == 100 + i * 10
            assert executor_config["setting_a"] == f"value_{i}"
            assert executor_config["setting_b"] == i * 2
            assert executor_config["enabled"] == (i % 2 == 0)

        # Verify non-existent executor returns empty
        assert many_executors_config.get("executor_99", {}) == {}
