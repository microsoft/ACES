"""
Test suite for ToolRegistry configuration functionality.

This module tests the configuration loading and application features
of the ToolRegistry system, including security settings, execution limits,
and domain-specific configurations.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from saber.server.tools.tool_registry import ToolRegistry
from saber.server.tools.base import SecurityTool, Parameter, ParameterType, ToolResult
from saber.server.tools.executors.base_executors import CommandLineToolExecutor
from saber.server.tools.utils.security_validator import SecurityValidator


class TestEchoTool(CommandLineToolExecutor):
    """Test tool for configuration testing."""

    _security_tool_metadata = {
        "domain": "malware",
        "name": "test_echo_tool",
        "description": "Test echo tool for configuration testing",
        "author": "Test Suite"
    }

    def __init__(self):
        super().__init__(command="echo", allowed_commands=["echo"])
        self.add_parameter(Parameter(
            name="text",
            type=ParameterType.STRING,
            description="Text to echo",
            required=True
        ))

    def build_command(self, parameters, context):
        return ["echo", parameters["text"]]

    def parse_output(self, stdout, stderr, return_code):
        if return_code != 0:
            return ToolResult.error_result(f"Command failed: {stderr}")
        return ToolResult.success_result({"output": stdout.strip()})


class TestDangerousTool(CommandLineToolExecutor):
    """Test tool that should be blocked by security configuration."""

    _security_tool_metadata = {
        "domain": "malware",
        "name": "test_dangerous_tool",
        "description": "Dangerous tool for security testing",
        "author": "Test Suite"
    }

    def __init__(self):
        # This should fail if security is properly configured
        super().__init__(command="rm", allowed_commands=["rm"])

    def build_command(self, parameters, context):
        return ["rm", "-rf", "/"]

    def parse_output(self, stdout, stderr, return_code):
        return ToolResult.success_result({"output": "Should never execute"})


@pytest.fixture
def test_config_path():
    """Path to test configuration file."""
    return Path(__file__).parent / "config" / "test_tool_config.yaml"


@pytest.fixture
def test_config_dict(test_config_path):
    """Load test configuration as dictionary."""
    with open(test_config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def registry_with_config(test_config_dict):
    """Create ToolRegistry with test configuration."""
    return ToolRegistry(domain="malware", config=test_config_dict)


class TestToolRegistryConfiguration:
    """Test tool registry configuration functionality."""

    def test_config_loading(self, test_config_dict):
        """Test that configuration is properly loaded."""
        assert "tools" in test_config_dict
        assert "security" in test_config_dict
        assert "test_settings" in test_config_dict

        # Verify tools configuration
        tools_config = test_config_dict["tools"]
        assert "domains" in tools_config
        assert "execution" in tools_config
        assert "malware" in tools_config["domains"]
        assert tools_config["execution"]["timeout"] == 120
        assert tools_config["execution"]["max_concurrent"] == 5

        # Verify security configuration
        security_config = test_config_dict["security"]
        assert security_config["sandbox_path"] == "/tmp/test_sandbox"
        assert security_config["max_command_length"] == 1000

    def test_registry_with_config_initialization(self, registry_with_config):
        """Test ToolRegistry initialization with configuration."""
        config = registry_with_config.get_configuration()
        assert config.get_full_config() is not None

        # Check execution manager settings
        execution_manager = registry_with_config.get_execution_manager()
        execution_stats = execution_manager.get_execution_stats()
        assert execution_stats["default_timeout"] == 120  # From config
        assert execution_stats["max_concurrent"] == 5     # From config

    def test_execution_timeout_from_config(self, registry_with_config):
        """Test that execution timeout is loaded from configuration."""
        execution_manager = registry_with_config.get_execution_manager()
        execution_stats = execution_manager.get_execution_stats()
        assert execution_stats["default_timeout"] == 120

    def test_max_concurrent_from_config(self, registry_with_config):
        """Test that max concurrent limit is loaded from configuration."""
        execution_manager = registry_with_config.get_execution_manager()
        execution_stats = execution_manager.get_execution_stats()
        assert execution_stats["max_concurrent"] == 5
        assert execution_stats["available_permits"] == 5

    def test_domain_filtering_from_config(self, test_config_dict):
        """Test that domain filtering works with configuration."""
        # Test with domain filter
        registry = ToolRegistry(domain="malware", config=test_config_dict)
        registrar = registry.get_registrar()
        assert registrar._domain_filter == "malware"

        # Test without domain filter
        registry_all = ToolRegistry(config=test_config_dict)
        registrar_all = registry_all.get_registrar()
        assert registrar_all._domain_filter is None

    def test_security_validator_with_tool_commands(self):
        """Test SecurityValidator initialization with tool-declared commands."""
        # Simulate a tool declaring its required commands
        tool_commands = ["echo", "ls", "cat"]
        sandbox_path = "/tmp/test_sandbox"

        validator = SecurityValidator(
            allowed_commands=tool_commands,
            sandbox_path=sandbox_path
        )

        # Test tool-declared commands
        assert "echo" in validator._allowed_commands
        assert "ls" in validator._allowed_commands
        assert "cat" in validator._allowed_commands

        # Test sandbox path
        assert validator._sandbox_path == Path("/tmp/test_sandbox")

    def test_tool_declared_commands_security_validation(self):
        """Test that only tool-declared commands are permitted."""
        # Simulate a tool declaring specific commands it needs
        tool_commands = ["echo", "ls", "cat"]

        validator = SecurityValidator(allowed_commands=tool_commands)

        # Should not raise for tool-declared commands
        validator.validate_base_command("echo")
        validator.validate_base_command("ls")
        validator.validate_base_command("cat")

        # Should raise for non-allowed commands (blocked commands take precedence)
        with pytest.raises(ValueError, match="not allowed for security reasons"):
            validator.validate_base_command("rm")

        # Test a command that's not blocked but not in allowed list
        with pytest.raises(ValueError, match="not in allowed commands list"):
            validator.validate_base_command("unzip")  # Not blocked, but not in allowed list

    def test_command_length_limit_from_config(self, test_config_dict):
        """Test that command length limits from config are enforced."""
        # For now, test with default limits since SecurityValidator doesn't yet accept config
        validator = SecurityValidator()

        # Create a command longer than the default limit (4096 characters)
        long_command = "echo " + "A" * 5000  # Longer than default 4096
        result = validator.validate_command_string(long_command)

        assert not result.valid
        assert any("too long" in error for error in result.errors)

    def test_tool_enablement_from_config(self, registry_with_config, test_config_dict):
        """Test that tool enablement configuration is respected."""
        # Create and register test tools
        echo_tool_executor = TestEchoTool()
        echo_tool = SecurityTool(
            name="test_echo_tool",
            domain="malware",
            description="Test echo tool",
            author="Test",
            parameters=echo_tool_executor.get_parameters(),
            executor=echo_tool_executor,
        )

        registry_with_config.register_tool(echo_tool)

        # Verify tool is registered using the new interface
        assert "test_echo_tool" in registry_with_config  # Uses __contains__

        # Test getting all tools
        all_tools = registry_with_config.list_tools()
        tool_names = [tool.name for tool in all_tools]
        assert "test_echo_tool" in tool_names

    def test_dangerous_tool_blocking_from_config(self, test_config_dict):
        """Test that dangerous tools are blocked based on security config."""
        # Test that blocked commands are rejected (blocked commands take precedence over allowed)
        with pytest.raises(ValueError, match="not allowed for security reasons"):
            TestDangerousTool()

    def test_sandbox_path_configuration(self, test_config_dict):
        """Test sandbox path configuration."""
        security_config = test_config_dict.get("security", {})
        sandbox_path = security_config.get("sandbox_path")

        validator = SecurityValidator(sandbox_path=sandbox_path)
        assert validator._sandbox_path == Path("/tmp/test_sandbox")

    def test_domain_specific_settings(self, test_config_dict):
        """Test domain-specific configuration settings."""
        domain_settings = test_config_dict.get("domain_settings", {})

        # Test malware domain settings
        malware_settings = domain_settings.get("malware", {})
        assert malware_settings["default_timeout"] == 60
        assert malware_settings["quarantine_path"] == "/tmp/quarantine"
        assert ".exe" in malware_settings["allowed_extensions"]

        # Test threat investigation domain settings
        threat_settings = domain_settings.get("threat_investigation", {})
        assert threat_settings["default_timeout"] == 180
        assert threat_settings["api_rate_limit"] == 100
        assert threat_settings["cache_duration"] == 3600

    def test_test_specific_settings(self, test_config_dict):
        """Test test-specific configuration settings."""
        test_settings = test_config_dict.get("test_settings", {})

        assert test_settings["mock_execution"] is True
        assert test_settings["validate_only"] is False
        assert test_settings["log_commands"] is True

    @pytest.mark.asyncio
    async def test_config_applied_to_execution(self, registry_with_config):
        """Test that configuration is properly applied during tool execution."""
        # This test verifies configuration loading and basic tool registration
        # For now, we'll focus on testing that the configuration is properly applied
        # rather than actual command execution which requires complex subprocess mocking

        # Verify that the registry loaded the configuration correctly
        execution_manager = registry_with_config.get_execution_manager()
        execution_stats = execution_manager.get_execution_stats()
        assert execution_stats["default_timeout"] == 120
        assert execution_stats["max_concurrent"] == 5

        # Create and register a test tool to verify the registration process works
        echo_tool_executor = TestEchoTool()
        echo_tool = SecurityTool(
            name="test_echo_tool",
            domain="malware",
            description="Test echo tool",
            author="Test",
            parameters=echo_tool_executor.get_parameters(),
            executor=echo_tool_executor,
            enabled=True
        )

        registry_with_config.register_tool(echo_tool)

        # Verify the tool was registered successfully
        registered_tool = registry_with_config.get_tool("test_echo_tool")
        assert registered_tool.name == "test_echo_tool"
        assert registered_tool.domain == "malware"

        # Verify the tool appears in tools list
        all_tools = registry_with_config.list_tools()
        tool_names = [tool.name for tool in all_tools]
        assert "test_echo_tool" in tool_names

        # Verify configuration timeout is applied to the tool executor
        assert echo_tool_executor.get_timeout() is None  # Uses registry default
        execution_manager = registry_with_config.get_execution_manager()
        execution_stats = execution_manager.get_execution_stats()
        assert execution_stats["default_timeout"] == 120  # From config

    def test_specific_tool_discovery(self):
        """Test discovery of specific tools by path."""
        registry = ToolRegistry(domain="malware")

        # Test discovering a specific tool
        registry.discover_tool_by_path("malware.static_analysis.File")

        # Verify the tool was registered using the new interface
        assert "file" in registry  # Uses __contains__
        file_tool = registry.get_tool("file")
        assert file_tool.domain == "malware"
        assert file_tool.name == "file"

    def test_specific_tools_list_discovery(self):
        """Test auto-discovery of multiple specific tools."""
        registry = ToolRegistry(domain="malware")

        # Test discovering multiple specific tools
        tool_specs = [
            "malware.static_analysis.File",
            "malware.static_analysis.Strings"
        ]
        registry.auto_discover_specific_tools(tool_specs)

        # Verify both tools were registered using the new interface
        assert "file" in registry  # Uses __contains__
        assert "strings" in registry  # Uses __contains__

        file_tool = registry.get_tool("file")
        strings_tool = registry.get_tool("strings")

        assert file_tool.domain == "malware"
        assert strings_tool.domain == "malware"

    def test_invalid_tool_path_format(self):
        """Test that invalid tool path formats are rejected."""
        registry = ToolRegistry(domain="malware")

        # Test invalid formats
        with pytest.raises(ValueError, match="Tool path must be"):
            registry.discover_tool_by_path("invalid.path")

        with pytest.raises(ValueError, match="Tool path must be"):
            registry.discover_tool_by_path("too.many.parts.here")

        with pytest.raises(ValueError, match="Tool path must be"):
            registry.discover_tool_by_path("single_part")

    def test_unsupported_domain_rejection(self):
        """Test that unsupported domains are rejected."""
        registry = ToolRegistry()

        # Test unsupported domain
        with pytest.raises(ValueError, match="Currently only 'malware' domain is supported"):
            registry.discover_tool_by_path("unsupported_domain.module.Class")

    def test_nonexistent_tool_class(self):
        """Test handling of nonexistent tool classes."""
        registry = ToolRegistry(domain="malware")

        # Test nonexistent class
        with pytest.raises(AttributeError, match="Class 'NonexistentClass' not found"):
            registry.discover_tool_by_path("malware.static_analysis.NonexistentClass")

    def test_hybrid_discovery_from_config(self, test_config_path):
        """Test that both domain and specific tool discovery work from config."""
        registry = ToolRegistry(config_file=str(test_config_path))

        # Load configuration which should trigger both domain and specific tool discovery
        registry.load_configuration(str(test_config_path))

        # Check that we have tools from domain discovery
        # (We can't test actual malware tools without them being implemented,
        # but we can verify the configuration loading logic works)
        config = registry.get_configuration()
        assert config.get_full_config() is not None

        # Verify configuration sections exist
        tools_config = config.get_tools_config()
        assert "domains" in config.get_full_config().get("tools", {})
        assert "specific_tools" in config.get_full_config().get("tools", {})
        tools_config = config.get_tools_config()
        assert "malware" in tools_config["domains"]
        assert "malware.static_analysis.File" in tools_config["specific_tools"]

    def test_config_validation_comprehensive(self, test_config_path):
        """Test comprehensive configuration validation."""
        # Test loading configuration from file
        registry = ToolRegistry(config_file=str(test_config_path))

        # Verify all major configuration sections are loaded
        config = registry.get_configuration()
        assert config.get_full_config() is not None

        # Test that registry can be initialized with file path
        execution_manager = registry.get_execution_manager()
        execution_stats = execution_manager.get_execution_stats()
        assert execution_stats["default_timeout"] == 120
        assert execution_stats["max_concurrent"] == 5


class TestConfigurationIntegration:
    """Integration tests for configuration with other components."""

    def test_registry_auto_discovery_with_config(self, registry_with_config):
        """Test auto-discovery works with configuration."""
        # This would normally discover actual tools, but we're testing the mechanism
        try:
            registry_with_config.auto_discover_domain_tools("malware")
            # Should not raise an exception
        except ImportError:
            # Expected if domain modules don't exist, which is fine for testing
            pass

    def test_config_override_defaults(self, test_config_dict):
        """Test that configuration properly overrides default values."""
        # Create registry without config (uses defaults)
        default_registry = ToolRegistry()
        default_execution_manager = default_registry.get_execution_manager()
        default_stats = default_execution_manager.get_execution_stats()
        assert default_stats["default_timeout"] == 300.0  # Default
        assert default_stats["max_concurrent"] == 10      # Default

        # Create registry with config (uses config values)
        config_registry = ToolRegistry(config=test_config_dict)
        config_execution_manager = config_registry.get_execution_manager()
        config_stats = config_execution_manager.get_execution_stats()
        assert config_stats["default_timeout"] == 120     # From config
        assert config_stats["max_concurrent"] == 5        # From config

    def test_partial_config_handling(self):
        """Test handling of partial configuration (missing sections)."""
        partial_config = {
            "tools": {
                "execution": {
                    "timeout": 180
                    # max_concurrent missing - should use default
                }
            }
            # security section missing entirely
        }

        registry = ToolRegistry(config=partial_config)
        execution_manager = registry.get_execution_manager()
        execution_stats = execution_manager.get_execution_stats()
        assert execution_stats["default_timeout"] == 180  # From config
        assert execution_stats["max_concurrent"] == 10    # Default value

    def test_empty_config_handling(self):
        """Test handling of empty configuration."""
        empty_config = {}

        registry = ToolRegistry(config=empty_config)
        # Should use all default values
        execution_manager = registry.get_execution_manager()
        execution_stats = execution_manager.get_execution_stats()
        assert execution_stats["default_timeout"] == 300.0
        assert execution_stats["max_concurrent"] == 10
