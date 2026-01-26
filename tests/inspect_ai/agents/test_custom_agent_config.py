"""Tests for CustomAgentConfig Pydantic model.

Tests cover:
- Valid config creation
- Serialization to SDK wire format
- Tool mapping integration
- Validation edge cases
"""

import pytest
from pydantic import ValidationError


class TestCustomAgentConfig:
    """Tests for CustomAgentConfig model."""

    def test_minimal_valid_config(self):
        """Test creating config with only required fields."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        config = CustomAgentConfig(
            name="test-agent",
            prompt="You are a helpful assistant.",
        )
        assert config.name == "test-agent"
        assert config.prompt == "You are a helpful assistant."
        assert config.display_name is None
        assert config.description is None
        assert config.tools is None
        assert config.infer is True

    def test_full_config(self):
        """Test creating config with all fields."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        config = CustomAgentConfig(
            name="red-team-agent",
            display_name="Red Team Agent",
            description="Execute security testing tasks",
            prompt="# Red Team Agent\n\nYou are a security researcher.",
            tools=["bash", "read_file", "write_file"],
            infer=False,
        )
        assert config.name == "red-team-agent"
        assert config.display_name == "Red Team Agent"
        assert config.description == "Execute security testing tasks"
        assert config.tools == ["bash", "read_file", "write_file"]
        assert config.infer is False

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        config = CustomAgentConfig(
            name="test-agent",
            prompt="Test prompt",
        )
        with pytest.raises(ValidationError):
            config.name = "new-name"  # type: ignore[misc]

    def test_empty_name_raises(self):
        """Test that empty name is rejected."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        with pytest.raises(ValidationError) as exc_info:
            CustomAgentConfig(name="", prompt="Test")
        assert "name" in str(exc_info.value).lower()

    def test_empty_prompt_raises(self):
        """Test that empty prompt is rejected."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        with pytest.raises(ValidationError) as exc_info:
            CustomAgentConfig(name="test", prompt="")
        assert "prompt" in str(exc_info.value).lower()

    def test_to_sdk_dict_minimal(self):
        """Test serialization to SDK format with minimal fields."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        config = CustomAgentConfig(
            name="test-agent",
            prompt="Test prompt",
        )
        sdk_dict = config.to_sdk_dict()

        assert sdk_dict == {
            "name": "test-agent",
            "prompt": "Test prompt",
            "infer": True,
        }

    def test_to_sdk_dict_full(self):
        """Test serialization to SDK format with all fields."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        config = CustomAgentConfig(
            name="red-team-agent",
            display_name="Red Team Agent",
            description="Security testing",
            prompt="Test prompt",
            tools=["bash", "read_file"],
            infer=False,
        )
        sdk_dict = config.to_sdk_dict()

        assert sdk_dict == {
            "name": "red-team-agent",
            "displayName": "Red Team Agent",
            "description": "Security testing",
            "prompt": "Test prompt",
            "tools": ["bash", "read_file"],
            "infer": False,
        }

    def test_to_sdk_dict_with_tool_mapping(self):
        """Test that tool mapping is applied during serialization."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        config = CustomAgentConfig(
            name="test-agent",
            prompt="Test",
            tools=["Bash", "Read", "Write"],  # Claude tool names
        )

        # Provide Claude -> SABER tool mapping
        tool_map = {
            "Bash": "bash",
            "Read": "read_file",
            "Write": "write_file",
        }
        sdk_dict = config.to_sdk_dict(tool_map=tool_map)

        assert sdk_dict["tools"] == ["bash", "read_file", "write_file"]

    def test_to_sdk_dict_null_tools_when_none(self):
        """Test that None tools becomes null in SDK format (means all tools)."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig

        config = CustomAgentConfig(
            name="test-agent",
            prompt="Test",
            tools=None,
        )
        sdk_dict = config.to_sdk_dict()

        # tools should not be in dict when None (SDK interprets missing as "all tools")
        assert "tools" not in sdk_dict


class TestAgentFileMetadata:
    """Tests for AgentFileMetadata model."""

    def test_minimal_metadata(self):
        """Test metadata with only required fields."""
        from saber.inspect_ai.agents.registry.custom_agent import AgentFileMetadata

        metadata = AgentFileMetadata(name="test-agent")
        assert metadata.name == "test-agent"
        assert metadata.description is None
        assert metadata.version is None
        assert metadata.model is None
        assert metadata.tools is None

    def test_full_metadata(self):
        """Test metadata with all fields."""
        from saber.inspect_ai.agents.registry.custom_agent import (
            AgentFileMetadata,
            AgentToolsConfig,
        )

        metadata = AgentFileMetadata(
            name="Red Team Agent",
            description="Execute security testing tasks",
            version="2.0.0",
            model="claude-sonnet-4-5-20250929",
            tools=AgentToolsConfig(
                allowed=["Bash", "Read", "Write"],
                disallowed=["WebSearch"],
            ),
            permission_mode="acceptEdits",
            max_turns=100,
        )
        assert metadata.name == "Red Team Agent"
        assert metadata.tools is not None
        assert metadata.tools.allowed == ["Bash", "Read", "Write"]
        assert metadata.tools.disallowed == ["WebSearch"]


class TestAgentToolsConfig:
    """Tests for AgentToolsConfig model."""

    def test_allowed_only(self):
        """Test config with only allowed tools."""
        from saber.inspect_ai.agents.registry.custom_agent import AgentToolsConfig

        config = AgentToolsConfig(allowed=["Bash", "Read"])
        assert config.allowed == ["Bash", "Read"]
        assert config.disallowed is None

    def test_disallowed_only(self):
        """Test config with only disallowed tools."""
        from saber.inspect_ai.agents.registry.custom_agent import AgentToolsConfig

        config = AgentToolsConfig(disallowed=["WebSearch"])
        assert config.allowed is None
        assert config.disallowed == ["WebSearch"]

    def test_empty_config(self):
        """Test empty tools config (all tools allowed)."""
        from saber.inspect_ai.agents.registry.custom_agent import AgentToolsConfig

        config = AgentToolsConfig()
        assert config.allowed is None
        assert config.disallowed is None
