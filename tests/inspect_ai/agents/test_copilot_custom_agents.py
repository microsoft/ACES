"""Tests for Copilot custom agents integration.

Tests cover:
- CopilotSessionConfig includes custom_agents in to_dict() output
- Custom agents are properly serialized to SDK format
- AgentType.supports_agent_persona() method
- AgentPromptKwargs includes agent_persona field
"""

from unittest.mock import MagicMock

import pytest


class TestCopilotSessionConfigCustomAgents:
    """Tests for CopilotSessionConfig with custom_agents field."""

    def test_session_config_without_custom_agents(self):
        """Test that to_dict() works without custom_agents."""
        from saber.inspect_ai.agents.registry.models import (
            CopilotSessionConfig,
            SystemMessageConfig,
        )

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        config = CopilotSessionConfig(
            model="gpt-4o",
            tools=[mock_tool],
            available_tools=["test_tool", "report_intent"],
            system_message=SystemMessageConfig(
                content="Test system message",
                mode="append",
            ),
            streaming=False,
            custom_agents=None,
        )

        result = config.to_dict()

        assert "custom_agents" not in result
        assert result["model"] == "gpt-4o"
        assert result["streaming"] is False

    def test_session_config_with_custom_agents(self):
        """Test that to_dict() includes custom_agents when set."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig
        from saber.inspect_ai.agents.registry.models import (
            CopilotSessionConfig,
            SystemMessageConfig,
        )

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        custom_agent = CustomAgentConfig(
            name="test-agent",
            display_name="Test Agent",
            description="A test agent",
            prompt="You are a test agent.",
            tools=["bash", "view"],
            infer=True,
        )

        config = CopilotSessionConfig(
            model="gpt-4o",
            tools=[mock_tool],
            available_tools=["test_tool", "report_intent"],
            system_message=SystemMessageConfig(
                content="Test system message",
                mode="append",
            ),
            streaming=False,
            custom_agents=[custom_agent],
        )

        result = config.to_dict()

        # SDK expects camelCase 'customAgents'
        assert "customAgents" in result
        assert len(result["customAgents"]) == 1
        assert result["customAgents"][0]["name"] == "test-agent"
        assert result["customAgents"][0]["displayName"] == "Test Agent"
        assert result["customAgents"][0]["description"] == "A test agent"
        assert result["customAgents"][0]["prompt"] == "You are a test agent."
        assert result["customAgents"][0]["tools"] == ["bash", "view"]
        assert result["customAgents"][0]["infer"] is True

    def test_session_config_with_multiple_custom_agents(self):
        """Test that to_dict() handles multiple custom agents."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig
        from saber.inspect_ai.agents.registry.models import (
            CopilotSessionConfig,
            SystemMessageConfig,
        )

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        agents = [
            CustomAgentConfig(
                name="agent-1",
                prompt="You are agent 1.",
            ),
            CustomAgentConfig(
                name="agent-2",
                prompt="You are agent 2.",
                tools=["bash"],
            ),
        ]

        config = CopilotSessionConfig(
            model="gpt-4o",
            tools=[mock_tool],
            available_tools=["test_tool"],
            system_message=SystemMessageConfig(content="Test", mode="append"),
            custom_agents=agents,
        )

        result = config.to_dict()

        # SDK expects camelCase 'customAgents'
        assert len(result["customAgents"]) == 2
        assert result["customAgents"][0]["name"] == "agent-1"
        assert result["customAgents"][1]["name"] == "agent-2"
        assert result["customAgents"][1]["tools"] == ["bash"]


class TestCopilotSessionConfigCreate:
    """Tests for CopilotSessionConfig.create() with custom_agents parameter."""

    def test_create_without_custom_agents(self):
        """Test factory method without custom_agents."""
        from saber.inspect_ai.agents.registry.models import CopilotSessionConfig

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        config = CopilotSessionConfig.create(
            model="gpt-4o",
            tools=[mock_tool],
            system_content="Test content",
        )

        assert config.custom_agents is None
        result = config.to_dict()
        assert "custom_agents" not in result

    def test_create_with_custom_agents(self):
        """Test factory method with custom_agents."""
        from saber.inspect_ai.agents.registry.custom_agent import CustomAgentConfig
        from saber.inspect_ai.agents.registry.models import CopilotSessionConfig

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        custom_agent = CustomAgentConfig(
            name="red-team-agent",
            display_name="Red Team Agent",
            prompt="You are a security researcher.",
        )

        config = CopilotSessionConfig.create(
            model="gpt-4o",
            tools=[mock_tool],
            system_content="Test content",
            custom_agents=[custom_agent],
        )

        assert config.custom_agents is not None
        assert len(config.custom_agents) == 1
        assert config.custom_agents[0].name == "red-team-agent"

        result = config.to_dict()
        # SDK expects camelCase 'customAgents'
        assert "customAgents" in result
        assert result["customAgents"][0]["name"] == "red-team-agent"


class TestAgentTypeSupportsAgentPersona:
    """Tests for AgentType.supports_agent_persona() method."""

    def test_copilot_supports_agent_persona(self):
        """Test that copilot agent supports agent_persona."""
        from saber.inspect_ai.agents.registry.models import AgentType

        assert AgentType.supports_agent_persona("copilot") is True

    def test_react_does_not_support_agent_persona(self):
        """Test that react agent does not support agent_persona."""
        from saber.inspect_ai.agents.registry.models import AgentType

        assert AgentType.supports_agent_persona("react") is False

    def test_claude_code_does_not_support_agent_persona(self):
        """Test that claude_code agent does not support agent_persona."""
        from saber.inspect_ai.agents.registry.models import AgentType

        assert AgentType.supports_agent_persona("claude_code") is False

    def test_unknown_agent_does_not_support_agent_persona(self):
        """Test that unknown agent does not support agent_persona."""
        from saber.inspect_ai.agents.registry.models import AgentType

        assert AgentType.supports_agent_persona("unknown") is False


class TestAgentPromptKwargs:
    """Tests for AgentPromptKwargs TypedDict."""

    def test_agent_prompt_kwargs_accepts_agent_persona(self):
        """Test that AgentPromptKwargs accepts agent_persona field."""
        from saber.inspect_ai.agents.registry.models import AgentPromptKwargs

        # TypedDict should accept agent_persona as a valid key
        kwargs: AgentPromptKwargs = {
            "instruction_prompt": "Test instructions",
            "agent_persona": "/path/to/agent.md",
        }

        assert kwargs["agent_persona"] == "/path/to/agent.md"

    def test_agent_prompt_kwargs_agent_persona_optional(self):
        """Test that agent_persona is optional in AgentPromptKwargs."""
        from saber.inspect_ai.agents.registry.models import AgentPromptKwargs

        # Should work without agent_persona
        kwargs: AgentPromptKwargs = {
            "instruction_prompt": "Test instructions",
        }

        assert "agent_persona" not in kwargs
