#!/usr/bin/env python3
"""
Tests for ReAct Agent

Unit tests for the ReAct agent implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from saber.client.agents.react_agent import ReActAgent


class TestReActAgent:
    """Test ReAct agent functionality."""

    def test_agent_initialization(self):
        """Test ReAct agent initialization."""
        mock_mcp = MagicMock()
        mock_llm = MagicMock()

        agent = ReActAgent(mock_mcp, mock_llm, max_iterations=10)

        assert agent.mcp_client == mock_mcp
        assert agent.llm_client == mock_llm
        assert agent.max_iterations == 10
        assert agent.available_tools == []
        assert agent.conversation_history == []
        assert agent.flags_found == []

    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        """Test dynamic tool discovery."""
        mock_mcp = AsyncMock()
        mock_llm = MagicMock()

        # Mock tools response
        mock_tools = [
            {"name": "cli_executor", "description": "Execute CLI commands"},
            {"name": "python_executor", "description": "Execute Python code"}
        ]
        mock_mcp.list_tools.return_value = {"tools": mock_tools}

        agent = ReActAgent(mock_mcp, mock_llm)
        await agent._discover_tools()

        assert len(agent.available_tools) == 2
        assert agent.available_tools[0]["name"] == "cli_executor"
        assert agent.available_tools[1]["name"] == "python_executor"

    def test_flag_detection(self):
        """Test flag detection in observations."""
        mock_mcp = MagicMock()
        mock_llm = MagicMock()

        agent = ReActAgent(mock_mcp, mock_llm)

        # Test various flag formats
        test_cases = [
            ("Found FLAG: test123", True, "test123"),
            ("flag: another_flag", True, "another_flag"),
            ("No flag here", False, None),
            ("FLAG:embedded_flag", True, "embedded_flag"),
            ("Multiple FLAG: first FLAG: second", True, ["first", "second"]),  # Should find both
        ]

        for observation, should_find, expected_flag in test_cases:
            agent.flags_found = []  # Reset
            result = agent._check_completion(observation)

            if should_find:
                assert result is True
                if isinstance(expected_flag, list):
                    assert len(agent.flags_found) == len(expected_flag)
                    for flag in expected_flag:
                        assert flag in agent.flags_found
                else:
                    assert len(agent.flags_found) == 1
                    assert agent.flags_found[0] == expected_flag
            else:
                assert result is False
                assert len(agent.flags_found) == 0

    def test_fallback_action(self):
        """Test fallback action when decision fails."""
        mock_mcp = MagicMock()
        mock_llm = MagicMock()

        agent = ReActAgent(mock_mcp, mock_llm)

        # Test with CLI tool available
        agent.available_tools = [
            {"name": "cli_executor", "description": "Execute CLI commands"}
        ]

        action = agent._fallback_action()
        assert action is not None
        assert action["tool"] == "cli_executor"
        assert "command" in action["arguments"]
        assert "reasoning" in action

    def test_history_management(self):
        """Test conversation history management."""
        mock_mcp = MagicMock()
        mock_llm = MagicMock()

        agent = ReActAgent(mock_mcp, mock_llm)

        # Add system message
        agent.conversation_history = [
            {"role": "system", "content": "Initial prompt"}
        ]

        # Add interaction
        thought = "I need to investigate the target"
        action = {"tool": "cli_executor", "reasoning": "reconnaissance"}
        observation = "Target response data"

        agent._update_history(thought, action, observation)

        # Should have system message + 3 new messages
        assert len(agent.conversation_history) == 4
        assert "Thought:" in agent.conversation_history[1]["content"]
        assert "Action:" in agent.conversation_history[2]["content"]
        assert "Observation:" in agent.conversation_history[3]["content"]

    def test_tools_formatting(self):
        """Test tool formatting for prompts."""
        mock_mcp = MagicMock()
        mock_llm = MagicMock()

        agent = ReActAgent(mock_mcp, mock_llm)

        # Test with no tools
        formatted = agent._format_tools_for_prompt()
        assert formatted == "No tools available"

        # Test with tools
        agent.available_tools = [
            {
                "name": "cli_executor",
                "description": "Execute CLI commands",
                "inputSchema": {
                    "properties": {
                        "command": {"type": "string"}
                    }
                }
            }
        ]

        formatted = agent._format_tools_for_prompt()
        assert "cli_executor" in formatted
        assert "Execute CLI commands" in formatted
        assert "command" in formatted

    @pytest.mark.asyncio
    async def test_harness_compliance(self):
        """Test that agent is compliant with harness interface."""
        mock_mcp = AsyncMock()
        mock_llm = AsyncMock()

        # Mock LLM to return flag-finding response
        mock_llm.chat = AsyncMock(return_value={"content": "I found it"})
        mock_mcp.list_tools.return_value = {"tools": []}
        mock_mcp.call_tool.return_value = {
            "content": [{"text": "FLAG: test_flag_123"}]
        }

        agent = ReActAgent(mock_mcp, mock_llm, max_iterations=2)

        # Test shutdown check function
        shutdown_called = False
        def shutdown_check():
            return shutdown_called

        # Run agent (should find flag quickly due to mock)
        result = await agent.run("Test mission", shutdown_check)

        # Verify harness-compliant response
        assert isinstance(result, dict)
        assert "success" in result
        assert "flag" in result
        assert "iterations" in result
        assert "reason" in result


if __name__ == "__main__":
    pytest.main([__file__])
