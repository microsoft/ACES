"""Tests for Copilot agent session changes from January 2026.

This module tests the fixes made during the January 2026 session:
1. Simplified submit tool that matches inspect_ai's native pattern
2. Model selection with --model flag working correctly with agents
3. BYOK (Bring Your Own Key) detection for Azure/OpenAI providers
4. Message construction without redundant initial user messages
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from typing import Any


# =============================================================================
# Tests for Submit Tool (inspect_ai native pattern)
# =============================================================================


class TestSubmitToolNativePattern:
    """Test that submit tool follows inspect_ai's native pattern."""

    def test_create_submit_tool_no_handler_required(self):
        """Test that create_submit_tool no longer requires a submission_handler."""
        from saber.inspect_ai.integration.copilot_tools import create_submit_tool

        # Should work without any handler - just returns the answer
        tool = create_submit_tool()

        assert tool.name == "submit"
        assert callable(tool.handler)

    def test_create_submit_tool_with_tracker(self):
        """Test that create_submit_tool accepts optional tracker."""
        from saber.inspect_ai.integration.copilot_tools import (
            create_submit_tool,
            ToolCallTracker,
        )

        tracker = ToolCallTracker()
        tool = create_submit_tool(tracker)

        assert tool.name == "submit"
        assert tool.parameters is not None
        assert "answer" in tool.parameters.get("properties", {})

    @pytest.mark.asyncio
    async def test_submit_tool_returns_answer_directly(self):
        """Test that submit tool returns the answer (like inspect_ai's pattern)."""
        from saber.inspect_ai.integration.copilot_tools import create_submit_tool

        tool = create_submit_tool()

        invocation = {
            "tool_call_id": "call-123",
            "arguments": {"answer": "The IP address is 192.168.1.1"},
        }

        result = await tool.handler(invocation)

        # Should return the answer in textResultForLlm (not a success message)
        assert result["resultType"] == "success"
        assert result["textResultForLlm"] == "The IP address is 192.168.1.1"

    @pytest.mark.asyncio
    async def test_submit_tool_records_to_tracker(self):
        """Test that submit tool records calls to tracker."""
        from saber.inspect_ai.integration.copilot_tools import (
            create_submit_tool,
            ToolCallTracker,
        )

        tracker = ToolCallTracker()
        tool = create_submit_tool(tracker)

        invocation = {
            "tool_call_id": "call-456",
            "arguments": {"answer": "My final answer"},
        }

        await tool.handler(invocation)

        # Check tracker recorded the call
        assert len(tracker.calls) == 1
        record = tracker.calls[0]
        assert record.tool_name == "submit"
        assert record.arguments == {"answer": "My final answer"}
        assert record.result == "My final answer"
        assert record.is_error is False

    @pytest.mark.asyncio
    async def test_submit_tool_handles_empty_answer(self):
        """Test that submit tool handles empty answer gracefully."""
        from saber.inspect_ai.integration.copilot_tools import create_submit_tool

        tool = create_submit_tool()

        invocation = {
            "tool_call_id": "call-789",
            "arguments": {"answer": ""},
        }

        result = await tool.handler(invocation)

        assert result["resultType"] == "success"
        assert result["textResultForLlm"] == ""


# =============================================================================
# Tests for Model Selection with --model flag
# =============================================================================


class TestModelSelectionWithFlag:
    """Test model selection works correctly with --model CLI flag."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock SABERExecutionContext."""
        from saber.inspect_ai.agents.solver_factory import SABERExecutionContext

        return SABERExecutionContext(
            instruction_prompt="Test instruction",
            assistant_prompt="Test assistant",
            submit_prompt="Test submit",
            continue_prompt="Test continue",
            session_id=None,
            episode_id=None,
            domain_slug=None,
            rest_url=None,
            task_id="test-task",
            sample_id="test-sample",
            role=None,
            transcript_config=None,
        )

    def test_select_model_with_azure_model(self, mock_context):
        """Test that azure/ prefixed models are correctly detected."""
        from saber.inspect_ai.agents.solver_factory import _select_model

        mock_model = Mock()
        mock_model.api.model_name = "openai/azure/gpt-5"

        with patch(
            "saber.inspect_ai.agents.solver_factory.active_model",
            return_value=mock_model,
        ):
            model, model_name = _select_model(mock_context, None, "copilot")

        assert model == mock_model
        assert model_name == "openai/azure/gpt-5"

    def test_select_model_with_openai_model(self, mock_context):
        """Test that openai/ prefixed models are correctly detected."""
        from saber.inspect_ai.agents.solver_factory import _select_model

        mock_model = Mock()
        mock_model.api.model_name = "openai/gpt-4o"

        with patch(
            "saber.inspect_ai.agents.solver_factory.active_model",
            return_value=mock_model,
        ):
            model, model_name = _select_model(mock_context, None, "copilot")

        assert model == mock_model
        assert model_name == "openai/gpt-4o"

    def test_select_model_with_anthropic_model(self, mock_context):
        """Test that anthropic/ prefixed models are correctly detected."""
        from saber.inspect_ai.agents.solver_factory import _select_model

        mock_model = Mock()
        mock_model.api.model_name = "anthropic/claude-3-opus"

        with patch(
            "saber.inspect_ai.agents.solver_factory.active_model",
            return_value=mock_model,
        ):
            model, model_name = _select_model(mock_context, None, "copilot")

        assert model == mock_model
        assert model_name == "anthropic/claude-3-opus"

    def test_select_model_without_flag_uses_agent_name(self, mock_context):
        """Test that without --model flag, agent name is used as identifier."""
        from saber.inspect_ai.agents.solver_factory import _select_model

        mock_model = Mock()
        mock_model.api.model_name = "agent/copilot"  # Fake agent model

        with patch(
            "saber.inspect_ai.agents.solver_factory.active_model",
            return_value=mock_model,
        ):
            model, model_name = _select_model(mock_context, None, "copilot")

        # Should fall back to agent/copilot identifier
        assert model_name == "agent/copilot"

    def test_select_model_with_none_active_model(self, mock_context):
        """Test graceful handling when active_model() returns None."""
        from saber.inspect_ai.agents.solver_factory import _select_model

        with patch(
            "saber.inspect_ai.agents.solver_factory.active_model",
            return_value=None,
        ):
            model, model_name = _select_model(mock_context, None, "copilot")

        # Should fall back to agent name
        assert model_name == "agent/copilot"


# =============================================================================
# Tests for BYOK Provider Configuration
# =============================================================================


class TestBYOKProviderConfig:
    """Test BYOK (Bring Your Own Key) provider configuration."""

    def test_build_provider_config_with_azure_model(self):
        """Test provider config is built correctly for Azure models."""
        from saber.inspect_ai.agents.registry.copilot import build_provider_config
        from saber.inspect_ai.agents.registry.models import AzureProviderConfig

        mock_model = Mock()
        mock_model.api.model_name = "openai/azure/gpt-5"
        mock_model.api.base_url = "https://myresource.openai.azure.com"
        mock_model.api.config.api_key = "my-api-key"
        mock_model.api.config.api_version = "2024-02-15-preview"

        with patch(
            "saber.inspect_ai.agents.registry.copilot.active_model",
            return_value=mock_model,
        ):
            config = build_provider_config(None, None, None, None)

        assert config is not None
        assert isinstance(config, AzureProviderConfig)
        config_dict = config.to_dict()
        assert config_dict.get("type") == "azure"

    def test_build_provider_config_with_explicit_params(self):
        """Test provider config uses explicit parameters when provided."""
        from saber.inspect_ai.agents.registry.copilot import build_provider_config
        from saber.inspect_ai.agents.registry.models import AzureProviderConfig

        config = build_provider_config(
            provider_type="azure",
            provider_base_url="https://explicit.azure.com",
            provider_api_key="explicit-key",
            provider_api_version="2024-01-01",
        )

        assert config is not None
        assert isinstance(config, AzureProviderConfig)
        config_dict = config.to_dict()
        assert config_dict.get("type") == "azure"
        assert config_dict.get("base_url") == "https://explicit.azure.com"
        assert config_dict.get("api_key") == "explicit-key"

    def test_build_provider_config_returns_none_for_copilot_models(self):
        """Test that provider config is None for Copilot-hosted models."""
        from saber.inspect_ai.agents.registry.copilot import build_provider_config

        # Mock a model that doesn't look like BYOK
        mock_model = Mock()
        mock_model.api.model_name = "agent/copilot"

        with patch(
            "saber.inspect_ai.agents.registry.copilot.active_model",
            return_value=mock_model,
        ):
            config = build_provider_config(None, None, None, None)

        # No provider config needed for Copilot-hosted models
        assert config is None


# =============================================================================
# Tests for Message Construction
# =============================================================================


class TestMessageConstruction:
    """Test that messages are constructed without redundancy."""

    def test_build_system_message_includes_all_parts(self):
        """Test system message includes assistant and submit prompts.

        Note: instruction_prompt now goes in the user message, not system message.
        """
        from saber.inspect_ai.agents.registry.copilot import build_system_message

        result = build_system_message(
            assistant_prompt="You are a security analyst.",
            submit_prompt="Submit your findings.",
            submit_enabled=True,
        )

        assert "You are a security analyst." in result
        assert "Submit your findings." in result

    def test_build_system_message_excludes_submit_when_disabled(self):
        """Test submit section excluded when submit_enabled=False."""
        from saber.inspect_ai.agents.registry.copilot import build_system_message

        result = build_system_message(
            assistant_prompt="You are a security analyst.",
            submit_prompt="Submit your findings.",
            submit_enabled=False,
        )

        assert "You are a security analyst." in result
        assert "Submit your findings." not in result

    def test_build_system_message_handles_empty_prompts(self):
        """Test system message handles empty prompts gracefully."""
        from saber.inspect_ai.agents.registry.copilot import build_system_message

        result = build_system_message(
            assistant_prompt="",
            submit_prompt="",
            submit_enabled=True,
        )

        # Empty prompts should result in empty message
        assert result == ""


# =============================================================================
# Tests for Setup Tools
# =============================================================================


class TestSetupTools:
    """Test tool setup functionality."""

    @pytest.fixture
    def mock_sandbox(self):
        """Create a mock sandbox."""
        sb = Mock()
        sb._sandbox = Mock()
        sb._sandbox._mcp_client = Mock()
        return sb

    @pytest.mark.asyncio
    async def test_setup_tools_adds_submit_when_enabled(self, mock_sandbox):
        """Test that submit tool is added when submit_enabled=True."""
        from saber.inspect_ai.agents.registry.copilot import setup_tools
        from saber.inspect_ai.integration.copilot_tools import ToolCallTracker

        tracker = ToolCallTracker()

        with patch(
            "saber.inspect_ai.agents.registry.copilot.get_saber_mcp_tools",
            return_value=[],
        ):
            tools = await setup_tools(mock_sandbox, tracker, submit_enabled=True)

        tool_names = [t.name for t in tools]
        assert "submit" in tool_names

    @pytest.mark.asyncio
    async def test_setup_tools_excludes_submit_when_disabled(self, mock_sandbox):
        """Test that submit tool is excluded when submit_enabled=False."""
        from saber.inspect_ai.agents.registry.copilot import setup_tools
        from saber.inspect_ai.integration.copilot_tools import ToolCallTracker

        tracker = ToolCallTracker()

        with patch(
            "saber.inspect_ai.agents.registry.copilot.get_saber_mcp_tools",
            return_value=[],
        ):
            tools = await setup_tools(mock_sandbox, tracker, submit_enabled=False)

        tool_names = [t.name for t in tools]
        assert "submit" not in tool_names


# =============================================================================
# Tests for Submission Detection in Agent Loop
# =============================================================================


class TestSubmissionDetection:
    """Test that submission is detected via tool calls."""

    def test_captured_message_stores_tool_arguments(self):
        """Test that CapturedMessage stores tool arguments correctly."""
        from saber.inspect_ai.agents.registry.copilot import CapturedMessage

        tc = CapturedMessage(
            message_id="msg-123",
            content="",
            message_type="tool_call",
            tool_call_id="call-123",
            tool_name="submit",
            tool_arguments={"answer": "The flag is CTF{test}"},
        )

        assert tc.tool_name == "submit"
        assert tc.tool_arguments["answer"] == "The flag is CTF{test}"

    def test_turn_capture_tracks_tool_calls(self):
        """Test that TurnCapture tracks tool calls via message groups."""
        from saber.inspect_ai.agents.registry.copilot import (
            TurnCapture,
            CapturedMessage,
            AssistantMessageGroup,
        )

        turn = TurnCapture()

        # Create message groups with tool calls
        group1 = AssistantMessageGroup(message_id="msg-1")
        group1.tool_calls.append(
            CapturedMessage(
                message_id="msg-1",
                content="",
                message_type="tool_call",
                tool_call_id="call-1",
                tool_name="bash",
                tool_arguments={"command": "ls"},
            )
        )

        group2 = AssistantMessageGroup(message_id="msg-2")
        group2.tool_calls.append(
            CapturedMessage(
                message_id="msg-2",
                content="",
                message_type="tool_call",
                tool_call_id="call-2",
                tool_name="submit",
                tool_arguments={"answer": "Done"},
            )
        )

        turn.message_groups.append(group1)
        turn.message_groups.append(group2)

        # Check we can find submit tool call via the property
        submit_calls = [tc for tc in turn.tool_calls if tc.tool_name == "submit"]
        assert len(submit_calls) == 1
        assert submit_calls[0].tool_arguments["answer"] == "Done"


# =============================================================================
# Integration-style Tests for CLI flag combinations
# =============================================================================


class TestCLIFlagCombinations:
    """Test various CLI flag combinations work correctly."""

    def test_agent_without_model_flag_scenario(self):
        """Test scenario: -T agent=copilot (no --model flag).

        In this case, the agent should use Copilot's hosted model (gpt-5 default).
        """
        from saber.inspect_ai.agents.solver_factory import _select_model, SABERExecutionContext

        context = SABERExecutionContext(
            instruction_prompt="Test",
            assistant_prompt="Test",
            submit_prompt="Test",
            continue_prompt="Test",
            session_id=None,
            episode_id=None,
            domain_slug=None,
            rest_url=None,
            task_id=None,
            sample_id=None,
            role=None,
            transcript_config=None,
        )

        # Simulate no --model flag: active_model returns fake agent model
        mock_model = Mock()
        mock_model.api.model_name = "agent/copilot"

        with patch(
            "saber.inspect_ai.agents.solver_factory.active_model",
            return_value=mock_model,
        ):
            _, model_name = _select_model(context, None, "copilot")

        # Should use agent identifier (copilot will use its default gpt-5)
        assert model_name == "agent/copilot"

    def test_agent_with_model_flag_scenario(self):
        """Test scenario: --model openai/azure/gpt-5 -T agent=copilot.

        In this case, the agent should use BYOK with the specified model.
        """
        from saber.inspect_ai.agents.solver_factory import _select_model, SABERExecutionContext

        context = SABERExecutionContext(
            instruction_prompt="Test",
            assistant_prompt="Test",
            submit_prompt="Test",
            continue_prompt="Test",
            session_id=None,
            episode_id=None,
            domain_slug=None,
            rest_url=None,
            task_id=None,
            sample_id=None,
            role=None,
            transcript_config=None,
        )

        # Simulate --model openai/azure/gpt-5: active_model returns real model
        mock_model = Mock()
        mock_model.api.model_name = "openai/azure/gpt-5"

        with patch(
            "saber.inspect_ai.agents.solver_factory.active_model",
            return_value=mock_model,
        ):
            model, model_name = _select_model(context, None, "copilot")

        # Should detect and use the real model
        assert model == mock_model
        assert model_name == "openai/azure/gpt-5"

    def test_react_agent_with_model_flag_scenario(self):
        """Test scenario: --model openai/gpt-4o -T agent=react.

        React agent should also correctly use the --model flag.
        """
        from saber.inspect_ai.agents.solver_factory import _select_model, SABERExecutionContext

        context = SABERExecutionContext(
            instruction_prompt="Test",
            assistant_prompt="Test",
            submit_prompt="Test",
            continue_prompt="Test",
            session_id=None,
            episode_id=None,
            domain_slug=None,
            rest_url=None,
            task_id=None,
            sample_id=None,
            role=None,
            transcript_config=None,
        )

        mock_model = Mock()
        mock_model.api.model_name = "openai/gpt-4o"

        with patch(
            "saber.inspect_ai.agents.solver_factory.active_model",
            return_value=mock_model,
        ):
            model, model_name = _select_model(context, None, "react")

        assert model_name == "openai/gpt-4o"


# =============================================================================
# Tests for ToolCallTracker
# =============================================================================


class TestToolCallTracker:
    """Test ToolCallTracker functionality."""

    def test_tracker_records_calls(self):
        """Test that tracker records tool calls correctly."""
        from saber.inspect_ai.integration.copilot_tools import ToolCallTracker

        tracker = ToolCallTracker()

        tracker.record_call(
            tool_name="bash",
            arguments={"command": "whoami"},
            result="root",
            is_error=False,
            tool_call_id="call-1",
        )

        assert len(tracker.calls) == 1
        assert tracker.calls[0].tool_name == "bash"
        assert tracker.calls[0].result == "root"

    def test_tracker_get_and_clear(self):
        """Test get_and_clear returns calls and clears tracker."""
        from saber.inspect_ai.integration.copilot_tools import ToolCallTracker

        tracker = ToolCallTracker()
        tracker.record_call("tool1", {}, "result1")
        tracker.record_call("tool2", {}, "result2")

        calls = tracker.get_and_clear()

        assert len(calls) == 2
        assert len(tracker.calls) == 0  # Should be cleared

    def test_tracker_generates_id_if_not_provided(self):
        """Test that tracker generates tool_call_id if not provided."""
        from saber.inspect_ai.integration.copilot_tools import ToolCallTracker

        tracker = ToolCallTracker()
        record = tracker.record_call("test_tool", {}, "result")

        assert record.tool_call_id is not None
        assert record.tool_call_id.startswith("call_")
