"""Integration tests for Copilot SDK with mocked LLM calls.

These tests use the real Copilot SDK types and event structures but mock
the actual LLM API calls. This allows us to test:
- EventCapture event handling
- TurnCapture data accumulation
- AssistantMessageGroup structuring
- Submission detection via tool calls
- Full solver loop behavior
"""

import pytest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Any

# Skip all tests if SDK not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("copilot", reason="Copilot SDK not installed"),
    reason="Copilot SDK not installed",
)


# Mock event classes that mimic the SDK's structure
class MockEventType(Enum):
    """Mock of SessionEventType enum."""
    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_INTENT = "assistant.intent"
    ASSISTANT_REASONING = "assistant.reasoning"
    TOOL_EXECUTION_COMPLETE = "tool.execution_complete"
    SESSION_IDLE = "session.idle"
    SESSION_ERROR = "session.error"


@dataclass
class MockToolRequest:
    """Mock of a tool request in assistant message data."""
    name: str
    arguments: dict
    tool_call_id: str


@dataclass
class MockAssistantMessageData:
    """Mock of assistant.message event data."""
    content: str | None = None
    message_id: str | None = None
    tool_requests: list[MockToolRequest] = field(default_factory=list)


@dataclass
class MockToolResultData:
    """Mock of tool.execution_complete event data."""
    tool_call_id: str
    result: str


@dataclass
class MockIntentData:
    """Mock of assistant.intent event data."""
    intent: str
    message_id: str | None = None


@dataclass
class MockEvent:
    """Mock of a SessionEvent from the SDK."""
    type: MockEventType
    data: Any | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TestEventCaptureWithRealEvents:
    """Test EventCapture with real SDK event structures."""

    @pytest.fixture
    def event_capture(self):
        """Create an EventCapture instance."""
        from saber.inspect_ai.agents.registry.copilot import EventCapture

        return EventCapture(debug_all_events=True)

    def test_capture_assistant_message_event(self, event_capture):
        """Test capturing an assistant.message event."""
        data = MockAssistantMessageData(
            content="I will analyze the security logs.",
            message_id="msg-1",
        )
        event = MockEvent(type=MockEventType.ASSISTANT_MESSAGE, data=data)

        event_capture._on_event(event)

        assert len(event_capture.current_turn.message_groups) == 1
        group = event_capture.current_turn.message_groups[0]
        assert group.text == "I will analyze the security logs."

    def test_capture_assistant_message_with_tool_calls(self, event_capture):
        """Test capturing an assistant.message with tool_calls."""
        tool_request = MockToolRequest(
            name="bash",
            arguments={"command": "ls -la"},
            tool_call_id="call-123",
        )
        data = MockAssistantMessageData(
            content="Let me check the system.",
            message_id="msg-2",
            tool_requests=[tool_request],
        )
        event = MockEvent(type=MockEventType.ASSISTANT_MESSAGE, data=data)

        event_capture._on_event(event)

        assert len(event_capture.current_turn.message_groups) == 1
        group = event_capture.current_turn.message_groups[0]
        assert group.text == "Let me check the system."
        assert len(group.tool_calls) == 1
        assert group.tool_calls[0].tool_name == "bash"
        assert group.tool_calls[0].tool_arguments == {"command": "ls -la"}

    def test_capture_tool_execution_complete(self, event_capture):
        """Test capturing tool.execution_complete event."""
        # First, create an assistant message with a tool call
        tool_request = MockToolRequest(
            name="bash",
            arguments={"command": "whoami"},
            tool_call_id="call-456",
        )
        msg_data = MockAssistantMessageData(
            content="",
            message_id="msg-3",
            tool_requests=[tool_request],
        )
        msg_event = MockEvent(type=MockEventType.ASSISTANT_MESSAGE, data=msg_data)
        event_capture._on_event(msg_event)

        # Now capture the tool result
        result_data = MockToolResultData(
            tool_call_id="call-456",
            result="root",
        )
        result_event = MockEvent(type=MockEventType.TOOL_EXECUTION_COMPLETE, data=result_data)
        event_capture._on_event(result_event)

        # Check the result was associated with the tool call
        group = event_capture.current_turn.message_groups[0]
        assert "call-456" in group.tool_results
        assert group.tool_results["call-456"] == "root"

    def test_capture_report_intent_tool(self, event_capture):
        """Test capturing report_intent tool calls (GPT-5 behavior)."""
        # report_intent appears as a tool call
        tool_request = MockToolRequest(
            name="report_intent",
            arguments={"intent": "I will analyze the network logs to find the attacker's IP"},
            tool_call_id="call-intent-1",
        )
        data = MockAssistantMessageData(
            content="",
            message_id="msg-4",
            tool_requests=[tool_request],
        )
        event = MockEvent(type=MockEventType.ASSISTANT_MESSAGE, data=data)
        event_capture._on_event(event)

        # The intent should be captured (prefixed with [Intent])
        group = event_capture.current_turn.message_groups[0]
        assert "[Intent]" in group.intent
        assert "analyze the network logs" in group.intent

    def test_capture_submit_tool_call(self, event_capture):
        """Test that submit tool calls are captured and detectable."""
        tool_request = MockToolRequest(
            name="submit",
            arguments={"answer": "192.168.1.100"},
            tool_call_id="call-submit-1",
        )
        data = MockAssistantMessageData(
            content="I found the answer.",
            message_id="msg-5",
            tool_requests=[tool_request],
        )
        event = MockEvent(type=MockEventType.ASSISTANT_MESSAGE, data=data)
        event_capture._on_event(event)

        # Should be able to detect submit in tool_calls
        submit_calls = [
            tc
            for tc in event_capture.current_turn.tool_calls
            if tc.tool_name == "submit"
        ]
        assert len(submit_calls) == 1
        assert submit_calls[0].tool_arguments == {"answer": "192.168.1.100"}

    def test_session_idle_triggers_completion(self, event_capture):
        """Test that session.idle event signals turn completion."""
        # Set up the idle event
        idle_event = MockEvent(type=MockEventType.SESSION_IDLE, data=None)

        # The idle event should be captured
        event_capture._on_event(idle_event)

        # Check that idle was captured in events
        idle_events = [
            e for e in event_capture.current_turn.events if e.get("type") == "session.idle"
        ]
        assert len(idle_events) == 1

    def test_reset_turn_clears_state(self, event_capture):
        """Test that reset_turn() clears all captured state."""
        # Add some events
        data = MockAssistantMessageData(
            content="Test message",
            message_id="msg-6",
        )
        event = MockEvent(type=MockEventType.ASSISTANT_MESSAGE, data=data)
        event_capture._on_event(event)

        assert len(event_capture.current_turn.message_groups) > 0

        # Reset
        event_capture.reset_turn()

        # Should be empty
        assert len(event_capture.current_turn.message_groups) == 0
        assert len(event_capture.current_turn.events) == 0


class TestTurnCaptureFormatting:
    """Test TurnCapture transcript formatting."""

    def test_format_transcript_basic(self):
        """Test basic transcript formatting."""
        from saber.inspect_ai.agents.registry.copilot import (
            TurnCapture,
            AssistantMessageGroup,
            CapturedMessage,
        )

        turn = TurnCapture()
        group = AssistantMessageGroup(message_id="msg-1")
        group.text = "I will analyze the logs."
        group.tool_calls.append(
            CapturedMessage(
                message_id="msg-1",
                content="",
                message_type="tool_call",
                tool_call_id="call-1",
                tool_name="bash",
                tool_arguments={"command": "cat /var/log/auth.log"},
            )
        )
        group.tool_results["call-1"] = "root logged in at 10:00"
        turn.message_groups.append(group)

        transcript = turn.format_transcript(
            system_message="You are a security analyst.",
            user_prompt="Analyze the incident.",
        )

        assert "<system>" in transcript
        assert "security analyst" in transcript
        assert "<user>" in transcript
        assert "Analyze the incident" in transcript
        assert "<assistant>" in transcript
        assert "analyze the logs" in transcript
        assert "<tool" in transcript
        assert "bash" in transcript
        assert "<tool_result" in transcript
        assert "root logged in" in transcript

    def test_format_transcript_with_intent(self):
        """Test transcript formatting with intent (GPT-5 style)."""
        from saber.inspect_ai.agents.registry.copilot import (
            TurnCapture,
            AssistantMessageGroup,
        )

        turn = TurnCapture()
        group = AssistantMessageGroup(message_id="msg-1")
        # Note: The [Intent] prefix is added by EventCapture._handle_assistant_message
        # when processing report_intent tool calls
        group.intent = "[Intent] I will check the authentication logs for suspicious activity."
        group.text = "Let me investigate."
        turn.message_groups.append(group)

        transcript = turn.format_transcript()

        assert "[Intent]" in transcript
        assert "authentication logs" in transcript
        assert "Let me investigate" in transcript


class TestCopilotClientWrapperIntegration:
    """Test CopilotClientWrapper with mocked SDK client."""

    @pytest.fixture
    def mock_copilot_client(self):
        """Create a mock CopilotClient."""
        client = AsyncMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        client.create_session = AsyncMock()
        return client

    @pytest.fixture
    def mock_session(self):
        """Create a mock CopilotSession."""
        session = AsyncMock()
        session.send = AsyncMock()
        session.on = Mock()
        session.destroy = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_client_wrapper_lifecycle(self, mock_copilot_client, mock_session):
        """Test CopilotClientWrapper start/stop lifecycle."""
        from saber.inspect_ai.agents.registry.copilot import CopilotClientWrapper

        mock_copilot_client.create_session.return_value = mock_session

        with patch(
            "saber.inspect_ai.agents.registry.copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            wrapper = CopilotClientWrapper({"auto_start": False})
            await wrapper.start()

            mock_copilot_client.start.assert_called_once()

            await wrapper.stop()
            mock_copilot_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_wrapper_create_session(self, mock_copilot_client, mock_session):
        """Test creating a session through the wrapper."""
        from saber.inspect_ai.agents.registry.copilot import CopilotClientWrapper

        mock_copilot_client.create_session.return_value = mock_session

        with patch(
            "saber.inspect_ai.agents.registry.copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            wrapper = CopilotClientWrapper({})
            await wrapper.start()

            session_config = {
                "model": "gpt-5",
                "tools": [],
                "systemMessage": {"content": "Test system", "mode": "append"},
            }
            session = await wrapper.create_session(session_config)

            mock_copilot_client.create_session.assert_called_once()
            assert session == mock_session


class TestSubmissionDetectionIntegration:
    """Test submission detection in the solver loop context."""

    def test_detect_submission_in_tool_calls(self):
        """Test that we can detect submission from TurnCapture tool_calls."""
        from saber.inspect_ai.agents.registry.copilot import (
            TurnCapture,
            AssistantMessageGroup,
            CapturedMessage,
        )

        turn = TurnCapture()
        group = AssistantMessageGroup(message_id="msg-1")
        group.tool_calls.append(
            CapturedMessage(
                message_id="msg-1",
                content="",
                message_type="tool_call",
                tool_call_id="call-submit",
                tool_name="submit",
                tool_arguments={"answer": "The attacker IP is 10.0.0.5"},
            )
        )
        turn.message_groups.append(group)

        # This is the same logic used in copilot_solver
        submitted = False
        answer = None
        for tc in turn.tool_calls:
            if tc.tool_name == "submit":
                answer = (tc.tool_arguments or {}).get("answer", "")
                if answer:
                    submitted = True
                    break

        assert submitted is True
        assert answer == "The attacker IP is 10.0.0.5"

    def test_no_submission_when_no_submit_tool(self):
        """Test that no submission is detected without submit tool call."""
        from saber.inspect_ai.agents.registry.copilot import (
            TurnCapture,
            AssistantMessageGroup,
            CapturedMessage,
        )

        turn = TurnCapture()
        group = AssistantMessageGroup(message_id="msg-1")
        group.tool_calls.append(
            CapturedMessage(
                message_id="msg-1",
                content="",
                message_type="tool_call",
                tool_call_id="call-1",
                tool_name="bash",
                tool_arguments={"command": "ls"},
            )
        )
        turn.message_groups.append(group)

        submitted = False
        for tc in turn.tool_calls:
            if tc.tool_name == "submit":
                submitted = True
                break

        assert submitted is False


class TestEventCaptureSessionAttachment:
    """Test EventCapture attachment to sessions."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock session with on() method."""
        session = Mock()
        session.on = Mock(return_value=Mock())  # Returns unsubscribe function
        return session

    def test_attach_registers_event_handler(self, mock_session):
        """Test that attach() registers an event handler on the session."""
        from saber.inspect_ai.agents.registry.copilot import EventCapture

        capture = EventCapture()
        capture.attach(mock_session)

        mock_session.on.assert_called_once()
        # First arg should be the event type pattern (e.g., "*" or specific events)
        call_args = mock_session.on.call_args
        assert call_args is not None

    def test_detach_unsubscribes(self, mock_session):
        """Test that detach() calls the unsubscribe function."""
        from saber.inspect_ai.agents.registry.copilot import EventCapture

        unsubscribe = Mock()
        mock_session.on.return_value = unsubscribe

        capture = EventCapture()
        capture.attach(mock_session)
        capture.detach()

        unsubscribe.assert_called_once()


class TestAssistantMessageGroupBehavior:
    """Test AssistantMessageGroup data structure behavior."""

    def test_all_results_received_empty_tools(self):
        """Test all_results_received with no tool calls."""
        from saber.inspect_ai.agents.registry.copilot import AssistantMessageGroup

        group = AssistantMessageGroup(message_id="msg-1")
        assert group.all_results_received() is True

    def test_all_results_received_pending(self):
        """Test all_results_received with pending results."""
        from saber.inspect_ai.agents.registry.copilot import (
            AssistantMessageGroup,
            CapturedMessage,
        )

        group = AssistantMessageGroup(message_id="msg-1")
        group.tool_calls.append(
            CapturedMessage(
                message_id="msg-1",
                content="",
                message_type="tool_call",
                tool_call_id="call-1",
                tool_name="bash",
                tool_arguments={},
            )
        )

        # No result yet
        assert group.all_results_received() is False

        # Add result
        group.tool_results["call-1"] = "output"
        assert group.all_results_received() is True

    def test_multiple_tool_calls_partial_results(self):
        """Test with multiple tool calls and partial results."""
        from saber.inspect_ai.agents.registry.copilot import (
            AssistantMessageGroup,
            CapturedMessage,
        )

        group = AssistantMessageGroup(message_id="msg-1")
        group.tool_calls.append(
            CapturedMessage("msg-1", "", "tool_call", "call-1", "bash", {})
        )
        group.tool_calls.append(
            CapturedMessage("msg-1", "", "tool_call", "call-2", "read_file", {})
        )

        # Only one result
        group.tool_results["call-1"] = "output1"
        assert group.all_results_received() is False

        # Both results
        group.tool_results["call-2"] = "output2"
        assert group.all_results_received() is True
