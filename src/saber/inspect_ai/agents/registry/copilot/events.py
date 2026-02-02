"""Copilot events.jsonl parser.

This module provides Pydantic models and utilities for parsing the Copilot SDK's
events.jsonl file, which contains a reliable record of all session events.

Key components:
- Pydantic event models with discriminated union for type-safe parsing
- SessionEventLog class for reading and parsing events.jsonl files
- events_to_chat_messages() for converting events to Inspect AI ChatMessage format

Usage:
    log = SessionEventLog(session_id="abc123")
    events = log.read_events()
    messages = events_to_chat_messages(events, system_content="You are a helpful assistant.")
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._chat_message import ToolCall
from pydantic import BaseModel, ConfigDict, Field, JsonValue

if TYPE_CHECKING:
    from inspect_ai.event._tool import ToolEvent

from .....logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


# =============================================================================
# Event Data Models
# =============================================================================


def _parse_event_timestamp(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None

    normalized = timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        logger.debug("Failed to parse event timestamp", extra={"timestamp": timestamp})
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed


def _extract_result_content(result: Mapping[str, JsonValue]) -> str:
    content = result.get("content")
    if content is None:
        return ""
    return str(content)


class ToolExecutionRecord(BaseModel):
    """Normalized tool execution record from Copilot events."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str | None = None
    arguments: Mapping[str, JsonValue] = Field(default_factory=dict)
    started: datetime | None = None
    completed: datetime | None = None
    success: bool | None = None
    result: Mapping[str, JsonValue] = Field(default_factory=dict)


class ToolRequest(BaseModel):
    """Tool call request from assistant."""

    model_config = ConfigDict(frozen=True)

    toolCallId: str
    name: str
    arguments: Mapping[str, JsonValue] = Field(default_factory=dict)
    type: Literal["function"] = "function"
    description: str | None = None  # Optional description of what the tool call does


class SessionStartData(BaseModel):
    """Data for session.start event."""

    model_config = ConfigDict(frozen=True)

    sessionId: str


class UserMessageData(BaseModel):
    """Data for user.message event."""

    model_config = ConfigDict(frozen=True)

    content: str


class AssistantMessageData(BaseModel):
    """Data for assistant.message event."""

    model_config = ConfigDict(frozen=True)

    messageId: str | None = None
    content: str = ""
    toolRequests: list[ToolRequest] = Field(default_factory=list)


class ToolExecutionStartData(BaseModel):
    """Data for tool.execution_start event."""

    model_config = ConfigDict(frozen=True)

    toolCallId: str
    toolName: str
    arguments: Mapping[str, JsonValue] = Field(default_factory=dict)


class ToolExecutionCompleteData(BaseModel):
    """Data for tool.execution_complete event."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    toolCallId: str
    success: bool
    result: Mapping[str, JsonValue] = Field(default_factory=dict)


class TurnData(BaseModel):
    """Data for turn_start/turn_end events."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    turnId: str


class AssistantTurnEndData(BaseModel):
    """Data for assistant.turn_end event."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    turnId: str | None = None
    summary: str | None = None
    content: str | None = None


# =============================================================================
# Event Models
# =============================================================================


class SessionStartEvent(BaseModel):
    """Session start event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["session.start"]
    timestamp: str | None = None
    data: SessionStartData


class UserMessageEvent(BaseModel):
    """User message event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["user.message"]
    timestamp: str | None = None
    data: UserMessageData


class AssistantMessageEvent(BaseModel):
    """Assistant message event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["assistant.message"]
    timestamp: str | None = None
    data: AssistantMessageData


class AssistantTurnStartEvent(BaseModel):
    """Assistant turn start event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["assistant.turn_start"]
    timestamp: str | None = None
    data: TurnData


class AssistantTurnEndEvent(BaseModel):
    """Assistant turn end event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["assistant.turn_end"]
    timestamp: str | None = None
    data: AssistantTurnEndData | None = None


class ToolExecutionStartEvent(BaseModel):
    """Tool execution start event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["tool.execution_start"]
    timestamp: str | None = None
    data: ToolExecutionStartData


class ToolExecutionCompleteEvent(BaseModel):
    """Tool execution complete event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["tool.execution_complete"]
    timestamp: str | None = None
    data: ToolExecutionCompleteData


class SessionIdleEvent(BaseModel):
    """Session idle event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["session.idle"]
    timestamp: str | None = None
    data: None = None


class SessionErrorEvent(BaseModel):
    """Session error event."""

    model_config = ConfigDict(frozen=True)

    type: Literal["session.error"]
    timestamp: str | None = None
    data: Mapping[str, str] | None = None


class UnknownEvent(BaseModel):
    """Unknown event type - fallback for unrecognized events."""

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str
    timestamp: str | None = None
    data: JsonValue | None = None


# Discriminated union of all event types
CopilotEvent = Annotated[
    SessionStartEvent
    | UserMessageEvent
    | AssistantMessageEvent
    | AssistantTurnStartEvent
    | AssistantTurnEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionCompleteEvent
    | SessionIdleEvent
    | SessionErrorEvent
    | UnknownEvent,
    Field(discriminator="type"),
]


# =============================================================================
# Event Parsing
# =============================================================================


def parse_event(event_dict: dict[str, object]) -> CopilotEvent:
    """Parse a single event from a dictionary.

    Args:
        event_dict: Dictionary representing an event from events.jsonl

    Returns:
        Typed event object
    """
    event_type = event_dict.get("type", "")

    # Map event types to their model classes
    # Using a union type for values to satisfy mypy return type checking
    EventClass = (
        type[SessionStartEvent]
        | type[UserMessageEvent]
        | type[AssistantMessageEvent]
        | type[AssistantTurnStartEvent]
        | type[AssistantTurnEndEvent]
        | type[ToolExecutionStartEvent]
        | type[ToolExecutionCompleteEvent]
        | type[SessionIdleEvent]
        | type[SessionErrorEvent]
        | type[UnknownEvent]
    )
    event_models: dict[str, EventClass] = {
        "session.start": SessionStartEvent,
        "user.message": UserMessageEvent,
        "assistant.message": AssistantMessageEvent,
        "assistant.turn_start": AssistantTurnStartEvent,
        "assistant.turn_end": AssistantTurnEndEvent,
        "tool.execution_start": ToolExecutionStartEvent,
        "tool.execution_complete": ToolExecutionCompleteEvent,
        "session.idle": SessionIdleEvent,
        "session.error": SessionErrorEvent,
    }

    model_class = event_models.get(str(event_type), UnknownEvent)

    try:
        return model_class.model_validate(event_dict)
    except Exception as e:
        logger.debug(f"Failed to parse event as {model_class.__name__}: {e}")
        # Fall back to UnknownEvent for any parse errors
        return UnknownEvent.model_validate(event_dict)


# =============================================================================
# Session Event Log Reader
# =============================================================================


class SessionEventLog:
    """Reads and parses Copilot session events from events.jsonl.

    The Copilot SDK saves all session events to:
    ~/.copilot/session-state/<session-id>/events.jsonl

    This class provides methods to read and parse these events.
    """

    def __init__(self, session_id: str, base_path: Path | None = None) -> None:
        """Initialize the session event log reader.

        Args:
            session_id: The session ID to read events for
            base_path: Base path for session state files (defaults to ~/.copilot/session-state)
        """
        self.session_id = session_id
        self._base_path = base_path or Path.home() / ".copilot" / "session-state"
        self.events_path = self._base_path / session_id / "events.jsonl"

    def read_events(self) -> list[CopilotEvent]:
        """Read all events from the log file.

        Returns:
            List of parsed events, or empty list if file doesn't exist
        """
        if not self.events_path.exists():
            logger.debug(f"Events file not found: {self.events_path}")
            return []

        events: list[CopilotEvent] = []
        with open(self.events_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    event_dict = json.loads(line)
                    event = parse_event(event_dict)
                    events.append(event)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping corrupted line {line_num} in events.jsonl: {e}")
                    continue

        logger.info(f"Read {len(events)} events from {self.events_path}")
        return events

    def find_submission(self) -> str | None:
        """Find the submit tool call and return the answer.

        Returns:
            The submission answer, or None if no submit call was found
        """
        events = self.read_events()

        for event in events:
            if isinstance(event, AssistantMessageEvent):
                for tool_request in event.data.toolRequests:
                    if tool_request.name == "submit":
                        raw_answer = tool_request.arguments.get("answer", "")
                        if raw_answer:
                            answer = str(raw_answer)
                            logger.info(f"Found submission answer: {answer[:100]}...")
                            return answer

        return None


# =============================================================================
# ChatMessage Conversion
# =============================================================================


def events_to_chat_messages(
    events: list[CopilotEvent],
    system_content: str,
) -> list[ChatMessage]:
    """Convert parsed events to Inspect AI ChatMessage list.

    Args:
        events: List of parsed Copilot events
        system_content: Content for the system message

    Returns:
        List of ChatMessage objects in conversation order
    """
    messages: list[ChatMessage] = [ChatMessageSystem(content=system_content)]
    logger.debug(f"Starting events_to_chat_messages with {len(events)} events")

    for event in events:
        if isinstance(event, UserMessageEvent):
            messages.append(ChatMessageUser(content=event.data.content))
            logger.debug(f"Added UserMessage, content length: {len(event.data.content)}")

        elif isinstance(event, AssistantMessageEvent):
            # Build tool calls list
            tool_calls: list[ToolCall] | None = None
            if event.data.toolRequests:
                tool_calls = [
                    ToolCall(
                        id=tr.toolCallId,
                        function=tr.name,
                        arguments=dict(tr.arguments),  # Convert to mutable dict
                        type="function",
                    )
                    for tr in event.data.toolRequests
                ]

            # Use the assistant content, or fall back to tool request descriptions
            assistant_content = event.data.content
            if not assistant_content and event.data.toolRequests:
                descriptions = [tr.description for tr in event.data.toolRequests if tr.description]
                if descriptions:
                    assistant_content = "; ".join(descriptions)

            messages.append(
                ChatMessageAssistant(
                    content=assistant_content,
                    tool_calls=tool_calls,
                )
            )
            logger.debug(
                f"Added AssistantMessage, content length: {len(assistant_content)}, "
                f"tool_calls: {len(tool_calls) if tool_calls else 0}"
            )

        elif isinstance(event, ToolExecutionCompleteEvent):
            content = _extract_result_content(event.data.result)
            messages.append(
                ChatMessageTool(
                    tool_call_id=event.data.toolCallId,
                    content=content,
                )
            )
            logger.debug(f"Added ToolMessage for {event.data.toolCallId}, content length: {len(content)}")

    logger.info(
        f"Converted {len(events)} events to {len(messages)} ChatMessages",
        extra={
            "event_count": len(events),
            "message_count": len(messages),
            "message_types": {
                type(m).__name__: sum(1 for x in messages if type(x).__name__ == type(m).__name__) for m in messages
            },
        },
    )

    return messages


# =============================================================================
# Inspect AI ToolEvent Conversion
# =============================================================================


def _collect_tool_execution_records(events: list[CopilotEvent]) -> list[ToolExecutionRecord]:
    records_by_id: dict[str, ToolExecutionRecord] = {}
    ordered_ids: list[str] = []

    for event in events:
        if isinstance(event, AssistantMessageEvent) and event.data.toolRequests:
            for tool_request in event.data.toolRequests:
                tool_call_id = tool_request.toolCallId
                if tool_call_id not in records_by_id:
                    ordered_ids.append(tool_call_id)
                    records_by_id[tool_call_id] = ToolExecutionRecord(tool_call_id=tool_call_id)

                existing = records_by_id[tool_call_id]
                update: dict[str, object] = {}
                if existing.tool_name is None:
                    update["tool_name"] = tool_request.name
                if not existing.arguments:
                    update["arguments"] = tool_request.arguments
                if update:
                    records_by_id[tool_call_id] = existing.model_copy(update=update)

        elif isinstance(event, ToolExecutionStartEvent):
            tool_call_id = event.data.toolCallId
            if tool_call_id not in records_by_id:
                ordered_ids.append(tool_call_id)
                records_by_id[tool_call_id] = ToolExecutionRecord(tool_call_id=tool_call_id)

            existing = records_by_id[tool_call_id]
            records_by_id[tool_call_id] = existing.model_copy(
                update={
                    "tool_name": event.data.toolName,
                    "arguments": event.data.arguments,
                    "started": _parse_event_timestamp(event.timestamp),
                }
            )

        elif isinstance(event, ToolExecutionCompleteEvent):
            tool_call_id = event.data.toolCallId
            if tool_call_id not in records_by_id:
                ordered_ids.append(tool_call_id)
                records_by_id[tool_call_id] = ToolExecutionRecord(tool_call_id=tool_call_id)

            existing = records_by_id[tool_call_id]
            records_by_id[tool_call_id] = existing.model_copy(
                update={
                    "completed": _parse_event_timestamp(event.timestamp),
                    "success": event.data.success,
                    "result": event.data.result,
                }
            )

    return [records_by_id[tool_call_id] for tool_call_id in ordered_ids]


def events_to_tool_events(events: list[CopilotEvent]) -> list[ToolEvent]:
    """Convert Copilot tool events to Inspect AI ToolEvents and add to transcript.

    Args:
        events: List of parsed Copilot events from events.jsonl

    Returns:
        List of ToolEvent objects added to the transcript
    """
    from datetime import datetime

    from inspect_ai.event._tool import ToolEvent
    from inspect_ai.log._transcript import transcript
    from inspect_ai.model._chat_message import ToolCallError

    tool_events: list[ToolEvent] = []
    for record in _collect_tool_execution_records(events):
        content = _extract_result_content(record.result)
        error_obj = ToolCallError(type="unknown", message=content) if record.success is False else None

        started = record.started
        completed = record.completed or record.started
        timestamp = started or completed or datetime.now(timezone.utc)

        tool_event = ToolEvent(
            id=record.tool_call_id,
            function=record.tool_name or "unknown",
            arguments=dict(record.arguments),
            result=content,
            error=error_obj,
            timestamp=timestamp,
            completed=completed,
        )
        transcript()._event(tool_event)
        tool_events.append(tool_event)

    return tool_events


# =============================================================================
# Inspect AI ModelEvent Conversion
# =============================================================================


def events_to_model_events(
    events: list[CopilotEvent],
    model_name: str = "copilot",
    system_content: str = "",
) -> None:
    """Convert Copilot events to Inspect AI ModelEvents and add to transcript.

    This function creates synthetic ModelEvent objects from Copilot SDK events
    and adds them to the Inspect AI transcript, allowing them to appear in the
    Transcript tab of the Inspect viewer.

    The function groups events into "turns" where each turn consists of:
    - Input messages (accumulated user/system/tool messages)
    - A single assistant response (possibly with tool calls)

    Timestamps are taken directly from assistant.message events when present.
    If a timestamp is missing, it is adjusted to appear before corresponding
    tool execution events so sorting preserves conversational order.

    Args:
        events: List of parsed Copilot events from events.jsonl
        model_name: Name of the model to use in events (default: "copilot")
        system_content: System message content for the first turn
    """
    from datetime import datetime, timedelta

    from inspect_ai.event._model import ModelEvent
    from inspect_ai.log._transcript import transcript
    from inspect_ai.model._generate_config import GenerateConfig
    from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput

    logger.debug(f"Converting {len(events)} Copilot events to ModelEvents")

    trans = transcript()

    tool_records = _collect_tool_execution_records(events)
    tool_timestamps: list[datetime | None] = [record.started or record.completed for record in tool_records]
    tool_event_index = 0

    def _first_timestamp_in_range(start: int, end: int) -> datetime | None:
        for idx in range(start, end):
            timestamp = tool_timestamps[idx]
            if timestamp is not None:
                return timestamp
        return None

    def _last_timestamp_in_range(start: int, end: int) -> datetime | None:
        for idx in range(end - 1, start - 1, -1):
            timestamp = tool_timestamps[idx]
            if timestamp is not None:
                return timestamp
        return None

    def _next_timestamp_from(start: int) -> datetime | None:
        for idx in range(start, len(tool_timestamps)):
            timestamp = tool_timestamps[idx]
            if timestamp is not None:
                return timestamp
        return None

    def _previous_timestamp_before(start: int) -> datetime | None:
        for idx in range(start - 1, -1, -1):
            timestamp = tool_timestamps[idx]
            if timestamp is not None:
                return timestamp
        return None

    known_tool_timestamps = sum(1 for ts in tool_timestamps if ts is not None)
    logger.debug(
        "Found %s tool execution timestamps in events.jsonl (%s total tool records)",
        known_tool_timestamps,
        len(tool_timestamps),
    )

    # Build conversation messages and track turns
    accumulated_input: list[ChatMessage] = []
    if system_content:
        accumulated_input.append(ChatMessageSystem(content=system_content))

    model_event_count = 0
    created_model_events: list[tuple[ModelEvent, int, bool]] = []  # (event, num_tool_calls, has_timestamp)
    last_tool_timestamp: datetime | None = None

    for event in events:
        if isinstance(event, UserMessageEvent):
            # Add user message to accumulated input
            accumulated_input.append(ChatMessageUser(content=event.data.content))

        elif isinstance(event, ToolExecutionCompleteEvent):
            content = _extract_result_content(event.data.result)
            # Add as tool message to input for next turn
            accumulated_input.append(
                ChatMessageTool(
                    tool_call_id=event.data.toolCallId,
                    content=content,
                )
            )

        elif isinstance(event, AssistantMessageEvent):
            # This is the assistant's response - create a ModelEvent

            # Build tool calls list
            tool_calls: list[ToolCall] | None = None
            num_tool_calls = 0
            if event.data.toolRequests:
                tool_calls = [
                    ToolCall(
                        id=tr.toolCallId,
                        function=tr.name,
                        arguments=dict(tr.arguments),
                        type="function",
                    )
                    for tr in event.data.toolRequests
                ]
                num_tool_calls = len(tool_calls)

            # Use the assistant content, or fall back to the first tool request's description
            # This provides meaningful context when the assistant doesn't include a text message
            assistant_content = event.data.content
            if not assistant_content and event.data.toolRequests:
                # Collect all descriptions from tool requests
                descriptions = [tr.description for tr in event.data.toolRequests if tr.description]
                if descriptions:
                    assistant_content = "; ".join(descriptions)

            # Create assistant message
            assistant_message = ChatMessageAssistant(
                content=assistant_content,
                tool_calls=tool_calls,
                model=model_name,
            )

            # Determine stop reason
            stop_reason = "tool_calls" if tool_calls else "stop"

            # Create ModelOutput
            output = ModelOutput(
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        message=assistant_message,
                        stop_reason=stop_reason,
                    )
                ],
            )

            parsed_timestamp = _parse_event_timestamp(event.timestamp)
            has_timestamp = parsed_timestamp is not None

            # Create ModelEvent with timestamp (or placeholder if missing)
            model_event = ModelEvent(
                model=model_name,
                role=None,
                input=list(accumulated_input),  # Copy current input
                tools=[],  # We don't track tool info
                tool_choice="auto",
                config=GenerateConfig(),  # Default config
                output=output,
                timestamp=parsed_timestamp or datetime.now(timezone.utc),
            )

            created_model_events.append((model_event, num_tool_calls, has_timestamp))
            model_event_count += 1

            logger.debug(
                f"Created ModelEvent #{model_event_count}: "
                f"input_msgs={len(accumulated_input)}, "
                f"tool_calls={num_tool_calls}"
            )

            # Add assistant message to accumulated input for next turn
            accumulated_input.append(assistant_message)

    # Now assign timestamps to model events that lacked timestamps
    # Process in order to maintain correct sequencing
    #
    # Strategy (fallback only):
    # 1. First model event should come BEFORE all tool events
    # 2. Model events WITH tool calls should come just BEFORE their corresponding tool events
    # 3. Model events WITHOUT tool calls should come AFTER the preceding tool events

    earliest_tool_timestamp = _next_timestamp_from(0)
    min_timestamp = datetime.min.replace(tzinfo=timezone.utc)

    for model_idx, (model_event, num_tool_calls, has_timestamp) in enumerate(created_model_events):
        tool_range_start = tool_event_index
        tool_range_end = tool_event_index + num_tool_calls
        if num_tool_calls > 0:
            range_last_timestamp = _last_timestamp_in_range(tool_range_start, tool_range_end)
            if range_last_timestamp is not None:
                last_tool_timestamp = range_last_timestamp
            tool_event_index = tool_range_end

        if has_timestamp:
            if num_tool_calls > 0:
                first_tool_timestamp = _first_timestamp_in_range(tool_range_start, tool_range_end)
                if first_tool_timestamp and model_event.timestamp and model_event.timestamp >= first_tool_timestamp:
                    model_event.timestamp = first_tool_timestamp - timedelta(milliseconds=1)
            ts_str = model_event.timestamp.isoformat() if model_event.timestamp else "None"
            logger.debug(f"Using Copilot timestamp for model event #{model_idx}: {ts_str}")
            continue

        if model_idx == 0:
            if earliest_tool_timestamp:
                model_event.timestamp = earliest_tool_timestamp - timedelta(milliseconds=10)
            else:
                model_event.timestamp = min_timestamp
        elif num_tool_calls > 0:
            first_tool_timestamp = _first_timestamp_in_range(tool_range_start, tool_range_end)
            if first_tool_timestamp:
                model_event.timestamp = first_tool_timestamp - timedelta(milliseconds=1)
            else:
                next_tool_timestamp = _next_timestamp_from(tool_range_end)
                prev_tool_timestamp = _previous_timestamp_before(tool_range_start)
                if next_tool_timestamp:
                    offset_ms = 1 + tool_range_start
                    model_event.timestamp = next_tool_timestamp - timedelta(milliseconds=offset_ms)
                elif prev_tool_timestamp:
                    model_event.timestamp = prev_tool_timestamp + timedelta(microseconds=500)
                else:
                    model_event.timestamp = min_timestamp + timedelta(microseconds=model_idx)
        elif num_tool_calls == 0:
            if last_tool_timestamp:
                model_event.timestamp = last_tool_timestamp + timedelta(microseconds=500)
            else:
                next_tool_timestamp = _next_timestamp_from(tool_event_index)
                if next_tool_timestamp:
                    model_event.timestamp = next_tool_timestamp - timedelta(milliseconds=2)
                else:
                    prev_tool_timestamp = _previous_timestamp_before(tool_event_index)
                    if prev_tool_timestamp:
                        model_event.timestamp = prev_tool_timestamp + timedelta(microseconds=500)
                    else:
                        model_event.timestamp = min_timestamp + timedelta(microseconds=model_idx)

        ts_str = model_event.timestamp.isoformat() if model_event.timestamp else "None"
        logger.debug(f"Set fallback timestamp for model event #{model_idx}: {ts_str}")

    # Add all model events to transcript
    for model_event, _, _ in created_model_events:
        trans._event(model_event)

    # Sort transcript events by timestamp to properly interleave model events with tool events
    if model_event_count > 0:
        try:
            # Sort events by timestamp
            trans._events.sort(key=lambda e: e.timestamp if e.timestamp else datetime.min)
            logger.debug(f"Sorted {len(trans._events)} transcript events by timestamp")
        except Exception as e:
            logger.warning(f"Failed to sort transcript events: {e}")

    logger.info(
        f"Created {model_event_count} ModelEvents from {len(events)} Copilot events",
        extra={
            "copilot_event_count": len(events),
            "model_event_count": model_event_count,
        },
    )


__all__ = [
    "CopilotEvent",
    "SessionStartEvent",
    "UserMessageEvent",
    "AssistantMessageEvent",
    "AssistantTurnStartEvent",
    "AssistantTurnEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionCompleteEvent",
    "SessionIdleEvent",
    "SessionErrorEvent",
    "UnknownEvent",
    "ToolRequest",
    "parse_event",
    "SessionEventLog",
    "events_to_chat_messages",
    "events_to_model_events",
    "events_to_tool_events",
]
