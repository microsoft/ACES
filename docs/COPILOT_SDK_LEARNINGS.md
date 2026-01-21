# Copilot SDK Integration Learnings

**Last Updated:** January 21, 2026
**SDK Version:** 0.0.388
**Status:** Partially working with known workarounds needed

## Overview

This document captures learnings from integrating the GitHub Copilot CLI SDK into SABER for agent benchmarking. The SDK has several bugs and limitations that require workarounds.

## Key Findings

### 1. `session.get_messages()` is BROKEN

The official SDK method `session.get_messages()` fails with an assertion error:

```python
# This fails in SDK 0.0.388
messages = await session.get_messages()  # AssertionError in session_event_from_dict()
```

**Root Cause:** The `session_event_from_dict()` function in `copilot/generated/session_events.py` cannot parse the `context` field when it's a dict (it expects string or None):

```
File "copilot/generated/session_events.py", line 388, in from_dict
    context = from_union([from_str, from_none], obj.get("context"))
AssertionError
```

**Workaround:** Make raw JSON-RPC requests directly, bypassing the broken parsing:

```python
async def _get_raw_session_messages(session, timeout: float = 5.0) -> list[dict]:
    """Get raw messages bypassing broken SDK parsing."""
    try:
        raw_response = await asyncio.wait_for(
            session._client.request(
                "session.getMessages",
                {"sessionId": session.session_id}
            ),
            timeout=timeout
        )
        return raw_response.get("events", [])
    except Exception:
        return []
```

### 2. Assistant Messages Have NO Content When Calling Tools

When the model decides to call tools, the `assistant.message` event contains:
- `message_id`: A UUID
- `content`: **Empty string** `""`
- `toolRequests`: Array of tool call objects

There is **NO separate reasoning/thinking text** - this is standard OpenAI API behavior. The tool call IS the assistant's message.

**Example from raw events:**
```json
{
  "type": "assistant.message",
  "data": {
    "messageId": "440fcc05-5467-4628-8108-02d4457021de",
    "content": "",
    "toolRequests": [
      {
        "toolCallId": "call_5wvlUs3058Yd6AoEY8WjslPQ",
        "name": "echo",
        "arguments": {"message": "Hello World"},
        "type": "function"
      }
    ]
  }
}
```

### 3. Raw Message Retrieval Times Out AFTER `send_and_wait()`

The session becomes unresponsive after `send_and_wait()` returns (after `session.idle` event). Attempting to call `_get_raw_session_messages()` at that point will timeout.

**Timing Issue:**
```
21:38:43 - session.idle received, send_and_wait returned
21:39:26 - Timeout (15.0s) getting raw session messages
```

**Solution Approaches (not yet implemented):**
1. **Background poller:** Run a background task that periodically calls `_get_raw_session_messages()` WHILE `send_and_wait()` is executing
2. **Event-based capture:** Rely entirely on the event handler to capture content (currently working but may miss some data)

### 4. Events DO Fire Correctly

Despite the `get_messages()` bug, the event system works. Register a handler with `session.on()`:

```python
def on_session_event(event):
    if event.type == SessionEventType.ASSISTANT_MESSAGE:
        content = getattr(event.data, 'content', None)
        if content:
            # Text response - capture it
            captured_messages.append(content)
        # Tool calls have no content - toolRequests are in event.data
```

**Event types we handle:**
- `assistant.message` - Final message (with content OR toolRequests)
- `assistant.message_delta` - Streaming chunks
- `assistant.turn_start` / `assistant.turn_end` - Turn boundaries
- `tool.execution_start` / `tool.execution_complete` - Tool execution
- `session.idle` - Agent finished processing

### 5. Tool Invocation Structure

When registering tools, the handler receives an invocation dict with **nested arguments**:

```python
async def tool_handler(invocation: dict) -> str:
    # invocation = {
    #     'session_id': '...',
    #     'tool_call_id': 'call_XXX',
    #     'tool_name': 'bash',
    #     'arguments': {'command': 'ls -la'}  # <-- Actual args nested here
    # }
    actual_args = invocation.get("arguments", {})
    command = actual_args.get("command")
```

## Current Implementation Status

### Working ✅
- Session creation with BYOK (Azure OpenAI)
- Tool registration and execution
- Event-based content capture for text responses
- Tool call tracking via `ToolCallTracker`
- Submit answer functionality

### Partially Working ⚠️
- Raw message retrieval (works during execution, times out after)
- Full transcript capture (missing tool-calling turn reasoning because it doesn't exist)

### Not Working ❌
- `session.get_messages()` - SDK bug
- Getting "reasoning" for tool-calling turns - doesn't exist in API

## Test Script

A test script exists at `scripts/test_copilot_sdk.py` that demonstrates:
1. Session creation with tools
2. Event handling
3. Raw message retrieval (working when called between turns)
4. The structure of assistant messages

Run with:
```bash
cd external/saber
uv run python scripts/test_copilot_sdk.py
```

## Files Modified

- `src/saber/inspect_ai/agents/registry/copilot.py` - Main agent implementation
  - Added `_get_raw_session_messages()` workaround function
  - Added extensive event handling
  - Added `ToolCallTracker` for capturing tool calls

- `scripts/test_copilot_sdk.py` - Test script for SDK behavior

## Recommended Next Steps

1. **Implement background poller:** Create an async task that polls `_get_raw_session_messages()` every 2-3 seconds DURING `send_and_wait()` execution, caching results. This ensures we capture raw messages while the session is still responsive.

2. **Report SDK bug:** File an issue with the Copilot SDK team about the `session_event_from_dict()` parsing failure on the `context` field.

3. **Accept limitation:** Document that tool-calling turns will not have reasoning text in the transcript - this is a fundamental limitation of the OpenAI API, not something we can work around.

4. **Consider hybrid approach:** Use events for real-time capture + raw messages for validation/backup when available.

## Architecture Notes

The Copilot agent flow:
```
1. Create CopilotClient
2. Create session with tools, BYOK provider config
3. Register event handler (on_session_event)
4. Loop:
   a. send_and_wait(prompt)
   b. Events fire during execution (tool calls, messages)
   c. Response returned when session.idle
   d. Process captured content
   e. Build state.messages for Inspect AI
5. Cleanup session
```

The key insight is that the session is only responsive DURING step 4a-c. After `send_and_wait()` returns, the session may be unresponsive for additional queries.
