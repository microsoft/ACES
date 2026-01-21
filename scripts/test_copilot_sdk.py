#!/usr/bin/env python3
"""Test script to exercise Copilot SDK APIs and understand their behavior.

This script tests the key APIs we use in the copilot agent:
1. Session creation with tools
2. Event handling (capturing assistant messages/reasoning)
3. send_and_wait() response structure
4. get_messages() for conversation history
5. RAW message retrieval during execution (bypassing broken parsing)

Run with:
    cd external/saber
    uv run python scripts/test_copilot_sdk.py
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from copilot import CopilotClient, Tool
from copilot.generated.session_events import SessionEventType


async def get_raw_messages(session: Any) -> list[dict]:
    """Get raw messages from session bypassing broken SDK parsing.

    This is a workaround for the SDK bug where session_event_from_dict()
    fails on the 'context' field parsing.
    """
    try:
        raw_response = await session._client.request(
            "session.getMessages",
            {"sessionId": session.session_id}
        )
        return raw_response.get("events", [])
    except Exception as e:
        print(f"  ERROR getting raw messages: {e}")
        return []


def extract_assistant_messages(raw_events: list[dict]) -> list[dict]:
    """Extract assistant messages with content or tool requests from raw events."""
    messages = []
    for evt in raw_events:
        if evt.get("type") == "assistant.message":
            data = evt.get("data", {})
            content = data.get("content", "")
            tool_requests = data.get("toolRequests", [])
            message_id = data.get("messageId", "")

            if content or tool_requests:
                messages.append({
                    "message_id": message_id,
                    "content": content,
                    "tool_requests": tool_requests,
                })
    return messages


def create_test_tool() -> Tool:
    """Create a simple test tool."""
    async def echo_handler(invocation: dict[str, Any]) -> str:
        """Echo the input back.

        The invocation dict contains:
        - session_id: str
        - tool_call_id: str
        - tool_name: str
        - arguments: dict[str, Any]  <-- The actual tool arguments are nested here
        """
        print(f"  [TOOL INVOCATION] {invocation}")

        # Extract the actual arguments from the invocation
        arguments = invocation.get("arguments", {})
        message = arguments.get("message", "no message provided")

        result = json.dumps({"echoed": message, "timestamp": datetime.now().isoformat()})
        print(f"  [TOOL RESULT] {result}")
        return result

    return Tool(
        name="echo",
        description="Echo a message back. Use this to test tool calling.",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back"
                }
            },
            "required": ["message"]
        },
        handler=echo_handler,
    )


async def main():
    print("=" * 60)
    print("Copilot SDK API Test Script")
    print("=" * 60)

    # Check for required environment variables
    azure_base_url = os.environ.get("AZUREAI_OPENAI_BASE_URL") or os.environ.get("AZURE_OPENAI_BASE_URL")
    azure_api_key = os.environ.get("AZUREAI_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")

    if not azure_base_url or not azure_api_key:
        print("ERROR: Missing Azure OpenAI credentials in environment")
        print("  Set AZUREAI_OPENAI_BASE_URL and AZUREAI_OPENAI_API_KEY")
        return

    # The base URL needs to include the deployment path for Azure
    # Format: https://<resource>.openai.azure.com/openai/deployments/<deployment>
    deployment_name = "gpt-4o"  # Adjust as needed
    if "/openai/deployments/" not in azure_base_url:
        azure_deployment_url = f"{azure_base_url.rstrip('/')}/openai/deployments/{deployment_name}"
    else:
        azure_deployment_url = azure_base_url

    print(f"\nUsing Azure endpoint: {azure_base_url[:50]}...")
    print(f"Deployment URL: {azure_deployment_url[:60]}...")

    # Create client
    print("\n1. Creating CopilotClient...")
    client = CopilotClient({"auto_start": True})
    await client.start()
    print("   Client started successfully")

    # Create tools
    print("\n2. Creating tools...")
    tools = [create_test_tool()]
    print(f"   Created {len(tools)} tool(s): {[t.name for t in tools]}")

    # Build session config
    print("\n3. Building session config...")
    session_config = {
        "model": deployment_name,  # Model/deployment name
        "systemMessage": {
            "mode": "append",
            "content": "You are a helpful assistant. When asked to echo something, use the echo tool.",
        },
        "availableTools": [t.name for t in tools],  # Restrict to only our tools
        "tools": tools,
        "streaming": True,
        "provider": {
            "type": "azure",
            "base_url": azure_deployment_url,  # Use snake_case
            "api_key": azure_api_key,
            "azure": {
                "api_version": "2024-12-01-preview",
            },
        },
    }
    print(f"   Config: model={session_config['model']}, tools={session_config['availableTools']}")

    # Create session
    print("\n4. Creating session...")
    session = await client.create_session(session_config)
    print("   Session created successfully")

    # Track events
    events_received: list[dict] = []
    assistant_content_from_events: list[str] = []

    def on_event(event: Any) -> None:
        """Capture all events."""
        if not event:
            return

        event_type = str(event.type.value) if hasattr(event.type, "value") else str(event.type)

        event_info = {
            "type": event_type,
            "has_data": hasattr(event, "data") and event.data is not None,
            "timestamp": datetime.now().isoformat(),
        }

        # Extract content fields if present
        if hasattr(event, "data") and event.data:
            data = event.data
            for field in ["content", "delta_content", "summary", "transformed_content", "message"]:
                val = getattr(data, field, None)
                if val:
                    event_info[field] = val[:100] + "..." if len(val) > 100 else val

        events_received.append(event_info)

        # Capture assistant content
        if event_type == "assistant.message":
            content = getattr(event.data, "content", None) if event.data else None
            if content:
                assistant_content_from_events.append(content)
                print(f"   [EVENT] assistant.message: {content[:100]}...")
        elif event_type == "assistant.message_delta":
            delta = getattr(event.data, "delta_content", None) if event.data else None
            if delta:
                print(f"   [EVENT] assistant.message_delta: {delta[:50]}...")
        elif event_type == "assistant.reasoning":
            content = getattr(event.data, "content", None) if event.data else None
            if content:
                print(f"   [EVENT] assistant.reasoning: {content[:100]}...")
        elif event_type in ("assistant.turn_start", "assistant.turn_end"):
            print(f"   [EVENT] {event_type}")
        elif event_type.startswith("tool."):
            print(f"   [EVENT] {event_type}")

    # Register event handler
    print("\n5. Registering event handler...")
    unsubscribe = session.on(on_event)
    print("   Event handler registered")

    # Test 1: Simple message
    print("\n" + "=" * 60)
    print("TEST 1: Simple message (no tool use)")
    print("=" * 60)

    prompt1 = "What is 2 + 2? Answer briefly."
    print(f"\nSending: {prompt1}")
    print("\nEvents during send_and_wait:")

    events_received.clear()
    response1 = await session.send_and_wait({"prompt": prompt1}, timeout=30)

    print(f"\nResponse object:")
    if response1:
        resp_type = str(response1.type.value) if hasattr(response1.type, "value") else str(response1.type)
        print(f"  type: {resp_type}")
        if hasattr(response1, "data") and response1.data:
            print(f"  data.content: {getattr(response1.data, 'content', None)}")
            print(f"  data.summary: {getattr(response1.data, 'summary', None)}")
    else:
        print("  Response is None")

    print(f"\nTotal events received: {len(events_received)}")
    print("Event types:", [e["type"] for e in events_received])

    # Get raw messages after turn 1
    print("\n--- RAW MESSAGES AFTER TURN 1 ---")
    raw_msgs_1 = await get_raw_messages(session)
    assistant_msgs_1 = extract_assistant_messages(raw_msgs_1)
    print(f"Total raw events: {len(raw_msgs_1)}")
    print(f"Assistant messages: {len(assistant_msgs_1)}")
    for i, msg in enumerate(assistant_msgs_1):
        content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        tool_info = f" + {len(msg['tool_requests'])} tool calls" if msg["tool_requests"] else ""
        print(f"  [{i}] content={repr(content_preview)}{tool_info}")

    # Track message count for incremental comparison
    prev_assistant_count = len(assistant_msgs_1)

    # Test 2: Tool use
    print("\n" + "=" * 60)
    print("TEST 2: Message requiring tool use")
    print("=" * 60)

    prompt2 = "Please use the echo tool to echo 'Hello World'"
    print(f"\nSending: {prompt2}")
    print("\nEvents during send_and_wait:")

    events_received.clear()
    response2 = await session.send_and_wait({"prompt": prompt2}, timeout=60)

    print(f"\nResponse object:")
    if response2:
        resp_type = str(response2.type.value) if hasattr(response2.type, "value") else str(response2.type)
        print(f"  type: {resp_type}")
        if hasattr(response2, "data") and response2.data:
            print(f"  data.content: {getattr(response2.data, 'content', None)}")
    else:
        print("  Response is None")

    print(f"\nTotal events received: {len(events_received)}")
    print("Event types:", [e["type"] for e in events_received])

    # Get raw messages after turn 2 (tool use)
    print("\n--- RAW MESSAGES AFTER TURN 2 (tool use) ---")
    raw_msgs_2 = await get_raw_messages(session)
    assistant_msgs_2 = extract_assistant_messages(raw_msgs_2)
    print(f"Total raw events: {len(raw_msgs_2)}")
    print(f"Assistant messages: {len(assistant_msgs_2)} (new: {len(assistant_msgs_2) - prev_assistant_count})")

    # Show only new messages
    new_msgs = assistant_msgs_2[prev_assistant_count:]
    for i, msg in enumerate(new_msgs):
        content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        if msg["tool_requests"]:
            tools = [f"{tr['name']}({tr['arguments']})" for tr in msg["tool_requests"]]
            print(f"  [NEW {prev_assistant_count + i}] TOOL CALL: {tools}")
        else:
            print(f"  [NEW {prev_assistant_count + i}] content={repr(content_preview)}")

    prev_assistant_count = len(assistant_msgs_2)

    # Test 3: Second turn (to verify message tracking)
    print("\n" + "=" * 60)
    print("TEST 3: Follow-up message (verify incremental tracking)")
    print("=" * 60)

    prompt3 = "What was my first question?"
    print(f"\nSending: {prompt3}")

    events_received.clear()
    response3 = await session.send_and_wait({"prompt": prompt3}, timeout=30)

    # Get raw messages after turn 3
    print("\n--- RAW MESSAGES AFTER TURN 3 ---")
    raw_msgs_3 = await get_raw_messages(session)
    assistant_msgs_3 = extract_assistant_messages(raw_msgs_3)
    print(f"Total raw events: {len(raw_msgs_3)}")
    print(f"Assistant messages: {len(assistant_msgs_3)} (new: {len(assistant_msgs_3) - prev_assistant_count})")

    # Show only new messages
    new_msgs = assistant_msgs_3[prev_assistant_count:]
    for i, msg in enumerate(new_msgs):
        content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        if msg["tool_requests"]:
            tools = [f"{tr['name']}({tr['arguments']})" for tr in msg["tool_requests"]]
            print(f"  [NEW {prev_assistant_count + i}] TOOL CALL: {tools}")
        else:
            print(f"  [NEW {prev_assistant_count + i}] content={repr(content_preview)}")

    # Test 4 removed - we're testing raw messages after each turn now

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Full conversation from raw messages")
    print("=" * 60)

    all_assistant_msgs = extract_assistant_messages(raw_msgs_3)
    print(f"\nTotal assistant messages: {len(all_assistant_msgs)}")
    for i, msg in enumerate(all_assistant_msgs):
        if msg["tool_requests"]:
            tools = [f"{tr['name']}({tr['arguments']})" for tr in msg["tool_requests"]]
            print(f"  [{i}] TOOL CALL: {tools}")
        else:
            content = msg["content"]
            print(f"  [{i}] TEXT: {content[:100]}{'...' if len(content) > 100 else ''}")

    # Also show user messages
    print("\nUser messages from raw events:")
    for evt in raw_msgs_3:
        if evt.get("type") == "user.message":
            content = evt.get("data", {}).get("content", "")
            print(f"  - {content[:100]}{'...' if len(content) > 100 else ''}")
    print("   Done!")


if __name__ == "__main__":
    asyncio.run(main())
