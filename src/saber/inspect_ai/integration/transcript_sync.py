"""Transcript synchronization for SABER agent execution.

This module handles pushing agent conversation transcripts back to the SABER
server for debugging and analysis purposes. Transcripts are synchronized after
each agent execution step.

Key features:
- Feature flag controlled (SABER_ENABLE_TRANSCRIPT_SYNC)
- Graceful degradation (failures don't crash agent execution)
- Retry logic with exponential backoff
- Payload size validation
- ChatMessage serialization for all message types
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict

import aiohttp
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.solver import TaskState

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys

logger = get_saber_logger(LogCategory.AGENT, __name__)


async def push_transcript_if_enabled(
    state: TaskState,
    metadata: Dict[str, Any],
    get_active_domain_func: Any,
) -> None:
    """Helper to push transcript if episode context is available.

    This function extracts episode context from metadata and active domain
    registry, then delegates to _push_transcript() if all required data is present.

    Args:
        state: TaskState after agent execution
        metadata: Sample metadata containing episode/session IDs and domain slug
        get_active_domain_func: Function to get active domain context

    Returns:
        None (gracefully handles missing context and errors)
    """
    try:
        # Extract episode context from metadata
        session_id = metadata.get(MetadataKeys.SESSION_ID)
        episode_id = metadata.get(MetadataKeys.EPISODE_ID)
        domain_slug = metadata.get(MetadataKeys.SABER_DOMAIN_SLUG)

        # Skip if episode context is missing
        if not session_id or not episode_id:
            return

        # Get REST URL from active domain registry
        if domain_slug:
            domain_context = get_active_domain_func(domain_slug)
            if domain_context:
                rest_url = domain_context.get("rest_url")
                if rest_url:
                    # Push transcript using extracted context
                    await _push_transcript(
                        state=state,
                        session_id=session_id,
                        episode_id=episode_id,
                        rest_url=rest_url,
                    )
    except Exception as e:
        # Log error but don't raise - transcript sync failures should not crash agent execution
        logger.warning(
            "Failed to push transcript, continuing with agent execution",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "episode_id": metadata.get(MetadataKeys.EPISODE_ID),
                "session_id": metadata.get(MetadataKeys.SESSION_ID),
            },
        )


def serialize_message(msg: ChatMessage) -> Dict[str, Any]:
    """Convert ChatMessage to JSON-safe dict.

    Args:
        msg: Inspect AI ChatMessage object (ChatMessageSystem, ChatMessageUser,
             ChatMessageAssistant, or ChatMessageTool)

    Returns:
        Dictionary with keys: role, content, tool_calls (optional),
        tool_call_id (optional), name (optional), reasoning (optional)

    Raises:
        ValueError: If message type is unknown or unsupported
    """
    result: Dict[str, Any] = {}

    # Extract role
    if isinstance(msg, ChatMessageSystem):
        result["role"] = "system"
    elif isinstance(msg, ChatMessageUser):
        result["role"] = "user"
    elif isinstance(msg, ChatMessageAssistant):
        result["role"] = "assistant"
    elif isinstance(msg, ChatMessageTool):
        result["role"] = "tool"
    else:
        raise ValueError(f"Unknown message type: {type(msg)}")

    # Extract content - handle both string and list of Content objects
    if isinstance(msg.content, str):
        result["content"] = msg.content
    elif isinstance(msg.content, list):
        # Extract text content parts and concatenate
        text_parts = []
        reasoning_text = None

        for content_item in msg.content:
            if isinstance(content_item, ContentText):
                text_parts.append(content_item.text)
            elif isinstance(content_item, ContentReasoning):
                reasoning_text = content_item.reasoning

        result["content"] = "".join(text_parts)

        # Add reasoning if present (for assistant messages)
        if reasoning_text:
            result["reasoning"] = reasoning_text
    else:
        result["content"] = ""

    # Handle assistant-specific fields
    if isinstance(msg, ChatMessageAssistant):
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": tc.function,
                    "arguments": tc.arguments,
                }
                for tc in msg.tool_calls
            ]

    # Handle tool message-specific fields
    if isinstance(msg, ChatMessageTool):
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if msg.function:
            result["name"] = msg.function

    return result


async def _push_transcript(
    state: TaskState,
    session_id: str,
    episode_id: str,
    rest_url: str,
) -> None:
    """Push current transcript to SABER server.

    Args:
        state: Current TaskState with messages
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")

    Returns:
        None (logs errors but doesn't raise exceptions)

    Side Effects:
        - Logs success/failure
        - Makes HTTP POST request to server
        - Retries on transient failures (max 3 attempts)
    """
    # Constants
    MAX_PAYLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB (matches server limit)
    REQUEST_TIMEOUT_SECONDS = 5.0
    MAX_RETRIES = 3
    RETRY_DELAYS_SECONDS = [0.5, 1.0, 2.0]
    FEATURE_FLAG_ENV_VAR = "SABER_ENABLE_TRANSCRIPT_SYNC"
    FEATURE_FLAG_DEFAULT = "true"

    # Check feature flag (evaluate at runtime, not module load time)
    if os.getenv(FEATURE_FLAG_ENV_VAR, FEATURE_FLAG_DEFAULT).lower() != "true":
        return

    # Validate rest_url format (defense against SSRF)
    if not rest_url.startswith(("http://", "https://")):
        logger.warning(
            "Invalid rest_url format - must start with http:// or https://",
            extra={
                "episode_id": episode_id,
                "rest_url": rest_url,
            },
        )
        return

    # Serialize all messages
    try:
        messages = [serialize_message(msg) for msg in state.messages]
    except Exception as e:
        logger.warning(
            "Failed to serialize transcript messages",
            extra={
                "episode_id": episode_id,
                "error": str(e),
            },
        )
        return

    # Prepare metadata
    metadata = {
        "step_number": len(state.messages),
        "timestamp": datetime.utcnow().isoformat(),
        "source": "inspect_ai",
    }

    # Prepare payload
    payload = {
        "messages": messages,
        "metadata": metadata,
    }

    # Check payload size before sending (client-side validation)
    try:
        payload_json = json.dumps(payload)
        payload_size = len(payload_json.encode("utf-8"))

        if payload_size > MAX_PAYLOAD_SIZE_BYTES:
            logger.warning(
                "Transcript payload too large, skipping push",
                extra={
                    "episode_id": episode_id,
                    "payload_size_bytes": payload_size,
                    "max_size_bytes": MAX_PAYLOAD_SIZE_BYTES,
                    "message_count": len(messages),
                },
            )
            return
    except Exception as e:
        logger.warning(
            "Failed to validate payload size",
            extra={
                "episode_id": episode_id,
                "error": str(e),
            },
        )
        return

    # Construct URL
    url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/transcript"

    # Retry configuration
    max_retries = MAX_RETRIES
    retry_delays = RETRY_DELAYS_SECONDS

    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        logger.info(
                            "Transcript pushed successfully",
                            extra={
                                "episode_id": episode_id,
                                "message_count": len(messages),
                                "response_status": resp.status,
                            },
                        )
                        return
                    elif resp.status == 422:
                        # Validation error - don't retry
                        error_text = await resp.json()
                        logger.warning(
                            "Failed to push transcript - validation error",
                            extra={
                                "episode_id": episode_id,
                                "status_code": resp.status,
                                "error": error_text,
                            },
                        )
                        return
                    else:
                        # Server error - retry
                        error_text = await resp.text()
                        logger.warning(
                            f"Failed to push transcript (attempt {attempt + 1}/{max_retries})",
                            extra={
                                "episode_id": episode_id,
                                "status_code": resp.status,
                                "error": error_text,
                            },
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delays[attempt])

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout pushing transcript (attempt {attempt + 1}/{max_retries})",
                extra={
                    "episode_id": episode_id,
                    "error": "Request timeout",
                },
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])

        except aiohttp.ClientError as e:
            logger.warning(
                f"Network error pushing transcript (attempt {attempt + 1}/{max_retries})",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])

        except Exception as e:
            logger.warning(
                f"Unexpected error pushing transcript (attempt {attempt + 1}/{max_retries})",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])

    # If we get here, all retries failed
    logger.error(
        "Failed to push transcript after all retry attempts",
        extra={
            "episode_id": episode_id,
            "max_retries": max_retries,
        },
    )


async def pull_injected_messages(
    state: TaskState,
    session_id: str,
    episode_id: str,
    rest_url: str,
) -> None:
    """Pull and inject pending messages from SABER server (Phase 4: Red team message injection).

    Retrieves pending messages from the server's injection queue and appends them to
    the agent's conversation state. This enables server-side message injection for
    adversarial testing and red team operations.

    Args:
        state: Current TaskState to inject messages into
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")

    Returns:
        None (modifies state.messages in-place, logs errors but doesn't raise exceptions)

    Side Effects:
        - Appends injected messages to state.messages
        - Logs injection events
        - Makes HTTP GET request to server
    """
    # Constants
    REQUEST_TIMEOUT_SECONDS = 5.0
    FEATURE_FLAG_ENV_VAR = "SABER_ENABLE_MESSAGE_INJECTION"
    FEATURE_FLAG_DEFAULT = "true"

    # Check feature flag
    if os.getenv(FEATURE_FLAG_ENV_VAR, FEATURE_FLAG_DEFAULT).lower() != "true":
        return

    # Validate rest_url format (defense against SSRF)
    if not rest_url.startswith(("http://", "https://")):
        logger.warning(
            "Invalid rest_url format for message injection - must start with http:// or https://",
            extra={
                "episode_id": episode_id,
                "rest_url": rest_url,
            },
        )
        return

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject"
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

            async with session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    messages = data.get("messages", [])

                    # Deserialize and inject messages
                    for msg_data in messages:
                        try:
                            # Convert dict to ChatMessage object
                            chat_msg = _deserialize_message(msg_data)
                            state.messages.append(chat_msg)

                            logger.info(
                                "Injected message from server",
                                extra={
                                    "episode_id": episode_id,
                                    "role": msg_data.get("role"),
                                    "content_preview": msg_data.get("content", "")[:50],
                                },
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to deserialize injected message",
                                extra={
                                    "episode_id": episode_id,
                                    "error": str(e),
                                    "message_data": msg_data,
                                },
                            )

                    if messages:
                        logger.info(
                            f"Successfully pulled {len(messages)} injected message(s)",
                            extra={
                                "episode_id": episode_id,
                                "message_count": len(messages),
                            },
                        )

                elif resp.status == 404:
                    logger.debug(
                        "Episode not found when pulling injections (may have ended)",
                        extra={
                            "episode_id": episode_id,
                            "status_code": resp.status,
                        },
                    )
                else:
                    logger.warning(
                        "Failed to pull injected messages",
                        extra={
                            "episode_id": episode_id,
                            "status_code": resp.status,
                        },
                    )

    except asyncio.TimeoutError:
        logger.warning(
            "Timeout pulling injected messages",
            extra={
                "episode_id": episode_id,
                "error": "Request timeout",
            },
        )

    except aiohttp.ClientError as e:
        logger.warning(
            "Network error pulling injected messages",
            extra={
                "episode_id": episode_id,
                "error": str(e),
            },
        )

    except Exception as e:
        logger.warning(
            "Unexpected error pulling injected messages",
            extra={
                "episode_id": episode_id,
                "error": str(e),
            },
        )


def _deserialize_message(msg_data: Dict[str, Any]) -> ChatMessage:
    """Convert message dictionary to ChatMessage object.

    Inverse of serialize_message() - converts JSON-safe dict back to
    Inspect AI ChatMessage objects.

    Args:
        msg_data: Dictionary with keys: role, content, tool_calls (optional), etc.

    Returns:
        ChatMessage object (ChatMessageUser, ChatMessageAssistant, etc.)

    Raises:
        ValueError: If role is unknown or message structure is invalid
    """
    role = msg_data.get("role")
    content = msg_data.get("content", "")

    if role == "system":
        return ChatMessageSystem(content=content)

    elif role == "user":
        return ChatMessageUser(content=content)

    elif role == "assistant":
        # Assistant messages may have tool calls
        tool_calls_data = msg_data.get("tool_calls")
        if tool_calls_data:
            # Convert tool call dicts to ToolCall objects
            from inspect_ai.tool import ToolCall

            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    function=tc["function"],
                    arguments=tc["arguments"],
                    type="function",
                )
                for tc in tool_calls_data
            ]
            return ChatMessageAssistant(content=content, tool_calls=tool_calls)
        else:
            return ChatMessageAssistant(content=content)

    elif role == "tool":
        # Tool messages need tool_call_id and function name
        tool_call_id = msg_data.get("tool_call_id")
        function_name = msg_data.get("name")
        if not tool_call_id or not function_name:
            raise ValueError(f"Tool message missing tool_call_id or name: {msg_data}")
        return ChatMessageTool(
            content=content,
            tool_call_id=tool_call_id,
            function=function_name,
        )

    else:
        raise ValueError(f"Unknown message role: {role}")
