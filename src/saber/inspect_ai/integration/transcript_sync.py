"""Transcript synchronization for SABER agent execution.

This module handles pushing agent conversation transcripts back to the SABER
server for debugging and analysis purposes. Transcripts are synchronized after
each agent execution step.

Key features:
- Graceful degradation (failures don't crash agent execution)
- Retry logic with exponential backoff
- Payload size validation
- ChatMessage serialization for all message types
- Per-iteration transcript syncing via Generate wrapper
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.solver import Generate, TaskState

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ...models.rest.config import TranscriptSyncConfig

logger = get_saber_logger(LogCategory.AGENT, __name__)


def create_transcript_syncing_generate(
    original_generate: Generate,
    session_id: Optional[str],
    episode_id: Optional[str],
    rest_url: Optional[str],
) -> Generate:
    """Create a Generate wrapper that pushes transcript after each model call.

    This wrapper intercepts generate() calls and pushes the updated transcript
    to the SABER server after each model response, ensuring the server has
    fresh context before tool calls are executed.

    Args:
        original_generate: The original Generate callable to wrap
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")

    Returns:
        Wrapped Generate callable that pushes transcripts after each call

    Example:
        wrapped_generate = create_transcript_syncing_generate(
            original_generate=generate,
            session_id="session_123",
            episode_id="episode_456",
            rest_url="http://localhost:8000",
        )
        result = await wrapped_generate(state)
    """

    async def transcript_syncing_generate(
        state: TaskState,
        **kwargs: Any,
    ) -> TaskState:
        """Wrapped generate that pushes transcript after model response.

        Flow:
        1. Call original generate (model responds)
        2. Calculate delta (new messages since last push)
        3. Push ONLY new messages to server in APPEND mode (differential)
        4. Return result to agent

        This ensures server has latest assistant message before tools execute.
        Uses true differential sync to only push new messages since last call.
        """
        # Call original generate to get model response
        result = await original_generate(state, **kwargs)

        # Push transcript if all required context is available
        if session_id and episode_id and rest_url:
            try:
                # Get last pushed index from state store
                last_index = result.store.get("_last_transcript_index", 0)
                current_count = len(result.messages)

                # Only push if there are new messages
                if current_count > last_index:
                    # Create a temporary state with only new messages
                    new_messages = result.messages[last_index:]

                    # Use append mode for differential sync (only new messages)
                    await _push_transcript_delta(
                        messages=new_messages,
                        session_id=session_id,
                        episode_id=episode_id,
                        rest_url=rest_url,
                    )

                    # Update last pushed index
                    result.store.set("_last_transcript_index", current_count)

                    logger.debug(
                        "Pushed transcript delta",
                        extra={
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "new_messages": len(new_messages),
                            "total_messages": current_count,
                        },
                    )
            except Exception as e:
                # Log but don't crash - graceful degradation
                logger.warning(
                    "Failed to push transcript after generate, continuing execution",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "session_id": session_id,
                        "episode_id": episode_id,
                    },
                )

        return result

    return transcript_syncing_generate


async def push_transcript_if_enabled(
    state: TaskState,
    metadata: Dict[str, Any],
    get_active_domain_func: Any,
) -> None:
    """Helper to push transcript if episode context is available.

    This function extracts episode context from metadata and active domain
    registry, then delegates to _push_transcript() if all required data is present.
    Uses differential sync (append mode) by tracking last pushed message index.

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
                    # Use append mode for differential sync (only new messages since last push)
                    await _push_transcript(
                        state=state,
                        session_id=session_id,
                        episode_id=episode_id,
                        rest_url=rest_url,
                        mode="append",
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


async def _push_single_message(
    message: ChatMessage,
    session_id: str,
    episode_id: str,
    rest_url: str,
) -> None:
    """Push a single message to SABER server (used by model wrapper).

    This is a convenience wrapper around _push_transcript_delta for pushing
    a single message (typically an assistant message right after generation).

    Args:
        message: Single ChatMessage to push
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")

    Returns:
        None (logs errors but doesn't raise exceptions)
    """
    await _push_transcript_delta(
        messages=[message],
        session_id=session_id,
        episode_id=episode_id,
        rest_url=rest_url,
    )


async def _push_transcript_delta(
    messages: list[ChatMessage],
    session_id: str,
    episode_id: str,
    rest_url: str,
) -> None:
    """Push only new messages to SABER server in append mode.

    This is the differential sync implementation that only sends messages
    that haven't been pushed before.

    Args:
        messages: List of new ChatMessage objects to push
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")

    Returns:
        None (logs errors but doesn't raise exceptions)

    Side Effects:
        - Logs success/failure
        - Makes HTTP POST request to server in append mode
        - Retries on transient failures (max 3 attempts)
    """
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
        serialized_messages = [serialize_message(msg) for msg in messages]
    except Exception as e:
        logger.warning(
            "Failed to serialize transcript delta messages",
            extra={
                "episode_id": episode_id,
                "error": str(e),
            },
        )
        return

    # Prepare metadata
    metadata = {
        MetadataKeys.TRANSCRIPT_STEP_NUMBER: len(messages),
        MetadataKeys.TRANSCRIPT_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
        MetadataKeys.TRANSCRIPT_SOURCE: TranscriptSyncConfig.DEFAULT_SOURCE,
    }

    # Prepare payload with append mode
    payload = {
        "messages": serialized_messages,
        "mode": "append",
        "metadata": metadata,
    }

    # Check payload size before sending (client-side validation)
    try:
        payload_json = json.dumps(payload)
        payload_size = len(payload_json.encode("utf-8"))

        if payload_size > TranscriptSyncConfig.MAX_PAYLOAD_SIZE_BYTES:
            logger.warning(
                "Transcript delta payload too large, skipping push",
                extra={
                    "episode_id": episode_id,
                    "payload_size_bytes": payload_size,
                    "max_size_bytes": TranscriptSyncConfig.MAX_PAYLOAD_SIZE_BYTES,
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

    # Retry configuration - exponential backoff with unlimited retries
    attempt = 0
    retry_delay = TranscriptSyncConfig.INITIAL_RETRY_DELAY_SECONDS

    while True:
        try:
            timeout = aiohttp.ClientTimeout(total=TranscriptSyncConfig.REQUEST_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        logger.debug(
                            "Transcript delta pushed successfully",
                            extra={
                                "episode_id": episode_id,
                                "message_count": len(messages),
                                "attempts": attempt + 1,
                            },
                        )
                        return
                    elif resp.status == 422:
                        # Validation error - don't retry
                        error_text = await resp.json()
                        logger.warning(
                            "Failed to push transcript delta - validation error",
                            extra={
                                "episode_id": episode_id,
                                "status_code": resp.status,
                                "error": error_text,
                            },
                        )
                        return
                    else:
                        # Server error - retry with exponential backoff
                        error_text = await resp.text()
                        logger.warning(
                            f"Failed to push transcript delta (attempt {attempt + 1}), retrying in {retry_delay:.1f}s",
                            extra={
                                "episode_id": episode_id,
                                "status_code": resp.status,
                                "error": error_text,
                                "retry_delay": retry_delay,
                            },
                        )
                        await asyncio.sleep(retry_delay)
                        attempt += 1
                        retry_delay = min(
                            retry_delay * TranscriptSyncConfig.BACKOFF_MULTIPLIER,
                            TranscriptSyncConfig.MAX_RETRY_DELAY_SECONDS,
                        )

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout pushing transcript delta (attempt {attempt + 1}), retrying in {retry_delay:.1f}s",
                extra={
                    "episode_id": episode_id,
                    "error": "Request timeout",
                    "retry_delay": retry_delay,
                },
            )
            await asyncio.sleep(retry_delay)
            attempt += 1
            retry_delay = min(
                retry_delay * TranscriptSyncConfig.BACKOFF_MULTIPLIER,
                TranscriptSyncConfig.MAX_RETRY_DELAY_SECONDS,
            )

        except aiohttp.ClientError as e:
            logger.warning(
                f"Network error pushing transcript delta (attempt {attempt + 1}), retrying in {retry_delay:.1f}s",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                    "retry_delay": retry_delay,
                },
            )
            await asyncio.sleep(retry_delay)
            attempt += 1
            retry_delay = min(
                retry_delay * TranscriptSyncConfig.BACKOFF_MULTIPLIER,
                TranscriptSyncConfig.MAX_RETRY_DELAY_SECONDS,
            )

        except Exception as e:
            logger.warning(
                f"Unexpected error pushing transcript delta (attempt {attempt + 1}), retrying in {retry_delay:.1f}s",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                    "retry_delay": retry_delay,
                },
            )
            await asyncio.sleep(retry_delay)
            attempt += 1
            retry_delay = min(
                retry_delay * TranscriptSyncConfig.BACKOFF_MULTIPLIER,
                TranscriptSyncConfig.MAX_RETRY_DELAY_SECONDS,
            )


async def _push_transcript(
    state: TaskState,
    session_id: str,
    episode_id: str,
    rest_url: str,
    mode: str = "replace",
) -> None:
    """Push current transcript to SABER server.

    Args:
        state: Current TaskState with messages
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")
        mode: Push mode - 'replace' for full transcript, 'append' for differential

    Returns:
        None (logs errors but doesn't raise exceptions)

    Side Effects:
        - Logs success/failure
        - Makes HTTP POST request to server
        - Retries on transient failures (max 3 attempts)
    """
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
        MetadataKeys.TRANSCRIPT_STEP_NUMBER: len(state.messages),
        MetadataKeys.TRANSCRIPT_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
        MetadataKeys.TRANSCRIPT_SOURCE: TranscriptSyncConfig.DEFAULT_SOURCE,
    }

    # Prepare payload
    payload = {
        "messages": messages,
        "mode": mode,
        "metadata": metadata,
    }

    # Check payload size before sending (client-side validation)
    try:
        payload_json = json.dumps(payload)
        payload_size = len(payload_json.encode("utf-8"))

        if payload_size > TranscriptSyncConfig.MAX_PAYLOAD_SIZE_BYTES:
            logger.warning(
                "Transcript payload too large, skipping push",
                extra={
                    "episode_id": episode_id,
                    "payload_size_bytes": payload_size,
                    "max_size_bytes": TranscriptSyncConfig.MAX_PAYLOAD_SIZE_BYTES,
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

    # Retry configuration - exponential backoff with unlimited retries
    attempt = 0
    retry_delay = TranscriptSyncConfig.INITIAL_RETRY_DELAY_SECONDS

    while True:
        try:
            timeout = aiohttp.ClientTimeout(total=TranscriptSyncConfig.REQUEST_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        logger.debug(
                            "Transcript pushed successfully",
                            extra={
                                "episode_id": episode_id,
                                "message_count": len(messages),
                                "attempts": attempt + 1,
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
                        # Server error - retry with exponential backoff
                        error_text = await resp.text()
                        logger.warning(
                            f"Failed to push transcript (attempt {attempt + 1}), retrying in {retry_delay:.1f}s",
                            extra={
                                "episode_id": episode_id,
                                "status_code": resp.status,
                                "error": error_text,
                                "retry_delay": retry_delay,
                            },
                        )
                        await asyncio.sleep(retry_delay)
                        attempt += 1
                        retry_delay = min(
                            retry_delay * TranscriptSyncConfig.BACKOFF_MULTIPLIER,
                            TranscriptSyncConfig.MAX_RETRY_DELAY_SECONDS,
                        )

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout pushing transcript (attempt {attempt + 1}), retrying in {retry_delay:.1f}s",
                extra={
                    "episode_id": episode_id,
                    "error": "Request timeout",
                    "retry_delay": retry_delay,
                },
            )
            await asyncio.sleep(retry_delay)
            attempt += 1
            retry_delay = min(
                retry_delay * TranscriptSyncConfig.BACKOFF_MULTIPLIER,
                TranscriptSyncConfig.MAX_RETRY_DELAY_SECONDS,
            )

        except aiohttp.ClientError as e:
            logger.warning(
                f"Network error pushing transcript (attempt {attempt + 1}), retrying in {retry_delay:.1f}s",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                    "retry_delay": retry_delay,
                },
            )
            await asyncio.sleep(retry_delay)
            attempt += 1
            retry_delay = min(
                retry_delay * TranscriptSyncConfig.BACKOFF_MULTIPLIER,
                TranscriptSyncConfig.MAX_RETRY_DELAY_SECONDS,
            )

        except Exception as e:
            logger.warning(
                f"Unexpected error pushing transcript (attempt {attempt + 1}), retrying in {retry_delay:.1f}s",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                    "retry_delay": retry_delay,
                },
            )
            await asyncio.sleep(retry_delay)
            attempt += 1
            retry_delay = min(
                retry_delay * TranscriptSyncConfig.BACKOFF_MULTIPLIER,
                TranscriptSyncConfig.MAX_RETRY_DELAY_SECONDS,
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
            timeout = aiohttp.ClientTimeout(total=TranscriptSyncConfig.REQUEST_TIMEOUT_SECONDS)

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

                            logger.debug(
                                "Injected message from server",
                                extra={
                                    "episode_id": episode_id,
                                    "role": msg_data.get("role"),
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
                        logger.debug(
                            f"Pulled {len(messages)} injected message(s)",
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
