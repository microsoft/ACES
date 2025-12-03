"""Model wrapper for transcript synchronization.

This module provides a transparent wrapper around inspect_ai Model objects that
automatically pushes transcripts to the SABER server after each generate() call.

The wrapper intercepts model.generate() calls at the lowest level, ensuring that
transcripts are synchronized BEFORE tools execute, solving the race condition where
the server needs assistant messages for context extraction.

Key features:
- Works with ANY agent type (react, basic_agent, custom agents, etc.)
- Transparent delegation - behaves exactly like the wrapped model
- Differential sync - only pushes new messages
- Graceful error handling - failures don't crash agent execution
- BlockingTranscriptSyncingModelWrapper: Blocks and waits for transcript modifications
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    Model,
    ModelOutput,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from ...client.api.rest_client import SABERRestClient
from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ...models.rest.config import TranscriptSyncConfig

logger = get_saber_logger(LogCategory.AGENT, __name__)


class TranscriptSyncingModelWrapper:
    """Wraps a Model to push transcripts after each generate() call.

    This wrapper intercepts at the model level, making it compatible with all
    agent implementations without modification. The agent calls get_model().generate(),
    which now goes through this wrapper, ensuring the server has fresh context
    before any tool executions occur.

    Attributes:
        base_model: The original Model being wrapped
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API

    Example:
        >>> original_model = get_model()
        >>> wrapped = TranscriptSyncingModelWrapper(
        ...     base_model=original_model,
        ...     session_id="session_123",
        ...     episode_id="episode_456",
        ...     rest_url="http://localhost:8000"
        ... )
        >>> active_model_context_var.set(wrapped)
        >>> # Now any agent will use wrapped model automatically
    """

    def __init__(
        self,
        base_model: Model,
        session_id: str,
        episode_id: str,
        rest_url: str,
    ):
        """Initialize the model wrapper.

        Args:
            base_model: The Model to wrap
            session_id: SABER session identifier
            episode_id: SABER episode identifier
            rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")
        """
        self._base_model = base_model
        self._session_id = session_id
        self._episode_id = episode_id
        self._rest_url = rest_url
        self._client = SABERRestClient(saber_server_url=rest_url)

        logger.debug(
            "Created TranscriptSyncingModelWrapper",
            extra={
                "session_id": session_id,
                "episode_id": episode_id,
                "base_model_type": type(base_model).__name__,
            },
        )

    async def generate(
        self,
        input: str | list[ChatMessage],
        tools: Optional[list] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Generate model output and push transcript.

        This method:
        1. Calls the base model's generate() method
        2. Pushes the assistant message to SABER server (append mode)
        3. Returns the original output

        The transcript push happens AFTER generation but BEFORE the agent
        processes tool calls, ensuring the server has context.

        Args:
            input: Messages or text input to the model
            tools: Optional list of tools available to the model
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput from the base model

        Raises:
            Any exceptions from the base model's generate() method.
            Transcript push failures are logged but not raised.
        """
        # Call original model generate
        output = await self._base_model.generate(input, tools, **kwargs)

        # Push the assistant message that was just generated
        # This happens BEFORE tools execute, solving the race condition
        try:
            # Call module-level function for easier mocking in tests
            await _push_single_message(self, output.message)

            logger.debug(
                "Pushed assistant message after generate",
                extra={
                    "episode_id": self._episode_id,
                    "has_tool_calls": bool(output.message.tool_calls),
                },
            )

        except Exception as e:
            # Log but don't crash - graceful degradation
            logger.warning(
                "Failed to push transcript after model.generate(), continuing execution",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                },
            )

        return output

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the base model.

        This makes the wrapper transparent - it behaves exactly like the
        wrapped model for all attributes/methods except generate().

        Args:
            name: Attribute name being accessed

        Returns:
            The attribute from the base model
        """
        return getattr(self._base_model, name)

    def __repr__(self) -> str:
        """Return string representation showing wrapped model."""
        return f"TranscriptSyncingModelWrapper({self._base_model!r})"

    @staticmethod
    def _serialize_message(msg: ChatMessage) -> Dict[str, Any]:
        """Convert ChatMessage to JSON-safe dict.

        Args:
            msg: Inspect AI ChatMessage object

        Returns:
            Dictionary with keys: role, content, tool_calls (optional), etc.

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

    @staticmethod
    def _deserialize_message(msg_data: Dict[str, Any]) -> ChatMessage:
        """Convert message dictionary to ChatMessage object.

        Inverse of _serialize_message() - converts JSON-safe dict back to
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

    async def _push_single_message(self, message: ChatMessage) -> None:
        """Push a single message to SABER server.

        This is a convenience wrapper around client.push_episode_transcript for pushing
        a single message (typically an assistant message right after generation).

        Args:
            message: Single ChatMessage to push

        Returns:
            None (logs errors but doesn't raise exceptions)
        """
        try:
            serialized = self._serialize_message(message)
            metadata: Dict[str, Any] = {
                str(MetadataKeys.TRANSCRIPT_TIMESTAMP): datetime.now(timezone.utc).isoformat(),
                str(MetadataKeys.TRANSCRIPT_SOURCE): TranscriptSyncConfig.DEFAULT_SOURCE,
            }

            await self._client.push_episode_transcript(
                session_id=self._session_id,
                episode_id=self._episode_id,
                messages=[serialized],
                mode="append",
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(
                "Failed to push single message",
                extra={
                    "error": str(e),
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                },
            )

    @staticmethod
    async def pull_injected_messages(
        state: TaskState,
        session_id: str,
        episode_id: str,
        rest_url: str,
    ) -> None:
        """Pull and inject pending messages from SABER server (Red team message injection).

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
            - Makes HTTP GET request to server via REST client
        """
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
            # Build URL for message injection endpoint
            url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject"

            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=TranscriptSyncConfig.REQUEST_TIMEOUT_SECONDS)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        messages = data.get("messages", [])

                        # Deserialize and inject messages
                        for msg_data in messages:
                            try:
                                chat_msg = TranscriptSyncingModelWrapper._deserialize_message(msg_data)
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

        except Exception as e:
            logger.warning(
                "Unexpected error pulling injected messages",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )


class BlockingTranscriptSyncingModelWrapper(TranscriptSyncingModelWrapper):
    """Model wrapper that blocks and waits for transcript modifications before each generate() call.

    Used by blue team agents in orchestrated red/blue scenarios where the blue agent
    should wait for red team transcript injections between generate() calls.

    Blocking flow:
    1. Before calling base_model.generate():
       - Poll TRANSCRIPT_LAST_MODIFIED_AT timestamp
       - Wait for timestamp to change (indicates red team made modification)
       - Pull modified transcript
       - Replace conversation history in input parameter
    2. Call base_model.generate() with modified input
    3. Push assistant message to transcript (inherited from parent)

    Attributes:
        _skip_first_iteration: Skip blocking on first generate() call
        _first_call: Tracks if this is the first generate() call

    Note:
        Polls indefinitely until episode ends with hardcoded 2-second intervals
        and 60-second HTTP timeouts for REST calls.
    """

    POLL_INTERVAL = 2.0  # Hardcoded: poll every 2 seconds
    HTTP_TIMEOUT = 60.0  # Hardcoded: 60 second timeout for REST calls

    def __init__(
        self,
        base_model: Model,
        session_id: str,
        episode_id: str,
        rest_url: str,
        skip_first_iteration: bool = True,
    ):
        """Initialize blocking transcript syncing wrapper.

        Args:
            base_model: Underlying Model to wrap
            session_id: SABER session ID
            episode_id: SABER episode ID
            rest_url: Base URL of SABER REST API
            skip_first_iteration: Skip blocking on first generate() call
        """
        super().__init__(base_model, session_id, episode_id, rest_url)
        self._skip_first_iteration = skip_first_iteration
        self._first_call = True
        self._client = SABERRestClient(saber_server_url=rest_url)

        logger.info(
            "Created BlockingTranscriptSyncingModelWrapper",
            extra={
                "event": "blocking_wrapper_created",
                "session_id": session_id,
                "episode_id": episode_id,
                "poll_interval": self.POLL_INTERVAL,
                "http_timeout": self.HTTP_TIMEOUT,
                "skip_first_iteration": skip_first_iteration,
            },
        )

    async def generate(
        self,
        input: str | list[ChatMessage],
        tools: Optional[list] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Generate with blocking for transcript modifications.

        Blocks before each generate() call (except optionally the first) to wait
        for red team transcript modifications.

        Args:
            input: Messages or text input to the model
            tools: Optional list of tools available to the model
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput from the base model

        Raises:
            RuntimeError: If blocking timeout or max iterations exceeded
        """
        # Skip blocking on first iteration if configured
        if self._skip_first_iteration and self._first_call:
            self._first_call = False
            logger.debug(
                "Skipping blocking on first iteration",
                extra={
                    "event": "blocking_wrapper_first_iteration_skipped",
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                },
            )
        else:
            # Block and wait for transcript modification
            if isinstance(input, list):
                modified_input = await self._block_and_pull_transcript(input)
                input = modified_input
            else:
                logger.warning(
                    "Blocking wrapper received non-list input, cannot replace with modified transcript",
                    extra={
                        "event": "blocking_wrapper_non_list_input",
                        "input_type": type(input).__name__,
                    },
                )

        # Call parent's generate() which will:
        # 1. Call base_model.generate() with (possibly modified) input
        # 2. Push assistant message to transcript
        return await super().generate(input, tools, **kwargs)

    async def _block_and_pull_transcript(
        self,
        current_input: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Block and poll for transcript modification, then pull updated transcript.

        Args:
            current_input: Current conversation history

        Returns:
            Modified conversation history from transcript (or original if no changes)

        Raises:
            RuntimeError: If blocking timeout is exceeded
        """
        start_time = time.time()

        # Get initial timestamp
        try:
            initial_timestamp = await self._get_transcript_timestamp()
        except Exception as exc:
            logger.warning(
                "Failed to get initial transcript timestamp, proceeding without blocking",
                extra={
                    "event": "blocking_wrapper_timestamp_error",
                    "error": str(exc),
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                },
            )
            return current_input

        logger.info(
            "Blocking for transcript modification (indefinite polling)",
            extra={
                "event": "blocking_wrapper_started",
                "session_id": self._session_id,
                "episode_id": self._episode_id,
                "initial_timestamp": initial_timestamp,
                "poll_interval": self.POLL_INTERVAL,
            },
        )

        # Poll indefinitely for timestamp change
        iteration = 0
        while True:
            await asyncio.sleep(self.POLL_INTERVAL)
            iteration += 1
            try:
                current_timestamp = await self._get_transcript_timestamp()

                if current_timestamp != initial_timestamp:
                    logger.info(
                        "Transcript modification detected",
                        extra={
                            "event": "blocking_wrapper_modification_detected",
                            "session_id": self._session_id,
                            "episode_id": self._episode_id,
                            "initial_timestamp": initial_timestamp,
                            "new_timestamp": current_timestamp,
                            "elapsed_seconds": time.time() - start_time,
                            "iterations": iteration + 1,
                        },
                    )

                    # Pull modified transcript
                    modified_transcript = await self._pull_transcript()
                    return modified_transcript

            except Exception as exc:
                logger.warning(
                    "Error polling transcript timestamp",
                    extra={
                        "event": "blocking_wrapper_poll_error",
                        "error": str(exc),
                        "iteration": iteration,
                        "session_id": self._session_id,
                        "episode_id": self._episode_id,
                    },
                )
                # Continue polling on error

    async def _get_transcript_timestamp(self) -> Any:
        """Get the last modification timestamp of the episode transcript.

        Returns:
            ISO format timestamp string

        Raises:
            Exception: If metadata retrieval fails
        """
        try:
            metadata = await get_episode_metadata(self._client, self._session_id, self._episode_id)
            return metadata.get(MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT, "")
        except Exception as exc:
            logger.error(
                "Failed to get episode metadata for timestamp",
                extra={
                    "event": "blocking_wrapper_metadata_error",
                    "error": str(exc),
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                },
            )
            raise

    async def _pull_transcript(self) -> list[ChatMessage]:
        """Pull complete transcript and convert to ChatMessage list.

        Returns:
            List of ChatMessage objects from transcript

        Raises:
            Exception: If transcript pull or conversion fails
        """
        try:
            # Call module-level function for easier mocking in tests
            transcript_data = await pull_episode_transcript(self._client, self._session_id, self._episode_id)

            # Convert transcript to ChatMessage list
            # transcript_data["messages"] is list of {"role": str, "content": str}
            messages = [self._deserialize_message(msg) for msg in transcript_data.get("messages", [])]

            logger.info(
                "Pulled modified transcript",
                extra={
                    "event": "blocking_wrapper_transcript_pulled",
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                    "message_count": len(messages),
                },
            )

            return messages

        except Exception as exc:
            logger.error(
                "Failed to pull transcript",
                extra={
                    "event": "blocking_wrapper_pull_error",
                    "error": str(exc),
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                },
            )
            raise


__all__ = ["TranscriptSyncingModelWrapper", "BlockingTranscriptSyncingModelWrapper"]


# Module-level helper functions for easier patching in tests
async def _push_single_message(wrapper: "TranscriptSyncingModelWrapper", message: ChatMessage) -> None:
    """Module-level wrapper for pushing a single message (for easier test mocking)."""
    await wrapper._push_single_message(message)


async def get_episode_metadata(client: "SABERRestClient", session_id: str, episode_id: str) -> Dict[str, Any]:
    """Module-level wrapper for getting episode metadata (for easier test mocking)."""
    return await client.get_episode_metadata(session_id, episode_id)


async def pull_episode_transcript(client: "SABERRestClient", session_id: str, episode_id: str) -> Dict[str, Any]:
    """Module-level wrapper for pulling episode transcript (for easier test mocking)."""
    return await client.pull_episode_transcript(session_id, episode_id)
