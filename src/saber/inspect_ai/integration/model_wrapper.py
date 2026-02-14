"""Model wrapper for transcript push.

This module provides a transparent wrapper around inspect_ai Model objects that
automatically pushes transcripts to the SABER server via WebSocket.

The wrapper intercepts model.generate() calls at the lowest level, ensuring that
transcripts are pushed to the server for evaluation and context extraction.

Key features:
- Works with ANY agent type (react, basic_agent, custom agents, etc.)
- Transparent delegation - behaves exactly like the wrapped model
- WebSocket push for transcript messages
- Graceful error handling - failures don't crash agent execution
- Unified coordination - all agents use WebSocketTranscriptSyncingModelWrapper

Architecture:
- Uses generic TranscriptSyncClient from saber.client.transcript
- Provides InspectAIMessageSerializer for ChatMessage serialization
- WebSocketTranscriptSyncingModelWrapper orchestrates Model + TranscriptSyncClient

Logging category: AGENT.
"""

from typing import Any, ClassVar, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Model,
    ModelOutput,
)

from ...client.transcript import MessageSerializer, TranscriptSyncClient
from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MessageRole
from ...models.rest.websocket_config import WebSocketConfig

logger = get_saber_logger(LogCategory.AGENT, __name__)


# Mapping from role string to ChatMessage subclass for deserialization
_ROLE_TO_MESSAGE_CLASS: dict[str, type[ChatMessage]] = {
    MessageRole.SYSTEM.value: ChatMessageSystem,
    MessageRole.USER.value: ChatMessageUser,
    MessageRole.ASSISTANT.value: ChatMessageAssistant,
    MessageRole.TOOL.value: ChatMessageTool,
}


class InspectAIMessageSerializer(MessageSerializer[ChatMessage]):
    """MessageSerializer implementation for Inspect AI ChatMessage.

    Uses Pydantic's model_dump()/model_validate() for complete field preservation,
    including metadata, source, model, and other fields that may be added in future
    versions of inspect_ai.
    """

    def serialize(self, message: ChatMessage) -> dict[str, Any]:
        """Convert ChatMessage to JSON-safe dict using Pydantic's native serialization.

        Uses model_dump() to preserve ALL fields including metadata, source, model,
        id, and any future fields added to ChatMessage types.

        Args:
            message: Inspect AI ChatMessage object

        Returns:
            Dictionary representation of the message with all fields preserved.

        Raises:
            ValueError: If message type is unknown or unsupported
        """
        if not isinstance(message, (ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool)):
            raise ValueError(f"Unknown message type: {type(message)}")

        result: dict[str, Any] = message.model_dump()
        return result

    def deserialize(self, data: dict[str, Any]) -> ChatMessage:
        """Convert message dictionary to ChatMessage object using Pydantic's native deserialization.

        Uses model_validate() to restore ALL fields including metadata, source, model,
        id, and any future fields added to ChatMessage types.

        Args:
            data: Dictionary representation of a ChatMessage (from serialize or JSON)

        Returns:
            ChatMessage object (ChatMessageUser, ChatMessageAssistant, etc.)

        Raises:
            ValueError: If role is unknown or message structure is invalid
        """
        role_str = data.get("role")

        if role_str not in _ROLE_TO_MESSAGE_CLASS:
            raise ValueError(f"Unknown message role: {role_str}")

        message_class = _ROLE_TO_MESSAGE_CLASS[role_str]

        try:
            return message_class.model_validate(data)
        except Exception as e:
            raise ValueError(f"Failed to deserialize {role_str} message: {e}") from e

    def get_role(self, message: ChatMessage) -> str:
        """Get the role of a message.

        Args:
            message: ChatMessage object

        Returns:
            Role string (one of: "system", "user", "assistant", "tool")
        """
        return cast(str, message.role)

    def has_tool_calls(self, message: ChatMessage) -> bool:
        """Check if an assistant message has tool calls.

        Args:
            message: ChatMessage object

        Returns:
            True if message has tool_calls, False otherwise
        """
        if isinstance(message, ChatMessageAssistant):
            return bool(message.tool_calls)
        return False

    def get_tool_call_ids(self, message: ChatMessage) -> list[str]:
        """Get tool call IDs from an assistant message.

        Args:
            message: ChatMessage object (typically assistant role)

        Returns:
            List of tool_call IDs (empty if none or not an assistant message)
        """
        if isinstance(message, ChatMessageAssistant) and message.tool_calls:
            return [tc.id for tc in message.tool_calls if tc.id]
        return []

    def get_tool_call_id(self, message: ChatMessage) -> str | None:
        """Get the tool_call_id for a tool message.

        Args:
            message: ChatMessage (typically tool role)

        Returns:
            tool_call_id if present, None otherwise
        """
        if isinstance(message, ChatMessageTool):
            return cast(str | None, message.tool_call_id)
        return None


class WebSocketTranscriptSyncingModelWrapper:
    """Model wrapper with WebSocket transcript push.

    Provides:
    - Persistent WebSocket connection (established on first generate())
    - Push messages to server with retry logic
    - Automatic reconnection on connection loss

    Flow:
    1. First generate() → establish WebSocket connection
    2. Push tool results before generate
    3. Call base model generate
    4. Push assistant output after generate
    5. Connection reused for entire episode

    Uses the generic TranscriptSyncClient with InspectAIMessageSerializer.
    """

    # Class-level registry to track wrappers by episode_id for cleanup
    _wrappers_by_episode: ClassVar[dict[str, "WebSocketTranscriptSyncingModelWrapper"]] = {}

    @classmethod
    def get_wrapper_for_episode(cls, episode_id: str) -> "WebSocketTranscriptSyncingModelWrapper | None":
        """Get the wrapper instance for a given episode_id.

        Used by sandbox cleanup to find and cleanup the wrapper.

        Args:
            episode_id: The episode ID to look up

        Returns:
            The wrapper instance if found, None otherwise
        """
        return cls._wrappers_by_episode.get(episode_id)

    @classmethod
    def _register_wrapper(cls, episode_id: str, wrapper: "WebSocketTranscriptSyncingModelWrapper") -> None:
        """Register a wrapper instance for cleanup lookup."""
        cls._wrappers_by_episode[episode_id] = wrapper
        logger.debug(
            "Registered wrapper for episode",
            extra={"episode_id": episode_id, "total_registered": len(cls._wrappers_by_episode)},
        )

    @classmethod
    def _unregister_wrapper(cls, episode_id: str) -> None:
        """Unregister a wrapper instance after cleanup."""
        if episode_id in cls._wrappers_by_episode:
            del cls._wrappers_by_episode[episode_id]
            logger.debug(
                "Unregistered wrapper for episode",
                extra={"episode_id": episode_id, "total_registered": len(cls._wrappers_by_episode)},
            )

    def __init__(
        self,
        base_model: Model,
        session_id: str,
        episode_id: str,
        rest_url: str,
        ws_config: WebSocketConfig | None = None,
    ):
        """Initialize WebSocket transcript syncing wrapper.

        Args:
            base_model: Underlying Model to wrap
            session_id: SABER session ID
            episode_id: SABER episode ID
            rest_url: Base URL of SABER REST API
            ws_config: WebSocket configuration (uses defaults if None)
        """
        self._base_model = base_model
        self._session_id = session_id
        self._episode_id = episode_id
        self._rest_url = rest_url
        self._ws_config = ws_config or WebSocketConfig()

        # Create the generic transcript sync client with Inspect AI serializer
        self._client: TranscriptSyncClient[ChatMessage] = TranscriptSyncClient(
            episode_id=episode_id,
            rest_url=rest_url,
            serializer=InspectAIMessageSerializer(),
            ws_config=self._ws_config,
        )

        # Register this wrapper for cleanup lookup
        self._register_wrapper(episode_id, self)

        logger.debug(
            "Created WebSocketTranscriptSyncingModelWrapper",
            extra={
                "session_id": session_id,
                "episode_id": episode_id,
                "ws_config": {
                    "connection_timeout": self._ws_config.connection_timeout,
                    "pull_event_timeout": self._ws_config.pull.event_timeout,
                    "push_confirmation_timeout": self._ws_config.push.confirmation_timeout,
                    "reconnect_enabled": self._ws_config.reconnect_enabled,
                },
            },
        )

    # =========================================================================
    # Connection lifecycle
    # =========================================================================

    async def __aenter__(self) -> "WebSocketTranscriptSyncingModelWrapper":
        """Context manager entry - establish WebSocket connection."""
        await self._client.ensure_connected()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit - cleanup WebSocket connection."""
        await self.cleanup()
        return False  # Don't suppress exceptions

    # =========================================================================
    # Main generate method
    # =========================================================================

    async def generate(
        self,
        input: str | list[ChatMessage],
        tools: list | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Generate with WebSocket-based push.

        Pushes tool results before generation and assistant output after.
        All push/connection logic is delegated to the TranscriptSyncClient
        public API, which handles push.enabled checks and connection management
        internally.

        Args:
            input: Messages or text input to the model
            tools: Optional list of tools available to the model
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput from the base model
        """
        # Push any tool results that Inspect AI added to input
        if isinstance(input, list):
            await self._client.push_messages_if_new(input)

        # Generate with base model
        output = await self._base_model.generate(input, tools, **kwargs)

        # Push assistant output (client handles push.enabled check and local tracking)
        success = await self._client.push_message(output.message, context="output_push")
        if not success:
            logger.warning(
                "Failed to push assistant output to server",
                extra={
                    "episode_id": self._episode_id,
                    "message_role": output.message.role,
                },
            )

        return output

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self) -> None:
        """Close WebSocket connection and release resources when episode ends."""
        logger.info(
            "[CLEANUP] cleanup() CALLED - starting shutdown sequence",
            extra={"episode_id": self._episode_id},
        )

        # Delegate cleanup to the client
        await self._client.cleanup()

        # Unregister from class-level registry
        self._unregister_wrapper(self._episode_id)

        logger.info(
            "[CLEANUP] cleanup() COMPLETE",
            extra={"episode_id": self._episode_id},
        )

    # =========================================================================
    # Delegation to base model
    # =========================================================================

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
        return f"WebSocketTranscriptSyncingModelWrapper({self._base_model!r})"


__all__ = [
    "WebSocketTranscriptSyncingModelWrapper",
    "InspectAIMessageSerializer",
]
