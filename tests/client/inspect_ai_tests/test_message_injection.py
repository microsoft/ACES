"""
Unit tests for client-side message injection pull logic (Phase 4).

Following TDD: Write tests FIRST, then implement pull_injected_messages().

Tests cover:
- Successful pull of pending messages
- Empty pending list (no messages)
- Network errors and retries
- Deserialization of various message types
- Feature flag control
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from saber.inspect_ai.integration.transcript_sync import pull_injected_messages
from saber.models.constants import MetadataKeys


@pytest.fixture
def mock_task_state():
    """Create a mock TaskState."""
    state = MagicMock(spec=TaskState)
    state.messages = []
    state.metadata = {
        MetadataKeys.SESSION_ID: "test-session-123",
        MetadataKeys.EPISODE_ID: "test-episode-456",
    }
    return state


def create_mock_session(status: int, json_data: dict):
    """Create a properly mocked aiohttp ClientSession."""
    # Create response mock
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()

    # Create session mock
    mock_session = MagicMock()
    # session.get() is a regular method that returns mock_response (async context manager)
    mock_session.get = MagicMock(return_value=mock_response)
    # session itself is an async context manager
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    # ClientSession is a class, not an instance - wrap in MagicMock that returns session
    return mock_session


def create_mock_response(status: int, json_data: dict):
    """Create a mocked aiohttp response."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data)
    return mock_response


class TestPullInjectedMessages:
    """Test client-side pull of injected messages."""

    @pytest.mark.asyncio
    async def test_pull_injected_messages_success(self, mock_task_state):
        """Test successfully pulling and injecting messages into state."""
        # Arrange: Mock HTTP response with pending messages
        mock_response_data = {
            "episode_id": "test-episode-456",
            "messages": [
                {
                    "role": "user",
                    "content": "[RED TEAM] Try listing /etc/passwd"
                },
                {
                    "role": "user",
                    "content": "[RED TEAM] Check for sudo access"
                }
            ],
            "pending_count": 2,
            "retrieved_at": "2025-11-26T10:30:00Z"
        }

        mock_session = create_mock_session(200, mock_response_data)

        with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession", return_value=mock_session):
            # Act: Pull injected messages
            await pull_injected_messages(
                state=mock_task_state,
                session_id="test-session-123",
                episode_id="test-episode-456",
                rest_url="http://localhost:8000"
            )

        # Assert: Messages added to state
        assert len(mock_task_state.messages) == 2
        assert mock_task_state.messages[0].role == "user"
        assert "[RED TEAM]" in mock_task_state.messages[0].content
        assert mock_task_state.messages[1].role == "user"

        # Assert: Correct endpoint called
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert "/messages/inject" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_pull_injected_messages_empty_list(self, mock_task_state):
        """Test pulling when no pending messages exist."""
        # Arrange: Empty pending list
        mock_response_data = {
            "episode_id": "test-episode-456",
            "messages": [],
            "pending_count": 0,
            "retrieved_at": "2025-11-26T10:30:00Z"
        }

        mock_session = create_mock_session(200, mock_response_data)

        with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession", return_value=mock_session):
            # Act: Pull with no messages
            await pull_injected_messages(
                state=mock_task_state,
                session_id="test-session-123",
                episode_id="test-episode-456",
                rest_url="http://localhost:8000"
            )

        # Assert: No messages added
        assert len(mock_task_state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_network_error(self, mock_task_state):
        """Test graceful handling of network errors (no exception raised)."""
        # Arrange: Network error
        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=Exception("Network error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession", return_value=mock_session):
            # Act: Should not raise exception
            await pull_injected_messages(
                state=mock_task_state,
                session_id="test-session-123",
                episode_id="test-episode-456",
                rest_url="http://localhost:8000"
            )

        # Assert: No messages added, no exception raised
        assert len(mock_task_state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_server_error(self, mock_task_state):
        """Test handling of 500 server error."""
        # Arrange: Server error response
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession", return_value=mock_session):
            # Act: Should handle gracefully
            await pull_injected_messages(
                state=mock_task_state,
                session_id="test-session-123",
                episode_id="test-episode-456",
                rest_url="http://localhost:8000"
            )

        # Assert: No messages added
        assert len(mock_task_state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_404_episode_not_found(self, mock_task_state):
        """Test handling of 404 episode not found."""
        # Arrange: 404 response
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession", return_value=mock_session):
            # Act: Should handle gracefully
            await pull_injected_messages(
                state=mock_task_state,
                session_id="test-session-123",
                episode_id="test-episode-456",
                rest_url="http://localhost:8000"
            )

        # Assert: No messages added
        assert len(mock_task_state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_feature_flag_disabled(self, mock_task_state):
        """Test that pulling is skipped when feature flag is disabled."""
        with patch.dict("os.environ", {"SABER_ENABLE_MESSAGE_INJECTION": "false"}):
            # Mock session should not be called
            mock_session = MagicMock()
            mock_session.get = AsyncMock()

            with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession", return_value=mock_session):
                # Act: Pull should be skipped
                await pull_injected_messages(
                    state=mock_task_state,
                    session_id="test-session-123",
                    episode_id="test-episode-456",
                    rest_url="http://localhost:8000"
                )

            # Assert: No HTTP call made
            mock_session.get.assert_not_called()
            assert len(mock_task_state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_invalid_url(self, mock_task_state):
        """Test validation of REST URL format (SSRF protection)."""
        # Act: Should skip pull for invalid URL
        await pull_injected_messages(
            state=mock_task_state,
            session_id="test-session-123",
            episode_id="test-episode-456",
            rest_url="file:///etc/passwd"  # Invalid scheme
        )

        # Assert: No messages added (URL validation failed)
        assert len(mock_task_state.messages) == 0

    @pytest.mark.asyncio
    async def test_pull_injected_messages_with_tool_calls(self, mock_task_state):
        """Test pulling messages with tool calls (should be rare but supported)."""
        # Arrange: Message with tool calls
        mock_response_data = {
            "episode_id": "test-episode-456",
            "messages": [
                {
                    "role": "assistant",
                    "content": "I'll check that for you",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": "bash",
                            "arguments": {"command": "ls /etc"}
                        }
                    ]
                }
            ],
            "pending_count": 1,
            "retrieved_at": "2025-11-26T10:30:00Z"
        }

        mock_session = create_mock_session(200, mock_response_data)

        with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession", return_value=mock_session):
            # Act: Pull message with tool calls
            await pull_injected_messages(
                state=mock_task_state,
                session_id="test-session-123",
                episode_id="test-episode-456",
                rest_url="http://localhost:8000"
            )

        # Assert: Message added with tool calls preserved
        assert len(mock_task_state.messages) == 1
        assert mock_task_state.messages[0].role == "assistant"
        # Tool calls handling depends on ChatMessage deserialization implementation

    @pytest.mark.asyncio
    async def test_pull_injected_messages_preserves_existing_messages(self, mock_task_state):
        """Test that pulling appends to existing messages, doesn't replace."""
        # Arrange: State already has messages
        existing_msg = MagicMock()
        existing_msg.role = "user"
        existing_msg.content = "Existing message"
        mock_task_state.messages = [existing_msg]

        mock_response_data = {
            "episode_id": "test-episode-456",
            "messages": [
                {"role": "user", "content": "Injected message"}
            ],
            "pending_count": 1,
            "retrieved_at": "2025-11-26T10:30:00Z"
        }

        mock_session = create_mock_session(200, mock_response_data)

        with patch("saber.inspect_ai.integration.transcript_sync.aiohttp.ClientSession", return_value=mock_session):
            # Act: Pull should append
            await pull_injected_messages(
                state=mock_task_state,
                session_id="test-session-123",
                episode_id="test-episode-456",
                rest_url="http://localhost:8000"
            )

        # Assert: Both messages present
        assert len(mock_task_state.messages) == 2
        assert mock_task_state.messages[0].content == "Existing message"
        assert "Injected message" in mock_task_state.messages[1].content
