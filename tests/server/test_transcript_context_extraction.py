"""
Test that assistant messages and reasoning are extracted from the client transcript
instead of from tool arguments (Phase 5: Monkey-patch deprecation).

This test suite proves the new transcript-based approach works correctly.
The monkey-patch approach has been completely removed.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.models.constants import MetadataKeys
from saber.models.headers import HTTPHeaders
from saber.server.api.session_mcp_api import SessionMCPAPI
from saber.server.base import Action, CommandResult, Episode, EpisodeState


class TestTranscriptContextExtraction:
    """Test context extraction from client transcript (new approach)."""

    @pytest.fixture
    def session_manager(self):
        """Create mock SessionManager for testing."""
        mock_manager = MagicMock()
        mock_manager.domain_name = "test_domain"
        mock_manager.execution_manager = MagicMock()
        mock_manager.execute_action = AsyncMock()
        mock_manager._get_session = MagicMock()
        return mock_manager

    @pytest.fixture
    def mcp_api(self, session_manager):
        """Create SessionMCPAPI instance for testing."""
        return SessionMCPAPI(session_manager, host="127.0.0.1", port=3001)

    @pytest.fixture
    def episode_with_transcript(self):
        """Create episode with client transcript containing assistant message."""
        episode = Episode(
            episode_id="ep_test_123",
            task_id="task_456",
            session_id="session_789",
            start_time=datetime.utcnow(),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={
                MetadataKeys.CLIENT_TRANSCRIPT.value: [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": "List all files in the current directory.",
                    },
                    {
                        "role": "assistant",
                        "content": "I'll use the ls command to list files.",
                        "reasoning": "The user wants to see directory contents, so I should use the bash tool with ls.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": "bash",
                                "arguments": {"command": "ls -la"},
                            }
                        ],
                    },
                ]
            },
            metadata={},
            max_steps=10,
        )
        return episode

    @pytest.fixture
    def episode_without_transcript(self):
        """Create episode without client transcript."""
        episode = Episode(
            episode_id="ep_test_456",
            task_id="task_789",
            session_id="session_101",
            start_time=datetime.utcnow(),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={},  # No CLIENT_TRANSCRIPT
            metadata={},
            max_steps=10,
        )
        return episode

    @pytest.mark.asyncio
    async def test_extract_context_from_transcript_success(self, mcp_api, episode_with_transcript):
        """Test successful extraction of assistant message and reasoning from transcript."""
        # Test the helper method directly
        assistant_message, reasoning = mcp_api._extract_context_from_transcript(episode_with_transcript)

        # Verify extraction
        assert assistant_message is not None
        assert assistant_message == "I'll use the ls command to list files."
        assert reasoning is not None
        assert reasoning == "The user wants to see directory contents, so I should use the bash tool with ls."

    @pytest.mark.asyncio
    async def test_extract_context_no_transcript(self, mcp_api, episode_without_transcript):
        """Test extraction when transcript is missing."""
        assistant_message, reasoning = mcp_api._extract_context_from_transcript(episode_without_transcript)

        # Should return None values gracefully
        assert assistant_message is None
        assert reasoning is None

    @pytest.mark.asyncio
    async def test_extract_context_no_assistant_message(self, mcp_api):
        """Test extraction when transcript has no assistant messages."""
        episode = Episode(
            episode_id="ep_test_789",
            task_id="task_012",
            session_id="session_345",
            start_time=datetime.utcnow(),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={
                MetadataKeys.CLIENT_TRANSCRIPT.value: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ]
            },
            metadata={},
            max_steps=10,
        )

        assistant_message, reasoning = mcp_api._extract_context_from_transcript(episode)

        assert assistant_message is None
        assert reasoning is None

    @pytest.mark.asyncio
    async def test_extract_context_multiple_assistant_messages(self, mcp_api):
        """Test extraction picks the MOST RECENT assistant message."""
        episode = Episode(
            episode_id="ep_test_multi",
            task_id="task_multi",
            session_id="session_multi",
            start_time=datetime.utcnow(),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={
                MetadataKeys.CLIENT_TRANSCRIPT.value: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "First request"},
                    {
                        "role": "assistant",
                        "content": "First response",
                        "reasoning": "First reasoning",
                    },
                    {"role": "user", "content": "Second request"},
                    {
                        "role": "assistant",
                        "content": "Second response - most recent",
                        "reasoning": "Second reasoning - most recent",
                    },
                ]
            },
            metadata={},
            max_steps=10,
        )

        assistant_message, reasoning = mcp_api._extract_context_from_transcript(episode)

        # Should get the MOST RECENT assistant message
        assert assistant_message == "Second response - most recent"
        assert reasoning == "Second reasoning - most recent"

    @pytest.mark.asyncio
    async def test_convert_to_action_uses_transcript(self, mcp_api, episode_with_transcript):
        """Test that _convert_to_action extracts context from transcript."""
        action = mcp_api._convert_to_action(
            tool_name="bash",
            arguments={"command": "ls -la"},
            episode=episode_with_transcript,
        )

        # Verify context was extracted from transcript
        assert action.assistant_message == "I'll use the ls command to list files."
        assert action.reasoning == "The user wants to see directory contents, so I should use the bash tool with ls."

        # Verify regular parameters are present
        assert action.parameters["command"] == "ls -la"

    @pytest.mark.asyncio
    async def test_convert_to_action_no_transcript_no_context(self, mcp_api, episode_without_transcript):
        """Test that actions are created without context when transcript is missing."""
        action = mcp_api._convert_to_action(
            tool_name="bash",
            arguments={"command": "ls -la"},
            episode=episode_without_transcript,
        )

        # Should have no context when transcript is missing
        assert action.assistant_message is None
        assert action.reasoning is None
        assert action.parameters["command"] == "ls -la"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_full_tool_call_with_transcript(self, mcp_api, episode_with_transcript):
        """Integration test: Full MCP tool call using transcript context."""
        # Mock successful command execution
        command_result = CommandResult.success_result(data={"output": "file1.txt\nfile2.txt"})
        mcp_api.session_manager.execute_action.return_value = command_result
        mcp_api.session_manager.get_episode_by_id.return_value = episode_with_transcript

        # Mock headers
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_789",
                HTTPHeaders.EPISODE_ID: "ep_test_123",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone",
            }

            # Execute tool call (no monkey-patch arguments needed!)
            result = await mcp_api.handle_call_tool(
                name="bash",
                arguments={"command": "ls -la"},
            )

        # Verify result is successful
        assert result.isError is False

        # Verify execute_action was called with correct context
        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args
        action: Action = call_args[0][2]

        # CRITICAL: Verify context came from transcript
        assert action.assistant_message == "I'll use the ls command to list files."
        assert action.reasoning == "The user wants to see directory contents, so I should use the bash tool with ls."

    @pytest.mark.asyncio
    async def test_assistant_message_without_reasoning(self, mcp_api):
        """Test extraction when assistant message has no reasoning field."""
        episode = Episode(
            episode_id="ep_no_reasoning",
            task_id="task_no_reasoning",
            session_id="session_no_reasoning",
            start_time=datetime.utcnow(),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={
                MetadataKeys.CLIENT_TRANSCRIPT.value: [
                    {
                        "role": "assistant",
                        "content": "Simple message without reasoning",
                        # No "reasoning" field
                    }
                ]
            },
            metadata={},
            max_steps=10,
        )

        assistant_message, reasoning = mcp_api._extract_context_from_transcript(episode)

        assert assistant_message == "Simple message without reasoning"
        assert reasoning is None  # No reasoning present

    @pytest.mark.asyncio
    async def test_empty_transcript_array(self, mcp_api):
        """Test extraction with empty transcript array."""
        episode = Episode(
            episode_id="ep_empty",
            task_id="task_empty",
            session_id="session_empty",
            start_time=datetime.utcnow(),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={MetadataKeys.CLIENT_TRANSCRIPT.value: []},  # Empty array
            metadata={},
            max_steps=10,
        )

        assistant_message, reasoning = mcp_api._extract_context_from_transcript(episode)

        assert assistant_message is None
        assert reasoning is None

    @pytest.mark.asyncio
    async def test_malformed_transcript_graceful_handling(self, mcp_api):
        """Test graceful handling of malformed transcript data."""
        episode = Episode(
            episode_id="ep_malformed",
            task_id="task_malformed",
            session_id="session_malformed",
            start_time=datetime.utcnow(),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={
                MetadataKeys.CLIENT_TRANSCRIPT.value: [
                    "not_a_dict",  # Malformed entry
                    {"role": "assistant", "content": "Valid message"},
                ]
            },
            metadata={},
            max_steps=10,
        )

        # Should not crash, should handle gracefully
        assistant_message, reasoning = mcp_api._extract_context_from_transcript(episode)

        # Should still extract valid message
        assert assistant_message == "Valid message"
        assert reasoning is None


class TestEndToEndTranscriptFlow:
    """Integration tests for the complete transcript-based context flow."""

    @pytest.mark.asyncio
    async def test_e2e_transcript_push_and_action_creation(self):
        """
        End-to-end test: Client pushes transcript → Server creates Action with context.

        This test simulates:
        1. Client pushes transcript via POST /transcript
        2. Client calls tool via MCP
        3. Server extracts context from transcript (not from tool args)
        4. Action is created with assistant_message and reasoning
        """
        from saber.server.api.session_rest_api import SessionRestAPI
        from saber.models.rest import TranscriptPushRequest, ChatMessage as RestChatMessage

        # Setup mock session manager
        mock_manager = MagicMock()
        mock_manager.domain_name = "test_domain"
        mock_manager.execution_manager = MagicMock()
        mock_manager.execute_action = AsyncMock(return_value=CommandResult.success_result(data={"success": True}))

        # Create episode
        episode = Episode(
            episode_id="ep_e2e",
            task_id="task_e2e",
            session_id="session_e2e",
            start_time=datetime.utcnow(),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={},
            metadata={},
            max_steps=10,
        )
        mock_manager.get_episode_by_id.return_value = episode

        # Create REST API and MCP API
        rest_api = SessionRestAPI(mock_manager)
        mcp_api = SessionMCPAPI(mock_manager)

        # Step 1: Client pushes transcript
        transcript_request = TranscriptPushRequest(
            messages=[
                RestChatMessage(role="system", content="You are a helpful assistant."),
                RestChatMessage(role="user", content="Run ls command"),
                RestChatMessage(
                    role="assistant",
                    content="I'll execute the ls command for you.",
                    reasoning="The user requested a directory listing.",
                ),
            ]
        )

        # Manually update episode context (simulating what the endpoint would do)
        episode.context[MetadataKeys.CLIENT_TRANSCRIPT.value] = [msg.model_dump() for msg in transcript_request.messages]
        episode.context[MetadataKeys.TRANSCRIPT_UPDATED_AT.value] = datetime.utcnow().isoformat()

        # Verify transcript was stored in episode context
        assert MetadataKeys.CLIENT_TRANSCRIPT.value in episode.context
        stored_transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT.value]
        assert len(stored_transcript) == 3
        assert stored_transcript[2]["role"] == "assistant"
        assert stored_transcript[2]["content"] == "I'll execute the ls command for you."

        # Step 2: Client calls MCP tool (NO monkey-patch arguments)
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_e2e",
                HTTPHeaders.EPISODE_ID: "ep_e2e",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone",
            }

            result = await mcp_api.handle_call_tool(
                name="bash",
                arguments={"command": "ls -la"},  # No __saber_assistant_message__!
            )

        # Step 3: Verify Action was created with context from transcript
        mock_manager.execute_action.assert_called_once()
        action: Action = mock_manager.execute_action.call_args[0][2]

        assert action.assistant_message == "I'll execute the ls command for you."
        assert action.reasoning == "The user requested a directory listing."
        assert action.tool_name == "bash"
        assert action.parameters == {"command": "ls -la"}
