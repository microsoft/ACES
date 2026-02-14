"""
Test that assistant messages and reasoning are extracted from the client transcript
stored in Redis (via TranscriptCoordinator) instead of from tool arguments.

This test suite proves the Redis-based transcript approach works correctly.
The in-memory CLIENT_TRANSCRIPT has been removed; Redis is the single source of truth.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.models.headers import HTTPHeaders
from saber.server.api.session_mcp_api import SessionMCPAPI
from saber.server.base import Action, CommandResult, Episode, EpisodeState


class TestTranscriptContextExtraction:
    """Test context extraction from Redis transcript."""

    @pytest.fixture
    def session_manager(self):
        """Create mock SessionManager for testing."""
        mock_manager = MagicMock()
        mock_manager.domain_name = "test_domain"
        mock_manager.execution_manager = MagicMock()
        mock_manager.execute_action = AsyncMock()
        mock_manager._get_session = MagicMock()
        # Setup transcript coordinator mock
        mock_coordinator = MagicMock()
        mock_coordinator.get_transcript = AsyncMock(return_value=[])
        mock_manager.episode_manager = MagicMock()
        mock_manager.episode_manager.transcript_coordinator = mock_coordinator
        return mock_manager

    @pytest.fixture
    def mcp_api(self, session_manager):
        """Create SessionMCPAPI instance for testing."""
        return SessionMCPAPI(session_manager, host="127.0.0.1", port=3001)

    @pytest.fixture
    def transcript_with_assistant(self):
        """Sample transcript containing an assistant message with reasoning."""
        return [
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

    @pytest.fixture
    def episode_with_transcript(self, session_manager, transcript_with_assistant):
        """Create episode and configure coordinator to return transcript."""
        episode = Episode(
            episode_id="ep_test_123",
            task_id="task_456",
            session_id="session_789",
            start_time=datetime.now(timezone.utc),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={},
            metadata={},
            max_steps=10,
        )
        session_manager.episode_manager.transcript_coordinator.get_transcript.return_value = (
            transcript_with_assistant
        )
        return episode

    @pytest.fixture
    def episode_without_transcript(self, session_manager):
        """Create episode with empty transcript in Redis."""
        episode = Episode(
            episode_id="ep_test_456",
            task_id="task_789",
            session_id="session_101",
            start_time=datetime.now(timezone.utc),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={},
            metadata={},
            max_steps=10,
        )
        session_manager.episode_manager.transcript_coordinator.get_transcript.return_value = []
        return episode

    @pytest.mark.asyncio
    async def test_extract_context_from_transcript_success(self, mcp_api, episode_with_transcript):
        """Test successful extraction of assistant message and reasoning from transcript."""
        assistant_message, reasoning = await mcp_api._extract_context_from_transcript(
            episode_with_transcript.episode_id
        )

        assert assistant_message is not None
        assert assistant_message == "I'll use the ls command to list files."
        assert reasoning is not None
        assert reasoning == "The user wants to see directory contents, so I should use the bash tool with ls."

    @pytest.mark.asyncio
    async def test_extract_context_no_transcript(self, mcp_api, episode_without_transcript):
        """Test extraction when transcript is empty."""
        assistant_message, reasoning = await mcp_api._extract_context_from_transcript(
            episode_without_transcript.episode_id
        )

        assert assistant_message is None
        assert reasoning is None

    @pytest.mark.asyncio
    async def test_extract_context_no_assistant_message(self, mcp_api, session_manager):
        """Test extraction when transcript has no assistant messages."""
        session_manager.episode_manager.transcript_coordinator.get_transcript.return_value = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        assistant_message, reasoning = await mcp_api._extract_context_from_transcript("ep_test_789")

        assert assistant_message is None
        assert reasoning is None

    @pytest.mark.asyncio
    async def test_extract_context_multiple_assistant_messages(self, mcp_api, session_manager):
        """Test extraction picks the MOST RECENT assistant message."""
        session_manager.episode_manager.transcript_coordinator.get_transcript.return_value = [
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

        assistant_message, reasoning = await mcp_api._extract_context_from_transcript("ep_test_multi")

        assert assistant_message == "Second response - most recent"
        assert reasoning == "Second reasoning - most recent"

    @pytest.mark.asyncio
    async def test_convert_to_action_uses_transcript(self, mcp_api, episode_with_transcript):
        """Test that _convert_to_action extracts context from transcript."""
        action = await mcp_api._convert_to_action(
            tool_name="bash",
            arguments={"command": "ls -la"},
            episode=episode_with_transcript,
        )

        assert action.assistant_message == "I'll use the ls command to list files."
        assert action.reasoning == "The user wants to see directory contents, so I should use the bash tool with ls."
        assert action.parameters["command"] == "ls -la"

    @pytest.mark.asyncio
    async def test_convert_to_action_no_transcript_no_context(self, mcp_api, episode_without_transcript):
        """Test that actions are created without context when transcript is missing."""
        action = await mcp_api._convert_to_action(
            tool_name="bash",
            arguments={"command": "ls -la"},
            episode=episode_without_transcript,
        )

        assert action.assistant_message is None
        assert action.reasoning is None
        assert action.parameters["command"] == "ls -la"

    @pytest.mark.asyncio
    async def test_full_tool_call_with_transcript(self, mcp_api, episode_with_transcript, session_manager):
        """Integration test: Full MCP tool call using transcript context."""
        command_result = CommandResult.success_result(data={"output": "file1.txt\nfile2.txt"})
        mcp_api.session_manager.execute_action.return_value = command_result
        mcp_api.session_manager.get_episode_by_id.return_value = episode_with_transcript

        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_789",
                HTTPHeaders.EPISODE_ID: "ep_test_123",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone",
            }

            result = await mcp_api.handle_call_tool(
                name="bash",
                arguments={"command": "ls -la"},
            )

        assert result.isError is False

        mcp_api.session_manager.execute_action.assert_called_once()
        call_args = mcp_api.session_manager.execute_action.call_args
        action: Action = call_args[0][2]

        assert action.assistant_message == "I'll use the ls command to list files."
        assert action.reasoning == "The user wants to see directory contents, so I should use the bash tool with ls."

    @pytest.mark.asyncio
    async def test_assistant_message_without_reasoning(self, mcp_api, session_manager):
        """Test extraction when assistant message has no reasoning field."""
        session_manager.episode_manager.transcript_coordinator.get_transcript.return_value = [
            {
                "role": "assistant",
                "content": "Simple message without reasoning",
            }
        ]

        assistant_message, reasoning = await mcp_api._extract_context_from_transcript("ep_no_reasoning")

        assert assistant_message == "Simple message without reasoning"
        assert reasoning is None

    @pytest.mark.asyncio
    async def test_empty_transcript_array(self, mcp_api, session_manager):
        """Test extraction with empty transcript array."""
        session_manager.episode_manager.transcript_coordinator.get_transcript.return_value = []

        assistant_message, reasoning = await mcp_api._extract_context_from_transcript("ep_empty")

        assert assistant_message is None
        assert reasoning is None

    @pytest.mark.asyncio
    async def test_malformed_transcript_graceful_handling(self, mcp_api, session_manager):
        """Test graceful handling of malformed transcript data."""
        session_manager.episode_manager.transcript_coordinator.get_transcript.return_value = [
            "not_a_dict",  # Malformed entry
            {"role": "assistant", "content": "Valid message"},
        ]

        assistant_message, reasoning = await mcp_api._extract_context_from_transcript("ep_malformed")

        assert assistant_message == "Valid message"
        assert reasoning is None


class TestEndToEndTranscriptFlow:
    """Integration tests for the complete transcript-based context flow."""

    @pytest.mark.asyncio
    async def test_e2e_transcript_push_and_action_creation(self):
        """
        End-to-end test: Transcript is in Redis → Server creates Action with context.

        This test simulates:
        1. Coordinator returns transcript from Redis
        2. Client calls tool via MCP
        3. Server extracts context from transcript (via coordinator)
        4. Action is created with assistant_message and reasoning
        """
        # Setup mock session manager
        mock_manager = MagicMock()
        mock_manager.domain_name = "test_domain"
        mock_manager.execution_manager = MagicMock()
        mock_manager.execute_action = AsyncMock(return_value=CommandResult.success_result(data={"success": True}))

        # Setup transcript coordinator mock
        mock_coordinator = MagicMock()
        transcript_data = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Run ls command"},
            {
                "role": "assistant",
                "content": "I'll execute the ls command for you.",
                "reasoning": "The user requested a directory listing.",
            },
        ]
        mock_coordinator.get_transcript = AsyncMock(return_value=transcript_data)
        mock_manager.episode_manager = MagicMock()
        mock_manager.episode_manager.transcript_coordinator = mock_coordinator

        # Create episode
        episode = Episode(
            episode_id="ep_e2e",
            task_id="task_e2e",
            session_id="session_e2e",
            start_time=datetime.now(timezone.utc),
            state=EpisodeState.ACTIVE,
            steps=[],
            context={},
            metadata={},
            max_steps=10,
        )
        mock_manager.get_episode_by_id.return_value = episode

        # Create MCP API
        mcp_api = SessionMCPAPI(mock_manager)

        # Client calls MCP tool
        with patch("saber.server.api.session_mcp_api.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {
                HTTPHeaders.SESSION_ID: "session_e2e",
                HTTPHeaders.EPISODE_ID: "ep_e2e",
                HTTPHeaders.ORCHESTRATION_ENV: "standalone",
            }

            await mcp_api.handle_call_tool(
                name="bash",
                arguments={"command": "ls -la"},
            )

        # Verify Action was created with context from transcript
        mock_manager.execute_action.assert_called_once()
        action: Action = mock_manager.execute_action.call_args[0][2]

        assert action.assistant_message == "I'll execute the ls command for you."
        assert action.reasoning == "The user requested a directory listing."
        assert action.tool_name == "bash"
        assert action.parameters == {"command": "ls -la"}
        assert action.tool_name == "bash"
        assert action.parameters == {"command": "ls -la"}
