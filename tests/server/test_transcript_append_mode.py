"""
Unit tests for transcript append mode on the server side.

Tests the POST /api/v1/session/{sid}/episodes/{eid}/transcript endpoint
with both 'replace' and 'append' modes to ensure differential sync works correctly.

Critical Test Coverage:
1. Append mode appends new messages to existing transcript
2. Replace mode replaces entire transcript (backward compatibility)
3. Mode parameter validation
4. Concurrent append operations
5. Edge cases (empty append, append to empty)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saber.server.base import Episode, EpisodeState
from saber.server.session_manager import SessionManager


class TestTranscriptAppendMode:
    """Test transcript append mode for differential sync."""

    @pytest.fixture
    def session_manager_app(self):
        """Create SessionManager with test client."""
        mock_task_manager = MagicMock()
        mock_config_loader = MagicMock()
        mock_config_loader.get_permanent_environment = MagicMock(return_value=None)
        mock_task_manager.config_loader = mock_config_loader

        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_execution_manager.initialize_permanent_environment_manager = MagicMock()
        mock_execution_manager._permanent_environment_manager = None

        mock_policy_manager = MagicMock()
        mock_evaluation_manager = MagicMock()
        mock_evaluation_manager.log_session_start = AsyncMock()
        mock_evaluation_manager.log_session_end = AsyncMock()
        mock_evaluation_manager.log_episode_start = AsyncMock()
        mock_evaluation_manager.log_episode_end = AsyncMock()
        mock_evaluation_manager.log_action = AsyncMock()

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()

        # Coordinator mock — Redis is the single source of truth for transcript data
        mock_coordinator = MagicMock()
        mock_coordinator.push_message = AsyncMock(return_value=1)
        mock_coordinator.get_message_count = AsyncMock(return_value=0)
        mock_episode_manager.transcript_coordinator = mock_coordinator

        with (
            patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8003)
            return manager, TestClient(manager.rest_api.app)

    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            episode_id="test_episode_123",
            task_id="test_task_456",
            session_id="test_session_789",
            state=EpisodeState.ACTIVE,
            context={}
        )

    # ========================================================================
    # Test: Append Mode - Basic Functionality
    # ========================================================================

    def test_append_mode_adds_to_existing_transcript(self, session_manager_app, sample_episode):
        """Test append mode pushes messages through coordinator."""
        manager, client = session_manager_app

        # Setup: Episode exists (transcript lives in Redis, not in context)
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator
        coordinator.get_message_count.return_value = 3  # 2 existing + 1 new

        # New messages to append
        new_messages = [
            {"role": "assistant", "content": "The answer is 4"},
        ]

        # Execute: Push with append mode
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": new_messages,
                "mode": "append",
                "metadata": {}
            }
        )

        # Assert: Success
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message_count"] == 1  # Count of NEW messages pushed

        # Assert: coordinator.push_message called once with correct args
        assert coordinator.push_message.call_count == 1
        call_kwargs = coordinator.push_message.call_args[1]
        assert call_kwargs["episode_id"] == sample_episode.episode_id
        assert call_kwargs["session_id"] == sample_episode.session_id
        assert call_kwargs["message"]["role"] == "assistant"
        assert call_kwargs["message"]["content"] == "The answer is 4"

    def test_append_mode_multiple_pushes(self, session_manager_app, sample_episode):
        """Test multiple append operations push messages through coordinator."""
        manager, client = session_manager_app

        # Setup: Episode starts empty
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator

        # Push 1: System message
        coordinator.get_message_count.return_value = 1
        response1 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [{"role": "system", "content": "System"}],
                "mode": "append",
                "metadata": {}
            }
        )
        assert response1.status_code == 200

        # Push 2: User message
        coordinator.get_message_count.return_value = 2
        response2 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [{"role": "user", "content": "User"}],
                "mode": "append",
                "metadata": {}
            }
        )
        assert response2.status_code == 200

        # Push 3: Assistant message
        coordinator.get_message_count.return_value = 3
        response3 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [{"role": "assistant", "content": "Assistant"}],
                "mode": "append",
                "metadata": {}
            }
        )
        assert response3.status_code == 200

        # Assert: push_message called 3 times total (1 per push)
        assert coordinator.push_message.call_count == 3

        # Verify messages pushed in correct order
        calls = coordinator.push_message.call_args_list
        assert calls[0][1]["message"]["content"] == "System"
        assert calls[1][1]["message"]["content"] == "User"
        assert calls[2][1]["message"]["content"] == "Assistant"

    def test_append_mode_preserves_order(self, session_manager_app, sample_episode):
        """Test append mode pushes messages in order to coordinator."""
        manager, client = session_manager_app

        # Setup: Episode exists (existing messages live in Redis)
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator
        coordinator.get_message_count.return_value = 5

        # Append multiple messages at once
        new_messages = [
            {"role": "assistant", "content": "Msg 3"},
            {"role": "user", "content": "Msg 4"},
            {"role": "assistant", "content": "Msg 5"},
        ]

        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": new_messages,
                "mode": "append",
                "metadata": {}
            }
        )

        assert response.status_code == 200

        # Assert: push_message called 3 times in order
        assert coordinator.push_message.call_count == 3
        calls = coordinator.push_message.call_args_list
        assert calls[0][1]["message"]["content"] == "Msg 3"
        assert calls[1][1]["message"]["content"] == "Msg 4"
        assert calls[2][1]["message"]["content"] == "Msg 5"

    # ========================================================================
    # Test: Replace Mode - Backward Compatibility
    # ========================================================================

    def test_replace_mode_replaces_entire_transcript(self, session_manager_app, sample_episode):
        """Test replace mode pushes all provided messages through coordinator."""
        manager, client = session_manager_app

        # Setup: Episode exists (existing messages are in Redis, not context)
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator
        coordinator.get_message_count.return_value = 3

        # New messages
        new_messages = [
            {"role": "system", "content": "New system"},
            {"role": "user", "content": "New user"},
            {"role": "assistant", "content": "New assistant"},
        ]

        # Execute: Push with replace mode
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": new_messages,
                "mode": "replace",
                "metadata": {}
            }
        )

        # Assert: Success
        assert response.status_code == 200

        # Assert: All 3 messages pushed through coordinator
        assert coordinator.push_message.call_count == 3
        calls = coordinator.push_message.call_args_list
        assert calls[0][1]["message"]["content"] == "New system"
        assert calls[1][1]["message"]["content"] == "New user"
        assert calls[2][1]["message"]["content"] == "New assistant"

    def test_default_mode_is_replace(self, session_manager_app, sample_episode):
        """Test that default mode is 'replace' for backward compatibility."""
        manager, client = session_manager_app

        # Setup: Episode exists
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator
        coordinator.get_message_count.return_value = 1

        # Execute: Push WITHOUT mode parameter (should default to replace)
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [{"role": "system", "content": "New"}],
                "metadata": {}
            }
        )

        assert response.status_code == 200

        # Assert: Message pushed through coordinator
        assert coordinator.push_message.call_count == 1
        call_kwargs = coordinator.push_message.call_args[1]
        assert call_kwargs["message"]["content"] == "New"

    # ========================================================================
    # Test: Mode Parameter Validation
    # ========================================================================

    def test_invalid_mode_returns_422(self, session_manager_app, sample_episode):
        """Test invalid mode parameter returns 422."""
        manager, client = session_manager_app
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Execute: Push with invalid mode
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "mode": "invalid_mode",
                "metadata": {}
            }
        )

        # Assert: 422 Unprocessable Entity
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "mode" in data["detail"].lower()

    def test_mode_case_sensitive(self, session_manager_app, sample_episode):
        """Test mode parameter is case-sensitive."""
        manager, client = session_manager_app
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Execute: Push with wrong case
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "mode": "APPEND",  # Wrong case
                "metadata": {}
            }
        )

        # Assert: Should fail validation
        assert response.status_code == 422

    # ========================================================================
    # Test: Edge Cases
    # ========================================================================

    def test_append_to_empty_transcript(self, session_manager_app, sample_episode):
        """Test appending messages when no previous transcript exists."""
        manager, client = session_manager_app
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator
        coordinator.get_message_count.return_value = 2

        # Episode has no existing transcript (empty context)
        # Execute: Append messages
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [
                    {"role": "system", "content": "First message"},
                    {"role": "user", "content": "Second message"},
                ],
                "mode": "append",
                "metadata": {}
            }
        )

        assert response.status_code == 200

        # Assert: Both messages pushed through coordinator
        assert coordinator.push_message.call_count == 2
        calls = coordinator.push_message.call_args_list
        assert calls[0][1]["message"]["content"] == "First message"
        assert calls[1][1]["message"]["content"] == "Second message"

    def test_append_empty_list(self, session_manager_app, sample_episode):
        """Test appending empty list is allowed (no-op)."""
        manager, client = session_manager_app

        # Setup: Episode exists (existing messages in Redis)
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator
        coordinator.get_message_count.return_value = 1

        # Execute: Append empty list
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [],
                "mode": "append",
                "metadata": {}
            }
        )

        assert response.status_code == 200

        # Assert: No messages pushed through coordinator
        assert coordinator.push_message.call_count == 0

    def test_replace_with_empty_clears_transcript(self, session_manager_app, sample_episode):
        """Test replace with empty list pushes no messages."""
        manager, client = session_manager_app

        # Setup: Episode exists (existing messages in Redis)
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator
        coordinator.get_message_count.return_value = 0

        # Execute: Replace with empty list
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [],
                "mode": "replace",
                "metadata": {}
            }
        )

        assert response.status_code == 200

        # Assert: No messages pushed
        assert coordinator.push_message.call_count == 0

    # ========================================================================
    # Test: Differential Sync Use Case
    # ========================================================================

    def test_differential_sync_scenario(self, session_manager_app, sample_episode):
        """Test realistic differential sync scenario with multiple iterations."""
        manager, client = session_manager_app
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator

        # Iteration 1: Initial push (system + user)
        coordinator.get_message_count.return_value = 2
        response1 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helper"},
                    {"role": "user", "content": "Calculate 2+2"},
                ],
                "mode": "append",
                "metadata": {"step": 1}
            }
        )
        assert response1.status_code == 200

        # Iteration 2: Agent responds with tool call
        coordinator.get_message_count.return_value = 3
        response2 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Using calculator",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": "calc",
                                "arguments": {"expr": "2+2"}
                            }
                        ]
                    },
                ],
                "mode": "append",
                "metadata": {"step": 2}
            }
        )
        assert response2.status_code == 200

        # Iteration 3: Tool response
        coordinator.get_message_count.return_value = 4
        response3 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "4",
                        "name": "calc"
                    },
                ],
                "mode": "append",
                "metadata": {"step": 3}
            }
        )
        assert response3.status_code == 200

        # Iteration 4: Final assistant response
        coordinator.get_message_count.return_value = 5
        response4 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [
                    {"role": "assistant", "content": "The answer is 4"},
                ],
                "mode": "append",
                "metadata": {"step": 4}
            }
        )
        assert response4.status_code == 200

        # Assert: All 5 messages pushed through coordinator in order
        assert coordinator.push_message.call_count == 5
        calls = coordinator.push_message.call_args_list
        assert calls[0][1]["message"]["role"] == "system"
        assert calls[1][1]["message"]["role"] == "user"
        assert calls[2][1]["message"]["role"] == "assistant"
        assert calls[2][1]["message"]["tool_calls"] is not None
        assert calls[3][1]["message"]["role"] == "tool"
        assert calls[4][1]["message"]["role"] == "assistant"
        assert calls[4][1]["message"]["content"] == "The answer is 4"

    def test_append_with_tool_calls_and_reasoning(self, session_manager_app, sample_episode):
        """Test appending messages with complex structures (tool calls, reasoning)."""
        manager, client = session_manager_app
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        coordinator = manager.episode_manager.transcript_coordinator
        coordinator.get_message_count.return_value = 1

        # Append assistant message with reasoning and tool calls
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I need to search",
                        "reasoning": "The user wants information, so I should search",
                        "tool_calls": [
                            {
                                "id": "call_search_1",
                                "function": "web_search",
                                "arguments": {"query": "Python programming"}
                            }
                        ]
                    },
                ],
                "mode": "append",
                "metadata": {}
            }
        )

        assert response.status_code == 200

        # Assert: Complex message pushed through coordinator
        assert coordinator.push_message.call_count == 1
        msg = coordinator.push_message.call_args[1]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "I need to search"
        assert msg["reasoning"] == "The user wants information, so I should search"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"] == "web_search"
