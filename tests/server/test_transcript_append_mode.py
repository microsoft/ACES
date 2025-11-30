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

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saber.models.constants import MetadataKeys
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
        """Test append mode adds new messages to existing transcript."""
        manager, client = session_manager_app

        # Setup: Episode with existing transcript
        existing_messages = [
            {"role": "system", "content": "You are a helper"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = existing_messages
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

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

        # Assert: Transcript has all messages (existing + new)
        stored_transcript = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored_transcript) == 3
        assert stored_transcript[0]["role"] == "system"
        assert stored_transcript[1]["role"] == "user"
        assert stored_transcript[2]["role"] == "assistant"
        assert stored_transcript[2]["content"] == "The answer is 4"

    def test_append_mode_multiple_pushes(self, session_manager_app, sample_episode):
        """Test multiple append operations accumulate messages."""
        manager, client = session_manager_app

        # Setup: Episode starts empty
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Push 1: System message
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
        response3 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [{"role": "assistant", "content": "Assistant"}],
                "mode": "append",
                "metadata": {}
            }
        )
        assert response3.status_code == 200

        # Assert: All 3 messages accumulated
        stored_transcript = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored_transcript) == 3
        assert stored_transcript[0]["content"] == "System"
        assert stored_transcript[1]["content"] == "User"
        assert stored_transcript[2]["content"] == "Assistant"

    def test_append_mode_preserves_order(self, session_manager_app, sample_episode):
        """Test append mode preserves message order."""
        manager, client = session_manager_app

        # Setup: Episode with existing messages
        existing = [
            {"role": "system", "content": "Msg 1"},
            {"role": "user", "content": "Msg 2"},
        ]
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = existing
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Append multiple messages
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

        # Assert: Order preserved
        stored = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored) == 5
        assert stored[0]["content"] == "Msg 1"
        assert stored[1]["content"] == "Msg 2"
        assert stored[2]["content"] == "Msg 3"
        assert stored[3]["content"] == "Msg 4"
        assert stored[4]["content"] == "Msg 5"

    # ========================================================================
    # Test: Replace Mode - Backward Compatibility
    # ========================================================================

    def test_replace_mode_replaces_entire_transcript(self, session_manager_app, sample_episode):
        """Test replace mode replaces entire transcript (default behavior)."""
        manager, client = session_manager_app

        # Setup: Episode with existing transcript
        existing_messages = [
            {"role": "system", "content": "Old system"},
            {"role": "user", "content": "Old user"},
        ]
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = existing_messages
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # New messages to replace with
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

        # Assert: Old transcript replaced with new
        stored_transcript = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored_transcript) == 3
        assert stored_transcript[0]["content"] == "New system"
        assert stored_transcript[1]["content"] == "New user"
        assert stored_transcript[2]["content"] == "New assistant"

    def test_default_mode_is_replace(self, session_manager_app, sample_episode):
        """Test that default mode is 'replace' for backward compatibility."""
        manager, client = session_manager_app

        # Setup: Episode with existing transcript
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = [
            {"role": "system", "content": "Old"}
        ]
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Execute: Push WITHOUT mode parameter (should default to replace)
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json={
                "messages": [{"role": "system", "content": "New"}],
                "metadata": {}
            }
        )

        assert response.status_code == 200

        # Assert: Transcript replaced (not appended)
        stored = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored) == 1
        assert stored[0]["content"] == "New"

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
        """Test appending to empty transcript works correctly."""
        manager, client = session_manager_app
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

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

        # Assert: Messages stored correctly
        stored = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored) == 2
        assert stored[0]["content"] == "First message"

    def test_append_empty_list(self, session_manager_app, sample_episode):
        """Test appending empty list is allowed (no-op)."""
        manager, client = session_manager_app

        # Setup: Episode with existing messages
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = [
            {"role": "system", "content": "Existing"}
        ]
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

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

        # Assert: Existing messages unchanged
        stored = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored) == 1
        assert stored[0]["content"] == "Existing"

    def test_replace_with_empty_clears_transcript(self, session_manager_app, sample_episode):
        """Test replace with empty list clears transcript."""
        manager, client = session_manager_app

        # Setup: Episode with existing messages
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = [
            {"role": "system", "content": "To be deleted"}
        ]
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

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

        # Assert: Transcript cleared
        stored = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert stored == []

    # ========================================================================
    # Test: Differential Sync Use Case
    # ========================================================================

    def test_differential_sync_scenario(self, session_manager_app, sample_episode):
        """Test realistic differential sync scenario with multiple iterations."""
        manager, client = session_manager_app
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

        # Iteration 1: Initial push (system + user)
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

        # Assert: All messages accumulated in order
        stored = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored) == 5
        assert stored[0]["role"] == "system"
        assert stored[1]["role"] == "user"
        assert stored[2]["role"] == "assistant"
        assert stored[2]["tool_calls"] is not None
        assert stored[3]["role"] == "tool"
        assert stored[4]["role"] == "assistant"
        assert stored[4]["content"] == "The answer is 4"

    def test_append_with_tool_calls_and_reasoning(self, session_manager_app, sample_episode):
        """Test appending messages with complex structures (tool calls, reasoning)."""
        manager, client = session_manager_app
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)

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

        # Assert: Complex message stored correctly
        stored = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored) == 1
        msg = stored[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "I need to search"
        assert msg["reasoning"] == "The user wants information, so I should search"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"] == "web_search"
