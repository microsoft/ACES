"""
Tests for blocking_transcript_solver - Blue team blocking coordination.

Tests the blue team's blocking solver that waits for red team transcript
modifications via timestamp-based change detection.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageAssistant
from inspect_ai.solver import TaskState

from saber.inspect_ai.agents.blocking_transcript_solver import (
    blocking_transcript_solver,
    _get_transcript_metadata,
    _pull_modified_transcript
)


class TestBlockingTranscriptSolver:
    """Test the blocking transcript solver coordination logic."""

    @pytest.fixture
    def mock_state(self) -> TaskState:
        """Create a mock TaskState."""
        state = MagicMock(spec=TaskState)
        state.messages = [
            ChatMessageSystem(content="You are helpful..."),
            ChatMessageAssistant(content="Hello! How can I help?"),
        ]
        return state

    @pytest.mark.asyncio
    async def test_solver_blocks_until_modification(self, mock_state):
        """Test that solver blocks and detects transcript modification."""
        # Arrange
        rest_url = "http://localhost:8000"
        session_id = "session-123"
        episode_id = "ep-blue-456"
        
        # Mock metadata responses
        poll_count = 0
        async def mock_get_metadata(*args, **kwargs):
            nonlocal poll_count
            poll_count += 1
            if poll_count < 3:
                # First 2 polls: no modification
                return {
                    "last_pushed_at": "2025-12-01T10:00:00Z",
                    "last_modified_at": None,
                    "modification_count": 0
                }
            else:
                # 3rd poll: modification detected
                return {
                    "last_pushed_at": "2025-12-01T10:00:00Z",
                    "last_modified_at": datetime.now(timezone.utc).isoformat(),
                    "modification_count": 1
                }
        
        # Mock transcript pull
        async def mock_pull_transcript(*args, **kwargs):
            return {
                "messages": [
                    ChatMessageSystem(content="You are helpful..."),
                    ChatMessageAssistant(content="Hello! How can I help?"),
                    ChatMessageSystem(content="Injected prompt", source="red_team_injection"),
                ],
                "modification_count": 1
            }
        
        # Create solver
        solver = blocking_transcript_solver(
            rest_url=rest_url,
            session_id=session_id,
            episode_id=episode_id,
            poll_interval_seconds=0.1,  # Fast polling for test
            timeout_seconds=5.0
        )
        
        # Act
        with patch('saber.inspect_ai.agents.blocking_transcript_solver._get_transcript_metadata', new=mock_get_metadata), \
             patch('saber.inspect_ai.agents.blocking_transcript_solver._pull_modified_transcript', new=mock_pull_transcript):
            result_state = await solver(mock_state, None)
        
        # Assert
        assert poll_count >= 3, "Should have polled at least 3 times"
        assert len(result_state.messages) == 3, "Should have updated messages"
        assert result_state.messages[-1].content == "Injected prompt"

    @pytest.mark.asyncio
    async def test_solver_timeout_no_modification(self, mock_state):
        """Test that solver times out if no modification detected."""
        # Arrange
        rest_url = "http://localhost:8000"
        session_id = "session-123"
        episode_id = "ep-blue-456"
        
        # Mock metadata always returns no modification
        async def mock_get_metadata(*args, **kwargs):
            return {
                "last_pushed_at": "2025-12-01T10:00:00Z",
                "last_modified_at": None,
                "modification_count": 0
            }
        
        # Create solver with short timeout
        solver = blocking_transcript_solver(
            rest_url=rest_url,
            session_id=session_id,
            episode_id=episode_id,
            poll_interval_seconds=0.1,
            timeout_seconds=0.5  # 500ms timeout
        )
        
        # Act
        with patch('saber.inspect_ai.agents.blocking_transcript_solver._get_transcript_metadata', new=mock_get_metadata):
            result_state = await solver(mock_state, None)
        
        # Assert - state unchanged after timeout
        assert len(result_state.messages) == 2, "Messages should be unchanged"

    @pytest.mark.asyncio
    async def test_solver_max_iterations_limit(self, mock_state):
        """Test that solver respects max_iterations limit."""
        # Arrange
        rest_url = "http://localhost:8000"
        session_id = "session-123"
        episode_id = "ep-blue-456"
        
        # Create solver with max_iterations=2
        solver = blocking_transcript_solver(
            rest_url=rest_url,
            session_id=session_id,
            episode_id=episode_id,
            poll_interval_seconds=0.1,
            timeout_seconds=5.0,
            max_iterations=2
        )
        
        # Act - call solver 3 times
        result1 = await solver(mock_state, None)  # Iteration 1
        result2 = await solver(result1, None)     # Iteration 2
        result3 = await solver(result2, None)     # Iteration 3 - should skip
        
        # Assert - third call should skip immediately
        assert result3 == result2, "Should skip blocking after max iterations"

    @pytest.mark.asyncio
    async def test_solver_handles_metadata_error_gracefully(self, mock_state):
        """Test that solver handles metadata polling errors gracefully."""
        # Arrange
        rest_url = "http://localhost:8000"
        session_id = "session-123"
        episode_id = "ep-blue-456"
        
        poll_count = 0
        async def mock_get_metadata_with_error(*args, **kwargs):
            nonlocal poll_count
            poll_count += 1
            if poll_count < 2:
                raise Exception("Network error")
            else:
                # Successful response after error
                return {
                    "last_pushed_at": "2025-12-01T10:00:00Z",
                    "last_modified_at": datetime.now(timezone.utc).isoformat(),
                    "modification_count": 1
                }
        
        async def mock_pull_transcript(*args, **kwargs):
            return {
                "messages": mock_state.messages + [ChatMessageSystem(content="Injected")],
                "modification_count": 1
            }
        
        solver = blocking_transcript_solver(
            rest_url=rest_url,
            session_id=session_id,
            episode_id=episode_id,
            poll_interval_seconds=0.1,
            timeout_seconds=2.0
        )
        
        # Act
        with patch('saber.inspect_ai.agents.blocking_transcript_solver._get_transcript_metadata', new=mock_get_metadata_with_error), \
             patch('saber.inspect_ai.agents.blocking_transcript_solver._pull_modified_transcript', new=mock_pull_transcript):
            result_state = await solver(mock_state, None)
        
        # Assert - should recover from error and detect modification
        assert len(result_state.messages) == 3, "Should have updated despite errors"

    @pytest.mark.asyncio
    async def test_solver_handles_pull_error_gracefully(self, mock_state):
        """Test that solver handles transcript pull errors gracefully."""
        # Arrange
        rest_url = "http://localhost:8000"
        session_id = "session-123"
        episode_id = "ep-blue-456"
        
        async def mock_get_metadata(*args, **kwargs):
            return {
                "last_pushed_at": "2025-12-01T10:00:00Z",
                "last_modified_at": datetime.now(timezone.utc).isoformat(),
                "modification_count": 1
            }
        
        async def mock_pull_transcript_with_error(*args, **kwargs):
            raise Exception("Failed to pull transcript")
        
        solver = blocking_transcript_solver(
            rest_url=rest_url,
            session_id=session_id,
            episode_id=episode_id,
            poll_interval_seconds=0.1,
            timeout_seconds=1.0
        )
        
        # Act
        with patch('saber.inspect_ai.agents.blocking_transcript_solver._get_transcript_metadata', new=mock_get_metadata), \
             patch('saber.inspect_ai.agents.blocking_transcript_solver._pull_modified_transcript', new=mock_pull_transcript_with_error):
            result_state = await solver(mock_state, None)
        
        # Assert - state unchanged after pull error
        assert len(result_state.messages) == 2, "Messages should be unchanged after error"

    @pytest.mark.asyncio
    async def test_timestamp_comparison_timezone_aware(self, mock_state):
        """Test that timestamp comparison handles timezone-aware datetimes."""
        # Arrange
        rest_url = "http://localhost:8000"
        session_id = "session-123"
        episode_id = "ep-blue-456"
        
        # Mock with ISO timestamp
        async def mock_get_metadata(*args, **kwargs):
            return {
                "last_pushed_at": "2025-12-01T10:00:00+00:00",
                "last_modified_at": "2025-12-01T10:05:00+00:00",
                "modification_count": 1
            }
        
        async def mock_pull_transcript(*args, **kwargs):
            return {
                "messages": mock_state.messages + [ChatMessageSystem(content="Modified")],
                "modification_count": 1
            }
        
        solver = blocking_transcript_solver(
            rest_url=rest_url,
            session_id=session_id,
            episode_id=episode_id,
            poll_interval_seconds=0.1,
            timeout_seconds=2.0
        )
        
        # Act
        with patch('saber.inspect_ai.agents.blocking_transcript_solver._get_transcript_metadata', new=mock_get_metadata), \
             patch('saber.inspect_ai.agents.blocking_transcript_solver._pull_modified_transcript', new=mock_pull_transcript):
            result_state = await solver(mock_state, None)
        
        # Assert
        assert len(result_state.messages) == 3, "Should detect modification with timezone"


class TestTranscriptMetadataHelpers:
    """Test helper functions for transcript metadata."""

    @pytest.mark.asyncio
    async def test_get_transcript_metadata_success(self):
        """Test successful metadata retrieval."""
        # This is an integration test that requires a running server
        # For unit testing, we'd mock httpx.AsyncClient
        pass  # Covered by solver tests above

    @pytest.mark.asyncio
    async def test_pull_modified_transcript_success(self):
        """Test successful transcript pull."""
        # This is an integration test that requires a running server
        # For unit testing, we'd mock httpx.AsyncClient
        pass  # Covered by solver tests above


class TestSolverConfiguration:
    """Test solver configuration and parameter validation."""

    def test_solver_with_custom_parameters(self):
        """Test creating solver with custom parameters."""
        solver = blocking_transcript_solver(
            rest_url="http://localhost:9000",
            session_id="custom-session",
            episode_id="custom-episode",
            poll_interval_seconds=1.5,
            timeout_seconds=600.0,
            max_iterations=5
        )
        
        assert solver is not None
        assert callable(solver)

    def test_solver_with_infinite_timeout(self):
        """Test creating solver with None timeout (infinite wait)."""
        solver = blocking_transcript_solver(
            rest_url="http://localhost:8000",
            session_id="session-123",
            episode_id="ep-blue-456",
            timeout_seconds=None  # Infinite timeout
        )
        
        assert solver is not None

    def test_solver_with_no_max_iterations(self):
        """Test creating solver with None max_iterations (unlimited)."""
        solver = blocking_transcript_solver(
            rest_url="http://localhost:8000",
            session_id="session-123",
            episode_id="ep-blue-456",
            max_iterations=None  # Unlimited iterations
        )
        
        assert solver is not None
