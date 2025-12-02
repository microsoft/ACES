"""
Tests for Phase 2c: Transparent Agent Interface for InjectPromptExecutor.

Tests automatic target resolution from orchestration metadata and injection strategies.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState, CommandResult
from saber.server.execution.executors.standard_registry.inject_prompt_executor import InjectPromptExecutor


class TestAutomaticTargetResolution:
    """Test automatic target episode ID resolution from orchestration metadata."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode with initial transcript."""
        return Episode(
            episode_id="ep-blue-123",
            task_id="blue-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                ],
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
                MetadataKeys.ORCHESTRATION_ROLE: "blue_team",
            },
        )

    @pytest.fixture
    def red_episode_with_orchestration(self, blue_episode: Episode) -> Episode:
        """Create a red team episode with orchestration metadata."""
        return Episode(
            episode_id="ep-red-456",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                # Phase 2c: Orchestration metadata
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: [blue_episode.episode_id],
                MetadataKeys.ORCHESTRATION_ROLE: "red_team",
            },
        )

    @pytest.fixture
    def red_episode_without_orchestration(self) -> Episode:
        """Create a red team episode without orchestration metadata."""
        return Episode(
            episode_id="ep-red-789",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={},
        )

    @pytest.fixture
    def mock_session_manager(self, blue_episode: Episode, red_episode_with_orchestration: Episode, red_episode_without_orchestration: Episode):
        """Create a mock session manager with episode lookup capability."""
        
        class MockEpisodeManager:
            def __init__(self, *episodes):
                self.episodes = {ep.episode_id: ep for ep in episodes}
            
            def get_episode_by_id(self, episode_id: str):
                return self.episodes.get(episode_id)
        
        class MockSessionManager:
            def __init__(self, *episodes):
                self.episode_manager = MockEpisodeManager(*episodes)
        
        return MockSessionManager(blue_episode, red_episode_with_orchestration, red_episode_without_orchestration)

    @pytest.mark.asyncio
    async def test_automatic_target_resolution_from_orchestration(
        self,
        blue_episode: Episode,
        red_episode_with_orchestration: Episode,
        mock_session_manager
    ):
        """Test that target episode ID is auto-resolved from ORCHESTRATION_TARGET_EPISODES."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=None,
            config={"max_injections_per_episode": 20}
        )
        executor._session_manager = mock_session_manager
        
        # No explicit target_episode_id - should be auto-resolved
        parameters = {
            "message": "Test injection",
            "injection_type": "system"
        }
        context = {
            "session_id": "session-789",
            "episode_id": red_episode_with_orchestration.episode_id,
            # NO target_episode_id parameter!
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        assert result.metadata["target_episode_id"] == blue_episode.episode_id
        assert len(blue_episode.context[MetadataKeys.CLIENT_TRANSCRIPT]) == 2  # System + injected

    @pytest.mark.asyncio
    async def test_explicit_target_overrides_orchestration(
        self,
        blue_episode: Episode,
        red_episode_with_orchestration: Episode,
        mock_session_manager
    ):
        """Test that explicit target_episode_id parameter overrides orchestration metadata."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=None,
            config={"max_injections_per_episode": 20}
        )
        executor._session_manager = mock_session_manager
        
        # Explicit target should take precedence
        parameters = {
            "message": "Test injection",
            "injection_type": "system",
            "target_episode_id": blue_episode.episode_id  # Explicit parameter
        }
        context = {
            "session_id": "session-789",
            "episode_id": red_episode_with_orchestration.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        assert result.metadata["target_episode_id"] == blue_episode.episode_id

    @pytest.mark.asyncio
    async def test_missing_target_without_orchestration(
        self,
        red_episode_without_orchestration: Episode,
        mock_session_manager
    ):
        """Test that executor fails gracefully when no target can be resolved."""
        # Arrange
        executor = InjectPromptExecutor(
            sandbox_manager=None,
            config={"max_injections_per_episode": 20}
        )
        executor._session_manager = mock_session_manager
        
        # No explicit target, no orchestration metadata
        parameters = {
            "message": "Test injection",
            "injection_type": "system"
        }
        context = {
            "session_id": "session-789",
            "episode_id": red_episode_without_orchestration.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is False
        assert "Could not resolve target_episode_id" in result.error

    @pytest.mark.asyncio
    async def test_multiple_target_episodes_uses_first(
        self,
        blue_episode: Episode,
        mock_session_manager
    ):
        """Test that when multiple targets exist, first one is used."""
        # Arrange
        red_episode = Episode(
            episode_id="ep-red-multi",
            task_id="red-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: [
                    blue_episode.episode_id,
                    "ep-blue-other",
                ],
                MetadataKeys.ORCHESTRATION_ROLE: "red_team",
            },
        )
        
        # Add to mock session manager
        mock_session_manager.episode_manager.episodes[red_episode.episode_id] = red_episode
        
        executor = InjectPromptExecutor(
            sandbox_manager=None,
            config={"max_injections_per_episode": 20}
        )
        executor._session_manager = mock_session_manager
        
        parameters = {"message": "Test", "injection_type": "system"}
        context = {
            "session_id": "session-789",
            "episode_id": red_episode.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        assert result.metadata["target_episode_id"] == blue_episode.episode_id  # First target


class TestInjectionStrategies:
    """Test different injection strategies for transcript modification."""

    @pytest.fixture
    def blue_episode_with_messages(self) -> Episode:
        """Create a blue team episode with multiple messages."""
        return Episode(
            episode_id="ep-blue-456",
            task_id="blue-task",
            session_id="session-xyz",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 is 4."},
                    {"role": "user", "content": "What is 3+3?"},
                ],
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.fixture
    def red_episode_with_target(self, blue_episode_with_messages: Episode) -> Episode:
        """Create a red team episode with target in orchestration metadata."""
        return Episode(
            episode_id="ep-red-strat",
            task_id="red-task",
            session_id="session-xyz",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: [blue_episode_with_messages.episode_id],
                MetadataKeys.ORCHESTRATION_ROLE: "red_team",
            },
        )

    @pytest.fixture
    def mock_session_manager(self, blue_episode_with_messages: Episode, red_episode_with_target: Episode):
        """Create a mock session manager."""
        
        class MockEpisodeManager:
            def __init__(self, *episodes):
                self.episodes = {ep.episode_id: ep for ep in episodes}
            
            def get_episode_by_id(self, episode_id: str):
                return self.episodes.get(episode_id)
        
        class MockSessionManager:
            def __init__(self, *episodes):
                self.episode_manager = MockEpisodeManager(*episodes)
        
        return MockSessionManager(blue_episode_with_messages, red_episode_with_target)

    @pytest.mark.asyncio
    async def test_append_strategy_default(
        self,
        blue_episode_with_messages: Episode,
        red_episode_with_target: Episode,
        mock_session_manager
    ):
        """Test append strategy (default) - adds message at end."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        original_length = len(blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT])
        
        parameters = {
            "message": "Injected message",
            "strategy": "append"
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode_with_target.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        transcript = blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == original_length + 1
        assert transcript[-1]["content"] == "Injected message"
        assert transcript[-1]["source"] == "red_team_injection"

    @pytest.mark.asyncio
    async def test_rewind_strategy_removes_then_appends(
        self,
        blue_episode_with_messages: Episode,
        red_episode_with_target: Episode,
        mock_session_manager
    ):
        """Test rewind strategy - removes last N messages then appends."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        original_length = len(blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT])
        assert original_length == 4
        
        parameters = {
            "message": "Injected after rewind",
            "strategy": "rewind",
            "rewind_count": 2  # Remove last 2 messages
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode_with_target.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        transcript = blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Original 4 - 2 removed + 1 injected = 3
        assert len(transcript) == 3
        assert transcript[-1]["content"] == "Injected after rewind"
        # Check that first 2 messages remain
        assert transcript[0]["role"] == "system"
        assert transcript[1]["content"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_rewrite_strategy_replaces_entire_transcript(
        self,
        blue_episode_with_messages: Episode,
        red_episode_with_target: Episode,
        mock_session_manager
    ):
        """Test rewrite strategy - replaces entire transcript."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        parameters = {
            "message": "New transcript content",
            "strategy": "rewrite"
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode_with_target.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        transcript = blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 1  # Only injected message
        assert transcript[0]["content"] == "New transcript content"

    @pytest.mark.asyncio
    async def test_insert_strategy_inserts_at_position(
        self,
        blue_episode_with_messages: Episode,
        red_episode_with_target: Episode,
        mock_session_manager
    ):
        """Test insert strategy - inserts at specific position."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        original_length = len(blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT])
        
        parameters = {
            "message": "Inserted at position 2",
            "strategy": "insert",
            "insert_position": 2  # Insert after first 2 messages
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode_with_target.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        transcript = blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == original_length + 1
        assert transcript[2]["content"] == "Inserted at position 2"
        # Verify surrounding messages
        assert transcript[1]["content"] == "What is 2+2?"
        assert transcript[3]["content"] == "2+2 is 4."

    @pytest.mark.asyncio
    async def test_invalid_strategy_fails(
        self,
        blue_episode_with_messages: Episode,
        red_episode_with_target: Episode,
        mock_session_manager
    ):
        """Test that invalid strategy is rejected."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        parameters = {
            "message": "Test",
            "strategy": "invalid_strategy"
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode_with_target.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is False
        assert "Invalid strategy" in result.error

    @pytest.mark.asyncio
    async def test_rewind_count_clamping(
        self,
        blue_episode_with_messages: Episode,
        red_episode_with_target: Episode,
        mock_session_manager
    ):
        """Test that rewind_count is clamped to valid range."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        original_length = len(blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT])
        
        # Try to rewind more messages than exist
        parameters = {
            "message": "Test",
            "strategy": "rewind",
            "rewind_count": 100  # More than exists
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode_with_target.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        transcript = blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should rewind all messages and add injected one
        assert len(transcript) == 1

    @pytest.mark.asyncio
    async def test_insert_position_clamping(
        self,
        blue_episode_with_messages: Episode,
        red_episode_with_target: Episode,
        mock_session_manager
    ):
        """Test that insert_position is clamped to valid range."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        original_length = len(blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT])
        
        # Try to insert beyond end
        parameters = {
            "message": "Test",
            "strategy": "insert",
            "insert_position": 100  # Beyond end
        }
        context = {
            "session_id": "session-xyz",
            "episode_id": red_episode_with_target.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        transcript = blue_episode_with_messages.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should insert at end
        assert len(transcript) == original_length + 1
        assert transcript[-1]["content"] == "Test"


class TestParameterSchemaUpdates:
    """Test that parameter schema includes new Phase 2c parameters."""

    def test_parameter_schema_includes_strategy(self):
        """Test that strategy parameter is in schema."""
        schema = InjectPromptExecutor.get_parameter_schema()
        
        assert "strategy" in schema
        assert schema["strategy"].required is False
        assert schema["strategy"].default == "append"

    def test_parameter_schema_includes_rewind_count(self):
        """Test that rewind_count parameter is in schema."""
        schema = InjectPromptExecutor.get_parameter_schema()
        
        assert "rewind_count" in schema
        assert schema["rewind_count"].required is False
        assert schema["rewind_count"].default == 1

    def test_parameter_schema_includes_insert_position(self):
        """Test that insert_position parameter is in schema."""
        schema = InjectPromptExecutor.get_parameter_schema()
        
        assert "insert_position" in schema
        assert schema["insert_position"].required is False
        assert schema["insert_position"].default == 0

    def test_parameter_schema_includes_optional_target(self):
        """Test that target_episode_id is optional in schema."""
        schema = InjectPromptExecutor.get_parameter_schema()
        
        assert "target_episode_id" in schema
        assert schema["target_episode_id"].required is False
        assert schema["target_episode_id"].default is None


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode."""
        return Episode(
            episode_id="ep-blue-legacy",
            task_id="blue-task",
            session_id="session-legacy",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                ],
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.fixture
    def red_episode(self) -> Episode:
        """Create a red team episode without orchestration."""
        return Episode(
            episode_id="ep-red-legacy",
            task_id="red-task",
            session_id="session-legacy",
            state=EpisodeState.ACTIVE,
            context={},
        )

    @pytest.fixture
    def mock_session_manager(self, blue_episode: Episode, red_episode: Episode):
        """Create a mock session manager."""
        
        class MockEpisodeManager:
            def __init__(self, *episodes):
                self.episodes = {ep.episode_id: ep for ep in episodes}
            
            def get_episode_by_id(self, episode_id: str):
                return self.episodes.get(episode_id)
        
        class MockSessionManager:
            def __init__(self, *episodes):
                self.episode_manager = MockEpisodeManager(*episodes)
        
        return MockSessionManager(blue_episode, red_episode)

    @pytest.mark.asyncio
    async def test_legacy_explicit_target_still_works(
        self,
        blue_episode: Episode,
        red_episode: Episode,
        mock_session_manager
    ):
        """Test that old code using explicit target_episode_id still works."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        # Old-style parameters with explicit target
        parameters = {
            "message": "Legacy injection",
            "injection_type": "system",
            "target_episode_id": blue_episode.episode_id  # Explicit parameter
        }
        context = {
            "session_id": "session-legacy",
            "episode_id": red_episode.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        assert len(blue_episode.context[MetadataKeys.CLIENT_TRANSCRIPT]) == 2

    @pytest.mark.asyncio
    async def test_legacy_context_target_still_works(
        self,
        blue_episode: Episode,
        red_episode: Episode,
        mock_session_manager
    ):
        """Test that old code using context target_episode_id still works (deprecated)."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        # Old-style context with target
        parameters = {
            "message": "Legacy injection",
            "injection_type": "system"
        }
        context = {
            "session_id": "session-legacy",
            "episode_id": red_episode.episode_id,
            "target_episode_id": blue_episode.episode_id  # In context
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        assert len(blue_episode.context[MetadataKeys.CLIENT_TRANSCRIPT]) == 2

    @pytest.mark.asyncio
    async def test_default_strategy_is_append(
        self,
        blue_episode: Episode,
        red_episode: Episode,
        mock_session_manager
    ):
        """Test that omitting strategy defaults to append (backward compatible)."""
        # Arrange
        executor = InjectPromptExecutor(sandbox_manager=None)
        executor._session_manager = mock_session_manager
        
        # No strategy parameter - should default to append
        parameters = {
            "message": "Default strategy test",
            "target_episode_id": blue_episode.episode_id
        }
        context = {
            "session_id": "session-legacy",
            "episode_id": red_episode.episode_id,
        }
        
        # Act
        result = await executor.execute(parameters, context)
        
        # Assert
        assert result.success is True
        transcript = blue_episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 2
        assert transcript[-1]["content"] == "Default strategy test"
