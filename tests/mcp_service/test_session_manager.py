"""
Tests for MCP Service Session Manager
"""

import pytest
from datetime import datetime, timezone

from saber.client.mcp_service.agent_registry import AgentSessionRegistry, AgentSession


@pytest.mark.asyncio
class TestAgentSessionRegistry:
    """Test session management functionality."""

    async def test_register_new_session(self):
        """Test registering a new agent session."""
        registry = AgentSessionRegistry()

        session = await registry.register_session(
            agent_id="test-agent-1",
            saber_session_id="saber-session-123",
            saber_episode_id="episode-456",
            task_id="task-456"
        )

        assert session.agent_id == "test-agent-1"
        assert session.saber_session_id == "saber-session-123"
        assert session.saber_episode_id == "episode-456"
        assert session.task_id == "task-456"
        assert session.is_active is True

        # Verify session is stored
        retrieved = await registry.get_session("test-agent-1")
        assert retrieved is not None
        assert retrieved.agent_id == "test-agent-1"

    async def test_register_duplicate_session_same_saber_id(self):
        """Test registering duplicate session with same SABER session ID updates existing."""
        registry = AgentSessionRegistry()

        # Register initial session
        session1 = await registry.register_session(
            agent_id="test-agent-1",
            saber_session_id="saber-session-123",
            saber_episode_id="episode-456",
            task_id="task-456"
        )

        # Register same agent with same SABER session but different task
        session2 = await registry.register_session(
            agent_id="test-agent-1",
            saber_session_id="saber-session-123",
            saber_episode_id="episode-456",
            task_id="task-789"
        )

        # Should be the same session object, updated
        assert session2.agent_id == "test-agent-1"
        assert session2.saber_session_id == "saber-session-123"
        assert session2.task_id == "task-789"  # Updated

        # Should only have one session
        sessions = await registry.list_active_sessions()
        assert len(sessions) == 1

    async def test_register_duplicate_session_different_saber_id(self):
        """Test registering duplicate session with different SABER session ID raises error."""
        registry = AgentSessionRegistry()

        # Register initial session
        await registry.register_session(
            agent_id="test-agent-1",
            saber_session_id="saber-session-123",
            saber_episode_id="episode-456"
        )

        # Try to register same agent with different SABER session
        with pytest.raises(ValueError, match="already registered with different session"):
            await registry.register_session(
                agent_id="test-agent-1",
                saber_session_id="saber-session-456",
                saber_episode_id="episode-789"
            )

    async def test_unregister_session(self):
        """Test unregistering a session."""
        registry = AgentSessionRegistry()

        # Register session
        await registry.register_session(
            agent_id="test-agent-1",
            saber_session_id="saber-session-123",
            saber_episode_id="episode-456"
        )

        # Verify it exists
        session = await registry.get_session("test-agent-1")
        assert session is not None

        # Unregister
        success = await registry.unregister_session("test-agent-1")
        assert success is True

        # Verify it's gone
        session = await registry.get_session("test-agent-1")
        assert session is None

        # Unregistering again should return False
        success = await registry.unregister_session("test-agent-1")
        assert success is False

    async def test_get_saber_session_id(self):
        """Test getting SABER session ID for routing."""
        registry = AgentSessionRegistry()

        # Non-existent agent
        saber_id = await registry.get_saber_session_id("nonexistent")
        assert saber_id is None

        # Register agent
        await registry.register_session(
            agent_id="test-agent-1",
            saber_session_id="saber-session-123",
            saber_episode_id="episode-456"
        )

        # Get SABER session ID
        saber_id = await registry.get_saber_session_id("test-agent-1")
        assert saber_id == "saber-session-123"

    async def test_list_active_sessions(self):
        """Test listing active sessions."""
        registry = AgentSessionRegistry()

        # Initially empty
        sessions = await registry.list_active_sessions()
        assert len(sessions) == 0

        # Register multiple sessions
        await registry.register_session("agent-1", "saber-1", "episode-1")
        await registry.register_session("agent-2", "saber-2", "episode-2")
        await registry.register_session("agent-3", "saber-3", "episode-3")

        # Deactivate one
        await registry.deactivate_session("agent-2")

        # List active (should exclude deactivated)
        sessions = await registry.list_active_sessions()
        assert len(sessions) == 2
        assert "agent-1" in sessions
        assert "agent-3" in sessions
        assert "agent-2" not in sessions

    async def test_session_stats(self):
        """Test getting session statistics."""
        registry = AgentSessionRegistry()

        # Initially empty
        stats = await registry.get_session_stats()
        assert stats["total_sessions"] == 0
        assert stats["active_sessions"] == 0
        assert stats["inactive_sessions"] == 0

        # Register sessions
        await registry.register_session("agent-1", "saber-1", "episode-1")
        await registry.register_session("agent-2", "saber-2", "episode-2")

        # Deactivate one
        await registry.deactivate_session("agent-1")

        # Check stats
        stats = await registry.get_session_stats()
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 1
        assert stats["inactive_sessions"] == 1


class TestAgentSession:
    """Test AgentSession data class."""

    def test_agent_session_creation(self):
        """Test creating an agent session."""
        session = AgentSession(
            agent_id="test-agent",
            saber_session_id="saber-123",
            saber_episode_id="episode-456",
            task_id="task-456"
        )

        assert session.agent_id == "test-agent"
        assert session.saber_session_id == "saber-123"
        assert session.saber_episode_id == "episode-456"
        assert session.task_id == "task-456"
        assert session.is_active is True
        assert isinstance(session.registered_at, datetime)
        assert isinstance(session.last_activity, datetime)

    def test_update_activity(self):
        """Test updating last activity timestamp."""
        session = AgentSession(
            agent_id="test-agent",
            saber_session_id="saber-123",
            saber_episode_id="episode-456"
        )

        original_time = session.last_activity

        # Wait a small amount and update
        import time
        time.sleep(0.01)
        session.update_activity()

        assert session.last_activity > original_time
