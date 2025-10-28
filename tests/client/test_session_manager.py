"""
Unit tests for ClientSessionManager with REST-only integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.client.client_session import ClientSessionManager
from saber.client.models import SessionManagerConfig


class TestSessionManagerConfig:
    """Test SessionManagerConfig model."""

    def test_session_manager_config_creation(self):
        """Test creating SessionManagerConfig."""
        config = SessionManagerConfig(
            base_url="http://test:8000",
            client_id="test-client",
            mcp_server_url="http://test:8001"
        )

        assert config.base_url == "http://test:8000"
        assert config.client_id == "test-client"
        assert config.mcp_server_url == "http://test:8001"
        assert config.rest_timeout == 300.0


class TestClientSessionManager:
    """Test ClientSessionManager functionality."""

    @pytest.fixture
    def session_config(self):
        """Create test session manager configuration."""
        return SessionManagerConfig(
            base_url="http://test:8000",
            client_id="test-client",
            mcp_server_url="http://test:8001"
        )

    @pytest.fixture
    def session_manager(self, session_config):
        """Create session manager for testing."""
        return ClientSessionManager(session_config)

    def test_session_manager_initialization(self, session_manager, session_config):
        """Test session manager initialization."""
        assert session_manager.config == session_config
        assert session_manager.base_url == "http://test:8000"
        assert session_manager.client_id == "test-client"
        assert session_manager._current_session_id is None

    @pytest.mark.asyncio
    async def test_create_session_success(self, session_manager):
        """Test successful session creation."""
        with patch.object(session_manager, 'create_session', return_value="test-session-123") as mock_create:
            session_id = await session_manager.create_session()

            assert session_id == "test-session-123"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_failure(self, session_manager):
        """Test session creation failure."""
        with patch.object(session_manager, 'create_session', side_effect=Exception("Failed to create session: 500 - Internal Server Error")):
            with pytest.raises(Exception) as exc_info:
                await session_manager.create_session()

            assert "Failed to create session: 500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_current_session_id(self, session_manager):
        """Test getting current session ID."""
        assert session_manager.get_current_session_id() is None

        session_manager._current_session_id = "test-session"
        assert session_manager.get_current_session_id() == "test-session"

    @pytest.mark.asyncio
    async def test_cleanup(self, session_manager):
        """Test session cleanup."""
        # This should not raise any exceptions
        await session_manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
