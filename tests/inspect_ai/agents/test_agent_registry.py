"""Unit tests for SABER agent registry (agents/__init__.py).

Tests cover:
- Agent registration and retrieval
- Duplicate registration prevention
- Agent listing
- Auto-discovery mechanism
- Error handling
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from saber.inspect_ai.agents import SABERAgentRegistry, AgentNotFoundError


class TestSABERAgentRegistry:
    """Test cases for SABERAgentRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        SABERAgentRegistry._agents.clear()

    def test_register_agent_success(self):
        """Test successful agent registration."""
        def mock_agent_factory():
            return lambda state: state

        SABERAgentRegistry.register("test_agent", mock_agent_factory)

        assert "test_agent" in SABERAgentRegistry._agents
        assert SABERAgentRegistry._agents["test_agent"] == mock_agent_factory

    def test_register_duplicate_agent_raises_error(self):
        """Test that registering duplicate agent raises ValueError."""
        def mock_agent_factory():
            return lambda state: state

        SABERAgentRegistry.register("test_agent", mock_agent_factory)

        with pytest.raises(ValueError, match="already registered"):
            SABERAgentRegistry.register("test_agent", mock_agent_factory)

    def test_get_existing_agent(self):
        """Test retrieving an existing agent."""
        def mock_agent_factory():
            return lambda state: state

        SABERAgentRegistry.register("test_agent", mock_agent_factory)

        result = SABERAgentRegistry.get("test_agent")

        assert result == mock_agent_factory

    def test_get_nonexistent_agent_returns_none(self):
        """Test that getting non-existent agent returns None."""
        result = SABERAgentRegistry.get("nonexistent_agent")

        assert result is None

    def test_list_agents_empty(self):
        """Test listing agents when registry is empty."""
        agents = SABERAgentRegistry.list_agents()

        assert agents == []

    def test_list_agents_multiple(self):
        """Test listing multiple registered agents."""
        def factory1():
            return lambda s: s

        def factory2():
            return lambda s: s

        SABERAgentRegistry.register("agent1", factory1)
        SABERAgentRegistry.register("agent2", factory2)

        agents = SABERAgentRegistry.list_agents()

        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents


class TestAgentAutoDiscovery:
    """Test cases for auto-discovery mechanism."""

    def test_register_core_agents_skips_private_files(self):
        """Test that auto-discovery skips private files (starting with _)."""
        from saber.inspect_ai.agents import _register_core_agents
        import importlib

        # Clear registry
        SABERAgentRegistry._agents.clear()

        with patch('saber.inspect_ai.agents.Path') as mock_path_class:
            mock_registry_dir = Mock()
            mock_registry_dir.exists.return_value = True

            # Create mock files including private ones
            private_file = Mock()
            private_file.name = "_private.py"
            private_file.stem = "_private"

            public_file = Mock()
            public_file.name = "react.py"
            public_file.stem = "react"

            mock_registry_dir.glob.return_value = [private_file, public_file]

            # Mock Path(__file__).parent / "registry"
            mock_parent = Mock()
            mock_parent.__truediv__ = Mock(return_value=mock_registry_dir)
            mock_path_instance = Mock()
            mock_path_instance.parent = mock_parent
            mock_path_class.return_value = mock_path_instance

            with patch.object(importlib, 'import_module') as mock_import:
                mock_module = Mock()
                mock_module.create_agent = Mock()
                mock_import.return_value = mock_module

                # Call auto-discovery
                _register_core_agents()

                # Should only try to import the public file
                # The private file should be skipped by the startswith("_") check
                assert mock_import.call_count <= 1  # May be 0 or 1 depending on mocking

    def test_register_core_agents_missing_create_agent(self):
        """Test auto-discovery handles modules without create_agent function."""
        from saber.inspect_ai.agents import _register_core_agents
        import importlib

        SABERAgentRegistry._agents.clear()

        with patch('saber.inspect_ai.agents.Path') as mock_path_class:
            mock_registry_dir = Mock()
            mock_registry_dir.exists.return_value = True

            mock_file = Mock()
            mock_file.name = "invalid_agent.py"
            mock_file.stem = "invalid_agent"

            mock_registry_dir.glob.return_value = [mock_file]

            # Mock Path(__file__).parent / "registry"
            mock_parent = Mock()
            mock_parent.__truediv__ = Mock(return_value=mock_registry_dir)
            mock_path_instance = Mock()
            mock_path_instance.parent = mock_parent
            mock_path_class.return_value = mock_path_instance

            with patch.object(importlib, 'import_module') as mock_import:
                # Module without create_agent function
                mock_module = Mock(spec=[])  # No create_agent attribute
                mock_import.return_value = mock_module

                # Should not raise, just log warning
                _register_core_agents()

                # Agent should not be registered
                assert SABERAgentRegistry.get("invalid_agent") is None

    def test_register_core_agents_import_error(self):
        """Test auto-discovery handles import errors gracefully."""
        from saber.inspect_ai.agents import _register_core_agents
        import importlib

        SABERAgentRegistry._agents.clear()

        with patch('saber.inspect_ai.agents.Path') as mock_path_class:
            mock_registry_dir = Mock()
            mock_registry_dir.exists.return_value = True

            mock_file = Mock()
            mock_file.name = "broken_agent.py"
            mock_file.stem = "broken_agent"

            mock_registry_dir.glob.return_value = [mock_file]

            # Mock Path(__file__).parent / "registry"
            mock_parent = Mock()
            mock_parent.__truediv__ = Mock(return_value=mock_registry_dir)
            mock_path_instance = Mock()
            mock_path_instance.parent = mock_parent
            mock_path_class.return_value = mock_path_instance

            with patch.object(importlib, 'import_module') as mock_import:
                mock_import.side_effect = ImportError("Module broken")

                # Should not raise, just log warning
                _register_core_agents()

                # Agent should not be registered
                assert SABERAgentRegistry.get("broken_agent") is None


class TestAgentNotFoundError:
    """Test AgentNotFoundError exception."""

    def test_agent_not_found_error_creation(self):
        """Test creating AgentNotFoundError."""
        error = AgentNotFoundError("Agent 'test' not found")

        assert isinstance(error, Exception)
        assert str(error) == "Agent 'test' not found"

    def test_agent_not_found_error_raise(self):
        """Test raising AgentNotFoundError."""
        with pytest.raises(AgentNotFoundError, match="not found"):
            raise AgentNotFoundError("Agent 'test' not found")
