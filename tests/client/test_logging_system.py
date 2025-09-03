#!/usr/bin/env python3
"""
Tests for SABER Client-Side Container Logging System

Comprehensive unit tests for the client-side container logging capabilities
including configuration validation, session management, log streaming, and
container log management.
"""

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, mock_open

import pytest

from saber.client.logging import (
    ClientLoggingConfig,
    ContainerLogManager,
    DockerLogCollector,
    SessionLoggingContext,
)


class TestClientLoggingConfig:
    """Test ClientLoggingConfig configuration class."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ClientLoggingConfig()

        assert config.base_log_dir == Path("./client/logs")
        assert config.enable_container_logging is True
        assert config.log_level == "INFO"
        assert config.max_log_size_mb == 50
        assert config.retention_days == 30
        assert config.compress_old_logs is True
        assert config.log_buffer_size == 8192
        assert config.log_collection_timeout == 30
        assert config.include_metadata is True
        assert config.log_format == "json"

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = ClientLoggingConfig(
            base_log_dir=Path("/custom/logs"),
            enable_container_logging=False,
            log_level="DEBUG",
            max_log_size_mb=100,
            retention_days=7,
            compress_old_logs=False,
            log_buffer_size=4096,
            log_collection_timeout=60,
            include_metadata=False,
            log_format="text"
        )

        assert config.base_log_dir == Path("/custom/logs")
        assert config.enable_container_logging is False
        assert config.log_level == "DEBUG"
        assert config.max_log_size_mb == 100
        assert config.retention_days == 7
        assert config.compress_old_logs is False
        assert config.log_buffer_size == 4096
        assert config.log_collection_timeout == 60
        assert config.include_metadata is False
        assert config.log_format == "text"

    def test_config_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = ClientLoggingConfig(base_log_dir="/test/path")
        assert config.base_log_dir == Path("/test/path")
        assert isinstance(config.base_log_dir, Path)

    def test_config_validation_invalid_log_level(self):
        """Test validation of invalid log level."""
        with pytest.raises(ValueError, match="Invalid log_level: INVALID"):
            ClientLoggingConfig(log_level="INVALID")

    def test_config_validation_invalid_log_format(self):
        """Test validation of invalid log format."""
        with pytest.raises(ValueError, match="Invalid log_format: invalid"):
            ClientLoggingConfig(log_format="invalid")

    def test_config_validation_negative_values(self):
        """Test validation of negative numeric values."""
        with pytest.raises(ValueError, match="max_log_size_mb must be positive"):
            ClientLoggingConfig(max_log_size_mb=-1)

        with pytest.raises(ValueError, match="retention_days must be positive"):
            ClientLoggingConfig(retention_days=0)

        with pytest.raises(ValueError, match="log_buffer_size must be positive"):
            ClientLoggingConfig(log_buffer_size=-100)

        with pytest.raises(ValueError, match="log_collection_timeout must be positive"):
            ClientLoggingConfig(log_collection_timeout=0)

    def test_config_case_insensitive_validation(self):
        """Test that log level and format validation are case insensitive."""
        # These should not raise exceptions
        config1 = ClientLoggingConfig(log_level="debug")
        assert config1.log_level == "debug"

        config2 = ClientLoggingConfig(log_format="JSON")
        assert config2.log_format == "JSON"


class TestSessionLoggingContext:
    """Test SessionLoggingContext data class."""

    def test_session_context_creation(self):
        """Test basic session context creation."""
        context = SessionLoggingContext(
            session_id="session-123",
            client_id="client-456",
            task_id="task-789",
            episode_id="episode-abc"
        )

        assert context.session_id == "session-123"
        assert context.client_id == "client-456"
        assert context.task_id == "task-789"
        assert context.episode_id == "episode-abc"

    def test_session_context_optional_fields(self):
        """Test session context with optional fields."""
        context = SessionLoggingContext(session_id="session-123")

        assert context.session_id == "session-123"
        assert context.client_id is None
        assert context.task_id is None
        assert context.episode_id is None

    def test_session_context_partial_fields(self):
        """Test session context with some optional fields set."""
        context = SessionLoggingContext(
            session_id="session-123",
            task_id="task-789"
        )

        assert context.session_id == "session-123"
        assert context.client_id is None
        assert context.task_id == "task-789"
        assert context.episode_id is None


class TestDockerLogCollector:
    """Test DockerLogCollector post-completion log collection functionality."""

    def test_docker_log_collector_initialization(self):
        """Test DockerLogCollector initialization."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        collector = DockerLogCollector(config, context)

        assert collector.config == config
        assert collector.session_context == context
        # Docker client should be initialized
        assert collector.docker_client is not None

    def test_collect_container_logs_success(self):
        """Test successful log collection from container."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")
        collector = DockerLogCollector(config, context)

        # Mock container
        mock_container = Mock()
        mock_container.id = "container123456789"
        mock_container.name = "test-container"
        mock_container.logs.return_value = b"2025-09-01T08:29:36.309108171Z INFO: Test log message\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "container.log"

            # Test log collection
            result = collector.collect_container_logs(
                mock_container,
                output_file,
                "sidecar"
            )

            # Verify success
            assert result is True
            assert output_file.exists()

            # Verify log content
            content = output_file.read_text()
            assert "INFO: Test log message" in content
            assert "test-container" in content

    def test_collect_container_logs_failure(self):
        """Test log collection when container logs fail."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")
        collector = DockerLogCollector(config, context)

        # Mock container that raises exception
        mock_container = Mock()
        mock_container.id = "container123456789"
        mock_container.name = "test-container"
        mock_container.logs.side_effect = Exception("Docker API error")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "container.log"

            # Test log collection
            result = collector.collect_container_logs(
                mock_container,
                output_file,
                "agent"
            )

            # Verify failure
            assert result is False

    def test_collect_logs_from_docker_path_success(self):
        """Test log collection from Docker internal path exists."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")
        collector = DockerLogCollector(config, context)

        container_id = "container123456789abcdef"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "output.log"

            # Since we can't easily mock the internal Docker paths,
            # just test that the method exists and handles missing files gracefully
            result = collector.collect_logs_from_docker_path(
                container_id,
                output_file,
                "sidecar"
            )

            # This should return False since the Docker log file doesn't exist
            assert result is False

    def test_collect_logs_from_docker_path_missing_file(self):
        """Test log collection when Docker log file doesn't exist."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")
        collector = DockerLogCollector(config, context)

        container_id = "nonexistent123456789"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "output.log"

            result = collector.collect_logs_from_docker_path(
                container_id,
                output_file,
                "agent"
            )

            # Verify failure (file doesn't exist)
            assert result is False


class TestContainerLogManager:
    """Test ContainerLogManager session and lifecycle management."""

    def test_container_log_manager_initialization(self):
        """Test ContainerLogManager initialization."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            assert manager.config == config
            assert manager.session_context == context
            assert manager._registered_containers == {}
            assert isinstance(manager.log_collector, DockerLogCollector)

    def test_session_directory_setup(self):
        """Test session directory structure creation."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session-123")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Check that timestamped session directory was created
            session_dirs = list(Path(temp_dir).glob("*_test-session-123"))
            assert len(session_dirs) == 1

            session_dir = session_dirs[0]
            assert session_dir.is_dir()

            # Check subdirectories
            expected_subdirs = [
                "container-logs/sidecar-containers",
                "container-logs/agent-containers",
                "container-events"
            ]

            for subdir in expected_subdirs:
                assert (session_dir / subdir).is_dir()

    def test_session_metadata_creation(self):
        """Test session metadata file creation."""
        config = ClientLoggingConfig(log_level="DEBUG", retention_days=7)
        context = SessionLoggingContext(
            session_id="test-session",
            client_id="test-client",
            task_id="test-task"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Find the meta.json file
            session_dirs = list(Path(temp_dir).glob("*_test-session"))
            meta_file = session_dirs[0] / "meta.json"

            assert meta_file.exists()

            with open(meta_file) as f:
                metadata = json.load(f)

            assert metadata["session_id"] == "test-session"
            assert metadata["client_id"] == "test-client"
            assert metadata["task_id"] == "test-task"
            assert metadata["config"]["log_level"] == "DEBUG"
            assert metadata["config"]["log_format"] == "json"
            assert metadata["config"]["include_metadata"] is True
            assert metadata["log_collection_method"] == "docker_post_completion"

    @pytest.mark.asyncio
    async def test_start_logging_container(self):
        """Test registering container for post-completion log collection."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Mock container
            mock_container = Mock()
            mock_container.id = "container123456789"
            mock_container.name = "test-container"

            result = await manager.start_logging_container(
                mock_container,
                "test-container.log",
                "sidecar"
            )

            # Should return True on success
            assert result is True

            # Check that the container is registered for log collection
            registered = manager.get_registered_containers()
            assert mock_container.id in registered

            registration = registered[mock_container.id]
            assert registration["container_name"] == "test-container"
            assert registration["component_type"] == "sidecar"
            assert "test-container.log" in registration["log_file_name"]

    @pytest.mark.asyncio
    async def test_stop_logging_container(self):
        """Test collecting logs for a registered container."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Mock container
            container_id = "container123456789"
            mock_container = Mock()
            mock_container.id = container_id
            mock_container.name = "test-container"

            # Register the container first
            await manager.start_logging_container(mock_container, "test.log", "agent")

            # Mock the log collection method to succeed
            with patch.object(manager, 'collect_container_logs', return_value=True) as mock_collect:
                result = await manager.stop_logging_container(container_id)

                # Verify collection was attempted
                mock_collect.assert_called_once_with(container_id)
                assert result is True

    @pytest.mark.asyncio
    async def test_cleanup_and_finalize(self):
        """Test cleanup and finalization of logging session."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Register some containers
            mock_container1 = Mock()
            mock_container1.id = "container1"
            mock_container1.name = "test-container-1"

            mock_container2 = Mock()
            mock_container2.id = "container2"
            mock_container2.name = "test-container-2"

            await manager.start_logging_container(mock_container1, "test1.log", "agent")
            await manager.start_logging_container(mock_container2, "test2.log", "sidecar")

            # Mock the log collection to succeed
            with patch.object(manager, 'collect_all_registered_logs', return_value={"container1": True, "container2": True}) as mock_collect:
                await manager.cleanup_and_finalize()

                # Verify all logs were collected
                mock_collect.assert_called_once()

                # Verify registrations were cleared
                assert manager.get_registered_containers() == {}

    @pytest.mark.asyncio
    async def test_log_container_lifecycle_event(self):
        """Test logging container lifecycle events."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Call with correct parameter structure
            manager.log_container_lifecycle_event(
                event_type="container_created",
                container_info={
                    "name": "test-container",
                    "image": "test-image:latest",
                    "component_type": "agent"
                },
                additional_data={"startup_time": "2025-09-01T08:00:00Z"}
            )

            # Check that container events directory was created
            session_dirs = list(Path(temp_dir).glob("*_test-session"))
            events_dir = session_dirs[0] / "container-events"

            assert events_dir.exists()

            # Check that event file was created
            event_files = list(events_dir.glob("*.jsonl"))
            assert len(event_files) > 0

            # Verify event content
            with open(event_files[0]) as f:
                event_content = f.read().strip()

            event_data = json.loads(event_content)
            assert event_data["event_type"] == "container_created"
            assert event_data["container_info"]["name"] == "test-container"
            assert event_data["container_info"]["image"] == "test-image:latest"


class TestLoggingSystemIntegration:
    """Integration tests for the complete logging system."""

    @pytest.mark.asyncio
    async def test_end_to_end_logging_flow(self):
        """Test complete logging flow from setup to cleanup."""
        config = ClientLoggingConfig(log_level="DEBUG", retention_days=1)
        context = SessionLoggingContext(
            session_id="integration-test",
            client_id="test-client",
            task_id="test-task",
            episode_id="test-episode"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            # Initialize manager
            manager = ContainerLogManager(config, context)

            # Verify session directory structure
            session_dirs = list(Path(temp_dir).glob("*_integration-test"))
            assert len(session_dirs) == 1
            session_dir = session_dirs[0]

            # Verify metadata
            meta_file = session_dir / "meta.json"
            assert meta_file.exists()

            # Mock container operations
            mock_container = Mock()
            mock_container.id = "container123456789"
            mock_container.name = "integration-test-container"

            # Log lifecycle event
            manager.log_container_lifecycle_event(
                "container_started",
                {"name": "integration-test-container", "component_type": "agent"}
            )

            # Start logging (new registration-based approach)
            result = await manager.start_logging_container(
                mock_container,
                "integration-test-container.log",
                "agent"
            )

            # Verify container is registered
            assert result is True
            registered = manager.get_registered_containers()
            assert mock_container.id in registered

            # Cleanup
            await manager.cleanup_and_finalize()

            # Verify cleanup
            assert manager.get_registered_containers() == {}

            # Verify system log exists
            events_dir = session_dir / "container-events"
            assert events_dir.exists()

    def test_logging_disabled_configuration(self):
        """Test behavior when logging is disabled."""
        config = ClientLoggingConfig(enable_container_logging=False)
        context = SessionLoggingContext(session_id="disabled-test")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            # Should still initialize without error
            manager = ContainerLogManager(config, context)

            # Should still create session directory (for consistency)
            session_dirs = list(Path(temp_dir).glob("*_disabled-test"))
            assert len(session_dirs) == 1

    def test_concurrent_session_isolation(self):
        """Test that concurrent sessions create isolated log directories."""
        config = ClientLoggingConfig()

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            # Create multiple sessions with same session_id but different timestamps
            context1 = SessionLoggingContext(session_id="shared-session")
            context2 = SessionLoggingContext(session_id="shared-session")

            manager1 = ContainerLogManager(config, context1)
            manager2 = ContainerLogManager(config, context2)

            # Should create separate timestamped directories
            session_dirs = list(Path(temp_dir).glob("*_shared-session"))
            assert len(session_dirs) == 2

            # Directories should be different
            assert manager1.session_log_dir != manager2.session_log_dir
