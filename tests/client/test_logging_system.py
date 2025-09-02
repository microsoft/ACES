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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from saber.client.logging import (
    ClientLoggingConfig,
    ContainerLogManager,
    LogStreamCollector,
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


class TestLogStreamCollector:
    """Test LogStreamCollector log streaming functionality."""

    def test_log_stream_collector_initialization(self):
        """Test LogStreamCollector initialization."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        collector = LogStreamCollector(config, context)

        assert collector.config == config
        assert collector.session_context == context
        assert collector._active_streams == {}

    @pytest.mark.asyncio
    async def test_stream_container_logs_setup(self):
        """Test log streaming setup and directory creation."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")
        collector = LogStreamCollector(config, context)

        # Mock container
        mock_container = Mock()
        mock_container.id = "container123456789"
        mock_container.logs.return_value = iter([b"test log line\n"])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test-logs" / "container.log"

            # Mock the internal streaming method to avoid Docker API calls
            with patch.object(collector, '_stream_logs_to_file', new_callable=AsyncMock) as mock_stream:
                await collector.stream_container_logs(
                    mock_container,
                    output_file,
                    "test-container",
                    "sidecar"
                )

                # Verify the streaming method was called with correct parameters
                mock_stream.assert_called_once_with(
                    mock_container,
                    output_file,
                    "test-container",
                    "sidecar"
                )

    @pytest.mark.asyncio
    async def test_stream_logs_cancellation(self):
        """Test that log streaming handles cancellation gracefully."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")
        collector = LogStreamCollector(config, context)

        mock_container = Mock()
        mock_container.id = "container123456789"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "container.log"

            # Mock the internal method to raise CancelledError
            with patch.object(collector, '_stream_logs_to_file', new_callable=AsyncMock) as mock_stream:
                mock_stream.side_effect = asyncio.CancelledError()

                # This should not raise an exception
                await collector.stream_container_logs(
                    mock_container,
                    output_file,
                    "test-container",
                    "agent"
                )

    def test_format_log_entry_json(self):
        """Test JSON log entry formatting."""
        config = ClientLoggingConfig(log_format="json")
        context = SessionLoggingContext(
            session_id="session-123",
            client_id="client-456",
            task_id="task-789"
        )
        collector = LogStreamCollector(config, context)

        # Mock log line with timestamp
        log_line = "2025-09-01T08:29:36.309108171Z INFO: Test log message"

        formatted = collector._format_log_entry(log_line, "test-container", "sidecar")

        # Parse the JSON to validate structure
        log_entry = json.loads(formatted)

        assert log_entry["container_name"] == "test-container"
        assert log_entry["component_type"] == "sidecar"
        assert log_entry["session_id"] == "session-123"
        assert log_entry["task_id"] == "task-789"
        assert log_entry["content"] == "INFO: Test log message"
        assert log_entry["level"] == "INFO"

    def test_format_log_entry_without_timestamp(self):
        """Test log entry formatting when no Docker timestamp is present."""
        config = ClientLoggingConfig(log_format="json")
        context = SessionLoggingContext(session_id="session-123")
        collector = LogStreamCollector(config, context)

        log_line = "Plain log message without timestamp"

        formatted = collector._format_log_entry(log_line, "test-container", "agent")
        log_entry = json.loads(formatted)

        assert log_entry["content"] == "Plain log message without timestamp"
        assert log_entry["container_name"] == "test-container"
        assert log_entry["component_type"] == "agent"
        # Timestamp should be generated
        assert "timestamp" in log_entry


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
            assert manager._active_loggers == {}
            assert isinstance(manager.log_collector, LogStreamCollector)

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
                "container-logs/client-containers",
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
            assert metadata["config"]["retention_days"] == 7

    @pytest.mark.asyncio
    async def test_start_logging_container(self):
        """Test starting container logging."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Mock container
            mock_container = Mock()
            mock_container.id = "container123456789"
            mock_container.name = "test-container"

            # Mock the log collector's stream method
            with patch.object(manager.log_collector, 'stream_container_logs', new_callable=AsyncMock) as mock_stream:
                result = await manager.start_logging_container(
                    mock_container,
                    "test-container.log",
                    "sidecar"
                )

                # Verify stream was called
                mock_stream.assert_called_once()

                # Should return True on success
                assert result is True

                # Check that the container is tracked using short container ID
                container_short_id = mock_container.id[:12]
                assert container_short_id in manager._active_loggers

    @pytest.mark.asyncio
    async def test_stop_logging_container(self):
        """Test stopping container logging."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Create a mock task that behaves like an asyncio.Task
            mock_task = Mock()
            mock_task.done.return_value = False
            mock_task.cancel = Mock()

            # Use exact container ID
            container_id = "container123"
            manager._active_loggers[container_id] = mock_task

            # Add mock timeout handling for the wait_for
            with patch('asyncio.wait_for', new_callable=AsyncMock) as mock_wait:
                mock_wait.side_effect = asyncio.CancelledError()  # Simulate expected cancellation

                result = await manager.stop_logging_container(container_id)

            # Verify task was cancelled and result is True
            mock_task.cancel.assert_called_once()
            assert result is True
            assert container_id not in manager._active_loggers

    @pytest.mark.asyncio
    async def test_cleanup_and_finalize(self):
        """Test cleanup and finalization of logging session."""
        config = ClientLoggingConfig()
        context = SessionLoggingContext(session_id="test-session")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_log_dir = Path(temp_dir)

            manager = ContainerLogManager(config, context)

            # Add mock active loggers
            mock_task1 = AsyncMock()
            mock_task2 = AsyncMock()
            mock_task1.done.return_value = False
            mock_task2.done.return_value = False
            manager._active_loggers["container1"] = mock_task1
            manager._active_loggers["container2"] = mock_task2

            # Mock the collector's stop method
            with patch.object(manager.log_collector, 'stop_all_streams', new_callable=AsyncMock) as mock_stop:
                await manager.cleanup_and_finalize()

                # Verify all streams were stopped
                mock_stop.assert_called_once()

                # Verify active loggers were cleared (the cleanup happens internally)
                assert manager._active_loggers == {}

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

            # Start logging (mocked)
            with patch.object(manager.log_collector, 'stream_container_logs', new_callable=AsyncMock):
                result = await manager.start_logging_container(
                    mock_container,
                    "integration-test-container.log",
                    "agent"
                )

                # Verify container is being tracked
                assert result is True
                container_short_id = mock_container.id[:12]
                assert container_short_id in manager._active_loggers

            # Cleanup
            await manager.cleanup_and_finalize()

            # Verify cleanup
            assert manager._active_loggers == {}

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
