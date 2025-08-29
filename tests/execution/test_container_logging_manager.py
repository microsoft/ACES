"""
Tests for the ContainerLoggingManager functionality.
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from saber.server.execution.logging import ContainerLoggingManager


class TestContainerLoggingManager:
    """Test ContainerLoggingManager functionality."""

    @pytest.fixture
    def temp_logs_dir(self):
        """Create a temporary directory for logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def logging_config(self, temp_logs_dir):
        """Create logging configuration for tests."""
        return {
            "domain": "test_domain",
            "logs_directory": temp_logs_dir,
            "enable_logging": True
        }

    @pytest.fixture
    def disabled_logging_config(self, temp_logs_dir):
        """Create disabled logging configuration for tests."""
        return {
            "domain": "test_domain",
            "logs_directory": temp_logs_dir,
            "enable_logging": False
        }

    def test_init_enabled(self, logging_config):
        """Test ContainerLoggingManager initialization with logging enabled."""
        manager = ContainerLoggingManager(logging_config)

        assert manager.enable_logging is True
        assert manager.domain == "test_domain"
        assert manager.logs_directory == Path(logging_config["logs_directory"])

        # Check base subdirectories were created
        # Note: Environment-specific subdirectories (permanent-environments, etc.)
        # are created on-demand when needed
        expected_subdirs = [
            "compose-configs",
            "container-logs"
        ]

        for subdir in expected_subdirs:
            assert (manager.logs_directory / subdir).exists()

    def test_init_disabled(self, disabled_logging_config):
        """Test ContainerLoggingManager initialization with logging disabled."""
        manager = ContainerLoggingManager(disabled_logging_config)

        assert manager.enable_logging is False

    def test_log_compose_config_enabled(self, logging_config):
        """Test logging docker compose configuration when enabled."""
        manager = ContainerLoggingManager(logging_config)

        compose_config = {
            "version": "3.8",
            "services": {
                "web": {
                    "image": "nginx:latest",
                    "ports": ["80:80"]
                }
            }
        }

        metadata = {
            "services_count": 1,
            "test_metadata": "value"
        }

        result_path = manager.log_compose_config(
            compose_config=compose_config,
            config_type="sandbox",
            identifier="test-session-123",
            additional_metadata=metadata
        )

        assert result_path is not None
        assert Path(result_path).exists()

        # Verify content
        with open(result_path, 'r') as f:
            logged_data = yaml.safe_load(f)

        assert logged_data["metadata"]["domain"] == "test_domain"
        assert logged_data["metadata"]["config_type"] == "sandbox"
        assert logged_data["metadata"]["identifier"] == "test-session-123"
        assert logged_data["metadata"]["services_count"] == 1
        assert logged_data["metadata"]["test_metadata"] == "value"
        assert logged_data["docker_compose_config"] == compose_config

    def test_log_compose_config_disabled(self, disabled_logging_config):
        """Test logging docker compose configuration when disabled."""
        manager = ContainerLoggingManager(disabled_logging_config)

        result = manager.log_compose_config(
            compose_config={"test": "config"},
            config_type="sandbox",
            identifier="test-session"
        )

        assert result is None

    @patch('saber.server.execution.logging.container_logging_manager.subprocess.run')
    def test_log_container_logs_success(self, mock_subprocess, logging_config):
        """Test successful container log collection."""
        manager = ContainerLoggingManager(logging_config)

        # Mock successful subprocess call
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Container log line 1\nContainer log line 2"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result_path = manager.log_container_logs(
            container_name="web",
            project_name="test-project",
            config_type="sandbox",
            tail_lines=100
        )

        assert result_path is not None
        assert Path(result_path).exists()

        # Verify subprocess was called correctly
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "-p" in call_args
        assert "test-project" in call_args
        assert "logs" in call_args
        assert "--tail=100" in call_args
        assert "web" in call_args

        # Verify log content
        with open(result_path, 'r') as f:
            content = f.read()

        assert "# Container Logs for web" in content
        assert "# Project: test-project" in content
        assert "# Type: sandbox" in content
        assert "Container log line 1" in content
        assert "Container log line 2" in content

    @patch('saber.server.execution.logging.container_logging_manager.subprocess.run')
    def test_log_container_logs_error(self, mock_subprocess, logging_config):
        """Test container log collection with error."""
        manager = ContainerLoggingManager(logging_config)

        # Mock failed subprocess call
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Container not found"
        mock_subprocess.return_value = mock_result

        result_path = manager.log_container_logs(
            container_name="nonexistent",
            project_name="test-project",
            config_type="sandbox"
        )

        assert result_path is not None
        assert Path(result_path).exists()

        # Verify error is logged
        with open(result_path, 'r') as f:
            content = f.read()

        assert "Error collecting logs (exit code 1):" in content
        assert "Container not found" in content

    def test_log_container_lifecycle_event(self, logging_config):
        """Test logging container lifecycle events."""
        manager = ContainerLoggingManager(logging_config)

        container_info = {
            "session_id": "test-session",
            "project_name": "test-project",
            "config_type": "sandbox"
        }

        additional_data = {
            "error": "Connection refused",
            "attempt_count": 3
        }

        manager.log_container_lifecycle_event(
            event_type="start_failure",
            container_info=container_info,
            additional_data=additional_data
        )

        # Check that event file was created
        date_str = datetime.now().strftime("%Y-%m-%d")
        events_file = manager.logs_directory / f"container-events-{date_str}.jsonl"
        assert events_file.exists()

        # Verify event content
        with open(events_file, 'r') as f:
            event_line = f.readline()

        event_data = json.loads(event_line)
        assert event_data["domain"] == "test_domain"
        assert event_data["event_type"] == "start_failure"
        assert event_data["container_info"] == container_info
        assert event_data["additional_data"] == additional_data
        assert "timestamp" in event_data

    @patch('saber.server.execution.logging.container_logging_manager.subprocess.run')
    def test_log_all_project_containers(self, mock_subprocess, logging_config):
        """Test logging all containers in a project."""
        manager = ContainerLoggingManager(logging_config)

        # Mock docker-compose ps call to return services
        mock_ps_result = Mock()
        mock_ps_result.returncode = 0
        mock_ps_result.stdout = "web\ndb\nredis"

        # Mock docker-compose logs calls
        mock_logs_result = Mock()
        mock_logs_result.returncode = 0
        mock_logs_result.stdout = "Service logs"

        mock_subprocess.side_effect = [mock_ps_result, mock_logs_result, mock_logs_result, mock_logs_result]

        log_paths = manager.log_all_project_containers(
            project_name="test-project",
            config_type="sandbox"
        )

        # Should return 3 log paths (for web, db, redis)
        assert len(log_paths) == 3

        # Verify all log files exist
        for log_path in log_paths:
            assert Path(log_path).exists()

    def test_get_logs_summary(self, logging_config):
        """Test getting logs summary."""
        manager = ContainerLoggingManager(logging_config)

        # Create some test files
        (manager.logs_directory / "test-file.log").touch()
        (manager.logs_directory / "compose-configs" / "test-compose.yml").touch()

        summary = manager.get_logs_summary()

        assert summary["enabled"] is True
        assert summary["domain"] == "test_domain"
        assert summary["logs_directory"] == str(manager.logs_directory)
        assert "subdirectories" in summary
        assert "recent_files" in summary

    def test_cleanup_old_logs(self, logging_config):
        """Test cleanup of old log files."""
        manager = ContainerLoggingManager(logging_config)

        # Create an old file and a new file
        old_file = manager.logs_directory / "old-log.log"
        new_file = manager.logs_directory / "new-log.log"

        old_file.touch()
        new_file.touch()

        # Modify timestamps
        import os
        import time

        # Set old file to 10 days ago
        old_time = time.time() - (10 * 24 * 60 * 60)
        os.utime(old_file, (old_time, old_time))

        # Clean up files older than 7 days
        manager.cleanup_old_logs(days_to_keep=7)

        # Old file should be deleted, new file should remain
        assert not old_file.exists()
        assert new_file.exists()

    def test_logging_disabled_no_operations(self, disabled_logging_config):
        """Test that no operations are performed when logging is disabled."""
        manager = ContainerLoggingManager(disabled_logging_config)

        # All methods should return None or empty lists when disabled
        assert manager.log_compose_config({}, "test", "id") is None
        assert manager.log_container_logs("container", "project", "type") is None
        assert manager.log_all_project_containers("project", "type") == []

        # Lifecycle events should not create files
        manager.log_container_lifecycle_event("test", {})

        # No event files should be created
        date_str = datetime.now().strftime("%Y-%m-%d")
        events_file = manager.logs_directory / f"container-events-{date_str}.jsonl"
        assert not events_file.exists()
