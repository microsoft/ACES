"""
Unit tests for health check failure handling in domain orchestrator.

Tests the enhanced _wait_for_services method that:
- Shows progress updates during health checks
- Retrieves and displays container logs on failure
- Finds and displays server log file errors
- Stops containers on health check failure
- Raises DockerError instead of silent failure
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, call
from saber.domain.orchestrator import DockerRunner
from saber.domain.exceptions import DockerError


@pytest.fixture
def mock_compose_file(tmp_path):
    """Create a temporary compose file."""
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services:\n  server:\n    image: test\n")
    return compose_file


@pytest.fixture
def mock_domains_root(tmp_path):
    """Create a temporary domains root with log files."""
    domains_root = tmp_path / "domains"
    domains_root.mkdir()

    test_domain = domains_root / "test_domain"
    test_domain.mkdir()

    # Create server logs directory
    server_logs_dir = test_domain / "server" / "logs" / "server-logs"
    server_logs_dir.mkdir(parents=True)

    # Create a mock log file with errors
    log_file = server_logs_dir / "saber-server-2025-12-11_10-00-00.log"
    log_content = """2025-12-11 10:00:00 INFO     Starting server
2025-12-11 10:00:01 ERROR    Failed to connect to database
2025-12-11 10:00:02 WARNING  Port already in use
2025-12-11 10:00:03 ERROR    Exception: Connection refused
2025-12-11 10:00:04 INFO     Retrying connection
2025-12-11 10:00:05 ERROR    Fatal error: Cannot start server
"""
    log_file.write_text(log_content)

    return domains_root


class TestHealthCheckFailureHandling:
    """Tests for _wait_for_services health check failure handling."""

    @patch('saber.domain.orchestrator.time.sleep')
    @patch('saber.domain.orchestrator.time.time')
    @patch('saber.domain.orchestrator.socket.socket')
    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('builtins.print')
    def test_health_check_timeout_shows_container_logs(
        self, mock_print, mock_subprocess, mock_socket, mock_time, mock_sleep,
        mock_compose_file, mock_domains_root
    ):
        """Test that health check timeout retrieves and displays container logs."""
        # Simulate timeout by making socket connection always fail
        mock_sock_instance = Mock()
        mock_sock_instance.connect_ex.return_value = 1  # Connection failed
        mock_socket.return_value.__enter__.return_value = mock_sock_instance

        # Simulate time passing to trigger timeout
        mock_time.side_effect = [0, 0, 10, 20, 30, 40, 50, 60, 70]  # Exceeds 60s timeout

        # Mock docker logs output
        mock_logs_result = Mock()
        mock_logs_result.stdout = "Server failed to start\nPort 8000 already in use"
        mock_logs_result.stderr = "Error: Connection refused"

        # Mock docker compose down (container stop)
        mock_down_result = Mock(returncode=0)

        def subprocess_side_effect(*args, **kwargs):
            cmd = args[0]
            if 'logs' in cmd:
                return mock_logs_result
            elif 'down' in cmd:
                return mock_down_result
            return Mock(returncode=0)

        mock_subprocess.side_effect = subprocess_side_effect

        env_vars = {
            "COMPOSE_PROFILES": "server",
            "REST_PORT": "8000",
            "DOMAINS_ROOT": str(mock_domains_root)
        }

        runner = DockerRunner(mock_compose_file, mock_domains_root)

        # Should raise DockerError on health check failure
        with pytest.raises(DockerError, match="(Server failed to become healthy|crashed during health check)"):
            runner._wait_for_services("test_domain", env_vars, timeout=60)

        # Verify container logs were retrieved
        logs_calls = [c for c in mock_subprocess.call_args_list
                      if c[0] and 'logs' in c[0][0]]
        assert len(logs_calls) == 1
        assert '--tail' in logs_calls[0][0][0]
        assert '100' in logs_calls[0][0][0]  # Changed from 50 to 100

        # Verify containers were stopped
        down_calls = [c for c in mock_subprocess.call_args_list
                      if c[0] and 'down' in c[0][0]]
        assert len(down_calls) == 1

        # Verify failure message was printed (check for crash indicator)
        print_calls = [str(c) for c in mock_print.call_args_list]
        # The code prints "❌ Container crashed" not "FAILED"
        assert any('crashed' in call.lower() or '❌' in call for call in print_calls)
        # The code prints "Container logs from" not "container logs"
        assert any('container logs' in call.lower() or 'diagnostic' in call.lower() for call in print_calls)

    @patch('saber.domain.orchestrator.time.sleep')
    @patch('saber.domain.orchestrator.time.time')
    @patch('saber.domain.orchestrator.socket.socket')
    @patch('saber.domain.orchestrator.subprocess.run')
    @patch('builtins.print')
    def test_health_check_timeout_displays_server_log_errors(
        self, mock_print, mock_subprocess, mock_socket, mock_time, mock_sleep,
        mock_compose_file, mock_domains_root
    ):
        """Test that health check timeout finds and displays server log file errors."""
        # Simulate timeout
        mock_sock_instance = Mock()
        mock_sock_instance.connect_ex.return_value = 1
        mock_socket.return_value.__enter__.return_value = mock_sock_instance

        mock_time.side_effect = [0, 0, 10, 20, 30, 40, 50, 60, 70]

        # Mock subprocess calls
        mock_subprocess.return_value = Mock(stdout="", stderr="", returncode=0)

        env_vars = {
            "COMPOSE_PROFILES": "server",
            "REST_PORT": "8000",
            "DOMAINS_ROOT": str(mock_domains_root)
        }

        runner = DockerRunner(mock_compose_file, mock_domains_root)

        with pytest.raises(DockerError):
            runner._wait_for_services("test_domain", env_vars, timeout=60)

        # Verify server log file path was printed
        print_calls = [str(c) for c in mock_print.call_args_list]
        assert any('saber-server-2025-12-11_10-00-00.log' in call for call in print_calls)

        # Verify error lines were printed
        assert any('Failed to connect to database' in call for call in print_calls)
        assert any('Connection refused' in call for call in print_calls)
        assert any('Fatal error' in call for call in print_calls)

    @patch('saber.domain.orchestrator.time.sleep')
    @patch('saber.domain.orchestrator.time.time')
    @patch('saber.domain.orchestrator.socket.socket')
    @patch('builtins.print')
    def test_health_check_success_no_error_handling(
        self, mock_print, mock_socket, mock_time, mock_sleep,
        mock_compose_file, mock_domains_root
    ):
        """Test that successful health check doesn't trigger error handling."""
        # Simulate successful connection
        mock_sock_instance = Mock()
        mock_sock_instance.connect_ex.return_value = 0  # Connection succeeded
        mock_socket.return_value.__enter__.return_value = mock_sock_instance

        # Provide enough time values for the while loop and elapsed calculations
        mock_time.side_effect = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]  # Multiple values for repeated calls

        env_vars = {
            "COMPOSE_PROFILES": "server",
            "REST_PORT": "8000",
            "DOMAINS_ROOT": str(mock_domains_root)
        }

        runner = DockerRunner(mock_compose_file, mock_domains_root)

        # Should not raise any exception
        runner._wait_for_services("test_domain", env_vars, timeout=60)

        # Verify success message was printed
        print_calls = [str(c) for c in mock_print.call_args_list]
        assert any('responding' in call.lower() for call in print_calls)

        # Verify no failure messages
        assert not any('FAILED' in call for call in print_calls)
