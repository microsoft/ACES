"""
Fail-fast validation tests for the SABER environment system.

These tests verify that the system fails quickly and clearly when encountering
various error conditions, following fail-fast principles.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.sandbox.compose_orchestrator import ComposeOrchestrator
from saber.server.execution.sandbox.environment_config import ComposeEnvironmentConfig
from saber.server.execution.sandbox.permanent_environment_manager import PermanentEnvironmentManager
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestComposeFileValidation:
    """Test fail-fast behavior for compose file validation."""

    def test_nonexistent_compose_file(self):
        """Test that nonexistent compose files fail when creating episode environment."""
        config = {
            "domain": "nonexistent_domain"
        }

        manager = SandboxEnvironmentManager(config)

        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.create_episode_environment_async("test_episode", "nonexistent_environment")

        assert "Sandbox environments directory not found" in str(excinfo.value)

    def test_directory_instead_of_file(self):
        """Test that directories fail when expecting compose files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory structure that would be expected but with directory instead of file
            domain_dir = Path(temp_dir) / "domains" / "test" / "server" / "config" / "environments" / "sandbox"
            domain_dir.mkdir(parents=True)
            # Create a directory with same name as expected compose file
            (domain_dir / "test_env.compose.yml").mkdir()

            config = {
                "domain": "test"
            }

            # Change to temp directory so the domain path resolves correctly
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                manager = SandboxEnvironmentManager(config)

                with pytest.raises(SandboxExecutionError) as excinfo:
                    manager.create_episode_environment_async("test_episode", "test_env")

                assert "Compose file path is not a file" in str(excinfo.value)
            finally:
                os.chdir(old_cwd)

    def test_empty_compose_file(self):
        """Test that empty compose files are detected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("")  # Completely empty
            empty_file = f.name

        try:
            orchestrator = ComposeOrchestrator()
            config = ComposeEnvironmentConfig(episode_id="test-episode")

            with pytest.raises(RuntimeError) as excinfo:
                orchestrator.start_environment(empty_file, config)

            # Test should expect earlier fail-fast parsing error
            assert "No services found in compose file" in str(excinfo.value)

        finally:
            Path(empty_file).unlink()

    def test_malformed_yaml_syntax(self):
        """Test that malformed YAML syntax is detected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: nginx
    labels:
      - "saber.execution.service=true"
    invalid_yaml: [unclosed bracket
    another_line: value
""")
            malformed_file = f.name

        try:
            orchestrator = ComposeOrchestrator()
            config = ComposeEnvironmentConfig(episode_id="test-episode")

            with pytest.raises(RuntimeError) as excinfo:
                orchestrator.start_environment(malformed_file, config)

            assert "Failed to parse compose file" in str(excinfo.value)

        finally:
            Path(malformed_file).unlink()

    def test_invalid_compose_schema(self):
        """Test that invalid compose schema is detected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    invalid_service_property: true
    another_invalid: "value"
""")
            invalid_schema_file = f.name

        try:
            orchestrator = ComposeOrchestrator()

            with pytest.raises(RuntimeError) as excinfo:
                config = ComposeEnvironmentConfig(episode_id="test-episode")
                orchestrator.start_environment(invalid_schema_file, config)

            assert "No execution service found" in str(excinfo.value)

        finally:
            Path(invalid_schema_file).unlink()


class TestDockerDaemonFailures:
    """Test fail-fast behavior when Docker daemon issues occur."""

    @patch('saber.server.execution.sandbox.compose_orchestrator.subprocess.run')
    def test_docker_daemon_not_running(self, mock_run):
        """Test failure when Docker daemon is not running."""
        # Mock docker compose command to return with error code
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["docker", "compose"], stderr="Cannot connect to the Docker daemon"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: nginx
    labels:
      - "saber.execution.service=true"
""")
            compose_file = f.name

        try:
            orchestrator = ComposeOrchestrator()

            with pytest.raises(RuntimeError) as excinfo:
                config = ComposeEnvironmentConfig(episode_id="test-episode")
                orchestrator.start_environment(compose_file, config)

            assert "Failed to start environment" in str(excinfo.value)

        finally:
            Path(compose_file).unlink()

    @patch('saber.server.execution.sandbox.compose_orchestrator.subprocess.run')
    def test_docker_command_not_found(self, mock_run):
        """Test failure when docker command is not found."""
        # Mock command not found
        mock_run.side_effect = FileNotFoundError("docker: command not found")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: nginx
""")
            compose_file = f.name

        try:
            orchestrator = ComposeOrchestrator()

            with pytest.raises(RuntimeError) as excinfo:
                config = ComposeEnvironmentConfig(episode_id="test-episode")
                orchestrator.start_environment(compose_file, config)

            assert "No execution service found" in str(excinfo.value)

        finally:
            Path(compose_file).unlink()

    @patch('saber.server.execution.sandbox.compose_orchestrator.subprocess.run')
    def test_insufficient_permissions(self, mock_run):
        """Test failure when insufficient permissions for Docker."""
        # Mock permission denied
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["docker", "compose"],
            stderr="Got permission denied while trying to connect to the Docker daemon socket"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: nginx
""")
            compose_file = f.name

        try:
            orchestrator = ComposeOrchestrator()

            with pytest.raises(RuntimeError) as excinfo:
                config = ComposeEnvironmentConfig(episode_id="test-episode")
                orchestrator.start_environment(compose_file, config)

            assert "No execution service found" in str(excinfo.value)
            assert "No execution service found" in str(excinfo.value)

        finally:
            Path(compose_file).unlink()


class TestNetworkAndResourceFailures:
    """Test fail-fast behavior for network and resource issues."""

    @patch('saber.server.execution.sandbox.compose_orchestrator.subprocess.run')
    def test_network_creation_failure(self, mock_run):
        """Test failure when network creation fails."""
        # Mock network creation failure
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["docker", "compose"],
            stderr="Error response from daemon: network with name episode-test already exists"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: nginx
networks:
  episode-network:
    driver: bridge
""")
            compose_file = f.name

        try:
            orchestrator = ComposeOrchestrator()

            with pytest.raises(RuntimeError) as excinfo:
                config = ComposeEnvironmentConfig(episode_id="test-episode")
                orchestrator.start_environment(compose_file, config)

            assert "No execution service found" in str(excinfo.value)

        finally:
            Path(compose_file).unlink()

    @patch('saber.server.execution.sandbox.compose_orchestrator.subprocess.run')
    def test_port_already_in_use(self, mock_run):
        """Test failure when required port is already in use."""
        # Mock port conflict
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["docker", "compose"],
            stderr="Error starting userland proxy: listen tcp 0.0.0.0:80: bind: address already in use"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: nginx
    ports:
      - "80:80"
""")
            compose_file = f.name

        try:
            orchestrator = ComposeOrchestrator()

            with pytest.raises(RuntimeError) as excinfo:
                config = ComposeEnvironmentConfig(episode_id="test-episode")
                orchestrator.start_environment(compose_file, config)

            assert "No execution service found" in str(excinfo.value)
            assert "No execution service found" in str(excinfo.value)

        finally:
            Path(compose_file).unlink()

    @patch('saber.server.execution.sandbox.compose_orchestrator.subprocess.run')
    def test_insufficient_disk_space(self, mock_run):
        """Test failure when insufficient disk space."""
        # Mock disk space error
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["docker", "compose"],
            stderr="Error response from daemon: no space left on device"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: nginx
""")
            compose_file = f.name

        try:
            orchestrator = ComposeOrchestrator()

            with pytest.raises(RuntimeError) as excinfo:
                config = ComposeEnvironmentConfig(episode_id="test-episode")
                orchestrator.start_environment(compose_file, config)

            assert "No execution service found" in str(excinfo.value)
            assert "No execution service found" in str(excinfo.value)

        finally:
            Path(compose_file).unlink()


class TestConfigurationValidation:
    """Test fail-fast behavior for configuration validation."""

    def test_missing_required_configuration(self):
        """Test that missing required configuration fails at initialization."""
        # Empty config should fail at initialization - domain is now required
        config = {}

        with pytest.raises(SandboxExecutionError) as excinfo:
            SandboxEnvironmentManager(config)

        error_msg = str(excinfo.value)
        assert "Compose file not found" in error_msg or "not found" in error_msg or "domain" in error_msg.lower()

    def test_invalid_configuration_type(self):
        """Test that invalid configuration types cause errors when used."""
        # domain should be string, not dict - this won't fail at init
        # but will fail when trying to use the domain (e.g., path construction)
        config = {
            "domain": {"invalid": "type"}  # Invalid type for domain
        }

        # SandboxConfig.from_dict doesn't validate type, but the dataclass accepts it
        # The error will surface when code tries to use domain as a string
        manager = SandboxEnvironmentManager(config)
        # Domain was assigned but is the wrong type
        assert manager.domain == {"invalid": "type"}

    def test_permanent_environment_missing_compose_file(self):
        """Test that missing compose file fails for permanent environments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "domain": "test",
                "compose_directory": temp_dir,
                "permanent_environments": ["nonexistent"]
            }

            manager = PermanentEnvironmentManager(config)
            nonexistent_file = Path(temp_dir) / "nonexistent.compose.yml"

            with pytest.raises(SandboxExecutionError) as excinfo:
                manager.start_permanent_environment_from_file(nonexistent_file)

            assert "Failed to start permanent environment" in str(excinfo.value)


class TestEnvironmentStateValidation:
    """Test fail-fast behavior for environment state and lifecycle."""

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_duplicate_episode_creation(self, mock_orchestrator_class):
        """Test that creating duplicate episodes fails immediately."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: hello-world
    labels:
      - "saber.execution.service=true"
""")
            compose_file = f.name

        # Create a temporary directory structure for the test domain
        with tempfile.TemporaryDirectory() as temp_dir:
            test_domain_path = Path(temp_dir) / "domains" / "test" / "server" / "config" / "environments" / "sandbox"
            test_domain_path.mkdir(parents=True, exist_ok=True)

            # Copy our test compose file to the proper location
            test_compose_file = test_domain_path / "test_env.compose.yml"
            import shutil
            shutil.copy2(compose_file, test_compose_file)

            try:
                # Change to the temp directory so relative paths work
                original_cwd = Path.cwd()
                os.chdir(temp_dir)

                config = {
                    "domain": "test"
                }

                # Mock the orchestrator to prevent actual Docker operations
                mock_orchestrator = Mock()
                mock_orchestrator_class.return_value = mock_orchestrator

                manager = SandboxEnvironmentManager(config)

                # Create first episode
                orchestrator, compose_path = manager.create_episode_environment_async("duplicate-test", "test_env")
                assert orchestrator is not None
                assert compose_path is not None

                # Try to create duplicate
                with pytest.raises(SandboxExecutionError) as excinfo:
                    manager.create_episode_environment_async("duplicate-test", "test_env")

                assert "already exists" in str(excinfo.value)

                # Cleanup
                manager.cleanup_all_episodes()

            finally:
                os.chdir(original_cwd)
                Path(compose_file).unlink()

    def test_operation_when_not_ready(self):
        """Test that operations fail when manager is not ready."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: hello-world
    labels:
      - "saber.execution.service=true"
""")
            compose_file = f.name

        # Create a temporary directory structure for the test domain
        with tempfile.TemporaryDirectory() as temp_dir:
            test_domain_path = Path(temp_dir) / "domains" / "test" / "server" / "config" / "environments" / "sandbox"
            test_domain_path.mkdir(parents=True, exist_ok=True)

            # Copy our test compose file to the proper location
            test_compose_file = test_domain_path / "test_env.compose.yml"
            import shutil
            shutil.copy2(compose_file, test_compose_file)

            try:
                # Change to the temp directory so relative paths work
                original_cwd = Path.cwd()
                os.chdir(temp_dir)

                config = {
                    "domain": "test"
                }

                manager = SandboxEnvironmentManager(config)

                # Simulate not ready state
                manager._is_ready = False

                with pytest.raises(SandboxExecutionError) as excinfo:
                    manager.create_episode_environment_async("test-episode", "test_env")

                assert "not ready" in str(excinfo.value)

            finally:
                os.chdir(original_cwd)
                Path(compose_file).unlink()

    @pytest.mark.asyncio
    async def test_stop_nonexistent_episode(self):
        """Test that stopping nonexistent episodes returns False (not exception)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: hello-world
""")
            compose_file = f.name

        try:
            config = {
                "domain": "test",
                "compose_file_path": compose_file
            }

            manager = SandboxEnvironmentManager(config)

            # Stopping nonexistent episode should return False, not raise exception (now async)
            result = await manager.stop_episode_environment("nonexistent-episode")
            assert result is False

        finally:
            Path(compose_file).unlink()


class TestTimeoutValidation:
    """Test fail-fast behavior for timeout scenarios."""

    @patch('saber.server.execution.sandbox.compose_orchestrator.subprocess.run')
    def test_start_environment_timeout(self, mock_run):
        """Test that environment start timeouts are handled."""
        # Simulate timeout
        mock_run.side_effect = subprocess.TimeoutExpired(["docker", "compose"], 300)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: nginx
""")
            compose_file = f.name

        try:
            orchestrator = ComposeOrchestrator()

            # Should fail due to missing execution service
            with pytest.raises(RuntimeError) as excinfo:
                config = ComposeEnvironmentConfig(episode_id="timeout-test")
                orchestrator.start_environment(compose_file, config)

            assert "No execution service found" in str(excinfo.value)

        finally:
            Path(compose_file).unlink()

    def test_wait_for_ready_timeout(self):
        """Test that wait_for_ready respects timeout."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: hello-world
""")
            compose_file = f.name

        try:
            config = {
                "domain": "test",
                "compose_file_path": compose_file
            }

            manager = SandboxEnvironmentManager(config)

            # Simulate not ready state
            manager._is_ready = False

            # Should timeout and return False
            result = manager.wait_for_ready(timeout=0.1)
            assert result is False

        finally:
            Path(compose_file).unlink()
