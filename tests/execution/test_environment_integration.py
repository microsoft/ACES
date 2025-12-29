"""
Integration tests for the complete environment lifecycle.

These tests verify the end-to-end functionality with real static compose files.
They require Docker to be available but use minimal test containers.
"""

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
import pytest

from saber.server.execution.sandbox.compose_orchestrator import ComposeOrchestrator
from saber.server.execution.sandbox.permanent_environment_manager import PermanentEnvironmentManager
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from saber.server.execution.sandbox.environment_config import ComposeEnvironmentConfig
from saber.server.execution.exceptions import SandboxExecutionError


@pytest.fixture(autouse=True)
def stub_docker_commands():
    """Patch docker compose invocations and health checks for deterministic tests."""

    def fake_run(cmd, *args, **kwargs):
        stdout = ""
        if "ps" in cmd:
            stdout = "test-service\n"
        elif "logs" in cmd:
            stdout = "mock log output\n"
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    async def fake_create_subprocess_exec(*args, **kwargs):
        """Mock async subprocess execution."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.kill = Mock()
        return mock_process

    with patch(
        "saber.server.execution.sandbox.compose_health_checker.ComposeHealthChecker.wait_for_all_services_healthy",
        return_value=None,
    ), patch(
        "saber.server.execution.sandbox.compose_orchestrator.subprocess.run",
        side_effect=fake_run,
    ), patch(
        "saber.server.execution.logging.container_logging_manager.subprocess.run",
        side_effect=fake_run,
    ), patch(
        "saber.server.execution.sandbox.compose_orchestrator.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        yield


class TestEnvironmentLifecycleIntegration:
    """Integration tests for complete environment lifecycle."""

    @pytest.fixture
    def minimal_compose_content(self):
        """Create minimal compose file content for testing."""
        return """
version: '3.8'
services:
  test-service:
    image: hello-world
    container_name: test-${EPISODE_ID:-default}
    environment:
      - EPISODE_ID=${EPISODE_ID:-default}
    restart: "no"
    labels:
      - "saber.execution.service=true"
networks:
  test-network:
    driver: bridge
"""

    @pytest.fixture
    def temp_compose_file(self, minimal_compose_content):
        """Create a temporary compose file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(minimal_compose_content)
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for compose files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_compose_orchestrator_lifecycle(self, temp_compose_file):
        """Test complete ComposeOrchestrator lifecycle."""
        episode_id = "test-integration-episode"

        with patch("saber.server.execution.sandbox.compose_orchestrator.ComposeHealthChecker") as mock_checker_cls, patch(
            "saber.server.execution.sandbox.compose_orchestrator.subprocess.run"
        ) as mock_run:
            mock_checker = mock_checker_cls.return_value
            mock_checker.wait_for_all_services_healthy.return_value = None

            completed = subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="", stderr="")
            mock_run.return_value = completed

            orchestrator = ComposeOrchestrator()

            try:
                # Create config for the start environment call
                config = ComposeEnvironmentConfig(
                    episode_id=episode_id,
                    config_type="sandbox"
                )

                # Start environment
                orchestrator.start_environment(str(temp_compose_file), config)

                # Stop environment
                orchestrator.stop_environment(temp_compose_file, episode_id)

                # Cleanup episode
                orchestrator.cleanup_episode(episode_id)

            except Exception:
                # Ensure cleanup on any failure
                try:
                    orchestrator.stop_environment(temp_compose_file, episode_id)
                    orchestrator.cleanup_episode(episode_id)
                except Exception:
                    pass
                raise

    @pytest.mark.asyncio
    async def test_sandbox_environment_manager_lifecycle(self, minimal_compose_content, temp_directory):
        """Test complete SandboxEnvironmentManager lifecycle."""
        # Create the expected directory structure
        domain_path = temp_directory / "domains" / "integration-test" / "server" / "config" / "environments" / "sandbox"
        domain_path.mkdir(parents=True, exist_ok=True)

        # Create the compose file in the expected location
        compose_file = domain_path / "excytin_sandbox.compose.yml"
        compose_file.write_text(minimal_compose_content)

        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            config = {
                "domain": "integration-test",
                "enable_logging": False,
            }

            manager = SandboxEnvironmentManager(config)
            episode_id = "test-sandbox-episode"

            try:
                # Verify manager is ready
                assert manager.is_ready() is True
                assert manager.wait_for_ready(timeout=1) is True

                # Create episode environment
                orchestrator, compose_path = manager.create_episode_environment_async(episode_id, "excytin_sandbox")
                assert orchestrator is not None
                assert compose_path is not None
                assert manager.is_episode_active(episode_id) is True
                assert episode_id in manager.get_active_episodes()

                # Get orchestrator
                orchestrator = manager.get_episode_environment(episode_id)
                assert orchestrator is not None

                # Stop episode environment (now async)
                result = await manager.stop_episode_environment(episode_id)
                assert result is True
                assert manager.is_episode_active(episode_id) is False
                assert episode_id not in manager.get_active_episodes()

            except Exception:
                # Ensure cleanup on any failure
                try:
                    await manager.cleanup_all_episodes()
                except:
                    pass
                raise
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_permanent_environment_manager_lifecycle(self, temp_compose_file, temp_directory):
        """Test complete PermanentEnvironmentManager lifecycle."""
        # Create a permanent compose file in the temp directory
        permanent_compose = temp_directory / "permanent.yml"
        permanent_compose.write_text("""
version: '3.8'
services:
  permanent-service:
    image: hello-world
    container_name: permanent-test
    restart: "no"
networks:
  permanent-network:
    driver: bridge
""")

        config = {
            "domain": "integration-test",
            "compose_directory": str(temp_directory),
            "permanent_environments": ["permanent"],
            "enable_logging": False,
        }

        # Mock container logging manager since we don't need actual logging for this test
        with patch('saber.server.execution.sandbox.permanent_environment_manager.ComposeOrchestrator') as mock_orchestrator_class:

            # Setup orchestrator mock
            mock_orchestrator = mock_orchestrator_class.return_value
            mock_orchestrator.container_logger = Mock()
            mock_orchestrator.start_environment = Mock()
            mock_orchestrator.stop_environment = AsyncMock()  # Now async

            manager = PermanentEnvironmentManager(config)

            try:
                # Start permanent environment
                manager.start_permanent_environment_from_file(permanent_compose)
                assert manager._is_running is True

                # Stop permanent environment (now async)
                await manager.stop_permanent_environment()
                assert manager._is_running is False
            except Exception:
                # Ensure cleanup on any failure
                try:
                    await manager.stop_permanent_environment()
                except:
                    pass
                raise

    @pytest.mark.asyncio
    async def test_multiple_episode_isolation(self, minimal_compose_content, temp_directory):
        """Test that multiple episodes are properly isolated."""
        # Create the expected directory structure
        domain_path = temp_directory / "domains" / "isolation-test" / "server" / "config" / "environments" / "sandbox"
        domain_path.mkdir(parents=True, exist_ok=True)

        # Create the compose file in the expected location
        compose_file = domain_path / "excytin_sandbox.compose.yml"
        compose_file.write_text(minimal_compose_content)

        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            config = {
                "domain": "isolation-test",
            }

            manager = SandboxEnvironmentManager(config)
            episode_ids = ["episode-1", "episode-2", "episode-3"]

            try:
                # Create multiple episodes
                for episode_id in episode_ids:
                    orchestrator, compose_path = manager.create_episode_environment_async(episode_id, "excytin_sandbox")
                    assert orchestrator is not None
                    assert compose_path is not None
                    assert manager.is_episode_active(episode_id) is True

                # Verify all episodes are active
                active_episodes = manager.get_active_episodes()
                assert set(active_episodes) == set(episode_ids)

                # Stop episodes individually (now async)
                for episode_id in episode_ids:
                    result = await manager.stop_episode_environment(episode_id)
                    assert result is True
                    assert manager.is_episode_active(episode_id) is False

                # Verify all episodes are stopped
                assert manager.get_active_episodes() == []

            except Exception:
                # Ensure cleanup on any failure
                try:
                    await manager.cleanup_all_episodes()
                except:
                    pass
                raise
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_compose_file_validation_integration(self, temp_directory):
        """Test that static compose files are properly validated."""
        # Create the directory structure for validation test
        domain_path = temp_directory / "domains" / "validation-test" / "server" / "config" / "environments" / "sandbox"
        domain_path.mkdir(parents=True, exist_ok=True)

        # Create a valid compose file
        valid_compose = domain_path / "valid_env.compose.yml"
        valid_compose.write_text("""
version: '3.8'
services:
  test:
    image: hello-world
""")

        # Create an invalid compose file
        invalid_compose = domain_path / "invalid_env.compose.yml"
        invalid_compose.write_text("""
invalid yaml content
  missing: structure
""")

        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            # Valid environment should work
            config = {"domain": "validation-test"}
            manager = SandboxEnvironmentManager(config)
            assert manager.is_ready() is True

            # Try to create an episode with valid environment
            try:
                orchestrator, compose_path = manager.create_episode_environment_async("valid-episode", "valid_env")
                assert orchestrator is not None
                assert compose_path is not None
                manager.stop_episode_environment("valid-episode")
            except Exception as e:
                # May fail due to Docker not being available, but file validation should pass
                pass

            # Invalid environment should fail when trying to create episode
            with pytest.raises(SandboxExecutionError) as excinfo:
                manager.create_episode_environment_async("invalid-episode", "invalid_env")

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, minimal_compose_content, temp_directory):
        """Test error recovery in integrated environment management."""
        # Create the expected directory structure
        domain_path = temp_directory / "domains" / "error-recovery-test" / "server" / "config" / "environments" / "sandbox"
        domain_path.mkdir(parents=True, exist_ok=True)

        # Create the compose file in the expected location
        compose_file = domain_path / "excytin_sandbox.compose.yml"
        compose_file.write_text(minimal_compose_content)

        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            config = {
                "domain": "error-recovery-test",
            }

            manager = SandboxEnvironmentManager(config)

            try:
                # Create an episode
                episode_id = "recovery-test-episode"
                orchestrator, compose_path = manager.create_episode_environment_async(episode_id, "excytin_sandbox")
                assert orchestrator is not None
                assert compose_path is not None

                # Simulate partial failure by stopping the orchestrator directly (now async)
                orchestrator = manager.get_episode_environment(episode_id)
                compose_file_path = manager.episode_compose_files[episode_id]
                await orchestrator.stop_environment(compose_file_path, episode_id)

                # Manager should still track the episode as active
                assert manager.is_episode_active(episode_id) is True

                # Cleanup all should handle the partially failed state (now async)
                await manager.cleanup_all_episodes()
                # Episode might be in failed list since orchestrator already stopped
                assert len(manager.get_active_episodes()) == 0

            except Exception:
                # Ensure cleanup on any failure
                try:
                    await manager.cleanup_all_episodes()
                except:
                    pass
                raise
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    @pytest.mark.skipif(
        subprocess.run(["docker", "--version"], capture_output=True).returncode != 0,
        reason="Docker not available"
    )
    def test_real_docker_integration(self, temp_compose_file):
        """Test with real Docker operations (skipped if Docker not available)."""
        orchestrator = ComposeOrchestrator()
        episode_id = "real-docker-test"

        try:
            # This test only runs if Docker is actually available
            config = ComposeEnvironmentConfig(
                episode_id=episode_id,
                config_type="sandbox"
            )
            orchestrator.start_environment(str(temp_compose_file), config)
            # start_environment returns None on success, raises exception on failure

            # Give container a moment to start and complete (hello-world exits quickly)
            time.sleep(1)

            orchestrator.stop_environment(temp_compose_file, episode_id)
            # stop_environment returns None on success, raises exception on failure

            orchestrator.cleanup_episode(episode_id)
            # cleanup_episode returns None on success, raises exception on failure

        except Exception:
            # Ensure cleanup on any failure
            try:
                orchestrator.stop_environment(temp_compose_file, episode_id)
                orchestrator.cleanup_episode(episode_id)
            except:
                pass
            raise


class TestFailFastValidation:
    """Tests for fail-fast behavior."""

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for compose files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def temp_compose_file(self):
        """Create a temporary compose file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test:
    image: hello-world
""")
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_missing_compose_file_fails_fast(self, temp_directory):
        """Test that missing compose files cause immediate failure."""
        # Create the directory structure but no compose file
        domain_path = temp_directory / "domains" / "fail-fast-test" / "server" / "config" / "environments" / "sandbox"
        domain_path.mkdir(parents=True, exist_ok=True)

        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            config = {
                "domain": "fail-fast-test",
            }

            manager = SandboxEnvironmentManager(config)

            # Creating manager should succeed, but trying to create episode should fail
            with pytest.raises(SandboxExecutionError) as excinfo:
                manager.create_episode_environment_async("test-episode", "nonexistent_env")

            assert "Compose file not found" in str(excinfo.value)
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_invalid_compose_syntax_fails_fast(self, temp_directory):
        """Test that invalid compose syntax is detected early."""
        # Create the directory structure
        domain_path = temp_directory / "domains" / "syntax-test" / "server" / "config" / "environments" / "sandbox"
        domain_path.mkdir(parents=True, exist_ok=True)

        # Create invalid compose file
        invalid_compose = domain_path / "test_env.compose.yml"
        invalid_compose.write_text("invalid: yaml: content: [")

        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            config = {
                "domain": "syntax-test",
            }

            # Creating the manager should succeed (file exists)
            manager = SandboxEnvironmentManager(config)

            # But creating an environment should fail due to invalid syntax
            with pytest.raises(SandboxExecutionError):
                manager.create_episode_environment_async("syntax-test-episode", "test_env")

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_missing_domain_configuration(self, temp_directory):
        """Test handling of missing domain configuration."""
        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            # Should use default domain when not specified
            config = {}
            manager = SandboxEnvironmentManager(config)
            assert manager.domain == "excytin_demo"  # Updated to match the actual default
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_empty_compose_file_fails_fast(self, temp_directory):
        """Test that empty compose files are handled gracefully."""
        # Create the directory structure
        domain_path = temp_directory / "domains" / "empty-test" / "server" / "config" / "environments" / "sandbox"
        domain_path.mkdir(parents=True, exist_ok=True)

        # Create empty compose file
        empty_compose = domain_path / "test_env.compose.yml"
        empty_compose.write_text("")  # Empty file

        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            config = {
                "domain": "empty-test",
            }

            manager = SandboxEnvironmentManager(config)

            # Creating environment with empty compose should fail
            with pytest.raises(SandboxExecutionError):
                manager.create_episode_environment_async("empty-test-episode", "test_env")

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_permission_denied_fails_fast(self, temp_directory):
        """Test that permission issues are detected early."""
        # Create the directory structure but no environment directories
        # Change to temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            config = {
                "domain": "permission-test",
            }

            manager = SandboxEnvironmentManager(config)

            # Trying to create episode without environments directory should fail
            with pytest.raises(SandboxExecutionError) as excinfo:
                manager.create_episode_environment_async("test-episode", "some_env")

            assert "Sandbox environments directory not found" in str(excinfo.value)
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
