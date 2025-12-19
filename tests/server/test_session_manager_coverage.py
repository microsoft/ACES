"""
Comprehensive test coverage for SessionManager to achieve >90% coverage.

Tests for:
- Health check metadata (get_health_metadata, _check_permanent_environment_health)
- Network cleanup with retry logic
- Episode cleanup methods (_cleanup_episode_network, _cleanup_saber_episode_networks)
- Session cleanup loop (_cleanup_inactive_sessions, _session_cleanup_loop)
- Permanent environment management (_start_permanent_environment)
- Episode finalization error paths
- Dependency resolution paths
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest
from fastapi import HTTPException

from saber.models.benchmark_task import SingleEpisodeTask
from saber.server.base import Episode, EpisodeState
from saber.server.episodes.constants import EpisodeTerminationReason
from saber.server.session_manager import ClientSession, SessionManager


class TestSessionManagerHealthChecks:
    """Test health check and metadata functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(
                domain_name="test_domain",
                config_dir="/tmp/config",
                host="127.0.0.1",
                port=8000,
                manifest={"domain": {"slug": "test-domain"}, "schemaVersion": "1.0", "capabilities": ["test"]},
                manifest_path="/tmp/manifest.yaml",
            )
            return manager

    def test_get_health_metadata_with_manifest(self, session_manager):
        """Test health metadata includes manifest information."""
        # Mock the permanent environment health check
        with patch.object(session_manager, "_check_permanent_environment_health") as mock_health:
            mock_health.return_value = {"healthy": True, "status": "not_configured"}
            metadata = session_manager.get_health_metadata()

        assert metadata["domain"] == "test_domain"
        assert metadata["domain_slug"] == "test-domain"
        assert metadata["schema_version"] == "1.0"
        assert metadata["capabilities"] == ["test"]
        assert metadata["manifest_path"] == "/tmp/manifest.yaml"

    def test_get_health_metadata_without_manifest(self):
        """Test health metadata without manifest."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", manifest=None)
            with patch.object(manager, "_check_permanent_environment_health") as mock_health:
                mock_health.return_value = {"healthy": True, "status": "not_configured"}
                metadata = manager.get_health_metadata()

            assert metadata["domain"] == "test_domain"
            assert "domain_slug" not in metadata
            assert "schema_version" not in metadata

    def test_get_health_metadata_with_config_checksum(self, session_manager, tmp_path):
        """Test health metadata includes config checksum."""
        # Create test config files
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "test.yaml").write_text("test: config")

        session_manager.config_dir = str(config_dir)
        metadata = session_manager.get_health_metadata()

        assert "config_checksum" in metadata
        assert isinstance(metadata["config_checksum"], str)

    def test_get_health_metadata_with_build_env_vars(self, session_manager, monkeypatch):
        """Test health metadata includes build environment variables."""
        monkeypatch.setenv("GIT_SHA", "abc123")
        monkeypatch.setenv("BUILD_TIMESTAMP", "2025-11-30")
        monkeypatch.setenv("IMAGE_TAG", "v1.0.0")

        metadata = session_manager.get_health_metadata()

        assert "build_metadata" in metadata
        assert metadata["build_metadata"]["git_sha"] == "abc123"
        assert metadata["build_metadata"]["build_timestamp"] == "2025-11-30"
        assert metadata["build_metadata"]["image_tag"] == "v1.0.0"

    def test_check_permanent_environment_health_not_configured(self, session_manager):
        """Test permanent environment health when not configured."""
        session_manager.benchmark_manager.config_loader.get_permanent_environment.return_value = None

        health = session_manager._check_permanent_environment_health()

        assert health["healthy"] is True
        assert health["status"] == "not_configured"
        assert "No permanent environment configured" in health["message"]

    def test_check_permanent_environment_health_manager_not_initialized(self, session_manager):
        """Test permanent environment health when manager not initialized."""
        session_manager.benchmark_manager.config_loader.get_permanent_environment.return_value = "test-env"
        session_manager.execution_manager._permanent_environment_manager = None
        session_manager._permanent_env_startup_complete = True  # Simulate completed startup

        health = session_manager._check_permanent_environment_health()

        assert health["healthy"] is False
        assert health["status"] == "manager_not_initialized"

    def test_check_permanent_environment_health_not_running(self, session_manager):
        """Test permanent environment health when not running."""
        session_manager.benchmark_manager.config_loader.get_permanent_environment.return_value = "test-env"
        mock_perm_env_manager = MagicMock()
        mock_perm_env_manager.is_running.return_value = False
        session_manager.execution_manager._permanent_environment_manager = mock_perm_env_manager
        session_manager._permanent_env_startup_complete = True  # Simulate completed startup

        health = session_manager._check_permanent_environment_health()

        assert health["healthy"] is False
        assert health["status"] == "not_running"

    def test_check_permanent_environment_health_compose_file_missing(self, session_manager, tmp_path):
        """Test permanent environment health when compose file missing."""
        session_manager.benchmark_manager.config_loader.get_permanent_environment.return_value = "test-env"
        mock_perm_env_manager = MagicMock()
        mock_perm_env_manager.is_running.return_value = True
        session_manager.execution_manager._permanent_environment_manager = mock_perm_env_manager
        session_manager.config_dir = str(tmp_path)
        session_manager._permanent_env_startup_complete = True  # Simulate completed startup

        health = session_manager._check_permanent_environment_health()

        assert health["healthy"] is False
        assert health["status"] == "compose_file_missing"

    def test_check_permanent_environment_health_healthy(self, session_manager, tmp_path):
        """Test permanent environment health when healthy."""
        # Setup
        session_manager.benchmark_manager.config_loader.get_permanent_environment.return_value = "test-env"
        mock_perm_env_manager = MagicMock()
        mock_perm_env_manager.is_running.return_value = True
        mock_perm_env_manager.compose_project_name = "test-project"
        session_manager.execution_manager._permanent_environment_manager = mock_perm_env_manager
        session_manager._permanent_env_startup_complete = True  # Simulate completed startup

        # Create compose file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        env_dir = config_dir / "environments" / "permanent"
        env_dir.mkdir(parents=True)
        compose_file = env_dir / "test-env.compose.yml"
        compose_file.write_text("version: '3'")
        session_manager.config_dir = str(config_dir)

        # Mock ComposeHealthChecker - need to patch where it's imported
        with patch("saber.server.execution.sandbox.compose_health_checker.ComposeHealthChecker") as mock_checker_class:
            mock_checker = MagicMock()
            mock_checker.get_service_health_summary.return_value = {
                "overall_healthy": True,
                "healthy_count": 2,
                "total_count": 2,
                "services": {"service1": "healthy", "service2": "healthy"},
            }
            mock_checker_class.return_value = mock_checker

            health = session_manager._check_permanent_environment_health()

            assert health["healthy"] is True
            assert health["status"] == "checked"
            assert health["healthy_services"] == 2
            assert health["total_services"] == 2

    def test_check_permanent_environment_health_unhealthy(self, session_manager, tmp_path):
        """Test permanent environment health when unhealthy."""
        # Setup
        session_manager.benchmark_manager.config_loader.get_permanent_environment.return_value = "test-env"
        mock_perm_env_manager = MagicMock()
        mock_perm_env_manager.is_running.return_value = True
        mock_perm_env_manager.compose_project_name = "test-project"
        session_manager.execution_manager._permanent_environment_manager = mock_perm_env_manager
        session_manager._permanent_env_startup_complete = True  # Simulate completed startup

        # Create compose file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        env_dir = config_dir / "environments" / "permanent"
        env_dir.mkdir(parents=True)
        compose_file = env_dir / "test-env.compose.yml"
        compose_file.write_text("version: '3'")
        session_manager.config_dir = str(config_dir)

        # Mock ComposeHealthChecker with unhealthy services
        with patch("saber.server.execution.sandbox.compose_health_checker.ComposeHealthChecker") as mock_checker_class:
            mock_checker = MagicMock()
            mock_checker.get_service_health_summary.return_value = {
                "overall_healthy": False,
                "healthy_count": 1,
                "total_count": 2,
                "services": {"service1": "healthy", "service2": "unhealthy"},
                "error": "Service2 is unhealthy",
            }
            mock_checker_class.return_value = mock_checker

            health = session_manager._check_permanent_environment_health()

            assert health["healthy"] is False
            assert health["healthy_services"] == 1
            assert health["total_services"] == 2


class TestSessionManagerNetworkCleanup:
    """Test network cleanup functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            return manager

    @pytest.mark.asyncio
    async def test_cleanup_episode_network_success(self, session_manager):
        """Test successful episode network cleanup."""
        episode_id = "test-episode-123"

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock network ls
            mock_ls_process = AsyncMock()
            mock_ls_process.communicate.return_value = (b"saber-episode-test-episode-123\n", b"")
            mock_ls_process.returncode = 0

            # Mock network inspect (no containers)
            mock_inspect_process = AsyncMock()
            mock_inspect_process.communicate.return_value = (b"", b"")
            mock_inspect_process.returncode = 0

            # Mock network rm
            mock_rm_process = AsyncMock()
            mock_rm_process.communicate.return_value = (b"", b"")
            mock_rm_process.returncode = 0

            mock_subprocess.side_effect = [mock_ls_process, mock_inspect_process, mock_rm_process]

            await session_manager._cleanup_episode_network(episode_id)

            # Verify all docker commands were called
            assert mock_subprocess.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup_episode_network_with_containers(self, session_manager):
        """Test episode network cleanup with attached containers."""
        episode_id = "test-episode-456"

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock network ls
            mock_ls_process = AsyncMock()
            mock_ls_process.communicate.return_value = (b"saber-episode-test-episode-456\n", b"")
            mock_ls_process.returncode = 0

            # Mock network inspect (with containers)
            mock_inspect_process = AsyncMock()
            mock_inspect_process.communicate.return_value = (b"container1 container2 ", b"")
            mock_inspect_process.returncode = 0

            # Mock container rm (2 times)
            mock_rm_container1 = AsyncMock()
            mock_rm_container1.communicate.return_value = (b"", b"")
            mock_rm_container2 = AsyncMock()
            mock_rm_container2.communicate.return_value = (b"", b"")

            # Mock network rm
            mock_rm_network = AsyncMock()
            mock_rm_network.communicate.return_value = (b"", b"")
            mock_rm_network.returncode = 0

            mock_subprocess.side_effect = [
                mock_ls_process,
                mock_inspect_process,
                mock_rm_container1,
                mock_rm_container2,
                mock_rm_network,
            ]

            await session_manager._cleanup_episode_network(episode_id)

            # Verify docker rm was called for containers and network
            assert mock_subprocess.call_count == 5

    @pytest.mark.asyncio
    async def test_cleanup_episode_network_not_found(self, session_manager):
        """Test cleanup when network doesn't exist."""
        episode_id = "nonexistent-episode"

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock network ls (network not found)
            mock_ls_process = AsyncMock()
            mock_ls_process.communicate.return_value = (b"other-network\n", b"")
            mock_ls_process.returncode = 0

            mock_subprocess.return_value = mock_ls_process

            # Should not raise exception
            await session_manager._cleanup_episode_network(episode_id)

            # Only ls should be called
            assert mock_subprocess.call_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_episode_network_with_retry_success(self, session_manager):
        """Test cleanup with retry succeeds on second attempt."""
        episode_id = "retry-episode"

        with patch.object(session_manager, "_cleanup_episode_network") as mock_cleanup:
            # Fail first time, succeed second time
            mock_cleanup.side_effect = [Exception("Network busy"), None]

            result = await session_manager._cleanup_episode_network_with_retry(episode_id, max_retries=3)

            assert result is True
            assert mock_cleanup.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_episode_network_with_retry_exhausted(self, session_manager):
        """Test cleanup with retry exhausts all attempts."""
        episode_id = "fail-episode"

        with patch.object(session_manager, "_cleanup_episode_network") as mock_cleanup:
            # Always fail
            mock_cleanup.side_effect = Exception("Permanent failure")

            result = await session_manager._cleanup_episode_network_with_retry(episode_id, max_retries=3)

            assert result is False
            assert mock_cleanup.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup_saber_episode_networks_success(self, session_manager):
        """Test bulk cleanup of SABER episode networks."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock network ls
            mock_ls_process = AsyncMock()
            mock_ls_process.communicate.return_value = (
                b"saber-episode-ep1\nsaber-episode-ep2\n",
                b"",
            )
            mock_ls_process.returncode = 0

            # Mock inspect for both networks (no containers)
            mock_inspect1 = AsyncMock()
            mock_inspect1.communicate.return_value = (b"", b"")
            mock_inspect1.returncode = 0

            mock_inspect2 = AsyncMock()
            mock_inspect2.communicate.return_value = (b"", b"")
            mock_inspect2.returncode = 0

            # Mock rm for both networks
            mock_rm1 = AsyncMock()
            mock_rm1.communicate.return_value = (b"", b"")
            mock_rm1.returncode = 0

            mock_rm2 = AsyncMock()
            mock_rm2.communicate.return_value = (b"", b"")
            mock_rm2.returncode = 0

            mock_subprocess.side_effect = [mock_ls_process, mock_inspect1, mock_inspect2, mock_rm1, mock_rm2]

            await session_manager._cleanup_saber_episode_networks()

            # Verify all operations were called
            assert mock_subprocess.call_count == 5

    @pytest.mark.asyncio
    async def test_cleanup_saber_episode_networks_no_networks(self, session_manager):
        """Test bulk cleanup when no SABER networks exist."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock network ls (no networks)
            mock_ls_process = AsyncMock()
            mock_ls_process.communicate.return_value = (b"", b"")
            mock_ls_process.returncode = 0

            mock_subprocess.return_value = mock_ls_process

            await session_manager._cleanup_saber_episode_networks()

            # Only ls should be called
            assert mock_subprocess.call_count == 1


# Cleanup loop tests removed - duplicates of test_session_manager_lifecycle.py


class TestSessionManagerPermanentEnvironment:
    """Test permanent environment management."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            return manager

    @pytest.mark.asyncio
    async def test_start_permanent_environment_not_configured(self, session_manager):
        """Test starting permanent environment when not configured."""
        session_manager.benchmark_manager.config_loader.get_permanent_environment.return_value = None

        # Should not raise exception
        await session_manager._start_permanent_environment()

        # Should not call execution manager
        session_manager.execution_manager.start_permanent_environment.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_permanent_environment_configured(self, session_manager, tmp_path):
        """Test starting permanent environment when configured."""
        session_manager.benchmark_manager.config_loader.get_permanent_environment.return_value = "test-env"

        # Create compose file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        env_dir = config_dir / "environments" / "permanent"
        env_dir.mkdir(parents=True)
        compose_file = env_dir / "test-env.compose.yml"
        compose_file.write_text("version: '3'")
        session_manager.config_dir = str(config_dir)

        # Mock the permanent environment manager
        mock_perm_env_mgr = MagicMock()
        session_manager.execution_manager._permanent_environment_manager = mock_perm_env_mgr

        await session_manager._start_permanent_environment()

        # Should call permanent environment manager's start method
        mock_perm_env_mgr.start_permanent_environment_from_file.assert_called_once()

# Shutdown tests removed - duplicates of test_session_manager_lifecycle.py


class TestSessionManagerEpisodeFinalization:
    """Test episode finalization error paths."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            # Setup required mocks
            manager.benchmark_manager.prompt_generator = MagicMock()
            manager.execution_manager.copy_initial_files_to_episode = AsyncMock()
            return manager

    @pytest.mark.asyncio
    async def test_cleanup_failed_episode_environment_success(self, session_manager):
        """Test cleanup of failed episode environment."""
        session = await session_manager.create_session("test-client")
        episode_id = "failed-episode"

        # Create episode
        episode = MagicMock()
        episode.episode_id = episode_id
        episode.session_id = session.session_id
        session.add_active_episode(episode_id)
        session_manager.episode_manager.get_episode_by_id.return_value = episode

        # Mock cleanup
        session_manager.execution_manager.cleanup_episode = MagicMock()

        await session_manager._cleanup_failed_episode_environment(episode_id)

        # Should remove from active episodes
        assert episode_id not in session.active_episode_ids
        # Should call cleanup
        session_manager.execution_manager.cleanup_episode.assert_called_once_with(episode_id)

    @pytest.mark.asyncio
    async def test_cleanup_failed_episode_environment_fallback(self, session_manager):
        """Test fallback cleanup when normal cleanup fails."""
        session = await session_manager.create_session("test-client")
        episode_id = "failed-episode"

        # Create episode
        episode = MagicMock()
        episode.episode_id = episode_id
        episode.session_id = session.session_id
        session.add_active_episode(episode_id)
        session_manager.episode_manager.get_episode_by_id.return_value = episode

        # Mock cleanup to fail
        session_manager.execution_manager.cleanup_episode = MagicMock(side_effect=Exception("Cleanup failed"))

        # Mock sandbox environment manager
        mock_sandbox_mgr = MagicMock()
        session_manager.execution_manager._sandbox_environment_manager = mock_sandbox_mgr

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            await session_manager._cleanup_failed_episode_environment(episode_id)

            # Should attempt fallback cleanup
            mock_run.assert_called_once()


class TestSessionManagerDependencies:
    """Test dependency resolution paths."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            # Setup required mocks
            manager.benchmark_manager.get_dependency_config = MagicMock(
                return_value={"wait_seconds": 30, "retry_interval": 1, "max_retry_interval": 5}
            )
            manager.benchmark_manager.prompt_generator = MagicMock()
            manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = {
                "instruction": "test"
            }
            # Mock get_single_episode_task to return proper SingleEpisodeTask
            single_episode_task = SingleEpisodeTask(
                task_id="dependent-task",
                domain="test_domain",
                title="Test Task",
                description="Test description",
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
                continue_prompt="test",
                episode_attempts=1,
            )
            manager.benchmark_manager.get_single_episode_task = MagicMock(return_value=single_episode_task)
            manager.execution_manager.configure_for_task_async = AsyncMock()
            manager.execution_manager.wait_for_episode_healthy = AsyncMock()
            manager.execution_manager.copy_initial_files_to_episode = AsyncMock()
            manager.episode_manager.configure_for_task = AsyncMock()
            # Mock initialize_episode_context to return proper dict
            manager.episode_manager.initialize_episode_context = MagicMock(return_value={})
            return manager

    @pytest.mark.asyncio
    async def test_start_episode_with_dependency_success(self, session_manager):
        """Test starting episode with successful dependency resolution."""
        session = await session_manager.create_session("test-client")

        # Create task with dependency
        # Note: After template expansion, dependency_template is cleared and depends_on_task_id is set
        task = MagicMock()
        task.task_id = "dependent-task"
        task.depends_on_task_id = "parent-task"  # Use correct attribute name (post-expansion)
        task.dependency_template = None  # Cleared after expansion
        task.initial_context = {}
        session_manager.benchmark_manager.get_task.return_value = task

        # Mock successful dependency resolution
        session_manager.episode_manager.find_available_episode_for_dependency_with_retry = AsyncMock(
            return_value="parent-episode-123"
        )
        session_manager.episode_manager.attach_episode_to_episode = MagicMock()

        episode = await session_manager.start_episode(session.session_id, "dependent-task")

        # Should attach to parent episode
        session_manager.episode_manager.attach_episode_to_episode.assert_called_once()

        # CRITICAL: Verify ORCHESTRATION_TARGET_EPISODES is set for executor resolution
        from saber.models.constants import MetadataKeys
        assert MetadataKeys.ORCHESTRATION_TARGET_EPISODES in episode.context
        assert episode.context[MetadataKeys.ORCHESTRATION_TARGET_EPISODES] == ["parent-episode-123"]

    @pytest.mark.asyncio
    async def test_start_episode_with_dependency_not_found(self, session_manager):
        """Test starting episode when dependency not found."""
        session = await session_manager.create_session("test-client")

        # Create task with dependency
        # Note: After template expansion, dependency_template is cleared and depends_on_task_id is set
        task = MagicMock()
        task.task_id = "dependent-task"
        task.depends_on_task_id = "parent-task"  # Use correct attribute name (post-expansion)
        task.dependency_template = None  # Cleared after expansion
        task.initial_context = {}
        session_manager.benchmark_manager.get_task.return_value = task

        # Mock dependency not found
        session_manager.episode_manager.find_available_episode_for_dependency_with_retry = AsyncMock(
            return_value=None
        )

        with pytest.raises(ValueError) as exc_info:
            await session_manager.start_episode(session.session_id, "dependent-task")

        assert "no available episodes" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_start_episode_with_dependency_validation_error(self, session_manager):
        """Test starting episode when dependency validation fails."""
        session = await session_manager.create_session("test-client")

        # Create task with dependency
        # Note: After template expansion, dependency_template is cleared and depends_on_task_id is set
        task = MagicMock()
        task.task_id = "dependent-task"
        task.depends_on_task_id = "parent-task"  # Use correct attribute name (post-expansion)
        task.dependency_template = None  # Cleared after expansion
        task.initial_context = {}
        session_manager.benchmark_manager.get_task.return_value = task

        # Mock dependency validation error
        session_manager.episode_manager.find_available_episode_for_dependency_with_retry = AsyncMock(
            side_effect=ValueError("Invalid dependency")
        )

        with pytest.raises(ValueError) as exc_info:
            await session_manager.start_episode(session.session_id, "dependent-task")

        assert "Invalid dependency" in str(exc_info.value)


# ClientSession tests removed - these should be in a dedicated ClientSession test file or test_session_manager_core.py
