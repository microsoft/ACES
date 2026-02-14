"""Unit tests for SABERSandboxEnvironment lifecycle methods.

Tests focus on increasing coverage for:
- task_init error paths and fresh start scenarios
- sample_init various code paths
- sample_cleanup error handling
- task_cleanup different scenarios
- Internal helper methods
- MCP client lifecycle
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ConfigDict, create_model

from saber.inspect_ai.core.types import DomainRegistryEntry
from saber.inspect_ai.saber import SABERSandboxEnvironment, SandboxError
from saber.models import MetadataKeys, SingleEpisodeTask


@pytest.fixture
def mock_config():
    """Create a mock SABER sandbox config."""
    SABERConfig = create_model(
        "SABERConfig",
        domain_slug=(str, ...),
        domains_root=(Path, ...),
        rest_port=(int, 8000),
        mcp_port=(int, 8001),
        compose_template_path=(Path | None, None),
        cleanup=(bool, False),
        enable_debug_logging=(bool, False),
        __config__=ConfigDict(frozen=True),
    )
    return SABERConfig(
        domain_slug="test_domain",
        domains_root=Path("/tmp/domains"),
        rest_port=8000,
        mcp_port=8001,
        cleanup=False,
        enable_debug_logging=False,
    )


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clear the registry before and after each test."""
    SABERSandboxEnvironment._registry.clear()
    yield
    SABERSandboxEnvironment._registry.clear()


class TestTaskInitErrorPaths:
    """Test task_init error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_task_init_none_config_raises_error(self):
        """Test that task_init raises SandboxError when config is None."""
        with pytest.raises(SandboxError, match="SABER sandbox config is required"):
            await SABERSandboxEnvironment.task_init("test_task", None)

    @pytest.mark.asyncio
    async def test_task_init_fresh_start_success(self, mock_config):
        """Test successful fresh start of domain (backward compat path)."""

        # No active domain from factory
        with patch('saber.inspect_ai.saber.get_active_domain') as mock_get_active:
            mock_get_active.return_value = None

            # Mock DomainController
            with patch('saber.inspect_ai.saber.DomainController') as mock_controller_class:
                mock_controller = MagicMock()
                mock_context = MagicMock()
                mock_context.rest_url = "http://localhost:8000"
                mock_context.mcp_url = "http://localhost:8001"
                mock_controller.start = AsyncMock(return_value=mock_context)
                mock_controller_class.return_value = mock_controller

                # Mock session creation
                with patch('saber.inspect_ai.server.session_manager.SessionLifecycleManager.create_session',
                          new_callable=AsyncMock) as mock_create_session:
                    mock_create_session.return_value = "fresh_session_123"

                    # Execute task_init
                    await SABERSandboxEnvironment.task_init("test_task", mock_config)

                    # Verify domain was started
                    mock_controller.start.assert_called_once()

                    # Verify session was created
                    mock_create_session.assert_called_once()

                    # Verify registry entry
                    assert "test_domain" in SABERSandboxEnvironment._registry
                    entry = SABERSandboxEnvironment._registry["test_domain"]
                    assert entry.owner == "test_task"
                    assert entry.ownership is True
                    assert entry.session_id == "fresh_session_123"

    @pytest.mark.asyncio
    async def test_task_init_fresh_start_failure_cleanup(self, mock_config):
        """Test that task_init cleans up on startup failure."""

        with patch('saber.inspect_ai.saber.get_active_domain') as mock_get_active:
            mock_get_active.return_value = None

            with patch('saber.inspect_ai.saber.DomainController') as mock_controller_class:
                mock_controller = MagicMock()
                # Simulate startup failure
                mock_controller.start = AsyncMock(side_effect=Exception("Startup failed"))
                mock_controller.stop = AsyncMock()
                mock_controller_class.return_value = mock_controller

                # Execute task_init - should raise error
                with pytest.raises(SandboxError, match="Failed to start SABER sandbox"):
                    await SABERSandboxEnvironment.task_init("test_task", mock_config)

                # Verify cleanup attempted
                mock_controller.stop.assert_called_once()

                # Verify registry is clean
                assert "test_domain" not in SABERSandboxEnvironment._registry

    @pytest.mark.asyncio
    async def test_task_init_ownership_transfer_success(self, mock_config):
        """Test successful ownership transfer from factory."""

        # Simulate domain started by factory
        mock_domain_data = {
            "controller": MagicMock(),
            "context": MagicMock(),
            "rest_port": 8000,
            "mcp_port": 8001,
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
        }

        with patch('saber.inspect_ai.saber.get_active_domain') as mock_get_active:
            mock_get_active.return_value = mock_domain_data

            with patch('saber.inspect_ai.server.session_manager.SessionLifecycleManager.create_session',
                      new_callable=AsyncMock) as mock_create_session:
                mock_create_session.return_value = "transfer_session_456"

                # Execute task_init
                await SABERSandboxEnvironment.task_init("test_task", mock_config)

                # Verify session was created
                mock_create_session.assert_called_once()

                # Verify registry entry
                assert "test_domain" in SABERSandboxEnvironment._registry
                entry = SABERSandboxEnvironment._registry["test_domain"]
                assert entry.owner == "test_task"
                assert entry.ownership is True
                assert entry.session_id == "transfer_session_456"

    @pytest.mark.asyncio
    async def test_task_init_enables_debug_logging(self):
        """Test that task_init enables debug logging when configured."""

        SABERConfig = create_model(
            "SABERConfig",
            domain_slug=(str, ...),
            domains_root=(Path, ...),
            rest_port=(int, 8000),
            mcp_port=(int, 8001),
            compose_template_path=(Path | None, None),
            cleanup=(bool, False),
            enable_debug_logging=(bool, True),  # Enable debug logging
            __config__=ConfigDict(frozen=True),
        )
        config = SABERConfig(
            domain_slug="test_domain",
            domains_root=Path("/tmp/domains"),
            rest_port=8000,
            mcp_port=8001,
            cleanup=False,
            enable_debug_logging=True,
        )

        with patch('saber.inspect_ai.saber.get_active_domain') as mock_get_active:
            mock_get_active.return_value = None

            with patch('saber.inspect_ai.saber.DomainController') as mock_controller_class:
                mock_controller = MagicMock()
                mock_context = MagicMock()
                mock_context.rest_url = "http://localhost:8000"
                mock_context.mcp_url = "http://localhost:8001"
                mock_controller.start = AsyncMock(return_value=mock_context)
                mock_controller_class.return_value = mock_controller

                with patch('saber.inspect_ai.server.session_manager.SessionLifecycleManager.create_session',
                          new_callable=AsyncMock) as mock_create_session:
                    mock_create_session.return_value = "session_123"

                    with patch('saber.inspect_ai.saber.enable_debug_logging') as mock_enable_debug:
                        # Execute task_init
                        await SABERSandboxEnvironment.task_init("test_task", config)

                        # Verify debug logging was enabled
                        mock_enable_debug.assert_called_once()


class TestSampleInitPaths:
    """Test sample_init various code paths."""

    @pytest.mark.asyncio
    async def test_sample_init_single_episode_task(self, mock_config):
        """Test sample_init for single episode task."""

        # Setup registry
        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="test_task",
            controller=MagicMock(),
            context=MagicMock(),
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="session_123",
        )

        # Create single episode task
        task = SingleEpisodeTask(
            benchmark_task_id="task_001",
            task_id="task_001",
            domain="test_domain",
            title="Test Task",
            description="Test task",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

        metadata = {
            MetadataKeys.BENCHMARK_TASK: task.model_dump(),
            MetadataKeys.SAMPLE_ID: "task_001__attempt_1",
            MetadataKeys.TASK_ID: "task_001",
        }

        with patch('saber.inspect_ai.saber.get_benchmark_task_handler') as mock_get_handler:
            from saber.inspect_ai.core.types import HandlerState

            mock_handler = MagicMock()
            mock_handler.initialize = AsyncMock(return_value=HandlerState(
                episode_ids=["episode_123"],
                primary_episode_id="episode_123",

            ))
            mock_handler.cleanup = AsyncMock()
            mock_get_handler.return_value = mock_handler

            with patch('saber.inspect_ai.core.mcp_factory.mcp_server_http') as mock_mcp_server:
                mock_mcp_server.return_value = MagicMock()

                with patch('saber.inspect_ai.saber.store'):
                    # Execute sample_init
                    result = await SABERSandboxEnvironment.sample_init(
                        "test_task", mock_config, metadata
                    )

                    # Verify instance was created
                    assert "default" in result
                    instance = result["default"]
                    assert instance._episode_id == "episode_123"
                    assert instance._task_id == "task_001"
                    assert instance._sample_id == "task_001__attempt_1"


class TestSampleCleanupPaths:
    """Test sample_cleanup error handling."""

    @pytest.mark.asyncio
    async def test_sample_cleanup_interrupted(self, mock_config):
        """Test sample_cleanup when interrupted."""

        # Create instance
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp/domains"),
            rest_port=8000,
            mcp_port=8001,
        )
        instance._episode_id = "episode_123"
        instance._task_id = "task_001"
        instance._sample_id = "sample_001"

        with patch.object(instance, '_cleanup_sample', new_callable=AsyncMock) as mock_cleanup:
            # Execute sample_cleanup with interrupted=True (classmethod signature)
            await SABERSandboxEnvironment.sample_cleanup(
                "test_task", mock_config, {"default": instance}, interrupted=True
            )

            # Verify cleanup was called with interrupted flag
            mock_cleanup.assert_called_once_with(interrupted=True)


class TestTaskCleanupPaths:
    """Test task_cleanup different scenarios."""

    @pytest.mark.asyncio
    async def test_task_cleanup_no_registry_entry(self, mock_config):
        """Test that task_cleanup handles missing registry entry gracefully."""

        # Registry empty
        assert len(SABERSandboxEnvironment._registry) == 0

        with patch('saber.inspect_ai.saber.remove_active_domain') as mock_remove:
            # Execute task_cleanup - should not raise
            await SABERSandboxEnvironment.task_cleanup("test_task", mock_config, cleanup=True)

            # Verify remove_active_domain was still called
            mock_remove.assert_called_once_with("test_domain")


class TestHelperMethods:
    """Test internal helper methods."""

    def test_default_concurrency_returns_sensible_default(self):
        """Test that default_concurrency returns None (defers to Inspect AI's --max-samples)."""
        assert SABERSandboxEnvironment.default_concurrency() is None

    def test_config_files_returns_empty_list(self):
        """Test that config_files returns empty list."""
        assert SABERSandboxEnvironment.config_files() == []

    def test_config_deserialize_creates_model(self):
        """Test that config_deserialize creates proper BaseModel."""
        config_dict = {
            "domain_slug": "test_domain",
            "domains_root": Path("/tmp/domains"),
            "rest_port": 8000,
            "mcp_port": 8001,
        }

        model = SABERSandboxEnvironment.config_deserialize(config_dict)

        assert model.domain_slug == "test_domain"
        assert model.domains_root == Path("/tmp/domains")
        assert model.rest_port == 8000

    @pytest.mark.asyncio
    async def test_task_init_environment_returns_empty_dict(self):
        """Test that task_init_environment returns empty dict."""
        result = await SABERSandboxEnvironment.task_init_environment(None, {})
        assert result == {}


class TestResetState:
    """Test _reset_state method."""

    def test_reset_state_clears_instance_variables(self):
        """Test that _reset_state clears all instance variables."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp/domains"),
        )

        # Set some state
        instance._episode_id = "episode_123"
        instance._task_id = "task_001"
        instance._sample_id = "sample_001"
        instance._mcp_client = MagicMock()

        # Reset state
        instance._reset_state()

        # Verify per-sample state cleared (but not session_id which persists)
        assert instance._episode_id is None
        assert instance._task_id is None
        assert instance._sample_id is None
        assert instance._mcp_client is None
        # Note: _handler is not reset by _reset_state, it's managed separately


class TestSessionManagement:
    """Test session creation and termination.

    Note: Tests for _create_session_for_task and _terminate_session_sync
    have been removed as these methods were extracted to SessionLifecycleManager
    in previous refactoring phases.
    """

    # TODO: Move these tests to tests/inspect_ai/server/test_session_lifecycle_manager.py
    # The methods being tested (_create_session_for_task, _terminate_session_sync)
    # were extracted to SessionLifecycleManager during Phase 3 refactoring.
    pass
