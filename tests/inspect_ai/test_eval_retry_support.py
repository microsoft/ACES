"""Unit tests for SABERSandboxEnvironment eval-retry support.

Tests verify that the sandbox properly handles:
1. Reusing existing domain ownership on retry (same task)
2. Detecting and skipping completed samples
3. Storing session/episode IDs in sample metadata
4. Preserving domain state when cleanup=False
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import create_model, ConfigDict

from saber.inspect_ai.saber import SABERSandboxEnvironment, SandboxError
from saber.inspect_ai.core.types import DomainRegistryEntry


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
        __config__=ConfigDict(frozen=True),
    )
    return SABERConfig(
        domain_slug="test_domain",
        domains_root=Path("/tmp/domains"),
        rest_port=8000,
        mcp_port=8001,
        cleanup=False,
    )


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clear the registry before and after each test."""
    SABERSandboxEnvironment._registry.clear()
    yield
    SABERSandboxEnvironment._registry.clear()


class TestEvalRetrySupport:
    """Test eval-retry functionality in SABERSandboxEnvironment."""

    @pytest.mark.asyncio
    async def test_task_init_reuses_existing_ownership_same_task(
        self, mock_config
    ):
        """Test that task_init reuses existing domain if same task (eval-retry scenario)."""

        # Setup: Simulate domain already initialized by same task
        with patch('saber.inspect_ai.server.domain_manager.get_active_domain') as mock_get_active:
            mock_get_active.return_value = None  # Not from factory

            with patch.object(
                SABERSandboxEnvironment, '_create_session_for_task',
                new_callable=AsyncMock
            ) as mock_create_session:
                mock_create_session.return_value = "test_session_123"

                # First init
                SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
                    domain_slug="test_domain",
                    owner="test_task_1",
                    controller=MagicMock(),
                    context=MagicMock(),
                    ownership=True,
                    rest_port=8000,
                    mcp_port=8001,
                    rest_url="http://localhost:8000",
                    mcp_url="http://localhost:8001",
                    session_id="test_session_123",
                )

                # Second init with SAME task name (eval-retry)
                await SABERSandboxEnvironment.task_init("test_task_1", mock_config)

                # Should succeed without raising error
                assert "test_domain" in SABERSandboxEnvironment._registry
                entry = SABERSandboxEnvironment._registry["test_domain"]
                assert entry.owner == "test_task_1"

                # Should not create new session (reusing existing)
                assert mock_create_session.call_count == 0

    @pytest.mark.asyncio
    async def test_task_init_raises_error_different_task(self, mock_config):
        """Test that task_init raises error if domain owned by different task."""

        # Setup: Simulate domain already initialized by different task
        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="other_task",
            controller=MagicMock(),
            context=MagicMock(),
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="other_session_456",
        )

        # Try to init with different task
        with pytest.raises(SandboxError) as exc_info:
            await SABERSandboxEnvironment.task_init("test_task_1", mock_config)

        assert "already owned by task 'other_task'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sample_init_stores_metadata_for_retry(self, mock_config):
        """Test that sample_init stores session/episode IDs in metadata."""
        from saber.models import SingleEpisodeTask, MetadataKeys

        # Setup: Domain initialized
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
            session_id="session_789",
        )

        # Create proper benchmark task
        task = SingleEpisodeTask(
            benchmark_task_id="task_001",
            task_id="task_001",
            domain="test_domain",
            title="Test Task",
            description="Test task for metadata",
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
        }

        # Mock the handler's initialize method to return expected state
        from saber.inspect_ai.core.types import HandlerState

        with patch('saber.inspect_ai.saber.get_benchmark_task_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.initialize = AsyncMock(return_value=HandlerState(
                episode_ids=["episode_abc"],
                primary_episode_id="episode_abc",
                semaphore_acquired=True,
            ))
            mock_handler.cleanup = AsyncMock()
            mock_get_handler.return_value = mock_handler

            with patch('saber.inspect_ai.core.mcp_factory.mcp_server_http'):
                # Mock the store to capture what's stored
                mock_store = MagicMock()
                stored_data = {}
                mock_store.set = lambda k, v: stored_data.update({k: v})
                mock_store.get = lambda k, default=None: stored_data.get(k, default)

                with patch('saber.inspect_ai.saber.store', return_value=mock_store):
                    # Initialize sample
                    result = await SABERSandboxEnvironment.sample_init(
                        "test_task", mock_config, metadata
                    )

                    # Verify SABER IDs were stored in inspect_ai store (not metadata dict)
                    # New code stores in store, not metadata, because inspect_ai copies metadata before sample_init
                    assert stored_data.get("saber_session_id") == "session_789"
                    assert stored_data.get("saber_task_id") == "task_001"
                    assert stored_data.get("saber_domain_slug") == "test_domain"

                    # Verify instance was created
                    assert "default" in result
                    instance = result["default"]
                    assert instance._session_id == "session_789"
                    assert instance._episode_id == "episode_abc"

    @pytest.mark.asyncio
    async def test_sample_init_skips_completed_sample(self, mock_config):
        """Test that sample_init creates minimal instance for completed sample (eval-retry)."""

        # Setup: Domain initialized
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
            session_id="session_789",
        )

        # Metadata from completed sample (has SABER IDs)
        metadata = {
            "task_id": "task_001",
            "saber_session_id": "old_session_123",
            "saber_episode_id": "old_episode_xyz",
            "saber_domain_slug": "test_domain",
        }

        # Mock handler to verify it's NOT called for completed samples
        with patch('saber.inspect_ai.saber.get_benchmark_task_handler') as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.initialize = AsyncMock()
            mock_get_handler.return_value = mock_handler

            # Initialize sample
            result = await SABERSandboxEnvironment.sample_init(
                "test_task", mock_config, metadata
            )

            # Verify handler was NOT called (completed sample detected early)
            assert mock_handler.initialize.call_count == 0

            # Verify minimal instance was created with old IDs
            assert "default" in result
            instance = result["default"]
            assert instance._session_id == "old_session_123"
            assert instance._episode_id == "old_episode_xyz"
            assert instance._task_id == "task_001"
            assert instance._mcp_client is None  # No MCP client for completed sample

    @pytest.mark.asyncio
    async def test_task_cleanup_preserves_domain_when_no_cleanup(self, mock_config):
        """Test that task_cleanup preserves domain when cleanup=False (eval-retry)."""

        # Setup: Domain initialized
        mock_controller = MagicMock()
        mock_controller.stop = AsyncMock()

        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="test_task",
            controller=mock_controller,
            context=MagicMock(),
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="session_789",
        )

        with patch('saber.inspect_ai.server.domain_manager.remove_active_domain'):
            with patch('saber.inspect_ai.server.session_manager.SessionLifecycleManager.terminate_session_sync'
                      ) as mock_terminate:
                # Test cleanup=False (--no-sandbox-cleanup scenario)
                await SABERSandboxEnvironment.task_cleanup(
                    "test_task", mock_config, cleanup=False
                )

                # Verify domain is still in registry (preserved for retry/inspection)
                assert "test_domain" in SABERSandboxEnvironment._registry

                # Verify session was NOT terminated (so episodes remain alive on server)
                # This is the key behavior for --no-sandbox-cleanup: everything stays alive
                assert mock_terminate.call_count == 0
                assert mock_controller.stop.call_count == 0

    @pytest.mark.asyncio
    async def test_task_cleanup_removes_domain_when_cleanup_true(self):
        """Test that task_cleanup removes domain when cleanup=True."""

        # Create config with cleanup=True for this test
        SABERConfig = create_model(
            "SABERConfig",
            domain_slug=(str, ...),
            domains_root=(Path, ...),
            rest_port=(int, 8000),
            mcp_port=(int, 8001),
            compose_template_path=(Path | None, None),
            cleanup=(bool, True),
            __config__=ConfigDict(frozen=True),
        )
        mock_config = SABERConfig(
            domain_slug="test_domain",
            domains_root=Path("/tmp/domains"),
            rest_port=8000,
            mcp_port=8001,
            cleanup=True,
        )

        # Setup: Domain initialized
        mock_controller = MagicMock()
        mock_controller.stop = AsyncMock()

        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="test_task",
            controller=mock_controller,
            context=MagicMock(),
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="session_789",
        )

        with patch('saber.inspect_ai.server.domain_manager.remove_active_domain'):
            with patch('saber.inspect_ai.server.session_manager.SessionLifecycleManager.terminate_session_sync'):
                # Cleanup with cleanup=True
                await SABERSandboxEnvironment.task_cleanup(
                    "test_task", mock_config, cleanup=True
                )

                # Verify domain was removed from registry
                assert "test_domain" not in SABERSandboxEnvironment._registry

                # Verify domain was stopped
                assert mock_controller.stop.call_count == 1

    def test_clear_stale_ownership(self):
        """Test manual cleanup of stale ownership."""

        # Setup: Domain with stale ownership
        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="stale_task",
            controller=MagicMock(),
            context=MagicMock(),
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="stale_session",
        )

        # Clear stale ownership
        result = SABERSandboxEnvironment.clear_stale_ownership("test_domain")

        assert result is True
        assert "test_domain" not in SABERSandboxEnvironment._registry

        # Try clearing again (should return False)
        result = SABERSandboxEnvironment.clear_stale_ownership("test_domain")
        assert result is False


class TestEvalRetryIntegration:
    """Integration-style tests for eval-retry workflow."""

    @pytest.mark.asyncio
    async def test_full_retry_workflow(self, mock_config):
        """Test complete eval-retry workflow from init to cleanup."""
        from saber.models import SingleEpisodeTask, MetadataKeys

        with patch('saber.inspect_ai.server.domain_manager.get_active_domain') as mock_get_active:
            mock_get_active.return_value = None

            with patch.object(
                SABERSandboxEnvironment, '_create_session_for_task',
                new_callable=AsyncMock
            ) as mock_create_session:
                mock_create_session.return_value = "session_001"

                # No need to patch _create_episode - it's handled by the task handler
                with patch('saber.inspect_ai.saber.get_benchmark_task_handler') as mock_get_handler:
                    from saber.inspect_ai.core.types import HandlerState

                    mock_handler = MagicMock()
                    mock_handler.initialize = AsyncMock(return_value=HandlerState(
                        episode_ids=["episode_001"],
                        primary_episode_id="episode_001",
                        semaphore_acquired=True,
                    ))
                    mock_handler.cleanup = AsyncMock()
                    mock_get_handler.return_value = mock_handler

                    with patch('saber.inspect_ai.core.mcp_factory.mcp_server_http'):
                        with patch('saber.inspect_ai.saber.store'):
                            with patch('saber.inspect_ai.server.domain_manager.remove_active_domain'):
                                # 1. First run: Initialize task
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
                                    session_id="session_001",
                                )

                                # Create proper benchmark task
                                task1 = SingleEpisodeTask(
                                    benchmark_task_id="task_001",
                                    task_id="task_001",
                                    domain="test_domain",
                                    title="Test Task 1",
                                    description="First test task",
                                    episode_attempts=1,
                                    max_steps=10,
                                    instruction_prompt="test",
                                    assistant_prompt="test",
                                    submit_prompt="test",
            continue_prompt="",
                                )

                                # 2. First run: Initialize sample
                                metadata1 = {
                                    MetadataKeys.BENCHMARK_TASK: task1.model_dump(),
                                    MetadataKeys.SAMPLE_ID: "task_001__attempt_1",
                                    MetadataKeys.TASK_ID: "task_001",
                                }
                                result1 = await SABERSandboxEnvironment.sample_init(
                                    "test_task", mock_config, metadata1
                                )

                                # Verify instance created (metadata stored in inspect_ai store, not dict)
                                instance1 = result1["default"]
                                assert instance1._session_id == "session_001"
                                assert instance1._episode_id == "episode_001"

                                # 3. Simulate failure and cleanup (preserve domain)
                                with patch('saber.inspect_ai.server.session_manager.SessionLifecycleManager.terminate_session_sync'):
                                    await SABERSandboxEnvironment.task_cleanup(
                                        "test_task", mock_config, cleanup=False
                                    )

                                # Domain preserved
                                assert "test_domain" in SABERSandboxEnvironment._registry

                                # 4. Retry: Re-initialize task (should reuse)
                                await SABERSandboxEnvironment.task_init(
                                    "test_task", mock_config
                                )

                                # Should still be in registry
                                assert "test_domain" in SABERSandboxEnvironment._registry

                                # 5. Retry: Skip completed sample
                                # Simulate inspect_ai adding SABER IDs from log to metadata
                                metadata1[MetadataKeys.SABER_SESSION_ID] = "session_001"
                                metadata1[MetadataKeys.SABER_EPISODE_ID] = "episode_001"
                                result = await SABERSandboxEnvironment.sample_init(
                                    "test_task", mock_config, metadata1
                                )

                                # Should create minimal instance
                                instance = result["default"]
                                assert instance._mcp_client is None

                                # 6. Retry: Process new sample
                                task2 = SingleEpisodeTask(
                                    benchmark_task_id="task_002",
                                    task_id="task_002",
                                    domain="test_domain",
                                    title="Test Task 2",
                                    description="Second test task",
                                    episode_attempts=1,
                                    max_steps=10,
                                    instruction_prompt="test",
                                    assistant_prompt="test",
                                    submit_prompt="test",
            continue_prompt="",
                                )
                                metadata2 = {
                                    MetadataKeys.BENCHMARK_TASK: task2.model_dump(),
                                    MetadataKeys.SAMPLE_ID: "task_002__attempt_1",
                                    MetadataKeys.TASK_ID: "task_002",
                                }
                                result2 = await SABERSandboxEnvironment.sample_init(
                                    "test_task", mock_config, metadata2
                                )

                                # Should create full instance
                                instance2 = result2["default"]
                                assert instance2._episode_id == "episode_001"
