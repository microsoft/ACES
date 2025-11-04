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
        with patch('saber.inspect_ai.tasks.get_active_domain') as mock_get_active:
            mock_get_active.return_value = None  # Not from factory

            with patch.object(
                SABERSandboxEnvironment, '_create_session_for_task',
                new_callable=AsyncMock
            ) as mock_create_session:
                mock_create_session.return_value = "test_session_123"

                # First init
                SABERSandboxEnvironment._registry["test_domain"] = {
                    "owner": "test_task_1",
                    "domain_slug": "test_domain",
                    "controller": MagicMock(),
                    "context": MagicMock(),
                    "ownership": True,
                    "rest_port": 8000,
                    "mcp_port": 8001,
                    "rest_url": "http://localhost:8000",
                    "mcp_url": "http://localhost:8001",
                    "session_id": "test_session_123",
                }

                # Second init with SAME task name (eval-retry)
                await SABERSandboxEnvironment.task_init("test_task_1", mock_config)

                # Should succeed without raising error
                assert "test_domain" in SABERSandboxEnvironment._registry
                entry = SABERSandboxEnvironment._registry["test_domain"]
                assert entry["owner"] == "test_task_1"

                # Should not create new session (reusing existing)
                assert mock_create_session.call_count == 0

    @pytest.mark.asyncio
    async def test_task_init_raises_error_different_task(self, mock_config):
        """Test that task_init raises error if domain owned by different task."""

        # Setup: Simulate domain already initialized by different task
        SABERSandboxEnvironment._registry["test_domain"] = {
            "owner": "other_task",
            "domain_slug": "test_domain",
            "controller": MagicMock(),
            "context": MagicMock(),
            "ownership": True,
            "rest_port": 8000,
            "mcp_port": 8001,
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
            "session_id": "other_session_456",
        }

        # Try to init with different task
        with pytest.raises(SandboxError) as exc_info:
            await SABERSandboxEnvironment.task_init("test_task_1", mock_config)

        assert "already owned by task 'other_task'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sample_init_stores_metadata_for_retry(self, mock_config):
        """Test that sample_init stores session/episode IDs in metadata."""

        # Setup: Domain initialized
        SABERSandboxEnvironment._registry["test_domain"] = {
            "owner": "test_task",
            "domain_slug": "test_domain",
            "controller": MagicMock(),
            "context": MagicMock(),
            "ownership": True,
            "rest_port": 8000,
            "mcp_port": 8001,
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
            "session_id": "session_789",
        }

        metadata = {"task_id": "task_001"}

        with patch.object(
            SABERSandboxEnvironment, '_create_episode',
            new_callable=AsyncMock
        ) as mock_create_episode:
            mock_create_episode.return_value = "episode_abc"

            with patch('saber.inspect_ai.saber.mcp_server_http'):
                with patch('saber.inspect_ai.saber.store'):
                    # Initialize sample
                    result = await SABERSandboxEnvironment.sample_init(
                        "test_task", mock_config, metadata
                    )

                    # Verify metadata was updated with SABER IDs
                    assert "saber_session_id" in metadata
                    assert "saber_episode_id" in metadata
                    assert "saber_domain_slug" in metadata

                    assert metadata["saber_session_id"] == "session_789"
                    assert metadata["saber_episode_id"] == "episode_abc"
                    assert metadata["saber_domain_slug"] == "test_domain"

                    # Verify instance was created
                    assert "default" in result
                    instance = result["default"]
                    assert instance._session_id == "session_789"
                    assert instance._episode_id == "episode_abc"

    @pytest.mark.asyncio
    async def test_sample_init_skips_completed_sample(self, mock_config):
        """Test that sample_init creates minimal instance for completed sample (eval-retry)."""

        # Setup: Domain initialized
        SABERSandboxEnvironment._registry["test_domain"] = {
            "owner": "test_task",
            "domain_slug": "test_domain",
            "controller": MagicMock(),
            "context": MagicMock(),
            "ownership": True,
            "rest_port": 8000,
            "mcp_port": 8001,
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
            "session_id": "session_789",
        }

        # Metadata from completed sample (has SABER IDs)
        metadata = {
            "task_id": "task_001",
            "saber_session_id": "old_session_123",
            "saber_episode_id": "old_episode_xyz",
            "saber_domain_slug": "test_domain",
        }

        # Mock episode creation - should NOT be called
        with patch.object(
            SABERSandboxEnvironment, '_create_episode',
            new_callable=AsyncMock
        ) as mock_create_episode:
            # Initialize sample
            result = await SABERSandboxEnvironment.sample_init(
                "test_task", mock_config, metadata
            )

            # Verify episode was NOT created (completed sample)
            assert mock_create_episode.call_count == 0

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

        SABERSandboxEnvironment._registry["test_domain"] = {
            "owner": "test_task",
            "domain_slug": "test_domain",
            "controller": mock_controller,
            "context": MagicMock(),
            "ownership": True,
            "rest_port": 8000,
            "mcp_port": 8001,
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
            "session_id": "session_789",
        }

        with patch('saber.inspect_ai.tasks.remove_active_domain'):
            with patch.object(
                SABERSandboxEnvironment, '_terminate_session_sync'
            ) as mock_terminate:
                # Cleanup with cleanup=False (eval-retry scenario)
                await SABERSandboxEnvironment.task_cleanup(
                    "test_task", mock_config, cleanup=False
                )

                # Verify domain is still in registry (preserved for retry)
                assert "test_domain" in SABERSandboxEnvironment._registry

                # Verify session was terminated but domain not stopped
                assert mock_terminate.call_count == 1
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

        SABERSandboxEnvironment._registry["test_domain"] = {
            "owner": "test_task",
            "domain_slug": "test_domain",
            "controller": mock_controller,
            "context": MagicMock(),
            "ownership": True,
            "rest_port": 8000,
            "mcp_port": 8001,
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
            "session_id": "session_789",
        }

        with patch('saber.inspect_ai.tasks.remove_active_domain'):
            with patch.object(
                SABERSandboxEnvironment, '_terminate_session_sync'
            ):
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
        SABERSandboxEnvironment._registry["test_domain"] = {
            "owner": "stale_task",
            "domain_slug": "test_domain",
            "controller": MagicMock(),
            "context": MagicMock(),
            "ownership": True,
            "rest_port": 8000,
            "mcp_port": 8001,
            "rest_url": "http://localhost:8000",
            "mcp_url": "http://localhost:8001",
            "session_id": "stale_session",
        }

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

        with patch('saber.inspect_ai.tasks.get_active_domain') as mock_get_active:
            mock_get_active.return_value = None

            with patch.object(
                SABERSandboxEnvironment, '_create_session_for_task',
                new_callable=AsyncMock
            ) as mock_create_session:
                mock_create_session.return_value = "session_001"

                with patch.object(
                    SABERSandboxEnvironment, '_create_episode',
                    new_callable=AsyncMock
                ) as mock_create_episode:
                    mock_create_episode.return_value = "episode_001"

                    with patch('saber.inspect_ai.saber.mcp_server_http'):
                        with patch('saber.inspect_ai.saber.store'):
                            with patch('saber.inspect_ai.tasks.remove_active_domain'):
                                # 1. First run: Initialize task
                                SABERSandboxEnvironment._registry["test_domain"] = {
                                    "owner": "test_task",
                                    "domain_slug": "test_domain",
                                    "controller": MagicMock(),
                                    "context": MagicMock(),
                                    "ownership": True,
                                    "rest_port": 8000,
                                    "mcp_port": 8001,
                                    "rest_url": "http://localhost:8000",
                                    "mcp_url": "http://localhost:8001",
                                    "session_id": "session_001",
                                }

                                # 2. First run: Initialize sample
                                metadata1 = {"task_id": "task_001"}
                                await SABERSandboxEnvironment.sample_init(
                                    "test_task", mock_config, metadata1
                                )

                                # Verify metadata stored
                                assert "saber_session_id" in metadata1
                                assert "saber_episode_id" in metadata1

                                # 3. Simulate failure and cleanup (preserve domain)
                                with patch.object(
                                    SABERSandboxEnvironment, '_terminate_session_sync'
                                ):
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
                                result = await SABERSandboxEnvironment.sample_init(
                                    "test_task", mock_config, metadata1
                                )

                                # Should create minimal instance
                                instance = result["default"]
                                assert instance._mcp_client is None

                                # 6. Retry: Process new sample
                                metadata2 = {"task_id": "task_002"}
                                result2 = await SABERSandboxEnvironment.sample_init(
                                    "test_task", mock_config, metadata2
                                )

                                # Should create full instance
                                instance2 = result2["default"]
                                assert "saber_episode_id" in metadata2
