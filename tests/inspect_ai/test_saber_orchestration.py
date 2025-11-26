"""Tests for orchestrated sub-task functionality in SABERSandboxEnvironment.

These tests cover the orchestration coordinator integration paths that were
previously uncovered.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from pydantic import create_model, ConfigDict

from saber.inspect_ai.saber import SABERSandboxEnvironment, SandboxError
from saber.models import MetadataKeys, OrchestratedTask, SubTaskDefinition, OrchestrationStrategy


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
        max_concurrent_episodes=(int | None, None),
        __config__=ConfigDict(frozen=True),
    )
    return SABERConfig(
        domain_slug="test_domain",
        domains_root=Path("/tmp/domains"),
        rest_port=8000,
        mcp_port=8001,
        cleanup=False,
        max_concurrent_episodes=2,
    )


@pytest.fixture
def orchestrated_metadata_root():
    """Metadata for root orchestrated sub-task (no dependency)."""
    return {
        MetadataKeys.ORCHESTRATION_ID: "orch_123",
        MetadataKeys.SUB_TASK_ROLE: "blue",
        MetadataKeys.TASK_ID: "blue_task",
        MetadataKeys.DEPENDS_ON_ROLE: None,
        MetadataKeys.ORDER: 1,
        MetadataKeys.SAMPLE_ID: "sample_blue",
        MetadataKeys.EXECUTION_MODE: "orchestrated_sub_task",
    }


@pytest.fixture
def orchestrated_metadata_dependent():
    """Metadata for dependent orchestrated sub-task."""
    return {
        MetadataKeys.ORCHESTRATION_ID: "orch_123",
        MetadataKeys.SUB_TASK_ROLE: "red",
        MetadataKeys.TASK_ID: "red_task",
        MetadataKeys.DEPENDS_ON_ROLE: "blue",
        MetadataKeys.ORDER: 2,
        MetadataKeys.SAMPLE_ID: "sample_red",
        MetadataKeys.EXECUTION_MODE: "orchestrated_sub_task",
    }


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clear the registry before and after each test."""
    SABERSandboxEnvironment._registry.clear()
    SABERSandboxEnvironment._episode_semaphore = None
    yield
    SABERSandboxEnvironment._registry.clear()
    SABERSandboxEnvironment._episode_semaphore = None


class TestInitOrchestratedSubTask:
    """Test _init_orchestrated_sub_task method."""

    @pytest.mark.asyncio
    async def test_init_root_sample_success(self, orchestrated_metadata_root):
        """Test initializing root orchestrated sub-task (no dependency)."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
            max_concurrent_episodes=2,
        )
        instance._sample_id = "sample_blue"
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()

        # Mock episode creation
        mock_episode_response = MagicMock()
        mock_episode_response.episode_id = "episode_blue_1"
        instance._session_manager.create_episode = AsyncMock(return_value=mock_episode_response)
        instance._session_manager.wait_for_episode_ready = AsyncMock()

        # Mock orchestration coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.register_root_sample = MagicMock(return_value=True)
        mock_coordinator.set_episode_id = MagicMock()

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            await instance._init_orchestrated_sub_task(orchestrated_metadata_root)

        # Verify coordinator was called correctly
        mock_coordinator.register_root_sample.assert_called_once_with(
            orchestration_id="orch_123",
            role="blue",
            sample_id="sample_blue",
            semaphore=instance._get_episode_semaphore(),
        )

        # Verify episode was created
        instance._session_manager.create_episode.assert_called_once_with(
            "session_456", "blue_task"
        )

        # Verify episode_id was set in coordinator
        mock_coordinator.set_episode_id.assert_called_once_with(
            orchestration_id="orch_123",
            role="blue",
            episode_id="episode_blue_1",
        )

        # Verify handler state was set
        assert instance._handler_state is not None
        assert instance._handler_state["episode_ids"] == ["episode_blue_1"]
        assert instance._handler_state["orchestration_id"] == "orch_123"
        assert instance._handler_state["sub_task_role"] == "blue"

    @pytest.mark.asyncio
    async def test_init_dependent_sample_success(self, orchestrated_metadata_dependent):
        """Test initializing dependent orchestrated sub-task."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
            max_concurrent_episodes=2,
        )
        instance._sample_id = "sample_red"
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()

        # Mock episode creation
        mock_episode_response = MagicMock()
        mock_episode_response.episode_id = "episode_red_1"
        instance._session_manager.create_episode = AsyncMock(return_value=mock_episode_response)
        instance._session_manager.wait_for_episode_ready = AsyncMock()

        # Mock orchestration coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.register_dependent_sample = AsyncMock(return_value=True)
        mock_coordinator.wait_for_dependency_ready = AsyncMock(return_value="episode_blue_1")
        mock_coordinator.set_episode_id = MagicMock()

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            await instance._init_orchestrated_sub_task(orchestrated_metadata_dependent)

        # Verify dependent registration
        mock_coordinator.register_dependent_sample.assert_called_once_with(
            orchestration_id="orch_123",
            role="red",
            sample_id="sample_red",
            depends_on_role="blue",
            order=2,
            timeout=60.0,
        )

        # Verify waited for dependency
        mock_coordinator.wait_for_dependency_ready.assert_called_once_with(
            orchestration_id="orch_123",
            role="red",
            session_manager=instance._session_manager,
            session_id="session_456",
            timeout=300.0,
        )

        # Verify episode was created
        assert instance._episode_id == "episode_red_1"

    @pytest.mark.asyncio
    async def test_init_root_sample_registration_failure(self, orchestrated_metadata_root):
        """Test that registration failure raises SandboxError."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
            max_concurrent_episodes=2,
        )
        instance._sample_id = "sample_blue"
        instance._session_id = "session_456"

        # Mock coordinator that fails registration
        mock_coordinator = MagicMock()
        mock_coordinator.register_root_sample = MagicMock(return_value=False)
        mock_coordinator.trigger_termination = MagicMock()

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            with pytest.raises(SandboxError, match="Failed to register root sample"):
                await instance._init_orchestrated_sub_task(orchestrated_metadata_root)

        # Verify termination was triggered
        mock_coordinator.trigger_termination.assert_called_once_with("orch_123")

    @pytest.mark.asyncio
    async def test_init_dependent_sample_registration_failure(self, orchestrated_metadata_dependent):
        """Test that dependent registration failure raises SandboxError."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
        )
        instance._sample_id = "sample_red"
        instance._session_id = "session_456"

        # Mock coordinator that fails dependent registration
        mock_coordinator = MagicMock()
        mock_coordinator.register_dependent_sample = AsyncMock(return_value=False)
        mock_coordinator.trigger_termination = MagicMock()

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            with pytest.raises(SandboxError, match="Failed to register dependent sample"):
                await instance._init_orchestrated_sub_task(orchestrated_metadata_dependent)

        # Verify termination was triggered
        mock_coordinator.trigger_termination.assert_called_once_with("orch_123")

    @pytest.mark.asyncio
    async def test_init_root_sample_semaphore_acquired_and_released_on_failure(self, orchestrated_metadata_root):
        """Test that semaphore is released if root sample init fails."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
            max_concurrent_episodes=2,
        )
        instance._sample_id = "sample_blue"
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()

        # Mock episode creation to fail
        instance._session_manager.create_episode = AsyncMock(side_effect=Exception("Episode creation failed"))

        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.register_root_sample = MagicMock(return_value=True)
        mock_coordinator.trigger_termination = MagicMock()

        semaphore = instance._get_episode_semaphore()
        initial_value = semaphore._value

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            with pytest.raises(SandboxError, match="Failed to initialize orchestrated sub-task"):
                await instance._init_orchestrated_sub_task(orchestrated_metadata_root)

        # Verify semaphore was released
        assert semaphore._value == initial_value


class TestCleanupOrchestratedSubTask:
    """Test _cleanup_orchestrated_sub_task method."""

    @pytest.mark.asyncio
    async def test_cleanup_orchestrated_sub_task_success(self):
        """Test successful cleanup of orchestrated sub-task."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
            max_concurrent_episodes=2,
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()
        instance._session_manager.end_episode = AsyncMock()

        # Set handler state as if orchestrated sub-task was initialized
        instance._handler_state = {
            "orchestration_id": "orch_123",
            "sub_task_role": "blue",
            "episode_ids": ["episode_blue_1"],
        }

        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.trigger_termination = MagicMock(return_value=[
            ("blue", "episode_blue_1"),
            ("red", "episode_red_1"),
        ])
        mock_coordinator.cleanup_sample = MagicMock(return_value=True)  # Should release semaphore

        semaphore = instance._get_episode_semaphore()
        initial_value = semaphore._value

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            await instance._cleanup_orchestrated_sub_task()

        # Verify trigger_termination was called
        mock_coordinator.trigger_termination.assert_called_once_with("orch_123")

        # Verify episodes were ended
        assert instance._session_manager.end_episode.call_count == 2

        # Verify cleanup_sample was called
        mock_coordinator.cleanup_sample.assert_called_once_with(
            orchestration_id="orch_123",
            role="blue",
            semaphore=semaphore,
        )

        # Verify semaphore was released (should_release=True)
        assert semaphore._value == initial_value + 1

    @pytest.mark.asyncio
    async def test_cleanup_orchestrated_sub_task_handler_state_missing(self):
        """Test that cleanup raises error when handler_state is None."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
        )
        instance._handler_state = None

        with pytest.raises(SandboxError, match="Handler state not initialized"):
            await instance._cleanup_orchestrated_sub_task()

    @pytest.mark.asyncio
    async def test_cleanup_orchestrated_sub_task_episode_end_failure_logged(self):
        """Test that episode end failures are logged but don't stop cleanup."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
            max_concurrent_episodes=2,
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()

        # First episode fails, second succeeds
        instance._session_manager.end_episode = AsyncMock(side_effect=[
            Exception("End failed"),
            None,
        ])

        instance._handler_state = {
            "orchestration_id": "orch_123",
            "sub_task_role": "blue",
        }

        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.trigger_termination = MagicMock(return_value=[
            ("blue", "episode_blue_1"),
            ("red", "episode_red_1"),
        ])
        mock_coordinator.cleanup_sample = MagicMock(return_value=False)  # Don't release

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            # Should not raise despite episode end failure
            await instance._cleanup_orchestrated_sub_task()

        # Verify both episodes were attempted
        assert instance._session_manager.end_episode.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_orchestrated_sub_task_no_semaphore_release(self):
        """Test cleanup when coordinator says not to release semaphore."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
            max_concurrent_episodes=2,
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()
        instance._session_manager.end_episode = AsyncMock()

        instance._handler_state = {
            "orchestration_id": "orch_123",
            "sub_task_role": "red",  # Dependent sample
        }

        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.trigger_termination = MagicMock(return_value=[])
        mock_coordinator.cleanup_sample = MagicMock(return_value=False)  # Don't release

        semaphore = instance._get_episode_semaphore()
        initial_value = semaphore._value

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            await instance._cleanup_orchestrated_sub_task()

        # Verify semaphore was NOT released (should_release=False)
        assert semaphore._value == initial_value


class TestCleanupSampleOrchestrated:
    """Test _cleanup_sample with orchestrated sub-tasks."""

    @pytest.mark.asyncio
    async def test_cleanup_sample_detects_orchestrated(self):
        """Test that _cleanup_sample calls orchestrated cleanup when appropriate."""
        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
            max_concurrent_episodes=2,
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()
        instance._sample_id = "sample_blue"

        # Set handler state with orchestration_id
        instance._handler_state = {
            "orchestration_id": "orch_123",
            "sub_task_role": "blue",
        }

        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.trigger_termination = MagicMock(return_value=[])
        mock_coordinator.cleanup_sample = MagicMock(return_value=True)

        with patch('saber.inspect_ai.orchestration_coordinator.OrchestrationCoordinator', return_value=mock_coordinator):
            with patch.object(instance, '_remove_episode_mapping'):
                await instance._cleanup_sample(interrupted=False)

        # Verify orchestrated cleanup was called
        mock_coordinator.trigger_termination.assert_called_once()
        mock_coordinator.cleanup_sample.assert_called_once()
