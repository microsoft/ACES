"""Tests for orchestrated sub-task functionality - refactored for Phase 5-6 architecture.

These tests verify orchestration works through SABERSandboxEnvironment._init_sample()
which delegates to OrchestrationInitializer.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from saber.inspect_ai.saber import SABERSandboxEnvironment
from saber.inspect_ai.core.types import DomainRegistryEntry
from saber.models import MetadataKeys
from saber.models import ExecutionMode


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clear the registry before and after each test."""
    SABERSandboxEnvironment._registry.clear()
    yield
    SABERSandboxEnvironment._registry.clear()


@pytest.fixture
def orchestrated_metadata_root():
    """Metadata for root orchestrated sub-task."""
    return {
        MetadataKeys.ORCHESTRATION_ID: "orch_123",
        MetadataKeys.SUB_TASK_ROLE: "blue",
        MetadataKeys.TASK_ID: "blue_task",
        MetadataKeys.DEPENDS_ON_ROLE: None,
        MetadataKeys.ORDER: 1,
        MetadataKeys.SAMPLE_ID: "sample_blue",
        MetadataKeys.EXECUTION_MODE: ExecutionMode.ORCHESTRATED_SUB_TASK,
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
        MetadataKeys.EXECUTION_MODE: ExecutionMode.ORCHESTRATED_SUB_TASK,
    }


class TestOrchestratedInit:
    """Test orchestrated sample initialization through _init_sample."""

    @pytest.mark.asyncio
    async def test_orchestrated_init_delegates_to_initializer(self, orchestrated_metadata_root):
        """Test that _init_sample delegates to OrchestrationInitializer for orchestrated tasks."""
        # Setup registry entry first
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
            session_id="session_456",
        )

        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()
        instance._episode_manager = AsyncMock()

        # Mock OrchestrationInitializer
        from saber.inspect_ai.core.types import OrchestrationSubTaskState

        mock_orch_init = AsyncMock()
        mock_handler_state = OrchestrationSubTaskState(
            episode_ids=["episode_blue"],
            primary_episode_id="episode_blue",
            orchestration_id="orch_123",
            sub_task_role="blue",
        )
        mock_orch_init.init_orchestrated_sub_task.return_value = mock_handler_state

        # Inject mock
        type(instance)._orchestration_initializer = mock_orch_init

        # Call _init_sample
        await instance._init_sample(orchestrated_metadata_root)

        # Verify OrchestrationInitializer was called
        mock_orch_init.init_orchestrated_sub_task.assert_called_once()
        assert instance._handler_state == mock_handler_state

    @pytest.mark.asyncio
    async def test_orchestrated_cleanup_delegates_to_initializer(self, orchestrated_metadata_root):
        """Test that _cleanup_sample delegates to OrchestrationInitializer for orchestrated tasks."""
        from saber.inspect_ai.core.types import OrchestrationSubTaskState

        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()
        instance._episode_manager = AsyncMock()
        instance._handler_state = OrchestrationSubTaskState(
            episode_ids=["episode_blue"],
            primary_episode_id="episode_blue",
            orchestration_id="orch_123",
            sub_task_role="blue",
        )

        # Mock OrchestrationInitializer
        mock_orch_init = AsyncMock()
        type(instance)._orchestration_initializer = mock_orch_init

        # Call _cleanup_sample
        await instance._cleanup_sample(interrupted=False)

        # Verify OrchestrationInitializer cleanup was called
        mock_orch_init.cleanup_orchestrated_sub_task.assert_called_once()


class TestOrchestratedIntegration:
    """Integration tests for orchestrated workflow."""

    @pytest.mark.asyncio
    async def test_full_orchestrated_workflow_root_sample(self, orchestrated_metadata_root):
        """Test complete init->cleanup flow for root orchestrated sample."""
        # Setup registry entry
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
            session_id="session_456",
        )

        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()
        instance._episode_manager = AsyncMock()

        # Mock OrchestrationInitializer for both init and cleanup
        from saber.inspect_ai.core.types import OrchestrationSubTaskState

        mock_orch_init = AsyncMock()
        mock_handler_state = OrchestrationSubTaskState(
            episode_ids=["episode_blue"],
            primary_episode_id="episode_blue",
            orchestration_id="orch_123",
            sub_task_role="blue",
        )
        mock_orch_init.init_orchestrated_sub_task.return_value = mock_handler_state
        mock_orch_init.cleanup_orchestrated_sub_task.return_value = None

        type(instance)._orchestration_initializer = mock_orch_init

        # Initialize
        await instance._init_sample(orchestrated_metadata_root)
        assert instance._handler_state == mock_handler_state

        # Cleanup
        await instance._cleanup_sample(interrupted=False)

        # Verify both init and cleanup were called
        assert mock_orch_init.init_orchestrated_sub_task.call_count == 1
        assert mock_orch_init.cleanup_orchestrated_sub_task.call_count == 1


class TestActiveSampleRegistration:
    """Test ActiveSample registration with OrchestrationCoordinator."""

    @pytest.mark.asyncio
    async def test_active_sample_registered_for_orchestrated_task(self, orchestrated_metadata_root):
        """Test that ActiveSample is registered with coordinator for orchestrated samples."""
        from saber.inspect_ai.core.types import OrchestrationSubTaskState

        # Setup registry entry
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
            session_id="session_456",
        )

        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()
        instance._episode_manager = AsyncMock()

        # Mock OrchestrationInitializer
        mock_orch_init = AsyncMock()
        mock_handler_state = OrchestrationSubTaskState(
            episode_ids=["episode_blue"],
            primary_episode_id="episode_blue",
            orchestration_id="orch_123",
            sub_task_role="blue",
        )
        mock_orch_init.init_orchestrated_sub_task.return_value = mock_handler_state
        type(instance)._orchestration_initializer = mock_orch_init

        # Mock sample_active to return an ActiveSample
        mock_active_sample = MagicMock()
        mock_active_sample.id = "active_sample_123"

        # Mock the coordinator
        with patch("saber.inspect_ai.saber.sample_active") as mock_sample_active, \
             patch("saber.inspect_ai.saber.OrchestrationCoordinator") as mock_coordinator_class:
            mock_sample_active.return_value = mock_active_sample
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            # Initialize
            await instance._init_sample(orchestrated_metadata_root)

            # Verify set_active_sample was called
            mock_coordinator.set_active_sample.assert_called_once_with(
                orchestration_id="orch_123",
                role="blue",
                active_sample=mock_active_sample,
            )

    @pytest.mark.asyncio
    async def test_active_sample_not_registered_when_none(self, orchestrated_metadata_root):
        """Test that set_active_sample is not called when sample_active returns None."""
        from saber.inspect_ai.core.types import OrchestrationSubTaskState

        # Setup registry entry
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
            session_id="session_456",
        )

        instance = SABERSandboxEnvironment(
            domain_slug="test_domain",
            domains_root=Path("/tmp"),
        )
        instance._session_id = "session_456"
        instance._session_manager = AsyncMock()
        instance._episode_manager = AsyncMock()

        # Mock OrchestrationInitializer
        mock_orch_init = AsyncMock()
        mock_handler_state = OrchestrationSubTaskState(
            episode_ids=["episode_blue"],
            primary_episode_id="episode_blue",
            orchestration_id="orch_123",
            sub_task_role="blue",
        )
        mock_orch_init.init_orchestrated_sub_task.return_value = mock_handler_state
        type(instance)._orchestration_initializer = mock_orch_init

        # Mock sample_active to return None
        with patch("saber.inspect_ai.saber.sample_active") as mock_sample_active, \
             patch("saber.inspect_ai.saber.OrchestrationCoordinator") as mock_coordinator_class:
            mock_sample_active.return_value = None
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            # Initialize
            await instance._init_sample(orchestrated_metadata_root)

            # Verify set_active_sample was NOT called
            mock_coordinator.set_active_sample.assert_not_called()
