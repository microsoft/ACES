"""Tests for orchestration initialization."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.inspect_ai.core.orchestration_init import OrchestrationInitializer, SandboxError
from saber.models import MetadataKeys


@pytest.fixture
def initializer():
    """Create orchestration initializer."""
    return OrchestrationInitializer()


@pytest.fixture
def session_manager():
    """Create mock session manager."""
    mock = MagicMock()
    mock.create_episode = AsyncMock()
    mock.wait_for_episode_ready = AsyncMock()
    mock.end_episode = AsyncMock()
    return mock


@pytest.fixture
def root_metadata():
    """Root sample metadata."""
    return {
        MetadataKeys.ORCHESTRATION_ID: "orch-123",
        MetadataKeys.SUB_TASK_ROLE: "blue",
        MetadataKeys.TASK_ID: "task-456",
        MetadataKeys.ORDER: "0",
        # No DEPENDS_ON_ROLE = root sample
    }


@pytest.fixture
def dependent_metadata():
    """Dependent sample metadata."""
    return {
        MetadataKeys.ORCHESTRATION_ID: "orch-123",
        MetadataKeys.SUB_TASK_ROLE: "red",
        MetadataKeys.TASK_ID: "task-456",
        MetadataKeys.DEPENDS_ON_ROLE: "blue",
        MetadataKeys.ORDER: "1",
    }


class TestOrchestrationInitializer:
    """Test orchestration initializer."""

    @pytest.mark.asyncio
    async def test_init_root_sample_success(
        self, initializer, root_metadata, session_manager
    ):
        """Test successful initialization of root sample."""
        semaphore = asyncio.Semaphore(2)
        sample_id = "sample-root-1"
        session_id = "session-789"

        # Mock episode creation
        episode_response = MagicMock()
        episode_response.episode_id = "episode-abc"
        session_manager.create_episode.return_value = episode_response

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.register_root_sample.return_value = True
            mock_coordinator.set_episode_id = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            result = await initializer.init_orchestrated_sub_task(
                metadata=root_metadata,
                session_id=session_id,
                session_manager=session_manager,
                sample_id=sample_id,
                semaphore=semaphore,
            )

        # Verify coordinator registration
        mock_coordinator.register_root_sample.assert_called_once_with(
            orchestration_id="orch-123",
            role="blue",
            sample_id=sample_id,
            semaphore=semaphore,
        )

        # Verify episode creation
        session_manager.create_episode.assert_called_once_with(session_id, "task-456")
        session_manager.wait_for_episode_ready.assert_called_once_with(session_id, "episode-abc")

        # Verify episode_id was set
        mock_coordinator.set_episode_id.assert_called_once_with(
            orchestration_id="orch-123",
            role="blue",
            episode_id="episode-abc",
        )

        # Verify handler state
        assert result.episode_ids == ["episode-abc"]
        assert result.primary_episode_id == "episode-abc"
        assert result.semaphore_acquired is True
        assert result.orchestration_id == "orch-123"
        assert result.sub_task_role == "blue"

        # Verify semaphore was acquired
        assert semaphore._value == 1  # Started at 2, acquired 1

    @pytest.mark.asyncio
    async def test_init_root_sample_no_semaphore(
        self, initializer, root_metadata, session_manager
    ):
        """Test root sample initialization without semaphore."""
        sample_id = "sample-root-2"
        session_id = "session-999"

        episode_response = MagicMock()
        episode_response.episode_id = "episode-xyz"
        session_manager.create_episode.return_value = episode_response

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.register_root_sample.return_value = True
            mock_coordinator.set_episode_id = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            result = await initializer.init_orchestrated_sub_task(
                metadata=root_metadata,
                session_id=session_id,
                session_manager=session_manager,
                sample_id=sample_id,
                semaphore=None,  # No semaphore
            )

        # Should succeed without semaphore
        assert result.semaphore_acquired is False
        assert result.orchestration_id == "orch-123"

    @pytest.mark.asyncio
    async def test_init_root_sample_registration_failure(
        self, initializer, root_metadata, session_manager
    ):
        """Test failure when root sample registration fails."""
        semaphore = asyncio.Semaphore(2)
        initial_value = semaphore._value

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.register_root_sample.return_value = False  # Failure
            mock_coordinator.trigger_termination = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            with pytest.raises(SandboxError) as exc_info:
                await initializer.init_orchestrated_sub_task(
                    metadata=root_metadata,
                    session_id="session-id",
                    session_manager=session_manager,
                    sample_id="sample-id",
                    semaphore=semaphore,
                )

        # Verify error message
        assert "Failed to register root sample" in str(exc_info.value)

        # Verify semaphore was released
        assert semaphore._value == initial_value

        # Verify termination was triggered
        mock_coordinator.trigger_termination.assert_called_once_with("orch-123")

    @pytest.mark.asyncio
    async def test_init_dependent_sample_success(
        self, initializer, dependent_metadata, session_manager
    ):
        """Test successful initialization of dependent sample."""
        sample_id = "sample-red-1"
        session_id = "session-dep"

        episode_response = MagicMock()
        episode_response.episode_id = "episode-red-abc"
        session_manager.create_episode.return_value = episode_response

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.register_dependent_sample = AsyncMock(return_value=True)
            mock_coordinator.wait_for_dependency_ready = AsyncMock(return_value="episode-blue-123")
            mock_coordinator.set_episode_id = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            result = await initializer.init_orchestrated_sub_task(
                metadata=dependent_metadata,
                session_id=session_id,
                session_manager=session_manager,
                sample_id=sample_id,
                semaphore=None,
            )

        # Verify dependent registration
        mock_coordinator.register_dependent_sample.assert_called_once_with(
            orchestration_id="orch-123",
            role="red",
            sample_id=sample_id,
            depends_on_role="blue",
            order=1,
            timeout=60,  # SandboxTimeouts.ORCHESTRATION_REGISTRATION_SECONDS
        )

        # Verify dependency wait
        mock_coordinator.wait_for_dependency_ready.assert_called_once()
        call_kwargs = mock_coordinator.wait_for_dependency_ready.call_args[1]
        assert call_kwargs["orchestration_id"] == "orch-123"
        assert call_kwargs["role"] == "red"
        assert call_kwargs["session_manager"] == session_manager
        assert call_kwargs["session_id"] == session_id

        # Verify episode creation
        session_manager.create_episode.assert_called_once_with(session_id, "task-456")

        # Verify handler state
        assert result.episode_ids == ["episode-red-abc"]
        assert result.primary_episode_id == "episode-red-abc"
        assert result.semaphore_acquired is False
        assert result.orchestration_id == "orch-123"
        assert result.sub_task_role == "red"

    @pytest.mark.asyncio
    async def test_init_dependent_sample_registration_failure(
        self, initializer, dependent_metadata, session_manager
    ):
        """Test failure when dependent sample registration fails."""
        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.register_dependent_sample = AsyncMock(return_value=False)  # Failure
            mock_coordinator.trigger_termination = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            with pytest.raises(SandboxError) as exc_info:
                await initializer.init_orchestrated_sub_task(
                    metadata=dependent_metadata,
                    session_id="session-id",
                    session_manager=session_manager,
                    sample_id="sample-id",
                    semaphore=None,
                )

        # Verify error message
        assert "Failed to register dependent sample red" in str(exc_info.value)

        # Verify termination was triggered
        mock_coordinator.trigger_termination.assert_called_once_with("orch-123")

    @pytest.mark.asyncio
    async def test_init_episode_creation_failure(
        self, initializer, root_metadata, session_manager
    ):
        """Test failure during episode creation."""
        semaphore = asyncio.Semaphore(2)
        initial_value = semaphore._value

        # Make episode creation fail
        session_manager.create_episode.side_effect = Exception("Episode creation failed")

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.register_root_sample.return_value = True
            mock_coordinator.trigger_termination = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            with pytest.raises(SandboxError) as exc_info:
                await initializer.init_orchestrated_sub_task(
                    metadata=root_metadata,
                    session_id="session-id",
                    session_manager=session_manager,
                    sample_id="sample-id",
                    semaphore=semaphore,
                )

        # Verify error propagation
        assert "Failed to initialize orchestrated sub-task" in str(exc_info.value)

        # Verify semaphore was released
        assert semaphore._value == initial_value

        # Verify termination was triggered
        mock_coordinator.trigger_termination.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_orchestrated_sub_task_success(
        self, initializer, session_manager
    ):
        """Test successful cleanup of orchestrated sub-task."""
        handler_state = {
            "episode_ids": ["episode-1"],
            "primary_episode_id": "episode-1",
            "semaphore_acquired": True,
            "orchestration_id": "orch-cleanup-1",
            "sub_task_role": "blue",
        }

        semaphore = asyncio.Semaphore(2)
        await semaphore.acquire()  # Simulate acquired semaphore
        initial_value = semaphore._value

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            # Return episodes to cleanup
            mock_coordinator.trigger_termination.return_value = [
                ("blue", "episode-1"),
                ("red", "episode-2"),
            ]
            mock_coordinator.cleanup_sample.return_value = True  # Should release semaphore
            mock_coordinator_class.return_value = mock_coordinator

            await initializer.cleanup_orchestrated_sub_task(
                handler_state=handler_state,
                session_id="session-cleanup",
                session_manager=session_manager,
                semaphore=semaphore,
            )

        # Verify termination was triggered with skip_role to avoid self-interrupt
        mock_coordinator.trigger_termination.assert_called_once_with(
            "orch-cleanup-1", skip_role="blue"
        )

        # Verify all episodes were ended
        assert session_manager.end_episode.call_count == 2
        session_manager.end_episode.assert_any_call("session-cleanup", "episode-1")
        session_manager.end_episode.assert_any_call("session-cleanup", "episode-2")

        # Verify cleanup was called
        mock_coordinator.cleanup_sample.assert_called_once_with(
            orchestration_id="orch-cleanup-1",
            role="blue",
            semaphore=semaphore,
        )

        # Verify semaphore was released
        assert semaphore._value == initial_value + 1

    @pytest.mark.asyncio
    async def test_cleanup_no_semaphore_release(
        self, initializer, session_manager
    ):
        """Test cleanup when semaphore should not be released."""
        handler_state = {
            "episode_ids": ["episode-1"],
            "primary_episode_id": "episode-1",
            "semaphore_acquired": False,
            "orchestration_id": "orch-cleanup-2",
            "sub_task_role": "red",
        }

        semaphore = asyncio.Semaphore(2)
        initial_value = semaphore._value

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.trigger_termination.return_value = [("red", "episode-1")]
            mock_coordinator.cleanup_sample.return_value = False  # Don't release semaphore
            mock_coordinator_class.return_value = mock_coordinator

            await initializer.cleanup_orchestrated_sub_task(
                handler_state=handler_state,
                session_id="session-cleanup",
                session_manager=session_manager,
                semaphore=semaphore,
            )

        # Verify semaphore was NOT released (value unchanged)
        assert semaphore._value == initial_value

    @pytest.mark.asyncio
    async def test_cleanup_no_semaphore(
        self, initializer, session_manager
    ):
        """Test cleanup without semaphore."""
        handler_state = {
            "episode_ids": ["episode-1"],
            "primary_episode_id": "episode-1",
            "semaphore_acquired": True,
            "orchestration_id": "orch-cleanup-3",
            "sub_task_role": "blue",
        }

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.trigger_termination.return_value = [("blue", "episode-1")]
            mock_coordinator.cleanup_sample.return_value = True
            mock_coordinator_class.return_value = mock_coordinator

            # Should not raise even without semaphore
            await initializer.cleanup_orchestrated_sub_task(
                handler_state=handler_state,
                session_id="session-cleanup",
                session_manager=session_manager,
                semaphore=None,
            )

        # Verify cleanup was still called
        mock_coordinator.cleanup_sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_episode_end_failure(
        self, initializer, session_manager
    ):
        """Test cleanup when episode ending fails."""
        handler_state = {
            "episode_ids": ["episode-1"],
            "primary_episode_id": "episode-1",
            "semaphore_acquired": False,
            "orchestration_id": "orch-cleanup-4",
            "sub_task_role": "blue",
        }

        # Make end_episode fail
        session_manager.end_episode.side_effect = Exception("End episode failed")

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.trigger_termination.return_value = [
                ("blue", "episode-1"),
                ("red", "episode-2"),
            ]
            mock_coordinator.cleanup_sample.return_value = False
            mock_coordinator_class.return_value = mock_coordinator

            # Should not raise - errors are logged
            await initializer.cleanup_orchestrated_sub_task(
                handler_state=handler_state,
                session_id="session-cleanup",
                session_manager=session_manager,
                semaphore=None,
            )

        # Verify both episodes were attempted
        assert session_manager.end_episode.call_count == 2

        # Verify cleanup still happened
        mock_coordinator.cleanup_sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_termination_trigger_failure(
        self, initializer, root_metadata, session_manager
    ):
        """Test that termination trigger failure during init is logged but doesn't raise."""
        semaphore = asyncio.Semaphore(2)

        session_manager.create_episode.side_effect = Exception("Episode failed")

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.register_root_sample.return_value = True
            # Make trigger_termination fail
            mock_coordinator.trigger_termination.side_effect = Exception("Trigger failed")
            mock_coordinator_class.return_value = mock_coordinator

            with pytest.raises(SandboxError):
                await initializer.init_orchestrated_sub_task(
                    metadata=root_metadata,
                    session_id="session-id",
                    session_manager=session_manager,
                    sample_id="sample-id",
                    semaphore=semaphore,
                )

        # Should have attempted termination despite failure
        mock_coordinator.trigger_termination.assert_called_once()


class TestCleanupWithSkipRole:
    """Test cleanup_orchestrated_sub_task with skip_role for sibling interruption."""

    @pytest.fixture
    def initializer(self):
        """Create orchestration initializer."""
        return OrchestrationInitializer()

    @pytest.fixture
    def session_manager(self):
        """Create mock session manager."""
        mock = MagicMock()
        mock.create_episode = AsyncMock()
        mock.wait_for_episode_ready = AsyncMock()
        mock.end_episode = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_cleanup_passes_skip_role_to_trigger_termination(
        self, initializer, session_manager
    ):
        """Test that cleanup passes the current role as skip_role to avoid self-interrupt."""
        handler_state = {
            "episode_ids": ["episode-blue"],
            "primary_episode_id": "episode-blue",
            "semaphore_acquired": False,
            "orchestration_id": "orch-skip-test",
            "sub_task_role": "blue",  # Blue is cleaning up
        }

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.trigger_termination.return_value = [
                ("blue", "episode-blue"),
                ("red", "episode-red"),
            ]
            mock_coordinator.cleanup_sample.return_value = False
            mock_coordinator_class.return_value = mock_coordinator

            await initializer.cleanup_orchestrated_sub_task(
                handler_state=handler_state,
                session_id="session-skip",
                session_manager=session_manager,
                semaphore=None,
            )

        # Verify trigger_termination was called with skip_role="blue"
        mock_coordinator.trigger_termination.assert_called_once_with(
            "orch-skip-test", skip_role="blue"
        )

    @pytest.mark.asyncio
    async def test_cleanup_with_orchestration_state_object(
        self, initializer, session_manager
    ):
        """Test cleanup with OrchestrationSubTaskState object."""
        from saber.inspect_ai.core.types import OrchestrationSubTaskState

        handler_state = OrchestrationSubTaskState(
            episode_ids=["episode-green"],
            primary_episode_id="episode-green",
            semaphore_acquired=False,
            orchestration_id="orch-state-obj",
            sub_task_role="green",
        )

        with patch("saber.inspect_ai.core.orchestration_init.OrchestrationCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator.trigger_termination.return_value = [("green", "episode-green")]
            mock_coordinator.cleanup_sample.return_value = True
            mock_coordinator_class.return_value = mock_coordinator

            await initializer.cleanup_orchestrated_sub_task(
                handler_state=handler_state,
                session_id="session-obj",
                session_manager=session_manager,
                semaphore=None,
            )

        # Verify trigger_termination was called with skip_role="green"
        mock_coordinator.trigger_termination.assert_called_once_with(
            "orch-state-obj", skip_role="green"
        )
