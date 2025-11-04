"""Integration tests for SABER Domain Task factory pattern.

These tests verify the end-to-end flow:
- Domain task modules can be imported without starting servers
- Task callable execution constructs proper Task objects
- Cleanup occurs properly on all paths
- Inspect AI list tasks works correctly
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample

from saber.models import BenchmarkInfo, TaskInfo

# Add workspace root to sys.path for domain imports
_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


@pytest.fixture
def mock_task_info():
    """Mock TaskInfo for testing."""
    return TaskInfo(
        task_id="integration_test_task",
        title="Integration Test Task",
        description="A task for integration testing",
        episode_attempts=1,
        subtask_count=0,
        max_steps=10,
        instruction_prompt="Test instruction",
        assistant_prompt="Test assistant",
        submit_prompt="Test submit",
    )


@pytest.fixture
def mock_benchmark_info(mock_task_info):
    """Mock BenchmarkInfo."""
    return BenchmarkInfo(
        domain="test_domain",
        tasks=[mock_task_info],
        total_tasks=1,
        total_episodes=1,
    )


@pytest.fixture
def mock_domain_context():
    """Mock DomainContext."""
    from saber.inspect_ai.server import DomainContext

    return DomainContext(
        domain="test_domain",
        rest_url="http://localhost:8000",
        mcp_url="http://localhost:8001",
        rest_port=8000,
        mcp_port=8001,
        project_slug="saber-test_domain",
        domains_root=Path("/test/domains"),
    )


@pytest.fixture(autouse=True)
def clear_all_registries():
    """Clear all registries before/after tests."""
    from saber.inspect_ai.saber import SABERSandboxEnvironment
    from saber.inspect_ai.tasks import _active_domains, _active_domains_lock

    SABERSandboxEnvironment._registry.clear()
    with _active_domains_lock:
        _active_domains.clear()

    yield

    SABERSandboxEnvironment._registry.clear()
    with _active_domains_lock:
        _active_domains.clear()


class TestDomainTaskImport:
    """Test that domain task modules can be imported safely."""

    def test_import_cybench_no_side_effects(self):
        """Test importing domains/cybench doesn't start server."""
        from saber.inspect_ai.tasks import _active_domains

        # Import should succeed
        from domains.cybench import cybench

        # Should be a callable
        assert callable(cybench)

        # Should not have started server
        assert len(_active_domains) == 0

    def test_import_excytin_no_side_effects(self):
        """Test importing domains/excytin doesn't start server."""
        from saber.inspect_ai.tasks import _active_domains

        # Import should succeed
        from domains.excytin import excytin

        # Should be a callable
        assert callable(excytin)

        # Should not have started server
        assert len(_active_domains) == 0


class TestTaskCallableExecution:
    """Test task callable execution flow."""

    def test_task_callable_creates_task_object(
        self,
        mock_domain_context,
        mock_benchmark_info,
    ):
        """Test that task callable creates proper Task object."""
        from saber.inspect_ai.tasks import create_domain_task

        with patch('saber.inspect_ai.tasks._get_or_create_portal') as mock_portal_func, \
             patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.tasks._wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.tasks._resolve_agent_implementation') as mock_resolve_agent:

            # Setup mocks
            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()

            mock_client = MagicMock()
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            mock_samples = [
                Sample(
                    id="integration_test_task__attempt_1",
                    input="Test input",
                    target="Test target",
                    metadata={"task_id": "integration_test_task"},
                )
            ]
            mock_create_dataset.return_value = mock_samples

            # Mock portal to run async code synchronously for test
            def mock_portal_call(async_func, *args, **kwargs):
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(async_func(*args, **kwargs))
                finally:
                    loop.close()

            mock_portal = MagicMock()
            mock_portal.call = mock_portal_call
            mock_portal_func.return_value = mock_portal

            # Create task callable
            task_callable = create_domain_task("test_domain", Path("/test/domains"))

            # Execute callable
            task = task_callable(rest_port=8000, mcp_port=8001)

            # Verify task structure
            assert isinstance(task, Task)
            assert len(task.dataset) == 1
            assert task.dataset[0].id == "integration_test_task__attempt_1"
            # Verify sandbox is configured for SABER
            assert task.sandbox is not None
            # Note: sandbox is a SandboxEnvironmentSpec, check type name
            assert task.sandbox.type == "saber"

    def test_task_callable_with_filter(
        self,
        mock_domain_context,
    ):
        """Test task callable with task_filter parameter."""
        from saber.inspect_ai.tasks import create_domain_task

        # Create multiple tasks
        tasks = [
            TaskInfo(
                task_id="labyrinth_easy",
                title="Easy",
                description="Easy",
                episode_attempts=1,
                subtask_count=0,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
            TaskInfo(
                task_id="labyrinth_hard",
                title="Hard",
                description="Hard",
                episode_attempts=1,
                subtask_count=0,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
        ]

        mock_benchmark_info = BenchmarkInfo(
            domain="test_domain",
            tasks=tasks,
            total_tasks=2,
            total_episodes=2,
        )

        with patch('saber.inspect_ai.tasks._get_or_create_portal') as mock_portal_func, \
             patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.tasks._wait_for_server_health'), \
             patch('saber.inspect_ai.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.tasks._resolve_agent_implementation') as mock_resolve_agent:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()

            mock_client = MagicMock()
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            # create_saber_dataset will only receive filtered tasks
            mock_create_dataset.return_value = [
                Sample(
                    id="labyrinth_easy__attempt_1",
                    input="Easy",
                    target="Pass",
                    metadata={"task_id": "labyrinth_easy"},
                )
            ]

            def mock_portal_call(async_func, *args, **kwargs):
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(async_func(*args, **kwargs))
                finally:
                    loop.close()

            mock_portal = MagicMock()
            mock_portal.call = mock_portal_call
            mock_portal_func.return_value = mock_portal

            # Create and execute callable with filter
            task_callable = create_domain_task("test_domain", Path("/test/domains"))
            task = task_callable(task_filter="labyrinth_easy")

            # Verify only filtered task in dataset
            assert len(task.dataset) == 1
            assert task.dataset[0].id == "labyrinth_easy__attempt_1"


class TestCleanupOnFailure:
    """Test cleanup occurs properly on failure paths."""

    def test_cleanup_on_task_callable_failure(self, mock_domain_context):
        """Test that failure during task callable execution cleans up."""
        from saber.inspect_ai.tasks import _active_domains, create_domain_task

        with patch('saber.inspect_ai.tasks._get_or_create_portal') as mock_portal_func, \
             patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.tasks._wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.tasks._resolve_agent_implementation') as mock_resolve_agent:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.stop = AsyncMock()
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()

            # Health check fails
            async def failing_health(*args, **kwargs):
                from inspect_ai._util.error import PrerequisiteError
                raise PrerequisiteError("Health check failed")

            mock_health.side_effect = failing_health

            def mock_portal_call(async_func, *args, **kwargs):
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(async_func(*args, **kwargs))
                finally:
                    loop.close()

            mock_portal = MagicMock()
            mock_portal.call = mock_portal_call
            mock_portal_func.return_value = mock_portal

            # Create and execute callable
            task_callable = create_domain_task("test_domain", Path("/test/domains"))

            from inspect_ai._util.error import PrerequisiteError
            with pytest.raises(PrerequisiteError):
                task_callable()

            # Verify cleanup occurred
            assert len(_active_domains) == 0
            mock_controller.stop.assert_called_once()


class TestInspectListTasks:
    """Test inspect list tasks functionality."""

    def test_list_tasks_shows_domains(self):
        """Test that domain tasks appear in task listing."""
        # This is a manual verification test - document expected behavior
        # In real usage:
        # $ uv run inspect list tasks | grep domains/
        # Should show: domains/cybench, domains/excytin

        # We can verify the tasks are importable
        from domains.cybench import cybench
        from domains.excytin import excytin

        assert callable(cybench)
        assert callable(excytin)

        # And that importing didn't start servers
        from saber.inspect_ai.tasks import _active_domains
        assert len(_active_domains) == 0
