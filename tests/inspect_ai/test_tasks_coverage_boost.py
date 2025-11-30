"""Additional tests to boost coverage for tasks.py to 90%+.

Focuses on uncovered lines:
- Build parameter validation errors (line 166)
- External domain detection and reuse (lines 300, 302, 311-312, 322, 332, 348)
- Orchestrated task logging (lines 444-446, 449)
- Cleanup error handling (lines 479-482, 541-542)
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.dataset import Sample

from saber.inspect_ai.core.tasks import _start_and_load_tasks, create_domain_task
from saber.inspect_ai.server.domain_manager import _active_domains, _active_domains_lock
from saber.models import BenchmarkInfo, SingleEpisodeTask, SubTaskDefinition
from saber.models.constants import MetadataKeys


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
def clear_active_domains():
    """Clear active domains registry before each test."""
    with _active_domains_lock:
        _active_domains.clear()
    yield
    with _active_domains_lock:
        _active_domains.clear()


class TestBuildParameterValidation:
    """Test build parameter validation (line 166)."""

    def test_mutually_exclusive_build_options_build_and_rebuild(self):
        """Test error when both build and rebuild are specified."""
        task_callable = create_domain_task("test_domain", Path("/test/domains"))

        with pytest.raises(PrerequisiteError) as exc_info:
            task_callable(build=True, rebuild="server")

        error_msg = str(exc_info.value)
        assert "Cannot specify multiple build options" in error_msg
        assert "build=true" in error_msg
        assert "rebuild=" in error_msg

    def test_mutually_exclusive_build_options_build_and_rebuild_all(self):
        """Test error when both build and rebuild_all are specified."""
        task_callable = create_domain_task("test_domain", Path("/test/domains"))

        with pytest.raises(PrerequisiteError) as exc_info:
            task_callable(build=True, rebuild_all=True)

        error_msg = str(exc_info.value)
        assert "Cannot specify multiple build options" in error_msg

    def test_mutually_exclusive_build_options_all_three(self):
        """Test error when all three build options are specified."""
        task_callable = create_domain_task("test_domain", Path("/test/domains"))

        with pytest.raises(PrerequisiteError) as exc_info:
            task_callable(build=True, rebuild="server", rebuild_all=True)

        error_msg = str(exc_info.value)
        assert "Cannot specify multiple build options" in error_msg


class TestExternalDomainDetection:
    """Test external domain detection and reuse (lines 300, 302, 311-312, 322, 332, 348)."""

    @pytest.mark.asyncio
    async def test_external_domain_same_domain_reuse(self, mock_domain_context):
        """Test reusing externally running domain (same domain on same ports)."""
        with patch('saber.inspect_ai.core.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.core.tasks.wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.core.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.core.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.core.tasks.resolve_agent_implementation') as mock_resolve_agent, \
             patch('saber.inspect_ai.core.tasks.register_domain') as mock_register:

            # Mock controller to return same domain name (external server running)
            mock_controller = AsyncMock()
            mock_controller.check_running_domain = AsyncMock(return_value="test_domain")  # Lines 300
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()
            mock_health.return_value = None

            mock_client = MagicMock()
            mock_benchmark_info = BenchmarkInfo(
                domain="test_domain",
                tasks=[
                    SingleEpisodeTask(
                        benchmark_task_id="task1",
                        task_id="task1",
                        domain="test_domain",
                        title="Task 1",
                        description="Test task",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="",
                        assistant_prompt="",
                        submit_prompt="",
                    )
                ],
                total_tasks=1,
                total_episodes=1,
            )
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            mock_samples = [
                Sample(
                    id="task1__attempt_1",
                    input="Test input",
                    target="Test target",
                    metadata={"task_id": "task1"},
                )
            ]
            mock_create_dataset.return_value = mock_samples

            # Execute - should detect and reuse external domain
            task = await _start_and_load_tasks(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
                rest_port=8000,
                mcp_port=8001,
                task_filter=None,
                agent_name="react",
                log_level="INFO",
                build=None,
                rebuild=None,
                compose_template_path=None,
                stop_saber_after=False,
                max_concurrent_episodes=3,
                run_preflight=False,
            )

            # Verify task was created (lines 302, 311-312)
            assert task is not None
            assert len(task.dataset) == 1

            # Verify domain was registered with ownership=False (line 322-332)
            mock_register.assert_called_once()
            call_kwargs = mock_register.call_args.kwargs
            assert call_kwargs["ownership"] is False  # We didn't start it

    @pytest.mark.asyncio
    async def test_external_domain_different_domain_conflict(self):
        """Test error when different domain is running on same ports."""
        with patch('saber.inspect_ai.core.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.core.tasks.resolve_agent_implementation') as mock_resolve_agent:

            # Mock controller to return different domain name (conflict)
            mock_controller = AsyncMock()
            mock_controller.check_running_domain = AsyncMock(return_value="different_domain")  # Line 332
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()

            # Execute - should raise error about port conflict
            with pytest.raises(PrerequisiteError) as exc_info:
                await _start_and_load_tasks(
                    domain_slug="test_domain",
                    domains_root=Path("/test/domains"),
                    rest_port=8000,
                    mcp_port=8001,
                    task_filter=None,
                    agent_name="react",
                    log_level="INFO",
                    build=None,
                    rebuild=None,
                    compose_template_path=None,
                    stop_saber_after=False,
                    max_concurrent_episodes=3,
                    run_preflight=False,
                )

            # Verify error message contains helpful info (lines 333-348)
            error_msg = str(exc_info.value)
            assert "different_domain" in error_msg
            assert "already running" in error_msg
            assert "8000" in error_msg
            assert "8001" in error_msg
            assert "Stop the running domain" in error_msg or "stop" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_preflight_check_execution(self, mock_domain_context):
        """Test preflight check is executed when requested (line 348)."""
        with patch('saber.inspect_ai.core.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.core.tasks.run_preflight_check') as mock_preflight, \
             patch('saber.inspect_ai.core.tasks.wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.core.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.core.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.core.tasks.resolve_agent_implementation') as mock_resolve_agent:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()
            mock_health.return_value = None
            mock_preflight.return_value = None  # Preflight succeeds

            mock_client = MagicMock()
            mock_benchmark_info = BenchmarkInfo(
                domain="test_domain",
                tasks=[],
                total_tasks=0,
                total_episodes=0,
            )
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            mock_samples = [
                Sample(
                    id="dummy__attempt_1",
                    input="Dummy",
                    target="Dummy",
                    metadata={},
                )
            ]
            mock_create_dataset.return_value = mock_samples

            # Execute with run_preflight=True
            task = await _start_and_load_tasks(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
                rest_port=8000,
                mcp_port=8001,
                task_filter=None,
                agent_name="react",
                log_level="INFO",
                build=None,
                rebuild=None,
                compose_template_path=None,
                stop_saber_after=False,
                max_concurrent_episodes=3,
                run_preflight=True,  # Enable preflight
            )

            # Verify preflight was called (line 348)
            mock_preflight.assert_called_once_with(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
                concurrency=3,
                timeout=180,
            )


class TestOrchestrationLogging:
    """Test orchestration metadata logging (lines 444-446, 449)."""

    @pytest.mark.asyncio
    async def test_orchestrated_task_logging(self, mock_domain_context):
        """Test logging when dataset contains orchestrated tasks."""
        with patch('saber.inspect_ai.core.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.core.tasks.wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.core.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.core.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.core.tasks.resolve_agent_implementation') as mock_resolve_agent, \
             patch('saber.inspect_ai.core.tasks.logger') as mock_logger:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()
            mock_health.return_value = None

            # Create orchestrated task with sub-tasks
            from saber.models import OrchestratedTask

            orchestrated_task = OrchestratedTask(
                benchmark_task_id="orch_task",
                domain="test_domain",
                title="Orchestrated Task",
                description="Multi-role orchestrated task",
                episode_attempts=1,
                sub_tasks=[
                    SubTaskDefinition(
                        task_id="red_task",
                        role="red",
                        order=0,
                        depends_on_role=None,
                        domain="test_domain",
                        title="Red Task",
                        description="Red team task",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="",
                        assistant_prompt="",
                        submit_prompt="",
                    ),
                    SubTaskDefinition(
                        task_id="blue_task",
                        role="blue",
                        order=1,
                        depends_on_role="red",
                        domain="test_domain",
                        title="Blue Task",
                        description="Blue team task",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="",
                        assistant_prompt="",
                        submit_prompt="",
                    ),
                ],
            )

            orchestrated_tasks = [orchestrated_task]

            mock_client = MagicMock()
            mock_benchmark_info = BenchmarkInfo(
                domain="test_domain",
                tasks=orchestrated_tasks,
                total_tasks=2,
                total_episodes=2,
            )
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            # Create samples with orchestration metadata
            mock_samples = [
                Sample(
                    id="red_task__attempt_1",
                    input="Red input",
                    target="Red target",
                    metadata={
                        "task_id": "red_task",
                        MetadataKeys.ORCHESTRATION_ID: "orch_1",  # Line 442
                    },
                ),
                Sample(
                    id="blue_task__attempt_1",
                    input="Blue input",
                    target="Blue target",
                    metadata={
                        "task_id": "blue_task",
                        MetadataKeys.ORCHESTRATION_ID: "orch_1",  # Line 442
                    },
                ),
            ]
            mock_create_dataset.return_value = mock_samples

            # Execute
            task = await _start_and_load_tasks(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
                rest_port=8000,
                mcp_port=8001,
                task_filter=None,
                agent_name="react",
                log_level="INFO",
                build=None,
                rebuild=None,
                compose_template_path=None,
                stop_saber_after=False,
                max_concurrent_episodes=3,
                run_preflight=False,
            )

            # Verify orchestration logging was triggered (lines 449)
            # Check that logger.info was called with orchestration message
            info_calls = [call for call in mock_logger.info.call_args_list]
            orchestration_log_found = any(
                "orchestrated task" in str(call).lower() or "orchestration" in str(call).lower()
                for call in info_calls
            )
            assert orchestration_log_found, "Expected orchestration logging to occur"


class TestCleanupErrorHandling:
    """Test cleanup error handling (lines 479-482, 541-542)."""

    @pytest.mark.asyncio
    async def test_controller_stop_failure_during_cleanup(self, mock_domain_context):
        """Test handling of controller.stop() failure during cleanup (lines 541-542)."""
        with patch('saber.inspect_ai.core.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.core.tasks.wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.core.tasks.resolve_agent_implementation') as mock_resolve_agent, \
             patch('saber.inspect_ai.core.tasks.logger') as mock_logger:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.check_running_domain = AsyncMock(return_value=None)

            # Make stop fail with an error
            mock_controller.stop = AsyncMock(side_effect=Exception("Failed to stop containers"))
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()

            # Make health check fail to trigger cleanup
            mock_health.side_effect = Exception("Health check failed")

            # Execute and expect failure
            with pytest.raises(PrerequisiteError):
                await _start_and_load_tasks(
                    domain_slug="test_domain",
                    domains_root=Path("/test/domains"),
                    rest_port=8000,
                    mcp_port=8001,
                    task_filter=None,
                    agent_name="react",
                    log_level="INFO",
                    build=None,
                    rebuild=None,
                    compose_template_path=None,
                    stop_saber_after=False,
                    max_concurrent_episodes=3,
                    run_preflight=False,
                )

            # Verify cleanup was attempted and warning was logged (lines 541-542)
            mock_controller.stop.assert_called_once_with("test_domain")

            # Check that warning was logged about stop failure
            warning_calls = [call for call in mock_logger.warning.call_args_list]
            stop_warning_found = any(
                "Failed to stop" in str(call) or "cleanup" in str(call).lower()
                for call in warning_calls
            )
            assert stop_warning_found, "Expected warning about stop failure"

    @pytest.mark.asyncio
    async def test_prerequisite_error_re_raised(self):
        """Test that PrerequisiteError is re-raised without wrapping (line 209, 547)."""
        with patch('saber.inspect_ai.core.tasks.resolve_agent_implementation') as mock_resolve_agent:

            # Make agent resolution fail with PrerequisiteError
            original_error = PrerequisiteError("Agent not found")
            mock_resolve_agent.side_effect = original_error

            # Execute and expect same PrerequisiteError
            with pytest.raises(PrerequisiteError) as exc_info:
                await _start_and_load_tasks(
                    domain_slug="test_domain",
                    domains_root=Path("/test/domains"),
                    rest_port=8000,
                    mcp_port=8001,
                    task_filter=None,
                    agent_name="nonexistent",
                    log_level="INFO",
                    build=None,
                    rebuild=None,
                    compose_template_path=None,
                    stop_saber_after=False,
                    max_concurrent_episodes=3,
                    run_preflight=False,
                )

            # Verify it's the original error, not wrapped (line 547)
            assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_generic_exception_wrapped_as_prerequisite_error(self):
        """Test that generic exceptions are wrapped in PrerequisiteError (line 549)."""
        with patch('saber.inspect_ai.core.tasks.resolve_agent_implementation') as mock_resolve_agent:

            # Make agent resolution fail with generic exception
            mock_resolve_agent.side_effect = ValueError("Invalid configuration")

            # Execute and expect PrerequisiteError wrapping
            with pytest.raises(PrerequisiteError) as exc_info:
                await _start_and_load_tasks(
                    domain_slug="test_domain",
                    domains_root=Path("/test/domains"),
                    rest_port=8000,
                    mcp_port=8001,
                    task_filter=None,
                    agent_name="react",
                    log_level="INFO",
                    build=None,
                    rebuild=None,
                    compose_template_path=None,
                    stop_saber_after=False,
                    max_concurrent_episodes=3,
                    run_preflight=False,
                )

            # Verify error message contains both wrapper and original (line 549)
            error_msg = str(exc_info.value)
            assert "Failed to construct task" in error_msg
            assert "Invalid configuration" in error_msg


class TestRoleConfigWithAllModels:
    """Test role config with all models defined (lines 479-482)."""

    @pytest.mark.asyncio
    async def test_task_with_all_roles_having_models(self, mock_domain_context):
        """Test Task model assignment when all roles have models (lines 479-482)."""
        from saber.client.models import RoleBasedConfig, RoleAgentConfig

        role_config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="mockllm/model"),
                "blue": RoleAgentConfig(agent="react", model="mockllm/model"),
            }
        )

        with patch('saber.inspect_ai.core.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.core.tasks.wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.core.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.core.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.core.tasks.resolve_agent_implementation') as mock_resolve_agent, \
             patch('saber.inspect_ai.core.tasks.logger') as mock_logger:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()
            mock_health.return_value = None

            mock_client = MagicMock()
            mock_benchmark_info = BenchmarkInfo(
                domain="test_domain",
                tasks=[],
                total_tasks=0,
                total_episodes=0,
            )
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            mock_samples = [
                Sample(
                    id="dummy__attempt_1",
                    input="Dummy",
                    target="Dummy",
                    metadata={},
                )
            ]
            mock_create_dataset.return_value = mock_samples

            # Execute with role_config
            task = await _start_and_load_tasks(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
                rest_port=8000,
                mcp_port=8001,
                task_filter=None,
                agent_name="react",
                log_level="INFO",
                build=None,
                rebuild=None,
                compose_template_path=None,
                stop_saber_after=False,
                max_concurrent_episodes=3,
                run_preflight=False,
                role_config=role_config,
            )

            # Verify Task model was set to first role's model (line 481)
            # Note: Inspect AI parses "mockllm/model" and sets model.name to "model"
            assert task.model.name == "model"  # First role's model (parsed from "mockllm/model")

            # Verify logging about model selection (lines 482)
            info_calls = [call for call in mock_logger.info.call_args_list]
            model_log_found = any(
                "All roles have models" in str(call) or "default" in str(call).lower()
                for call in info_calls
            )
            assert model_log_found, "Expected logging about model selection"
