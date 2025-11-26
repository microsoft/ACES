"""Unit tests for SABER Domain Task factory (tasks.py).

Tests cover:
- Successful task construction and dataset population
- Server startup failure handling and cleanup
- Health check retry logic
- Task filter exact and glob matching
- Concurrent domain evaluation blocking
- Preflight check for already-running instances
- Registry cleanup on all failure paths
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.dataset import Sample

from saber.inspect_ai.tasks import (
    _active_domains,
    _active_domains_lock,
    _apply_task_filter,
    _start_and_load_tasks,
    _wait_for_server_health,
    create_domain_task,
    get_active_domain,
    remove_active_domain,
)
from saber.models import BenchmarkInfo, SingleEpisodeTask


@pytest.fixture
def mock_task():
    """Mock SingleEpisodeTask for testing."""
    return SingleEpisodeTask(
        benchmark_task_id="test_task",
        task_id="test_task",
        domain="test_domain",
        title="Test Task",
        description="A test task",
        episode_attempts=2,
        max_steps=10,
        instruction_prompt="Test instruction",
        assistant_prompt="Test assistant",
        submit_prompt="Test submit",
    )


@pytest.fixture
def mock_benchmark_info(mock_task):
    """Mock BenchmarkInfo with tasks."""
    return BenchmarkInfo(
        domain="test_domain",
        tasks=[mock_task],
        total_tasks=1,
        total_episodes=2,
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
def clear_active_domains():
    """Clear active domains registry before each test."""
    with _active_domains_lock:
        _active_domains.clear()
    yield
    with _active_domains_lock:
        _active_domains.clear()


class TestCreateDomainTask:
    """Test create_domain_task factory function."""

    def test_create_domain_task_returns_callable(self):
        """Test that create_domain_task returns a callable."""
        task_callable = create_domain_task("test_domain", Path("/test/domains"))
        assert callable(task_callable)

    @patch('saber.inspect_ai.tasks._get_or_create_portal')
    def test_task_callable_uses_portal(self, mock_get_portal):
        """Test that task callable uses BlockingPortal to run async code."""
        mock_portal = MagicMock()
        mock_portal.call = MagicMock(side_effect=PrerequisiteError("test"))
        mock_get_portal.return_value = mock_portal

        task_callable = create_domain_task("test_domain", Path("/test/domains"))

        with pytest.raises(PrerequisiteError):
            task_callable()

        mock_portal.call.assert_called_once()


class TestStartAndLoadTasks:
    """Test _start_and_load_tasks async helper."""

    @pytest.mark.asyncio
    async def test_successful_task_construction(
        self,
        mock_domain_context,
        mock_benchmark_info,
    ):
        """Test successful task construction with dataset population."""
        with patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
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
            mock_health.return_value = None

            mock_client = MagicMock()
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            mock_samples = [
                Sample(
                    id="test_task__attempt_1",
                    input="Test input",
                    target="Test target",
                    metadata={"task_id": "test_task"},
                )
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

                # Verify
        assert task is not None
        assert len(task.dataset) == 1
        assert task.sandbox.type == "saber"
        assert task.sandbox.config.domain_slug == "test_domain"

        # Verify active domains registry
        assert "test_domain" in _active_domains
        assert _active_domains["test_domain"]["rest_url"] == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_concurrent_domain_reuse(self, mock_domain_context, mock_benchmark_info):
        """Test that existing active domains are reused (not blocked)."""
        with patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.tasks._wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.tasks._resolve_agent_implementation') as mock_resolve_agent:

            # Populate active domains with all required keys
            mock_controller = AsyncMock()
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            with _active_domains_lock:
                _active_domains["test_domain"] = {
                    "owner": "existing_task",
                    "domain_slug": "test_domain",
                    "rest_port": 8000,
                    "mcp_port": 8001,
                    "context": mock_domain_context,
                    "controller": mock_controller,
                }

            mock_controller_class.return_value = mock_controller
            mock_resolve_agent.return_value = MagicMock()
            mock_health.return_value = None

            mock_client = MagicMock()
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            mock_samples = [
                Sample(
                    id="test_task__attempt_1",
                    input="Test input",
                    target="Test target",
                    metadata={"task_id": "test_task"},
                )
            ]
            mock_create_dataset.return_value = mock_samples

            # Should reuse the existing domain, not raise an error
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

            # Verify task was created successfully
            assert task is not None
            assert len(task.dataset) == 1

    @pytest.mark.asyncio
    async def test_cleanup_on_controller_start_failure(self, mock_domain_context):
        """Test cleanup when controller.start() fails."""
        with patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.tasks._resolve_agent_implementation') as mock_resolve_agent:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(side_effect=Exception("Docker error"))
            mock_controller.stop = AsyncMock()
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()

            # Execute and expect failure
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

            assert "Docker error" in str(exc_info.value)

            # Verify cleanup - domain was never registered since start() failed
            assert "test_domain" not in _active_domains
            # controller.stop should NOT be called since the domain never started successfully
            mock_controller.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_on_health_check_failure(self, mock_domain_context):
        """Test cleanup when health check fails."""
        with patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.tasks._wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.tasks._resolve_agent_implementation') as mock_resolve_agent:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.stop = AsyncMock()
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()
            mock_health.side_effect = PrerequisiteError("Health check timeout")

            # Execute and expect failure
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

            assert "Health check timeout" in str(exc_info.value)

            # Verify cleanup
            assert "test_domain" not in _active_domains
            mock_controller.stop.assert_called_once_with("test_domain")


class TestPreflightCheck:
    """Test preflight check for already-running instances (now integrated into _start_and_load_tasks)."""

    @pytest.mark.asyncio
    async def test_preflight_passes_when_no_server_running(self, mock_domain_context):
        """Test preflight check passes when no server running."""
        with patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.tasks._wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.tasks._resolve_agent_implementation') as mock_resolve_agent:

            # Setup mocks
            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.check_running_domain = AsyncMock(return_value=None)  # No server running
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()
            mock_health.return_value = None

            mock_client = MagicMock()
            mock_client.get_benchmark_info = AsyncMock(return_value=BenchmarkInfo(
                domain="test_domain",
                tasks=[],
                total_tasks=0,
                total_episodes=0,
            ))
            mock_client_class.return_value = mock_client

            # Return at least one sample (Inspect AI requires non-empty datasets)
            mock_create_dataset.return_value = [
                Sample(
                    id="dummy_task__attempt_1",
                    input="Dummy input",
                    target="Dummy target",
                    metadata={"task_id": "dummy_task"},
                )
            ]

            # Should not raise - starts domain successfully
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

            # Verify controller was started
            mock_controller.start.assert_called_once()


class TestWaitForServerHealth:
    """Test _wait_for_server_health retry logic."""

    @pytest.mark.asyncio
    async def test_health_check_succeeds_first_attempt(self):
        """Test health check succeeds on first attempt."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"domain": "test_domain"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_client_session = Mock(return_value=mock_session)

        with patch('saber.inspect_ai.tasks.aiohttp.ClientSession', mock_client_session), \
             patch('saber.inspect_ai.tasks.aiohttp.ClientTimeout'), \
             patch('saber.inspect_ai.tasks.aiohttp.ClientError', Exception):
            # Should not raise
            await _wait_for_server_health("http://localhost:8000", max_retries=3, backoff=0.1)

    @pytest.mark.asyncio
    async def test_health_check_fails_after_max_retries(self):
        """Test health check fails after exceeding max retries."""
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        # Always fail
        mock_session.get = MagicMock(side_effect=Exception("Connection refused"))

        mock_client_session = Mock(return_value=mock_session)

        with patch('saber.inspect_ai.tasks.aiohttp.ClientSession', mock_client_session), \
             patch('saber.inspect_ai.tasks.aiohttp.ClientTimeout'), \
             patch('saber.inspect_ai.tasks.aiohttp.ClientError', Exception):
            with pytest.raises(PrerequisiteError) as exc_info:
                await _wait_for_server_health("http://localhost:8000", max_retries=3, backoff=0.01)

            assert "failed after 3 attempts" in str(exc_info.value)


class TestApplyTaskFilter:
    """Test _apply_task_filter with exact and glob matching."""

    def test_exact_match(self, mock_task):
        """Test exact task ID match."""
        tasks = [mock_task]

        result = _apply_task_filter(tasks, "test_task", "test_domain")

        assert len(result) == 1
        assert result[0].task_id == "test_task"

    def test_glob_pattern_prefix(self):
        """Test glob pattern with prefix match."""
        tasks = [
            SingleEpisodeTask(
                benchmark_task_id="labyrinth_easy",
                task_id="labyrinth_easy",
                domain="test_domain",
                title="Easy",
                description="Easy task",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
            SingleEpisodeTask(
                benchmark_task_id="labyrinth_hard",
                task_id="labyrinth_hard",
                domain="test_domain",
                title="Hard",
                description="Hard task",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
            SingleEpisodeTask(
                benchmark_task_id="other_task",
                task_id="other_task",
                domain="test_domain",
                title="Other",
                description="Other task",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
        ]

        result = _apply_task_filter(tasks, "labyrinth_*", "test_domain")

        assert len(result) == 2
        assert all(t.task_id.startswith("labyrinth_") for t in result)

    def test_glob_pattern_suffix(self):
        """Test glob pattern with suffix match."""
        tasks = [
            SingleEpisodeTask(
                benchmark_task_id="task_easy",
                task_id="task_easy",
                domain="test_domain",
                title="Easy",
                description="Easy task",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
            SingleEpisodeTask(
                benchmark_task_id="task_hard",
                task_id="task_hard",
                domain="test_domain",
                title="Hard",
                description="Hard task",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
        ]

        result = _apply_task_filter(tasks, "*_hard", "test_domain")

        assert len(result) == 1
        assert result[0].task_id == "task_hard"

    def test_no_match_raises_error(self, mock_task):
        """Test that no match raises helpful error with available tasks."""
        tasks = [mock_task]

        with pytest.raises(PrerequisiteError) as exc_info:
            _apply_task_filter(tasks, "nonexistent", "test_domain")

        error_msg = str(exc_info.value)
        assert "No tasks matched filter" in error_msg
        assert "nonexistent" in error_msg
        assert "test_task" in error_msg  # Shows available task


class TestRegistryHelpers:
    """Test get_active_domain and remove_active_domain helpers."""

    def test_get_active_domain_exists(self):
        """Test getting existing active domain."""
        with _active_domains_lock:
            _active_domains["test_domain"] = {"owner": "test_owner"}

        result = get_active_domain("test_domain")

        assert result is not None
        assert result["owner"] == "test_owner"

    def test_get_active_domain_not_exists(self):
        """Test getting non-existent active domain."""
        result = get_active_domain("nonexistent")
        assert result is None

    def test_remove_active_domain(self):
        """Test removing active domain."""
        with _active_domains_lock:
            _active_domains["test_domain"] = {"owner": "test_owner"}

        remove_active_domain("test_domain")

        assert "test_domain" not in _active_domains

    def test_remove_active_domain_not_exists(self):
        """Test removing non-existent domain (should not raise)."""
        remove_active_domain("nonexistent")  # Should not raise


class TestRoleBasedConfiguration:
    """Test role-based configuration and model selection."""

    @pytest.mark.asyncio
    async def test_all_roles_have_models_check(self):
        """Test the _all_roles_have_models validation function."""
        from saber.inspect_ai.tasks import _all_roles_have_models
        from saber.client.models import RoleBasedConfig, RoleAgentConfig

        # Case 1: All roles have models
        config1 = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4"),
                "blue": RoleAgentConfig(agent="react", model="gpt-3.5"),
            }
        )
        assert _all_roles_have_models(config1) is True

        # Case 2: Defaults has model (all roles inherit)
        config2 = RoleBasedConfig(
            defaults=RoleAgentConfig(agent="react", model="gpt-4"),
            roles={
                "red": RoleAgentConfig(agent="react"),
                "blue": RoleAgentConfig(agent="custom"),
            }
        )
        assert _all_roles_have_models(config2) is True

        # Case 3: Missing model in one role, no defaults
        config3 = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4"),
                "blue": RoleAgentConfig(agent="react"),  # No model
            }
        )
        assert _all_roles_have_models(config3) is False

        # Case 4: Empty config with defaults should work if defaults has model
        config4 = RoleBasedConfig(
            defaults=RoleAgentConfig(agent="react", model="gpt-4"),
            roles={"red": RoleAgentConfig(agent="react")}
        )
        assert _all_roles_have_models(config4) is True

    @pytest.mark.asyncio
    async def test_task_with_role_config_file(self, mock_domain_context, mock_benchmark_info, tmp_path):
        """Test task creation with role configuration from file."""
        # Create a temporary role config file
        roles_file = tmp_path / "roles.yaml"
        roles_file.write_text("""
defaults:
  agent: react
  model: gpt-4

roles:
  red:
    agent: react
    model: gpt-4o
  blue:
    agent: react
    model: gpt-3.5-turbo
""")

        with patch('saber.inspect_ai.tasks.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.tasks._wait_for_server_health') as mock_health, \
             patch('saber.inspect_ai.tasks.SABERRestClient') as mock_client_class, \
             patch('saber.inspect_ai.tasks.create_saber_dataset') as mock_create_dataset, \
             patch('saber.inspect_ai.tasks._resolve_agent_implementation') as mock_resolve_agent:

            mock_controller = AsyncMock()
            mock_controller.start = AsyncMock(return_value=mock_domain_context)
            mock_controller.check_running_domain = AsyncMock(return_value=None)
            mock_controller_class.return_value = mock_controller

            mock_resolve_agent.return_value = MagicMock()
            mock_health.return_value = None

            mock_client = MagicMock()
            mock_client.get_benchmark_info = AsyncMock(return_value=mock_benchmark_info)
            mock_client_class.return_value = mock_client

            mock_samples = [
                Sample(
                    id="test_task__attempt_1",
                    input="Test input",
                    target="Test target",
                    metadata={"task_id": "test_task"},
                )
            ]
            mock_create_dataset.return_value = mock_samples

            # Execute with roles_file parameter
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
                role_config=None,  # Will be loaded from file in task_callable
            )

            assert task is not None
            assert len(task.dataset) == 1

    # Note: test_task_with_inline_role_config removed as it requires actual API keys
    # to initialize OpenAI models. Role-based configuration is tested indirectly
    # through other tests that mock the model resolution.


class TestAgentResolution:
    """Test agent resolution and loading."""

    @pytest.mark.asyncio
    async def test_resolve_domain_local_agent(self, tmp_path):
        """Test resolving agent from domain's client folder."""
        from saber.inspect_ai.tasks import _load_domain_agent

        # Create a domain client directory with a custom agent
        domain_dir = tmp_path / "test_domain" / "client"
        domain_dir.mkdir(parents=True)

        agent_file = domain_dir / "custom_agent.py"
        agent_file.write_text("""
def create_agent():
    '''Custom domain agent factory.'''
    return lambda **kwargs: lambda state: state
""")

        # Test loading domain-local agent
        agent_factory = _load_domain_agent("test_domain", tmp_path, "custom_agent")

        assert agent_factory is not None
        assert callable(agent_factory)

    @pytest.mark.asyncio
    async def test_resolve_domain_local_agent_missing_create_agent(self, tmp_path):
        """Test domain agent file without create_agent function."""
        from saber.inspect_ai.tasks import _load_domain_agent

        # Create a domain client directory with an invalid agent
        domain_dir = tmp_path / "test_domain" / "client"
        domain_dir.mkdir(parents=True)

        agent_file = domain_dir / "invalid_agent.py"
        agent_file.write_text("""
# Missing create_agent function
def some_other_function():
    pass
""")

        # Should return None when create_agent is missing
        agent_factory = _load_domain_agent("test_domain", tmp_path, "invalid_agent")

        assert agent_factory is None

    @pytest.mark.asyncio
    async def test_resolve_agent_not_found(self):
        """Test agent resolution when agent doesn't exist."""
        from saber.inspect_ai.tasks import _resolve_agent_implementation
        from saber.inspect_ai.agents import AgentNotFoundError

        with pytest.raises(AgentNotFoundError) as exc_info:
            _resolve_agent_implementation("nonexistent_domain", Path("/tmp"), "nonexistent_agent")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "nonexistent_agent" in error_msg


class TestPreflightExecution:
    """Test preflight check execution."""

    @pytest.mark.asyncio
    async def test_run_preflight_check_success(self):
        """Test successful preflight check execution."""
        from saber.inspect_ai.tasks import _run_preflight_check

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.wait = AsyncMock(return_value=0)

        with patch('saber.inspect_ai.tasks.asyncio.create_subprocess_exec', return_value=mock_process) as mock_create:
            # Should not raise
            await _run_preflight_check("test_domain", Path("/test/domains"))

            # Verify command was called
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_preflight_check_failure(self):
        """Test preflight check failure handling."""
        from saber.inspect_ai.tasks import _run_preflight_check

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.wait = AsyncMock(return_value=1)

        with patch('saber.inspect_ai.tasks.asyncio.create_subprocess_exec', return_value=mock_process):
            with pytest.raises(PrerequisiteError) as exc_info:
                await _run_preflight_check("test_domain", Path("/test/domains"))

            assert "preflight check failed" in str(exc_info.value).lower()


class TestTaskFilterAdvanced:
    """Test advanced task filtering scenarios."""

    def test_apply_task_filter_list_input(self):
        """Test task filter with list input (from Inspect AI CLI parsing)."""
        from saber.inspect_ai.tasks import _apply_task_filter

        tasks = [
            SingleEpisodeTask(
                benchmark_task_id="task_a",
                task_id="task_a",
                domain="test",
                title="A",
                description="Task A",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
            SingleEpisodeTask(
                benchmark_task_id="task_b",
                task_id="task_b",
                domain="test",
                title="B",
                description="Task B",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
        ]

        # Test with list input (Inspect AI may parse "a,b" as ["a", "b"])
        result = _apply_task_filter(tasks, ["task_a", "task_b"], "test")

        assert len(result) == 2

    def test_apply_task_filter_empty_pattern(self):
        """Test task filter with empty patterns in list."""
        from saber.inspect_ai.tasks import _apply_task_filter

        tasks = [
            SingleEpisodeTask(
                benchmark_task_id="task_a",
                task_id="task_a",
                domain="test",
                title="A",
                description="Task A",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
        ]

        # Test with empty patterns (should skip them)
        result = _apply_task_filter(tasks, "task_a,,", "test")

        assert len(result) == 1


class TestPortalManagement:
    """Test BlockingPortal creation and management."""

    def test_get_or_create_portal_creates_once(self):
        """Test that portal is created once and reused."""
        from saber.inspect_ai.tasks import _get_or_create_portal, _portal

        # Reset portal state
        import saber.inspect_ai.tasks as tasks_module
        tasks_module._portal = None
        tasks_module._portal_cm = None

        with patch('saber.inspect_ai.tasks.anyio.from_thread.start_blocking_portal') as mock_portal:
            mock_cm = MagicMock()
            mock_portal_instance = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_portal_instance)
            mock_portal.return_value = mock_cm

            # First call creates portal
            portal1 = _get_or_create_portal()
            assert portal1 is not None
            mock_portal.assert_called_once()

            # Second call reuses portal
            portal2 = _get_or_create_portal()
            assert portal2 is portal1
            # Still only called once
            mock_portal.assert_called_once()
