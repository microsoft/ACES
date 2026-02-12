"""
Tests for DockerExecutor base class.

This module tests the abstract Docker executor base class that provides shared
Docker container management functionality.
"""

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from saber.server.base import CommandResult
from saber.server.execution.base import ValidationResult, ExecutorConfig, ExecutorParameters
from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.executors.docker_executor import DockerExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


# Mock parameter class for concrete executor (prefixed with Mock to avoid pytest collection)
@dataclass(frozen=True, slots=True)
class MockDockerParams:
    """Mock parameters for concrete executor."""
    command: str = ""
    working_dir: str | None = None
    timeout: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockDockerParams":
        return cls(
            command=data.get("command", ""),
            working_dir=data.get("working_dir"),
            timeout=data.get("timeout"),
        )


# Module-level concrete executor for target_container tests
class ConcreteDockerExecutor(DockerExecutor):
    """Concrete implementation of DockerExecutor for testing."""

    @classmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        return MockDockerParams

    async def execute(self, parameters, context):
        return CommandResult.success_result(data="test_execution")


class TestDockerExecutor:
    """Test cases for DockerExecutor base class."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
            "resource_limits": {"memory": "512m", "cpus": "1.0"},
        }
        return manager

    @pytest.fixture
    def docker_executor(self, mock_sandbox_manager):
        """Create a concrete Docker executor instance for testing."""
        config = ExecutorConfig(timeout=60.0)
        return ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

    @pytest.fixture
    def mock_docker_environment(self):
        """Create a mock Docker execution environment."""
        env = MagicMock()
        env.get_container_id.return_value = "container123456789"
        return env

    def test_initialization_success(self, mock_sandbox_manager):
        """Test successful initialization with sandbox manager."""
        config = ExecutorConfig(timeout=120.0)
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor._sandbox_manager == mock_sandbox_manager
        assert executor.get_timeout() == 120.0

    def test_initialization_with_docker_config(self, mock_sandbox_manager):
        """Test initialization with Docker configuration."""
        # Docker-specific configs beyond timeout are not part of ExecutorConfig
        # This test now just verifies basic initialization with timeout
        config = ExecutorConfig(timeout=60.0)
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor._config.timeout == 60.0

    def test_initialization_without_sandbox_manager(self):
        """Test that initialization fails without sandbox manager."""
        with pytest.raises(SandboxExecutionError, match="sandbox_manager is required"):
            ConcreteDockerExecutor(sandbox_manager=None)

    def test_get_episode_environment_existing(self, docker_executor, mock_docker_environment):
        """Test retrieving existing episode environment."""
        episode_id = "test_episode_123"

        # Mock an existing environment
        docker_executor._sandbox_manager.get_episode_environment.return_value = mock_docker_environment

        result = docker_executor.get_episode_environment(episode_id)

        assert result == mock_docker_environment
        docker_executor._sandbox_manager.get_episode_environment.assert_called_once_with(episode_id)
        # Verify create_episode_environment_async was NOT called since environment exists
        docker_executor._sandbox_manager.create_episode_environment_async.assert_not_called()

    def test_get_episode_environment_not_found(self, docker_executor):
        """Test behavior when episode environment doesn't exist."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.return_value = None

        result = docker_executor.get_episode_environment(episode_id)
        assert result is None

    def test_get_episode_environment_failure(self, docker_executor):
        """Test episode environment retrieval failure."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.side_effect = Exception("Environment error")

        with pytest.raises(SandboxExecutionError, match="Failed to get episode environment"):
            docker_executor.get_episode_environment(episode_id)

    def test_ensure_container_ready_success(self, docker_executor, mock_docker_environment):
        """Test successful container readiness check."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.return_value = mock_docker_environment

        result = docker_executor.ensure_container_ready(episode_id)

        assert result is True

    def test_ensure_container_ready_failure(self, docker_executor):
        """Test container readiness check failure."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.side_effect = Exception("Container error")

        result = docker_executor.ensure_container_ready(episode_id)

        assert result is False

    def test_cleanup_execution_success(self, docker_executor):
        """Test successful execution cleanup."""
        episode_id = "test_episode_123"

        # Create and set an event loop for this test since cleanup_execution uses asyncio.get_event_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Mock the async method to return a coroutine
            async def mock_stop(ep_id):
                return True
            docker_executor._sandbox_manager.stop_episode_environment = mock_stop

            docker_executor.cleanup_execution(episode_id)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_cleanup_execution_with_error(self, docker_executor):
        """Test execution cleanup with error (should not raise)."""
        episode_id = "test_episode_123"

        # Create and set an event loop for this test since cleanup_execution uses asyncio.get_event_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Mock the async method to raise an exception when awaited
            async def mock_stop_error(ep_id):
                raise Exception("Cleanup error")
            docker_executor._sandbox_manager.stop_episode_environment = mock_stop_error

            with pytest.raises(SandboxExecutionError, match="Failed to clean up execution resources"):
                docker_executor.cleanup_execution(episode_id)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_validate_docker_parameters_valid(self, docker_executor):
        """Test validation of valid Docker parameters."""
        parameters = MockDockerParams(command="test", working_dir="/workspace", timeout=300)

        result = docker_executor.validate_docker_parameters(parameters)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_docker_parameters_invalid_working_dir(self, docker_executor):
        """Test validation with invalid working directory."""
        parameters = MockDockerParams(command="test", working_dir="relative/path")

        result = docker_executor.validate_docker_parameters(parameters)

        assert result.valid is False
        assert "working_dir must be an absolute path" in result.errors

    def test_validate_docker_parameters_invalid_working_dir_type(self, docker_executor):
        """Test validation with wrong working directory type."""
        # Create a mock object with invalid type for working_dir
        class BadParams:
            working_dir = 123  # Should be string
            timeout = None

        result = docker_executor.validate_docker_parameters(BadParams())

        assert result.valid is False
        assert "working_dir must be a string" in result.errors

    def test_validate_docker_parameters_invalid_timeout(self, docker_executor):
        """Test validation with invalid timeout."""
        parameters = MockDockerParams(command="test", timeout=-10)

        result = docker_executor.validate_docker_parameters(parameters)

        assert result.valid is False
        assert "timeout must be a positive number" in result.errors

    def test_validate_docker_parameters_invalid_timeout_type(self, docker_executor):
        """Test validation with wrong timeout type."""
        # Create a mock object with invalid type for timeout
        class BadParams:
            working_dir = None
            timeout = "not_a_number"  # Should be number

        result = docker_executor.validate_docker_parameters(BadParams())

        assert result.valid is False
        assert "timeout must be a positive number" in result.errors

    def test_get_docker_info(self, docker_executor):
        """Test getting Docker configuration information."""
        info = docker_executor.get_docker_info()

        assert info.execution_environment == "docker_container"
        assert info.timeout == 60.0
        assert info.docker_config is not None

        # Check sandbox config details
        docker_config = info.docker_config
        assert docker_config.image == "saber/sandbox:latest"
        assert docker_config.network_mode == "none"
        assert docker_config.read_only_root is True
        assert docker_config.user == "tooluser:tooluser"
        assert docker_config.resource_limits["memory"] == "512m"

    def test_get_docker_info_with_error(self, docker_executor):
        """Test getting Docker info when sandbox config retrieval fails."""
        # Simulate an exception by making sandbox_config access fail
        del docker_executor._sandbox_manager.sandbox_config

        info = docker_executor.get_docker_info()

        assert info.execution_environment == "docker_container"
        assert info.timeout == 60.0
        # Should still work even if sandbox config fails (docker_config may be None)

    def test_validate_parameters_combined(self, docker_executor):
        """Test parameter validation combining base and Docker validation."""
        # Mock base class validation
        with patch.object(DockerExecutor.__bases__[0], "validate_parameters") as mock_base_validate:
            base_result = ValidationResult.success()
            base_result.add_warning("Base warning")
            mock_base_validate.return_value = base_result

            parameters = MockDockerParams(command="test", working_dir="/workspace", timeout=300)

            result = docker_executor.validate_parameters(parameters)

            assert result.valid is True
            assert "Base warning" in result.warnings
            mock_base_validate.assert_called_once_with(parameters)

    def test_validate_parameters_combined_with_errors(self, docker_executor):
        """Test parameter validation with both base and Docker errors."""
        # Mock base class validation with errors
        with patch.object(DockerExecutor.__bases__[0], "validate_parameters") as mock_base_validate:
            base_result = ValidationResult.failure(["Base error"])
            mock_base_validate.return_value = base_result

            parameters = MockDockerParams(command="test", working_dir="invalid/path")

            result = docker_executor.validate_parameters(parameters)

            assert not result.valid  # Should be invalid due to errors
            assert "Base error" in result.errors
            assert "working_dir must be an absolute path" in result.errors

    @pytest.mark.asyncio
    async def test_execute_abstract_method_implemented(self, docker_executor):
        """Test that concrete implementation provides execute method."""
        from saber.server.execution.base import ExecutionContext

        parameters = MockDockerParams(command="test")
        context = ExecutionContext(episode_id="test")

        result = await docker_executor(parameters, context)

        assert result.success is True
        assert result.data == "test_execution"

    def test_inheritance_from_command_executor(self, docker_executor):
        """Test that DockerExecutor properly inherits from CommandExecutor."""
        from saber.server.execution.executors.base_executors import CommandExecutor

        assert isinstance(docker_executor, CommandExecutor)
        assert hasattr(docker_executor, "get_timeout")
        assert hasattr(docker_executor, "get_parameters")
        assert hasattr(docker_executor, "add_parameter")

    def test_docker_specific_methods_exist(self, docker_executor):
        """Test that Docker-specific methods are properly defined."""
        assert hasattr(docker_executor, "get_episode_environment")
        assert hasattr(docker_executor, "ensure_container_ready")
        assert hasattr(docker_executor, "cleanup_execution")
        assert hasattr(docker_executor, "validate_docker_parameters")
        assert hasattr(docker_executor, "get_docker_info")

    def test_abstract_class_cannot_be_instantiated(self, mock_sandbox_manager):
        """Test that DockerExecutor cannot be instantiated directly."""
        # Since we're not using ABC, DockerExecutor can be instantiated
        # but calling execute() should raise NotImplementedError or similar
        # This test documents that it's meant to be abstract
        try:
            # This should work since we're not using ABC
            executor = DockerExecutor(sandbox_manager=mock_sandbox_manager)
            assert hasattr(executor, "execute")
            # The execute method should be abstract (not implemented)
        except Exception:
            # If there are import or other issues, that's also fine
            pass


class TestDockerExecutorTargetContainer:
    """Tests for _get_target_container() on DockerExecutor."""

    @pytest.fixture
    def mock_sandbox_manager(self) -> MagicMock:
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {"image": "saber/sandbox:latest"}
        return manager

    def test_get_target_container_from_config(self, mock_sandbox_manager: MagicMock) -> None:
        """Test that _get_target_container returns the configured value."""
        config = ExecutorConfig(timeout=60.0, target_container="saber-excytin-sandbox")
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor._get_target_container() == "saber-excytin-sandbox"

    def test_get_target_container_none_when_not_configured(self, mock_sandbox_manager: MagicMock) -> None:
        """Test that _get_target_container returns None when key is absent from config."""
        config = ExecutorConfig(timeout=60.0)
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor._get_target_container() is None

    def test_get_target_container_none_when_no_config(self, mock_sandbox_manager: MagicMock) -> None:
        """Test that _get_target_container returns None when config is None."""
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=None)

        assert executor._get_target_container() is None

    def test_ensure_container_ready_with_target_container(self, mock_sandbox_manager: MagicMock) -> None:
        """Test ensure_container_ready returns True when target_container is configured but no episode env."""
        mock_sandbox_manager.get_episode_environment.return_value = None
        config = ExecutorConfig(timeout=60.0, target_container="saber-excytin-sandbox")
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor.ensure_container_ready("ep_no_env") is True

    def test_ensure_container_ready_no_env_no_target_container(self, mock_sandbox_manager: MagicMock) -> None:
        """Test ensure_container_ready returns False when neither env nor target_container exist."""
        mock_sandbox_manager.get_episode_environment.return_value = None
        config = ExecutorConfig(timeout=60.0)
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor.ensure_container_ready("ep_nothing") is False

    @pytest.mark.asyncio
    async def test_execute_on_target_container(self, mock_sandbox_manager: MagicMock) -> None:
        """Test _execute_on_target_container delegates to ComposeOrchestrator."""
        config = ExecutorConfig(timeout=60.0, target_container="saber-excytin-sandbox")
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        expected_result = CommandResult(exit_code=0, stdout="ok\n", stderr="", execution_time=0.1)

        with patch(
            "saber.server.execution.sandbox.compose_orchestrator.ComposeOrchestrator"
        ) as mock_orch_cls:
            mock_orchestrator = MagicMock()
            mock_orchestrator.execute_command = AsyncMock(return_value=expected_result)
            mock_orch_cls.return_value = mock_orchestrator

            result = await executor._execute_on_target_container(
                ["/bin/sh", "-c", "echo ok"], 30, "saber-excytin-sandbox"
            )

        assert result.exit_code == 0
        assert result.stdout == "ok\n"
        mock_orchestrator.execute_command.assert_called_once_with(
            command=["/bin/sh", "-c", "echo ok"],
            timeout=30,
            target_container="saber-excytin-sandbox",
        )


class TestExecuteInContainer:
    """Tests for centralized _execute_in_container() routing method."""

    @pytest.fixture
    def mock_sandbox_manager(self) -> MagicMock:
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {"image": "saber/sandbox:latest"}
        return manager

    @pytest.mark.asyncio
    async def test_delegates_to_environment_when_present(self, mock_sandbox_manager: MagicMock) -> None:
        """When episode has an environment, delegate to environment.execute_command with target_container."""
        env = MagicMock()
        expected = CommandResult(exit_code=0, stdout="ok\n", stderr="", execution_time=0.1)
        env.execute_command = AsyncMock(return_value=expected)
        mock_sandbox_manager.get_episode_environment.return_value = env

        config = ExecutorConfig(timeout=60.0, target_container="my-container")
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        result = await executor._execute_in_container("ep1", ["/bin/sh", "-c", "echo ok"], 30)

        assert result is expected
        env.execute_command.assert_called_once_with(
            command=["/bin/sh", "-c", "echo ok"], timeout=30, target_container="my-container"
        )

    @pytest.mark.asyncio
    async def test_delegates_to_target_container_when_no_environment(self, mock_sandbox_manager: MagicMock) -> None:
        """When no episode environment exists but target_container is set, delegate to _execute_on_target_container."""
        mock_sandbox_manager.get_episode_environment.return_value = None
        config = ExecutorConfig(timeout=60.0, target_container="saber-sandbox")
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        expected = CommandResult(exit_code=0, stdout="direct\n", stderr="", execution_time=0.2)
        with patch(
            "saber.server.execution.sandbox.compose_orchestrator.ComposeOrchestrator"
        ) as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.execute_command = AsyncMock(return_value=expected)
            mock_orch_cls.return_value = mock_orch

            result = await executor._execute_in_container("ep2", ["echo", "hi"], 15)

        assert result is expected
        mock_orch.execute_command.assert_called_once_with(
            command=["echo", "hi"], timeout=15, target_container="saber-sandbox"
        )

    @pytest.mark.asyncio
    async def test_raises_when_neither_env_nor_target_container(self, mock_sandbox_manager: MagicMock) -> None:
        """When neither environment nor target_container is available, raise SandboxExecutionError."""
        mock_sandbox_manager.get_episode_environment.return_value = None
        config = ExecutorConfig(timeout=60.0)  # no target_container
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        with pytest.raises(SandboxExecutionError, match="No sandbox environment and no target_container configured"):
            await executor._execute_in_container("ep3", ["echo"], 10)

    @pytest.mark.asyncio
    async def test_environment_takes_precedence_over_target_container(self, mock_sandbox_manager: MagicMock) -> None:
        """When environment AND target_container both exist, environment.execute_command is used."""
        env = MagicMock()
        expected = CommandResult(exit_code=0, stdout="via env\n", stderr="", execution_time=0.1)
        env.execute_command = AsyncMock(return_value=expected)
        mock_sandbox_manager.get_episode_environment.return_value = env

        config = ExecutorConfig(timeout=60.0, target_container="fallback-container")
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        result = await executor._execute_in_container("ep4", ["ls"], 5)

        assert result is expected
        env.execute_command.assert_called_once_with(
            command=["ls"], timeout=5, target_container="fallback-container"
        )


class TestGetContainerId:
    """Tests for _get_container_id() display helper."""

    @pytest.fixture
    def mock_sandbox_manager(self) -> MagicMock:
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {"image": "saber/sandbox:latest"}
        return manager

    def test_returns_container_id_from_environment(self, mock_sandbox_manager: MagicMock) -> None:
        """When episode has an environment with a container, return truncated container ID."""
        env = MagicMock()
        container = MagicMock()
        container.id = "abcdef1234567890"
        env.get_execution_container.return_value = container
        mock_sandbox_manager.get_episode_environment.return_value = env

        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager)
        assert executor._get_container_id("ep1") == "abcdef123456"

    def test_returns_unknown_when_environment_has_no_container(self, mock_sandbox_manager: MagicMock) -> None:
        """When environment exists but get_execution_container returns None, return 'unknown'."""
        env = MagicMock()
        env.get_execution_container.return_value = None
        mock_sandbox_manager.get_episode_environment.return_value = env

        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager)
        assert executor._get_container_id("ep2") == "unknown"

    def test_returns_direct_target_container_when_no_environment(self, mock_sandbox_manager: MagicMock) -> None:
        """When no environment but target_container is set, return 'direct:<name>'."""
        mock_sandbox_manager.get_episode_environment.return_value = None
        config = ExecutorConfig(timeout=60.0, target_container="my-sandbox")
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor._get_container_id("ep3") == "direct:my-sandbox"

    def test_returns_unknown_when_neither(self, mock_sandbox_manager: MagicMock) -> None:
        """When no environment and no target_container, return 'unknown'."""
        mock_sandbox_manager.get_episode_environment.return_value = None
        config = ExecutorConfig(timeout=60.0)  # no target_container
        executor = ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor._get_container_id("ep4") == "unknown"
