"""
Unit tests for DockerSandboxEnvironment.

Tests the multi-container Docker sandbox environment management.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from docker.models.containers import Container

from saber.server.execution.sandbox.docker_sandbox_environment import (
    DockerSandboxEnvironment,
    CommandResult
)
from saber.server.execution.sandbox.environment_spec import (
    EnvironmentSpec,
    ServiceSpec,
    NetworkSpec,
    HealthCheck
)
from saber.server.execution.exceptions import (
    ContainerCreationError,
    SandboxExecutionError,
    InvalidEnvironmentSpecException,
    ContainerCommunicationError
)


class TestDockerSandboxEnvironment:
    """Test DockerSandboxEnvironment functionality."""

    @pytest.fixture
    def sample_environment_spec(self):
        """Create a sample environment specification."""
        network = NetworkSpec(
            name="test_network",
            driver="bridge",
            internal=True
        )

        webapp_service = ServiceSpec(
            name="webapp",
            container="webapp_container",
            image="nginx:latest",
            ports=["80"],
            health_check=HealthCheck(test=["CMD", "curl", "-f", "http://localhost/"])
        )

        return EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={
                "image": "ubuntu:latest",
                "working_dir": "/workspace",
                "user": "user:user"
            },
            target_services=[webapp_service]
        )

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_init_success(self, mock_docker, sample_environment_spec):
        """Test successful initialization."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)

        assert env.session_id == "test_session"
        assert env.environment_spec == sample_environment_spec
        assert env.active_services == {}
        assert env.compose_project_name == "saber-session-test_session"
        assert env.docker_client == mock_client

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_init_docker_error(self, mock_docker, sample_environment_spec):
        """Test initialization with Docker client error."""
        mock_docker.side_effect = Exception("Docker not available")

        with pytest.raises(ContainerCreationError, match="Failed to initialize Docker client"):
            DockerSandboxEnvironment("test_session", sample_environment_spec)

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_init_validation_error(self, mock_docker):
        """Test initialization with invalid environment spec."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Create invalid spec (missing execution service)
        network = NetworkSpec(name="test_network")
        invalid_spec = EnvironmentSpec(
            network=network,
            execution_service="",  # Invalid empty string
            execution_config={}
        )

        with pytest.raises(InvalidEnvironmentSpecException):
            DockerSandboxEnvironment("test_session", invalid_spec)

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    @patch('saber.server.execution.sandbox.docker_sandbox_environment.tempfile.NamedTemporaryFile')
    @patch('saber.server.execution.sandbox.docker_sandbox_environment.subprocess.run')
    def test_start_success(self, mock_subprocess, mock_temp_file, mock_docker, sample_environment_spec):
        """Test successful environment start."""
        # Setup mocks
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_file = Mock()
        mock_file.name = "/tmp/test_compose.yml"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        mock_subprocess.return_value = Mock(stdout="Services started", stderr="")

        # Mock container tracking
        mock_container1 = Mock(spec=Container)
        mock_container1.labels = {"com.docker.compose.service": "execution"}
        mock_container2 = Mock(spec=Container)
        mock_container2.labels = {"com.docker.compose.service": "webapp"}

        mock_client.containers.list.return_value = [mock_container1, mock_container2]

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)

        # Mock service health checks
        with patch.object(env, 'is_service_healthy', return_value=True):
            env.start()

        # Verify compose file was written
        mock_temp_file.assert_called_once()

        # Verify docker compose up was called (modern docker syntax)
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "docker" in args and "compose" in args
        assert "up" in args
        assert "-d" in args

        # Verify containers were tracked
        assert len(env.active_services) == 2
        assert "execution" in env.active_services
        assert "webapp" in env.active_services

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_start_compose_error(self, mock_docker, sample_environment_spec):
        """Test start with docker-compose error."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)

        with patch('saber.server.execution.sandbox.docker_sandbox_environment.subprocess.run') as mock_subprocess:
            mock_subprocess.side_effect = Exception("Compose failed")

            with pytest.raises(ContainerCreationError, match="Failed to start sandbox environment"):
                env.start()

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_execute_command_success(self, mock_docker, sample_environment_spec):
        """Test successful command execution."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        # Setup execution container
        mock_container = Mock(spec=Container)
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.output = (b"Hello World", b"")
        mock_container.exec_run.return_value = mock_result

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["execution"] = mock_container

        result = env.execute_command(["echo", "Hello World"])

        assert isinstance(result, CommandResult)
        assert result.exit_code == 0
        assert result.stdout == "Hello World"
        assert result.stderr == ""
        assert result.execution_time > 0

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_execute_command_no_environment(self, mock_docker, sample_environment_spec):
        """Test command execution without started environment."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)

        with pytest.raises(SandboxExecutionError, match="Environment not started"):
            env.execute_command(["echo", "test"])

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_execute_command_no_execution_container(self, mock_docker, sample_environment_spec):
        """Test command execution without execution container."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["webapp"] = Mock()  # Only webapp, no execution

        with pytest.raises(SandboxExecutionError, match="Execution container not available"):
            env.execute_command(["echo", "test"])

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_execute_command_container_error(self, mock_docker, sample_environment_spec):
        """Test command execution with container error."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.exec_run.side_effect = Exception("Container error")

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["execution"] = mock_container

        with pytest.raises(SandboxExecutionError, match="Command execution failed"):
            env.execute_command(["echo", "test"])

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_copy_to_container_success(self, mock_docker, sample_environment_spec):
        """Test successful file copy to container."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.put_archive.return_value = True

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["execution"] = mock_container

        with patch('saber.server.execution.sandbox.docker_sandbox_environment.tarfile.open'), \
             patch('saber.server.execution.sandbox.docker_sandbox_environment.io.BytesIO'), \
             patch('builtins.open', mock_open(read_data=b"test file content")):
            env.copy_to_container("/local/file", "/container/file")

        mock_container.put_archive.assert_called_once()

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_copy_to_container_no_execution_container(self, mock_docker, sample_environment_spec):
        """Test file copy without execution container."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)

        with pytest.raises(SandboxExecutionError, match="Execution container not available"):
            env.copy_to_container("/local/file", "/container/file")

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_copy_from_container_success(self, mock_docker, sample_environment_spec):
        """Test successful file copy from container."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.get_archive.return_value = (iter([b"tar data"]), {})

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["execution"] = mock_container

        with patch('saber.server.execution.sandbox.docker_sandbox_environment.tarfile.open'), \
             patch('saber.server.execution.sandbox.docker_sandbox_environment.io.BytesIO'), \
             patch('builtins.open', mock_open()) as mock_file:
            env.copy_from_container("/container/file", "/local/file")

        mock_container.get_archive.assert_called_once_with("/container/file")

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_get_execution_container(self, mock_docker, sample_environment_spec):
        """Test getting execution container."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["execution"] = mock_container

        result = env.get_execution_container()
        assert result == mock_container

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_get_service_container(self, mock_docker, sample_environment_spec):
        """Test getting specific service container."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["webapp"] = mock_container

        result = env.get_service_container("webapp")
        assert result == mock_container

        result = env.get_service_container("nonexistent")
        assert result is None

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_is_service_healthy_running_with_health(self, mock_docker, sample_environment_spec):
        """Test health check for running service with health status."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.status = "running"
        mock_container.attrs = {
            "State": {
                "Health": {
                    "Status": "healthy"
                }
            }
        }

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["webapp"] = mock_container

        assert env.is_service_healthy("webapp") is True

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_is_service_healthy_running_no_health(self, mock_docker, sample_environment_spec):
        """Test health check for running service without health check."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.status = "running"
        mock_container.attrs = {"State": {}}

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["webapp"] = mock_container

        assert env.is_service_healthy("webapp") is True

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_is_service_healthy_not_running(self, mock_docker, sample_environment_spec):
        """Test health check for non-running service."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.status = "exited"

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["webapp"] = mock_container

        assert env.is_service_healthy("webapp") is False

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_is_service_healthy_unhealthy(self, mock_docker, sample_environment_spec):
        """Test health check for unhealthy service."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.status = "running"
        mock_container.attrs = {
            "State": {
                "Health": {
                    "Status": "unhealthy"
                }
            }
        }

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["webapp"] = mock_container

        assert env.is_service_healthy("webapp") is False

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_is_service_healthy_service_not_found(self, mock_docker, sample_environment_spec):
        """Test health check for non-existent service."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)

        assert env.is_service_healthy("nonexistent") is False

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_get_service_logs(self, mock_docker, sample_environment_spec):
        """Test getting service logs."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.logs.return_value = b"Log line 1\nLog line 2\n"

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["webapp"] = mock_container

        logs = env.get_service_logs("webapp")
        assert logs == "Log line 1\nLog line 2\n"
        mock_container.logs.assert_called_once_with(tail=100, timestamps=True)

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_get_service_logs_not_found(self, mock_docker, sample_environment_spec):
        """Test getting logs for non-existent service."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)

        logs = env.get_service_logs("nonexistent")
        assert "Service nonexistent not found" in logs

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_list_active_services(self, mock_docker, sample_environment_spec):
        """Test listing active services."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["execution"] = Mock()
        env.active_services["webapp"] = Mock()

        services = env.list_active_services()
        assert set(services) == {"execution", "webapp"}

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    def test_get_service_info(self, mock_docker, sample_environment_spec):
        """Test getting service information."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        mock_container = Mock(spec=Container)
        mock_container.id = "container123"
        mock_container.status = "running"
        mock_container.image.tags = ["nginx:latest"]
        mock_container.ports = {"80/tcp": [{"HostPort": "8080"}]}

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.active_services["webapp"] = mock_container

        with patch.object(env, 'is_service_healthy', return_value=True):
            info = env.get_service_info()

        assert "webapp" in info
        webapp_info = info["webapp"]
        assert webapp_info["id"] == "container123"
        assert webapp_info["status"] == "running"
        assert webapp_info["image"] == "nginx:latest"
        assert webapp_info["healthy"] is True

    @patch('saber.server.execution.sandbox.docker_sandbox_environment.docker.from_env')
    @patch('saber.server.execution.sandbox.docker_sandbox_environment.subprocess.run')
    def test_stop_success(self, mock_subprocess, mock_docker, sample_environment_spec):
        """Test successful environment stop."""
        mock_client = Mock()
        mock_docker.return_value = mock_client

        env = DockerSandboxEnvironment("test_session", sample_environment_spec)
        env.compose_file_path = "/tmp/test_compose.yml"
        env.active_services["webapp"] = Mock()

        with patch('saber.server.execution.sandbox.docker_sandbox_environment.os.path.exists', return_value=True), \
             patch('saber.server.execution.sandbox.docker_sandbox_environment.os.unlink'):
            env.stop()

        # Verify docker compose down was called (modern docker syntax)
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "docker" in args and "compose" in args
        assert "down" in args

        # Verify cleanup
        assert len(env.active_services) == 0
