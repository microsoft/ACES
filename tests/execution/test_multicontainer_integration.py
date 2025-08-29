"""
Integration test for multi-container orchestration with real Docker containers.

This test verifies the complete multi-container workflow using actual Docker containers.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

import docker
from saber.server.execution.exceptions import ContainerCreationError, SandboxExecutionError
from saber.server.execution.sandbox.docker_sandbox_environment import DockerSandboxEnvironment
from saber.server.execution.sandbox.environment_spec import EnvironmentSpec, HealthCheck, NetworkSpec, ServiceSpec


class TestMultiContainerIntegration:
    """Integration tests for multi-container orchestration."""

    @pytest.fixture(scope="class")
    def docker_client(self):
        """Get Docker client and verify Docker is available."""
        try:
            client = docker.from_env()
            client.ping()
            return client
        except Exception as e:
            pytest.skip(f"Docker not available: {e}")

    @pytest.fixture
    def simple_environment_spec(self):
        """Create a simple environment spec for testing."""
        network = NetworkSpec(
            name="saber_test_network", driver="bridge", internal=False  # Allow external access for testing
        )

        # Simple nginx service for testing
        nginx_service = ServiceSpec(
            name="nginx",
            container="nginx_container",
            image="nginx:alpine",
            ports=["80"],
            health_check=HealthCheck(
                test=["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/"],
                interval="10s",
                timeout="5s",
                retries=3,
                start_period="10s",
            ),
        )

        return EnvironmentSpec(
            networks=[network],
            execution_service="ubuntu_executor",
            execution_config={
                "image": "ubuntu:latest",
                "working_dir": "/workspace",
                "command": ["sleep", "3600"],  # Keep container running
            },
            target_services=[nginx_service],
            resource_limits={"total_memory": "512m"},
        )

    @pytest.fixture
    def complex_environment_spec(self):
        """Create a complex environment spec with multiple services."""
        network = NetworkSpec(name="saber_complex_test_network", driver="bridge", internal=False)

        # Redis service
        redis_service = ServiceSpec(
            name="redis",
            container="redis_container",
            image="redis:alpine",
            ports=["6379"],
            health_check=HealthCheck(test=["CMD", "redis-cli", "ping"], interval="10s", timeout="3s", retries=3),
        )

        # Nginx service that depends on Redis
        nginx_service = ServiceSpec(
            name="nginx",
            container="nginx_container",
            image="nginx:alpine",
            ports=["80"],
            depends_on=["redis"],
            health_check=HealthCheck(
                test=["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/"],
                interval="10s",
                timeout="5s",
                retries=3,
            ),
        )

        return EnvironmentSpec(
            networks=[network],
            execution_service="ubuntu_executor",
            execution_config={"image": "ubuntu:latest", "working_dir": "/workspace", "command": ["sleep", "3600"]},
            target_services=[redis_service, nginx_service],
            resource_limits={"total_memory": "1g"},
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_single_service_environment(self, docker_client, simple_environment_spec):
        """Test creating and managing a simple single-service environment."""
        session_id = "test_simple_session"
        env = None

        try:
            # Create and start environment
            env = DockerSandboxEnvironment(session_id, simple_environment_spec)
            env.start()

            # Verify environment is running
            assert len(env.active_services) == 2  # execution + nginx
            assert "ubuntu_executor" in env.active_services
            assert "nginx" in env.active_services

            # Get execution container
            exec_container = env.get_execution_container()
            assert exec_container is not None

            # Test command execution
            result = await env.execute_command(["echo", "Hello from container"])
            assert result.exit_code == 0
            assert "Hello from container" in result.stdout

            # Test service health
            # Note: Health checks may take time to become healthy
            time.sleep(15)  # Wait for health checks

            # Check if nginx service is accessible
            nginx_healthy = env.is_service_healthy("nginx")
            if not nginx_healthy:
                # Print logs for debugging
                logs = env.get_service_logs("nginx")
                print(f"Nginx logs: {logs}")

            # Get service info
            service_info = env.get_service_info()
            assert "ubuntu_executor" in service_info
            assert "nginx" in service_info

            # Test list active services
            active_services = env.list_active_services()
            assert "ubuntu_executor" in active_services
            assert "nginx" in active_services

            print("✓ Simple environment test passed")

        except Exception as e:
            if env:
                # Print service info for debugging
                try:
                    info = env.get_service_info()
                    print(f"Service info during failure: {info}")
                    for service_name in env.list_active_services():
                        logs = env.get_service_logs(service_name, tail=50)
                        print(f"Logs for {service_name}:\n{logs}")
                except:
                    pass
            raise

        finally:
            if env:
                try:
                    env.stop()
                    time.sleep(2)  # Give time for cleanup
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complex_multi_service_environment(self, docker_client, complex_environment_spec):
        """Test creating and managing a complex multi-service environment."""
        session_id = "test_complex_session"
        env = None

        try:
            # Create and start environment
            env = DockerSandboxEnvironment(session_id, complex_environment_spec)
            env.start()

            # Verify all services are running
            assert len(env.active_services) == 3  # execution + redis + nginx
            assert "ubuntu_executor" in env.active_services
            assert "redis" in env.active_services
            assert "nginx" in env.active_services

            # Test command execution
            result = await env.execute_command(["apt-get", "update"])
            assert result.exit_code == 0

            # Install network tools for testing
            result = await env.execute_command(["apt-get", "install", "-y", "iputils-ping", "curl"])
            assert result.exit_code == 0

            # Test network connectivity between services
            # Ping redis from execution container
            result = await env.execute_command(["ping", "-c", "1", "redis"])
            assert result.exit_code == 0
            assert "1 packets transmitted, 1 received" in result.stdout

            # Ping nginx from execution container
            result = await env.execute_command(["ping", "-c", "1", "nginx"])
            assert result.exit_code == 0

            # Wait for services to be healthy
            time.sleep(20)

            # Check service health
            redis_healthy = env.is_service_healthy("redis")
            nginx_healthy = env.is_service_healthy("nginx")

            print(f"Redis healthy: {redis_healthy}")
            print(f"Nginx healthy: {nginx_healthy}")

            # Get comprehensive service info
            service_info = env.get_service_info()
            print(f"Service info: {service_info}")

            # Verify we can interact with Redis
            result = await env.execute_command(
                [
                    "bash",
                    "-c",
                    'echo \'import socket; s=socket.socket(); s.connect(("redis", 6379)); s.close(); print("Redis reachable")\' | python3',
                ]
            )
            if result.exit_code != 0:
                print(f"Redis connectivity test stdout: {result.stdout}")
                print(f"Redis connectivity test stderr: {result.stderr}")

            print("✓ Complex environment test passed")

        except Exception as e:
            if env:
                # Print detailed debugging info
                try:
                    info = env.get_service_info()
                    print(f"Service info during failure: {info}")
                    for service_name in env.list_active_services():
                        logs = env.get_service_logs(service_name, tail=50)
                        print(f"Logs for {service_name}:\n{logs}")
                except:
                    pass
            raise

        finally:
            if env:
                try:
                    env.stop()
                    time.sleep(2)  # Give time for cleanup
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_environment_lifecycle(self, docker_client, simple_environment_spec):
        """Test complete environment lifecycle: create, use, stop, recreate."""
        session_id = "test_lifecycle_session"

        # First lifecycle
        env1 = DockerSandboxEnvironment(session_id + "_1", simple_environment_spec)
        try:
            env1.start()

            result = await env1.execute_command(["echo", "first lifecycle"])
            assert result.exit_code == 0
            assert "first lifecycle" in result.stdout

            env1.stop()

            # Verify services are cleaned up
            assert len(env1.active_services) == 0

        except Exception as e:
            if env1:
                try:
                    env1.stop()
                except:
                    pass
            raise

        # Second lifecycle - should work independently
        env2 = DockerSandboxEnvironment(session_id + "_2", simple_environment_spec)
        try:
            env2.start()

            result = await env2.execute_command(["echo", "second lifecycle"])
            assert result.exit_code == 0
            assert "second lifecycle" in result.stdout

            env2.stop()

        except Exception as e:
            if env2:
                try:
                    env2.stop()
                except:
                    pass
            raise

        print("✓ Environment lifecycle test passed")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_file_operations(self, docker_client, simple_environment_spec):
        """Test basic file operations in container."""
        session_id = "test_file_ops_session"
        env = None

        try:
            # Create and start environment
            env = DockerSandboxEnvironment(session_id, simple_environment_spec)
            env.start()

            # Test creating and reading files in container
            test_content = "Hello from integration test!"
            container_path = "/workspace/test_file.txt"

            # Create file in container
            result = await env.execute_command(["bash", "-c", f"echo '{test_content}' > {container_path}"])
            assert result.exit_code == 0

            # Read file from container
            result = await env.execute_command(["cat", container_path])
            assert result.exit_code == 0
            assert test_content in result.stdout

            # Modify file in container
            new_content = "Modified in container"
            result = await env.execute_command(["bash", "-c", f"echo '{new_content}' > {container_path}"])
            assert result.exit_code == 0

            # Verify modification
            result = await env.execute_command(["cat", container_path])
            assert result.exit_code == 0
            assert new_content in result.stdout

            print("✓ File operations test passed")

        finally:
            if env:
                try:
                    env.stop()
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")

    @pytest.mark.integration
    def test_error_handling(self, docker_client):
        """Test error handling with invalid configurations."""
        # Test with invalid image
        network = NetworkSpec(name="test_error_network")
        invalid_spec = EnvironmentSpec(
            networks=[network],
            execution_service="invalid_executor",
            execution_config={"image": "nonexistent:image"},
            target_services=[],
        )

        env = DockerSandboxEnvironment("test_error_session", invalid_spec, )

        with pytest.raises(ContainerCreationError):
            env.start()

        print("✓ Error handling test passed")
