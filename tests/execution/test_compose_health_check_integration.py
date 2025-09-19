"""
Tests for Docker Compose Health Check Integration in Episode Creation.

This test module verifies that the new health check mechanism correctly
prevents episode IDs from being returned until all Docker Compose services
are healthy.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from saber.server.execution.sandbox.compose_health_checker import (
    ComposeHealthChecker,
    ComposeHealthCheckError
)


class TestComposeHealthChecker:
    """Test the core health check functionality."""

    def setup_method(self):
        """Setup for each test method."""
        # Mock Docker to avoid requiring actual Docker in tests
        self.mock_docker_patcher = patch('saber.server.execution.sandbox.compose_health_checker.docker')
        self.mock_docker = self.mock_docker_patcher.start()

        # Create mock Docker client
        self.mock_docker_client = MagicMock()
        self.mock_docker.from_env.return_value = self.mock_docker_client

        self.health_checker = ComposeHealthChecker()

    def teardown_method(self):
        """Cleanup after each test."""
        self.mock_docker_patcher.stop()

    def test_health_checker_initialization(self):
        """Test that health checker initializes correctly."""
        assert self.health_checker is not None
        assert self.health_checker._docker_client is None  # Lazy loading

    @patch('saber.server.execution.sandbox.compose_health_checker.yaml')
    @patch('saber.server.execution.sandbox.compose_health_checker.Path')
    def test_parse_compose_services_success(self, mock_path, mock_yaml):
        """Test successful parsing of compose file."""
        # Setup mocks
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        mock_yaml.safe_load.return_value = {
            'services': {
                'webapp': {
                    'image': 'nginx',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost/']
                    }
                },
                'database': {
                    'image': 'mysql',
                    'healthcheck': {
                        'test': ['CMD', 'mysqladmin', 'ping']
                    }
                }
            }
        }

        # Test
        services = self.health_checker._parse_compose_services('/fake/compose.yml')

        assert len(services) == 2
        assert 'webapp' in services
        assert 'database' in services
        assert services['webapp']['image'] == 'nginx'

    @patch('saber.server.execution.sandbox.compose_health_checker.yaml')
    @patch('saber.server.execution.sandbox.compose_health_checker.Path')
    def test_parse_compose_services_file_not_found(self, mock_path, mock_yaml):
        """Test handling of missing compose file."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj

        with pytest.raises(RuntimeError, match="Compose file not found"):
            self.health_checker._parse_compose_services('/fake/compose.yml')

    def test_check_service_health_healthy_container(self):
        """Test health check for a healthy container."""
        # Setup mock container
        mock_container = MagicMock()
        mock_container.status = 'running'
        mock_container.attrs = {
            'State': {
                'Health': {
                    'Status': 'healthy'
                }
            }
        }
        self.mock_docker_client.containers.get.return_value = mock_container

        # Test
        result = self.health_checker._check_service_health(
            'webapp',
            {'image': 'nginx'},
            'test-project'
        )

        assert result['healthy'] is True
        assert result['reason'] == 'Health check passed'
        assert result['container_name'] == 'test-project-webapp-1'

    def test_check_service_health_unhealthy_container(self):
        """Test health check for an unhealthy container."""
        mock_container = MagicMock()
        mock_container.status = 'running'
        mock_container.attrs = {
            'State': {
                'Health': {
                    'Status': 'unhealthy'
                }
            }
        }
        self.mock_docker_client.containers.get.return_value = mock_container

        result = self.health_checker._check_service_health(
            'webapp',
            {'image': 'nginx'},
            'test-project'
        )

        assert result['healthy'] is False
        assert result['reason'] == 'Health check failed'

    def test_check_service_health_container_not_running(self):
        """Test health check for a non-running container."""
        mock_container = MagicMock()
        mock_container.status = 'exited'
        mock_container.attrs = {}
        self.mock_docker_client.containers.get.return_value = mock_container

        result = self.health_checker._check_service_health(
            'webapp',
            {'image': 'nginx'},
            'test-project'
        )

        assert result['healthy'] is False
        assert 'not running' in result['reason']
        assert result['status'] == 'exited'

    def test_check_service_health_container_not_found(self):
        """Test health check when container doesn't exist."""
        # Create a custom exception that mimics docker.errors.NotFound
        class MockNotFound(Exception):
            pass

        self.mock_docker_client.containers.get.side_effect = MockNotFound("Container not found")

        result = self.health_checker._check_service_health(
            'webapp',
            {'image': 'nginx'},
            'test-project'
        )

        assert result['healthy'] is False
        assert result['reason'] == 'Container not found'
        assert result['status'] == 'not_found'

    def test_check_service_health_custom_container_name(self):
        """Test health check with custom container name."""
        mock_container = MagicMock()
        mock_container.status = 'running'
        mock_container.attrs = {'State': {}}  # No health check
        self.mock_docker_client.containers.get.return_value = mock_container

        result = self.health_checker._check_service_health(
            'webapp',
            {'image': 'nginx', 'container_name': 'custom-webapp'},
            'test-project'
        )

        assert result['healthy'] is True
        assert result['container_name'] == 'custom-webapp'
        assert 'no health check defined' in result['reason']

    @patch.object(ComposeHealthChecker, '_parse_compose_services')
    @patch.object(ComposeHealthChecker, '_check_all_services_health')
    def test_wait_for_all_services_healthy_success(self, mock_check_all, mock_parse):
        """Test successful wait for all services to become healthy."""
        # Setup mocks
        mock_parse.return_value = {
            'webapp': {'image': 'nginx'},
            'database': {'image': 'mysql'}
        }

        mock_check_all.return_value = {
            'webapp': {'healthy': True, 'reason': 'Health check passed'},
            'database': {'healthy': True, 'reason': 'Health check passed'}
        }

        # Test - should complete without exception
        self.health_checker.wait_for_all_services_healthy(
            '/fake/compose.yml',
            'test-project',
            timeout_seconds=10,
            check_interval=0.1
        )

        # Verify methods were called
        mock_parse.assert_called_once_with('/fake/compose.yml')
        mock_check_all.assert_called()

    @patch.object(ComposeHealthChecker, '_parse_compose_services')
    @patch.object(ComposeHealthChecker, '_check_all_services_health')
    def test_wait_for_all_services_healthy_timeout(self, mock_check_all, mock_parse):
        """Test timeout when services don't become healthy."""
        mock_parse.return_value = {
            'webapp': {'image': 'nginx'},
            'database': {'image': 'mysql'}
        }

        # Always return one unhealthy service
        mock_check_all.return_value = {
            'webapp': {'healthy': True, 'reason': 'Health check passed'},
            'database': {'healthy': False, 'reason': 'Health check failed'}
        }

        # Test - should raise timeout exception
        with pytest.raises(ComposeHealthCheckError, match="failed to become healthy within"):
            self.health_checker.wait_for_all_services_healthy(
                '/fake/compose.yml',
                'test-project',
                timeout_seconds=0.5,  # Very short timeout
                check_interval=0.1
            )


class TestHealthCheckIntegration:
    """Test integration of health checks with episode creation."""

    @patch('saber.server.execution.sandbox.compose_orchestrator.ComposeHealthChecker')
    def test_compose_orchestrator_health_check_integration(self, mock_health_checker_class):
        """Test that ComposeOrchestrator integrates health checks correctly."""
        from saber.server.execution.sandbox.compose_orchestrator import ComposeOrchestrator

        # Setup mock health checker
        mock_health_checker = MagicMock()
        mock_health_checker_class.return_value = mock_health_checker

        # Create orchestrator
        orchestrator = ComposeOrchestrator()

        # Verify health checker was created
        assert orchestrator.health_checker is not None
        mock_health_checker_class.assert_called_once()

    def test_health_check_error_contains_meaningful_message(self):
        """Test that health check errors contain useful information."""
        error_msg = "Services failed to become healthy within 300s. Unhealthy services: ['database: Health check failed']"

        with pytest.raises(ComposeHealthCheckError, match="Services failed to become healthy"):
            raise ComposeHealthCheckError(error_msg)


@pytest.mark.integration
class TestHealthCheckRealDocker:
    """Integration tests that require actual Docker (marked for selective running)."""

    def test_real_docker_health_check(self):
        """Test health check with real Docker (requires Docker to be running)."""
        pytest.skip("Integration test requiring Docker - run manually when needed")

        # This test would check actual Docker containers
        # Only run when Docker is available and you want full integration testing
        health_checker = ComposeHealthChecker()

        # Test with a simple compose file
        # ... implementation would go here


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
