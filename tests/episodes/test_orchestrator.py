#!/usr/bin/env python3
"""
Unit tests for the SABER Episode Container Orchestrator.

Tests the orchestrator's ability to:
1. Monitor episode status
2. Detect episode termination
3. Perform container cleanup with multiple strategies
4. Handle various failure scenarios
"""

import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from pathlib import Path
import sys

from saber.server.episodes.episode_orchestrator import EpisodeContainerOrchestrator


class TestEpisodeContainerOrchestrator(unittest.TestCase):
    """Test suite for the Episode Orchestrator."""

    def setUp(self):
        """Set up test environment variables."""
        self.original_env = os.environ.copy()

        # Set required environment variables for testing
        os.environ.update({
            "SABER_SESSION_ID": "test_session_123",
            "SABER_CLEANUP_TOKEN": "test_token_456",
            "SABER_HOST_URL": "http://localhost:8000",
            "SABER_COMPOSE_PROJECT": "saber-session-test_session_123",
            "SABER_POLL_INTERVAL": "5",  # Short interval for tests
            "SABER_GRACEFUL_TIMEOUT": "10"
        })

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_initialization_success(self):
        """Test successful orchestrator initialization."""
        orchestrator = EpisodeContainerOrchestrator()

        self.assertEqual(orchestrator.session_id, "test_session_123")
        self.assertEqual(orchestrator.cleanup_token, "test_token_456")
        self.assertEqual(orchestrator.host_url, "http://localhost:8000")
        self.assertEqual(orchestrator.compose_project, "saber-session-test_session_123")
        self.assertEqual(orchestrator.poll_interval, 5)
        self.assertTrue(orchestrator.running)

    def test_initialization_missing_session_id(self):
        """Test initialization fails without session ID."""
        del os.environ["SABER_SESSION_ID"]

        with self.assertRaises(ValueError) as context:
            EpisodeContainerOrchestrator()

        self.assertIn("SABER_SESSION_ID", str(context.exception))

    def test_initialization_missing_cleanup_token(self):
        """Test initialization fails without cleanup token."""
        del os.environ["SABER_CLEANUP_TOKEN"]

        with self.assertRaises(ValueError) as context:
            EpisodeContainerOrchestrator()

        self.assertIn("SABER_CLEANUP_TOKEN", str(context.exception))

    def test_initialization_missing_compose_project(self):
        """Test initialization fails without compose project."""
        del os.environ["SABER_COMPOSE_PROJECT"]

        with self.assertRaises(ValueError) as context:
            EpisodeContainerOrchestrator()

        self.assertIn("SABER_COMPOSE_PROJECT", str(context.exception))

    @patch('httpx.AsyncClient')
    async def test_check_episode_status_active(self, mock_client):
        """Test episode status check when episode is active."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"active": True}

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        orchestrator = EpisodeContainerOrchestrator()
        result = await orchestrator.check_episode_status()

        self.assertTrue(result)
        mock_client_instance.get.assert_called_once_with(
            "http://localhost:8000/internal/episode-status/test_session_123",
            params={"token": "test_token_456"}
        )

    @patch('httpx.AsyncClient')
    async def test_check_episode_status_inactive(self, mock_client):
        """Test episode status check when episode is inactive."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"active": False}

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        orchestrator = EpisodeContainerOrchestrator()
        result = await orchestrator.check_episode_status()

        self.assertFalse(result)

    @patch('httpx.AsyncClient')
    async def test_check_episode_status_http_error(self, mock_client):
        """Test episode status check with HTTP error."""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 404

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        orchestrator = EpisodeContainerOrchestrator()
        result = await orchestrator.check_episode_status()

        self.assertIsNone(result)

    @patch('httpx.AsyncClient')
    async def test_check_episode_status_network_error(self, mock_client):
        """Test episode status check with network error."""
        # Mock network exception
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = Exception("Network error")
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        orchestrator = EpisodeContainerOrchestrator()
        result = await orchestrator.check_episode_status()

        self.assertIsNone(result)

    @patch('subprocess.run')
    def test_get_episode_containers_success(self, mock_run):
        """Test successful container discovery."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "container1\ncontainer2\ncontainer3\n"

        orchestrator = EpisodeContainerOrchestrator()
        containers = orchestrator.get_episode_containers()

        self.assertEqual(containers, ["container1", "container2", "container3"])
        mock_run.assert_called_once_with(
            ["docker", "compose", "-p", "saber-session-test_session_123", "ps", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )

    @patch('subprocess.run')
    def test_get_episode_containers_failure(self, mock_run):
        """Test container discovery failure."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Error listing containers"

        orchestrator = EpisodeContainerOrchestrator()
        containers = orchestrator.get_episode_containers()

        self.assertEqual(containers, [])

    @patch('subprocess.run')
    def test_graceful_stop_containers_success(self, mock_run):
        """Test successful graceful container stop."""
        mock_run.return_value.returncode = 0

        orchestrator = EpisodeContainerOrchestrator()
        result = orchestrator.graceful_stop_containers()

        self.assertTrue(result)
        mock_run.assert_called_once_with(
            ["docker", "compose", "-p", "saber-session-test_session_123", "stop", "-t", "10"],
            capture_output=True,
            text=True,
            timeout=40
        )

    @patch('subprocess.run')
    def test_graceful_stop_containers_failure(self, mock_run):
        """Test graceful container stop failure."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Stop failed"

        orchestrator = EpisodeContainerOrchestrator()
        result = orchestrator.graceful_stop_containers()

        self.assertFalse(result)

    @patch('subprocess.run')
    def test_force_cleanup_containers_success(self, mock_run):
        """Test successful force cleanup."""
        mock_run.return_value.returncode = 0

        orchestrator = EpisodeContainerOrchestrator()
        result = orchestrator.force_cleanup_containers()

        self.assertTrue(result)
        mock_run.assert_called_once_with(
            [
                "docker", "compose", "-p", "saber-session-test_session_123",
                "down", "--remove-orphans", "--volumes", "--timeout", "10"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

    @patch('subprocess.run')
    def test_nuclear_cleanup_success(self, mock_run):
        """Test nuclear cleanup with containers found."""
        # Mock finding containers by label
        mock_run.side_effect = [
            # First call: find containers
            Mock(returncode=0, stdout="container1\ncontainer2\n"),
            # Subsequent calls: kill and remove
            Mock(returncode=0),  # kill container1
            Mock(returncode=0),  # rm container1
            Mock(returncode=0),  # kill container2
            Mock(returncode=0),  # rm container2
        ]

        orchestrator = EpisodeContainerOrchestrator()
        orchestrator.nuclear_cleanup()

        # Should have made 5 calls: 1 to find + 4 to kill/remove
        self.assertEqual(mock_run.call_count, 5)

    @patch('subprocess.run')
    def test_nuclear_cleanup_no_containers(self, mock_run):
        """Test nuclear cleanup with no containers found."""
        # Mock no containers found
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""

        orchestrator = EpisodeContainerOrchestrator()
        orchestrator.nuclear_cleanup()

        # Should only call once to find containers
        self.assertEqual(mock_run.call_count, 1)

    async def test_cleanup_episode_graceful_success(self):
        """Test successful graceful cleanup."""
        orchestrator = EpisodeContainerOrchestrator()

        with patch.object(orchestrator, 'get_episode_containers') as mock_get, \
             patch.object(orchestrator, 'graceful_stop_containers') as mock_graceful:

            mock_get.side_effect = [["container1"], []]  # Before and after cleanup
            mock_graceful.return_value = True

            await orchestrator.cleanup_episode("Test cleanup")

            mock_graceful.assert_called_once()

    async def test_cleanup_episode_force_fallback(self):
        """Test cleanup with force fallback."""
        orchestrator = EpisodeContainerOrchestrator()

        with patch.object(orchestrator, 'get_episode_containers') as mock_get, \
             patch.object(orchestrator, 'graceful_stop_containers') as mock_graceful, \
             patch.object(orchestrator, 'force_cleanup_containers') as mock_force:

            mock_get.side_effect = [["container1"], ["container1"], []]  # Before, after graceful, after force
            mock_graceful.return_value = False
            mock_force.return_value = True

            await orchestrator.cleanup_episode("Test cleanup")

            mock_graceful.assert_called_once()
            mock_force.assert_called_once()

    async def test_cleanup_episode_nuclear_fallback(self):
        """Test cleanup with nuclear fallback."""
        orchestrator = EpisodeContainerOrchestrator()

        with patch.object(orchestrator, 'get_episode_containers') as mock_get, \
             patch.object(orchestrator, 'graceful_stop_containers') as mock_graceful, \
             patch.object(orchestrator, 'force_cleanup_containers') as mock_force, \
             patch.object(orchestrator, 'nuclear_cleanup') as mock_nuclear:

            mock_get.side_effect = [["container1"], ["container1"], ["container1"]]  # All fail
            mock_graceful.return_value = False
            mock_force.return_value = False

            await orchestrator.cleanup_episode("Test cleanup")

            mock_graceful.assert_called_once()
            mock_force.assert_called_once()
            mock_nuclear.assert_called_once()

    async def test_monitor_episode_normal_termination(self):
        """Test normal episode termination detection."""
        orchestrator = EpisodeContainerOrchestrator()

        with patch.object(orchestrator, 'check_episode_status') as mock_check, \
             patch.object(orchestrator, 'cleanup_episode') as mock_cleanup:

            # Episode becomes inactive
            mock_check.return_value = False

            await orchestrator.monitor_episode()

            mock_cleanup.assert_called_once_with("Episode no longer active")
            self.assertFalse(orchestrator.running)

    async def test_monitor_episode_failure_threshold(self):
        """Test cleanup after max failures."""
        orchestrator = EpisodeContainerOrchestrator()

        with patch.object(orchestrator, 'check_episode_status') as mock_check, \
             patch.object(orchestrator, 'cleanup_episode') as mock_cleanup:

            # All status checks fail
            mock_check.return_value = None

            await orchestrator.monitor_episode()

            mock_cleanup.assert_called_once()
            self.assertFalse(orchestrator.running)
            # Should have failed max_failures times
            self.assertEqual(mock_check.call_count, orchestrator.max_failures)


class TestEpisodeContainerOrchestratorIntegration(unittest.TestCase):
    """Integration tests for the orchestrator with real episode manager."""

    def setUp(self):
        """Set up integration test environment."""
        self.original_env = os.environ.copy()

        from saber.server.episodes.episode_manager import EpisodeManager

        self.episode_manager = EpisodeManager()

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    async def test_full_coordination_flow(self):
        """Test full coordination between episode manager and orchestrator."""
        session_id = "integration_test_session"
        task_id = "integration_test_task"

        # Create episode with cleanup token
        episode = self.episode_manager.start_episode(session_id, task_id)
        cleanup_token = episode.metadata.get("cleanup_token")

        # Verify episode is active
        self.assertTrue(self.episode_manager.is_episode_active(session_id, cleanup_token))

        # Simulate error and remove episode
        test_error = Exception("Integration test error")
        self.episode_manager.remove_episode_on_error(session_id, test_error)

        # Verify episode is no longer active
        self.assertFalse(self.episode_manager.is_episode_active(session_id, cleanup_token))

        # Verify episode was removed from tracking
        self.assertIsNone(self.episode_manager.get_current_episode(session_id))


def run_tests():
    """Run all orchestrator tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEpisodeContainerOrchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestEpisodeContainerOrchestratorIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import asyncio

    # Run async tests
    async def main():
        success = run_tests()
        if success:
            print("\n✅ All orchestrator tests passed!")
        else:
            print("\n❌ Some tests failed!")
        return success

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
