"""
Test network attachment functionality for ComposeOrchestrator.

Tests the network auto-injection logic that enables episodes to share networks.
"""

import tempfile
import yaml
from pathlib import Path
import pytest

from saber.server.execution.sandbox.compose_orchestrator import ComposeOrchestrator
from saber.server.execution.sandbox.environment_config import ComposeEnvironmentConfig


class TestNetworkAttachment:
    """Test network-attached episodes functionality."""

    @pytest.fixture
    def sample_compose_content(self):
        """Sample compose file content for testing."""
        return """
version: '3.8'
services:
  test-service:
    image: hello-world
    networks:
      - saber-episode-network
    labels:
      - "saber.execution.service=true"

# No networks section - will be auto-injected
"""

    @pytest.fixture
    def temp_compose_file(self, sample_compose_content):
        """Create temporary compose file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(sample_compose_content)
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_inject_episode_network_normal_mode(self, temp_compose_file):
        """Test network injection for normal episodes (creates new isolated network)."""
        orchestrator = ComposeOrchestrator()

        config = ComposeEnvironmentConfig(
            episode_id="episode_123",
            config_type="sandbox"
        )

        # Test network injection
        processed_path = orchestrator._inject_episode_network(str(temp_compose_file), config)

        # Parse the processed compose file
        with open(processed_path, 'r') as f:
            processed_data = yaml.safe_load(f)

        # Verify network was injected correctly
        assert 'networks' in processed_data
        assert 'saber-episode-network' in processed_data['networks']

        network_config = processed_data['networks']['saber-episode-network']
        assert network_config['name'] == "saber-episode-episode_123"
        assert network_config['internal'] is True
        assert network_config['driver'] == 'bridge'
        assert 'saber.network.type=isolated' in network_config['labels']
        assert 'saber.episode.id=episode_123' in network_config['labels']

        # Cleanup processed file
        Path(processed_path).unlink()

    def test_inject_episode_network_attached_mode(self, temp_compose_file):
        """Test network injection for attached episodes (references existing network)."""
        orchestrator = ComposeOrchestrator()

        config = ComposeEnvironmentConfig(
            episode_id="episode_456",
            config_type="sandbox",
            target_episode_id="episode_123"
        )

        # Test network injection
        processed_path = orchestrator._inject_episode_network(str(temp_compose_file), config)

        # Parse the processed compose file
        with open(processed_path, 'r') as f:
            processed_data = yaml.safe_load(f)

        # Verify network was injected correctly
        assert 'networks' in processed_data
        assert 'saber-episode-network' in processed_data['networks']

        network_config = processed_data['networks']['saber-episode-network']
        assert network_config['external'] is True
        assert network_config['name'] == "saber-episode-episode_123"

        # Should not have internal, driver, or labels for external networks
        assert 'internal' not in network_config
        assert 'driver' not in network_config
        assert 'labels' not in network_config

        # Cleanup processed file
        Path(processed_path).unlink()

    def test_inject_episode_network_preserves_existing_networks(self, temp_compose_file):
        """Test that network injection preserves existing network definitions."""
        # Create compose file with existing networks
        compose_content = """
version: '3.8'
services:
  test-service:
    image: hello-world
    networks:
      - existing-network
      - saber-episode-network
    labels:
      - "saber.execution.service=true"

networks:
  existing-network:
    driver: bridge
    name: my-existing-network
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(compose_content)
            temp_path = f.name

        try:
            orchestrator = ComposeOrchestrator()

            config = ComposeEnvironmentConfig(
                episode_id="episode_789",
                config_type="sandbox"
            )

            # Test network injection
            processed_path = orchestrator._inject_episode_network(temp_path, config)

            # Parse the processed compose file
            with open(processed_path, 'r') as f:
                processed_data = yaml.safe_load(f)

            # Verify existing network is preserved
            assert 'networks' in processed_data
            assert 'existing-network' in processed_data['networks']
            assert processed_data['networks']['existing-network']['name'] == 'my-existing-network'

            # Verify injected network is added
            assert 'saber-episode-network' in processed_data['networks']
            network_config = processed_data['networks']['saber-episode-network']
            assert network_config['name'] == "saber-episode-episode_789"

            # Cleanup processed file
            Path(processed_path).unlink()
        finally:
            # Cleanup temp file
            Path(temp_path).unlink()

    def test_inject_episode_network_error_fallback(self):
        """Test that network injection falls back to original file on error."""
        orchestrator = ComposeOrchestrator()

        config = ComposeEnvironmentConfig(
            episode_id="episode_error",
            config_type="sandbox"
        )

        # Test with non-existent file
        nonexistent_path = "/nonexistent/compose.yml"
        result_path = orchestrator._inject_episode_network(nonexistent_path, config)

        # Should fall back to original path
        assert result_path == nonexistent_path
