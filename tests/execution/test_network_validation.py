"""
Tests for network naming validation in compose orchestrator.
Verifies fail-fast behavior when non-standard isolation networks are used.
"""

import tempfile
import yaml
import pytest

from saber.server.execution.sandbox.compose_orchestrator import ComposeOrchestrator
from saber.server.execution.sandbox.environment_config import ComposeEnvironmentConfig


class TestNetworkValidation:
    """Test network naming validation and fail-fast behavior."""

    def test_valid_saber_episode_network_passes(self):
        """Test that using saber-episode-network passes validation."""
        compose_content = {
            "services": {
                "test-service": {
                    "image": "test:latest",
                    "networks": ["external-db", "saber-episode-network"]
                }
            },
            "networks": {
                "external-db": {"external": True}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(compose_content, f)
            compose_path = f.name

        try:
            config = ComposeEnvironmentConfig(
                episode_id="test-episode",
                project_name="test-project"
            )

            orchestrator = ComposeOrchestrator()

            # Should not raise exception
            processed_path = orchestrator._inject_episode_network(compose_path, config)
            assert processed_path is not None

        finally:
            import os
            os.unlink(compose_path)

    def test_invalid_isolated_network_fails_fast(self):
        """Test that using non-standard isolation networks fails fast."""
        compose_content = {
            "services": {
                "test-service": {
                    "image": "test:latest",
                    "networks": ["external-db", "custom-isolated"]
                }
            },
            "networks": {
                "external-db": {"external": True}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(compose_content, f)
            compose_path = f.name

        try:
            config = ComposeEnvironmentConfig(
                episode_id="test-episode",
                project_name="test-project"
            )

            orchestrator = ComposeOrchestrator()

            # Should raise ValueError with descriptive message
            with pytest.raises(ValueError) as exc_info:
                orchestrator._inject_episode_network(compose_path, config)

            error_message = str(exc_info.value)
            assert "SABER Network Naming Error" in error_message
            assert "custom-isolated" in error_message
            assert "saber-episode-network" in error_message

        finally:
            import os
            os.unlink(compose_path)

    def test_invalid_episode_network_fails_fast(self):
        """Test that using custom episode networks fails fast."""
        compose_content = {
            "services": {
                "test-service": {
                    "image": "test:latest",
                    "networks": ["my-episode-net"]
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(compose_content, f)
            compose_path = f.name

        try:
            config = ComposeEnvironmentConfig(
                episode_id="test-episode",
                project_name="test-project"
            )

            orchestrator = ComposeOrchestrator()

            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                orchestrator._inject_episode_network(compose_path, config)

            error_message = str(exc_info.value)
            assert "my-episode-net" in error_message
            assert "saber-episode-network" in error_message

        finally:
            import os
            os.unlink(compose_path)

    def test_invalid_sandbox_network_fails_fast(self):
        """Test that using custom sandbox networks fails fast."""
        compose_content = {
            "services": {
                "test-service": {
                    "image": "test:latest",
                    "networks": ["custom-sandbox-net"]
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(compose_content, f)
            compose_path = f.name

        try:
            config = ComposeEnvironmentConfig(
                episode_id="test-episode",
                project_name="test-project"
            )

            orchestrator = ComposeOrchestrator()

            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                orchestrator._inject_episode_network(compose_path, config)

            error_message = str(exc_info.value)
            assert "custom-sandbox-net" in error_message

        finally:
            import os
            os.unlink(compose_path)

    def test_external_networks_allowed(self):
        """Test that external/shared networks are allowed alongside saber-episode-network."""
        compose_content = {
            "services": {
                "test-service": {
                    "image": "test:latest",
                    "networks": ["shared-database", "external-cache", "saber-episode-network"]
                }
            },
            "networks": {
                "shared-database": {"external": True},
                "external-cache": {"external": True}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(compose_content, f)
            compose_path = f.name

        try:
            config = ComposeEnvironmentConfig(
                episode_id="test-episode",
                project_name="test-project"
            )

            orchestrator = ComposeOrchestrator()

            # Should not raise exception
            processed_path = orchestrator._inject_episode_network(compose_path, config)
            assert processed_path is not None

        finally:
            import os
            os.unlink(compose_path)

    def test_network_dict_format_validation(self):
        """Test validation works with networks defined as dict format."""
        compose_content = {
            "services": {
                "test-service": {
                    "image": "test:latest",
                    "networks": {
                        "external-db": {"aliases": ["db"]},
                        "custom-isolated": {"priority": 1000}
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(compose_content, f)
            compose_path = f.name

        try:
            config = ComposeEnvironmentConfig(
                episode_id="test-episode",
                project_name="test-project"
            )

            orchestrator = ComposeOrchestrator()

            # Should raise ValueError for custom-isolated
            with pytest.raises(ValueError) as exc_info:
                orchestrator._inject_episode_network(compose_path, config)

            error_message = str(exc_info.value)
            assert "custom-isolated" in error_message

        finally:
            import os
            os.unlink(compose_path)
