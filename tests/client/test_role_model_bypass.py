"""Test that role-based model assignment bypasses --model requirement."""

import pytest
from saber.client.models import RoleAgentConfig, RoleBasedConfig
from saber.inspect_ai.tasks import _all_roles_have_models


class TestRoleModelBypass:
    """Test the logic for bypassing --model requirement when all roles have models."""

    def test_all_roles_have_models_true(self):
        """Test when all roles have models explicitly defined."""
        config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4"),
                "blue": RoleAgentConfig(agent="cot", model="claude-3"),
            }
        )

        assert _all_roles_have_models(config) is True

    def test_all_roles_have_models_via_defaults(self):
        """Test when default provides model for all roles."""
        config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react"),  # No model
                "blue": RoleAgentConfig(agent="cot"),   # No model
            },
            defaults=RoleAgentConfig(agent="react", model="gpt-4o-mini")
        )

        assert _all_roles_have_models(config) is True

    def test_some_roles_missing_models(self):
        """Test when some roles don't have models."""
        config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4"),
                "blue": RoleAgentConfig(agent="cot"),  # No model
            }
        )

        assert _all_roles_have_models(config) is False

    def test_no_roles_defined(self):
        """Test when no roles are defined."""
        config = RoleBasedConfig(
            roles={},
            defaults=RoleAgentConfig(agent="react", model="gpt-4o-mini")
        )

        # No roles means nothing to validate - should return False
        assert _all_roles_have_models(config) is False

    def test_mixed_explicit_and_default_models(self):
        """Test when some roles have models and defaults provide fallback."""
        config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4"),
                "blue": RoleAgentConfig(agent="cot"),  # Will inherit from defaults
                "green": RoleAgentConfig(agent="react"),  # Will inherit from defaults
            },
            defaults=RoleAgentConfig(agent="react", model="claude-3")
        )

        assert _all_roles_have_models(config) is True

    def test_no_models_anywhere(self):
        """Test when no models are defined anywhere."""
        config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react"),
                "blue": RoleAgentConfig(agent="cot"),
            }
        )

        assert _all_roles_have_models(config) is False
