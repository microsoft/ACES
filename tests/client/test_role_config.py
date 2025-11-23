"""Unit tests for role-based configuration system."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from saber.client.config_loader import RoleConfigLoader
from saber.client.models import AgentAssignment, RoleAgentConfig, RoleBasedConfig, SABERConfig, AgentCompositeKey


class TestRoleAgentConfig:
    """Test RoleAgentConfig model."""

    def test_create_role_agent_config_minimal(self):
        """Test creating RoleAgentConfig with minimal fields."""
        config = RoleAgentConfig(agent="react")

        assert config.agent == "react"
        assert config.model is None
        assert config.attempts is None
        assert config.kwargs == {}

    def test_create_role_agent_config_full(self):
        """Test creating RoleAgentConfig with all fields."""
        config = RoleAgentConfig(
            agent="react",
            model="gpt-4",
            attempts=3,
            kwargs={"temperature": 0.7, "max_tokens": 4000}
        )

        assert config.agent == "react"
        assert config.model == "gpt-4"
        assert config.attempts == 3
        assert config.kwargs["temperature"] == 0.7
        assert config.kwargs["max_tokens"] == 4000


class TestRoleBasedConfig:
    """Test RoleBasedConfig model."""

    def test_create_role_based_config_no_defaults(self):
        """Test creating RoleBasedConfig without defaults."""
        red_config = RoleAgentConfig(agent="react", model="gpt-4")
        blue_config = RoleAgentConfig(agent="cot", model="claude-3")

        config = RoleBasedConfig(
            roles={"red": red_config, "blue": blue_config}
        )

        assert len(config.roles) == 2
        assert config.roles["red"].model == "gpt-4"
        assert config.roles["blue"].model == "claude-3"
        assert config.defaults is None

    def test_create_role_based_config_with_defaults(self):
        """Test creating RoleBasedConfig with defaults."""
        red_config = RoleAgentConfig(agent="react", model="gpt-4")
        defaults = RoleAgentConfig(agent="react", model="gpt-4o-mini")

        config = RoleBasedConfig(
            roles={"red": red_config},
            defaults=defaults
        )

        assert config.defaults.model == "gpt-4o-mini"

    def test_get_config_for_role_existing(self):
        """Test get_config_for_role with existing role."""
        red_config = RoleAgentConfig(agent="react", model="gpt-4", attempts=3)
        defaults = RoleAgentConfig(agent="react", model="gpt-4o-mini", attempts=1)

        config = RoleBasedConfig(
            roles={"red": red_config},
            defaults=defaults
        )

        result = config.get_config_for_role("red")
        assert result.model == "gpt-4"
        assert result.attempts == 3

    def test_get_config_for_role_missing_with_defaults(self):
        """Test get_config_for_role with missing role falls back to defaults."""
        defaults = RoleAgentConfig(agent="react", model="gpt-4o-mini", attempts=1)

        config = RoleBasedConfig(
            roles={"red": RoleAgentConfig(agent="react", model="gpt-4")},
            defaults=defaults
        )

        result = config.get_config_for_role("unknown")
        assert result.model == "gpt-4o-mini"
        assert result.attempts == 1

    def test_get_config_for_role_partial_override(self):
        """Test get_config_for_role merges role config with defaults."""
        red_config = RoleAgentConfig(agent="react", model="gpt-4")  # No attempts
        defaults = RoleAgentConfig(agent="react", model="gpt-4o-mini", attempts=2)

        config = RoleBasedConfig(
            roles={"red": red_config},
            defaults=defaults
        )

        result = config.get_config_for_role("red")
        assert result.model == "gpt-4"  # From role config
        assert result.attempts == 2  # From defaults

    def test_to_agent_assignments(self):
        """Test converting RoleBasedConfig to agent assignments."""
        red_config = RoleAgentConfig(agent="react", model="gpt-4", attempts=3)
        blue_config = RoleAgentConfig(agent="cot", model="claude-3", attempts=2)

        config = RoleBasedConfig(
            roles={"red": red_config, "blue": blue_config}
        )

        assignments = config.to_agent_assignments()

        assert len(assignments) == 2

        red_assignment = next(a for a in assignments if a.role == "red")
        assert red_assignment.id == "react"
        assert red_assignment.model == "gpt-4"
        assert red_assignment.attempts == 3
        assert red_assignment.tasks == ["*"]

        blue_assignment = next(a for a in assignments if a.role == "blue")
        assert blue_assignment.id == "cot"
        assert blue_assignment.model == "claude-3"
        assert blue_assignment.attempts == 2


class TestRoleConfigLoader:
    """Test RoleConfigLoader class."""

    def test_load_from_dict(self):
        """Test loading role config from dictionary."""
        config_dict = {
            "roles": {
                "red": {"agent": "react", "model": "gpt-4"},
                "blue": {"agent": "cot", "model": "claude-3"}
            },
            "defaults": {"agent": "react", "model": "gpt-4o-mini"}
        }

        config = RoleConfigLoader.load(config_dict)

        assert len(config.roles) == 2
        assert config.roles["red"].model == "gpt-4"
        assert config.defaults.model == "gpt-4o-mini"

    def test_load_from_json_string(self):
        """Test loading role config from JSON string."""
        json_str = json.dumps({
            "roles": {
                "red": {"agent": "react", "model": "gpt-4"}
            }
        })

        config = RoleConfigLoader.load(json_str)

        assert len(config.roles) == 1
        assert config.roles["red"].model == "gpt-4"

    def test_load_from_yaml_file(self):
        """Test loading role config from YAML file."""
        yaml_content = """
roles:
  red:
    agent: react
    model: gpt-4
    attempts: 3
  blue:
    agent: cot
    model: claude-3
    attempts: 2
defaults:
  agent: react
  model: gpt-4o-mini
  attempts: 1
"""

        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            config = RoleConfigLoader.load(str(temp_path))

            assert len(config.roles) == 2
            assert config.roles["red"].model == "gpt-4"
            assert config.roles["red"].attempts == 3
            assert config.roles["blue"].model == "claude-3"
            assert config.defaults.model == "gpt-4o-mini"
        finally:
            temp_path.unlink()

    def test_load_from_json_file(self):
        """Test loading role config from JSON file."""
        config_dict = {
            "roles": {
                "red": {"agent": "react", "model": "gpt-4"}
            }
        }

        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = Path(f.name)

        try:
            config = RoleConfigLoader.load(str(temp_path))

            assert len(config.roles) == 1
            assert config.roles["red"].model == "gpt-4"
        finally:
            temp_path.unlink()

    def test_merge_with_overrides_existing_role(self):
        """Test merging overrides into existing role."""
        base_config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4")
            }
        )

        overrides = {
            "red_model": "gpt-4-turbo"  # CLI-style override
        }

        merged = RoleConfigLoader.merge_with_overrides(base_config, overrides)

        assert merged.roles["red"].model == "gpt-4-turbo"
        assert merged.roles["red"].agent == "react"

    def test_merge_with_overrides_new_role(self):
        """Test merging overrides creates new role."""
        base_config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4")
            }
        )

        overrides = {
            "blue_agent": "cot",  # CLI-style overrides
            "blue_model": "claude-3"
        }

        merged = RoleConfigLoader.merge_with_overrides(base_config, overrides)

        assert len(merged.roles) == 2
        assert merged.roles["blue"].agent == "cot"
        assert merged.roles["blue"].model == "claude-3"

    def test_merge_with_overrides_partial_fields(self):
        """Test merging overrides with partial field updates."""
        base_config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4", attempts=3)
            }
        )

        overrides = {
            "red_attempts": 5  # Only override attempts (CLI-style)
        }

        merged = RoleConfigLoader.merge_with_overrides(base_config, overrides)

        assert merged.roles["red"].model == "gpt-4"  # Preserved
        assert merged.roles["red"].agent == "react"  # Preserved
        assert merged.roles["red"].attempts == 5  # Updated


class TestSABERConfigIntegration:
    """Test SABERConfig integration with role-based config."""

    def test_saber_config_with_role_config(self):
        """Test SABERConfig can store and use role_config."""
        role_config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4", attempts=3),
                "blue": RoleAgentConfig(agent="cot", model="claude-3", attempts=2)
            }
        )

        saber_config = SABERConfig.create(
            model="gpt-4o-mini",
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            agents=[AgentAssignment(id="react", tasks=["task1"])]
        )

        saber_config.role_config = role_config

        assert saber_config.role_config is not None
        assert len(saber_config.role_config.roles) == 2

    def test_get_agent_assignments_merges_explicit_and_roles(self):
        """Test get_agent_assignments merges explicit and role-based assignments."""
        role_config = RoleBasedConfig(
            roles={
                "red": RoleAgentConfig(agent="react", model="gpt-4", attempts=3),
                "blue": RoleAgentConfig(agent="cot", model="claude-3", attempts=2)
            }
        )

        saber_config = SABERConfig.create(
            model="gpt-4o-mini",
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            agents=[
                AgentAssignment(id="react", tasks=["task1", "task2"])
            ]
        )

        saber_config.role_config = role_config

        all_assignments = saber_config.get_agent_assignments()

        # Should have 1 explicit + 2 role-based = 3 total
        assert len(all_assignments) == 3

        # Check explicit assignment is included
        explicit = next(a for a in all_assignments if a.role is None)
        assert explicit.id == "react"
        assert explicit.tasks == ["task1", "task2"]

        # Check role-based assignments are included
        role_assignments = [a for a in all_assignments if a.role is not None]
        assert len(role_assignments) == 2

        red_assignment = next(a for a in role_assignments if a.role == "red")
        assert red_assignment.id == "react"
        assert red_assignment.attempts == 3

    def test_get_agent_assignments_without_role_config(self):
        """Test get_agent_assignments returns only explicit when no role_config."""
        saber_config = SABERConfig.create(
            model="gpt-4o-mini",
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            agents=[
                AgentAssignment(id="react", tasks=["task1"])
            ]
        )

        all_assignments = saber_config.get_agent_assignments()

        assert len(all_assignments) == 1
        assert all_assignments[0].id == "react"


class TestAgentCompositeKey:
    """Test AgentCompositeKey type-safe composite keys."""

    def test_create_without_role(self):
        """Test creating composite key without role."""
        key = AgentCompositeKey.create(
            agent_id="react",
            tasks=["task1", "task2"]
        )

        assert key.agent_id == "react"
        assert key.role is None
        assert len(key.tasks_hash) == 8
        assert "|" in key.to_string()
        # Without role, should only have 1 pipe separator
        assert key.to_string().count("|") == 1

    def test_create_with_role(self):
        """Test creating composite key with role."""
        key = AgentCompositeKey.create(
            agent_id="react",
            tasks=["task1"],
            role="red"
        )

        assert key.agent_id == "react"
        assert key.role == "red"
        assert key.to_string().count("|") == 2

    def test_from_string_without_role(self):
        """Test parsing composite key string without role."""
        key_str = "react|a1b2c3d4"
        key = AgentCompositeKey.from_string(key_str)

        assert key.agent_id == "react"
        assert key.tasks_hash == "a1b2c3d4"
        assert key.role is None

    def test_from_string_with_role(self):
        """Test parsing composite key string with role."""
        key_str = "react|red|a1b2c3d4"
        key = AgentCompositeKey.from_string(key_str)

        assert key.agent_id == "react"
        assert key.role == "red"
        assert key.tasks_hash == "a1b2c3d4"

    def test_from_string_invalid_format(self):
        """Test parsing invalid composite key raises ValueError."""
        with pytest.raises(ValueError, match="Invalid composite key format"):
            AgentCompositeKey.from_string("invalid")

        with pytest.raises(ValueError, match="Invalid composite key format"):
            AgentCompositeKey.from_string("too|many|parts|here")

    def test_agent_id_with_underscores(self):
        """Test that agent IDs with underscores work correctly."""
        key = AgentCompositeKey.create(
            agent_id="custom_react_v2",
            tasks=["task1"],
            role="blue"
        )

        key_str = key.to_string()
        parsed = AgentCompositeKey.from_string(key_str)

        assert parsed.agent_id == "custom_react_v2"
        assert parsed.role == "blue"

    def test_hash_consistency(self):
        """Test that same tasks produce same hash."""
        key1 = AgentCompositeKey.create("react", ["task2", "task1"])
        key2 = AgentCompositeKey.create("react", ["task1", "task2"])

        # Should be same because tasks are sorted
        assert key1.tasks_hash == key2.tasks_hash

    def test_roundtrip(self):
        """Test that to_string and from_string roundtrip correctly."""
        original = AgentCompositeKey.create(
            agent_id="my_custom_agent",
            tasks=["task1", "task2", "task3"],
            role="attacker"
        )

        key_str = original.to_string()
        parsed = AgentCompositeKey.from_string(key_str)

        assert parsed.agent_id == original.agent_id
        assert parsed.role == original.role
        assert parsed.tasks_hash == original.tasks_hash


class TestRoleConfigValidation:
    """Test validation improvements for role configuration."""

    def test_role_config_requires_roles_or_defaults(self):
        """Test that RoleBasedConfig requires at least roles or defaults."""
        with pytest.raises(ValueError, match="must have at least one role"):
            RoleBasedConfig(roles={}, defaults=None)

    def test_get_config_for_role_missing_role_no_defaults(self):
        """Test get_config_for_role raises ValueError when role not found and no defaults."""
        config = RoleBasedConfig(
            roles={"red": RoleAgentConfig(agent="react", model="gpt-4")}
        )

        with pytest.raises(ValueError, match="Role 'unknown' not found"):
            config.get_config_for_role("unknown")

    def test_get_config_for_role_with_defaults_fallback(self):
        """Test get_config_for_role falls back to defaults for missing role."""
        defaults = RoleAgentConfig(agent="react", model="gpt-4o-mini", attempts=1)
        config = RoleBasedConfig(
            roles={"red": RoleAgentConfig(agent="react", model="gpt-4")},
            defaults=defaults
        )

        # Should not raise, should return defaults
        result = config.get_config_for_role("unknown")
        assert result.model == "gpt-4o-mini"
        assert result.attempts == 1

    def test_merge_with_overrides_unknown_field_strict(self):
        """Test merge_with_overrides raises error on unknown field when strict."""
        from saber.client.config_loader import RoleConfigLoader
        from saber.client.exceptions import InvalidRoleOverrideError

        base = RoleBasedConfig(
            roles={"red": RoleAgentConfig(agent="react")}
        )

        overrides = {"red_unknown_field": "value"}

        with pytest.raises(InvalidRoleOverrideError, match="Unknown field"):
            RoleConfigLoader.merge_with_overrides(base, overrides, strict=True)

    def test_merge_with_overrides_unknown_field_non_strict(self):
        """Test merge_with_overrides skips unknown field when not strict."""
        from saber.client.config_loader import RoleConfigLoader

        base = RoleBasedConfig(
            roles={"red": RoleAgentConfig(agent="react", model="gpt-4")}
        )

        overrides = {"red_unknown_field": "value", "red_model": "claude-3"}

        # Should not raise, should skip unknown field
        merged = RoleConfigLoader.merge_with_overrides(base, overrides, strict=False)

        assert merged.roles["red"].model == "claude-3"
        assert not hasattr(merged.roles["red"], "unknown_field")
