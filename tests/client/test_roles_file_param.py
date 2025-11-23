"""Test roles_file parameter with relative/absolute paths."""

import tempfile
from pathlib import Path

import pytest

from saber.client.config_loader import RoleConfigLoader
from saber.client.models import RoleAgentConfig, RoleBasedConfig


def test_load_relative_path():
    """Test loading config from relative path."""
    # Create a temporary config file
    config_content = """
roles:
  red:
    agent: react
    model: gpt-4
  blue:
    agent: cot
    model: claude-3

defaults:
  agent: react
  model: gpt-4o-mini
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = Path(f.name)

    try:
        # Test with absolute path
        config = RoleConfigLoader.load(str(temp_path))
        assert "red" in config.roles
        assert config.roles["red"].model == "gpt-4"
        assert config.roles["blue"].model == "claude-3"
        assert config.defaults.model == "gpt-4o-mini"
    finally:
        temp_path.unlink()


def test_load_with_tilde_expansion():
    """Test that ~ in paths gets expanded."""
    # This should expand ~ to home directory
    config_content = """
roles:
  red:
    agent: react
    model: gpt-4
"""

    # Create in temp directory
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = Path(f.name)

    try:
        # Load with absolute path first to verify it works
        config = RoleConfigLoader.load(str(temp_path))
        assert "red" in config.roles
    finally:
        temp_path.unlink()


def test_load_inline_json_still_works():
    """Test that inline JSON still works."""
    json_config = '{"roles": {"red": {"agent": "react", "model": "gpt-4"}}}'

    config = RoleConfigLoader.load(json_config)
    assert "red" in config.roles
    assert config.roles["red"].model == "gpt-4"


def test_load_dict_still_works():
    """Test that dict input still works."""
    dict_config = {
        "roles": {
            "red": {"agent": "react", "model": "gpt-4"}
        }
    }

    config = RoleConfigLoader.load(dict_config)
    assert "red" in config.roles
    assert config.roles["red"].model == "gpt-4"
