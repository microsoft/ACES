# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for saber.tools.inspectors — ToolInspector, KeyBasedInspector, etc."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saber.tools.inspectors import (
    FallbackInspector,
    KeyBasedInspector,
    ToolArgumentKey,
    ToolInspector,
    ToolInspectorConfig,
    ToolInspectorRegistry,
    bash_inspector,
    python_inspector,
)

# ── KeyBasedInspector ───────────────────────────────────────────────


class TestKeyBasedInspector:
    """Tests for the KeyBasedInspector class."""

    def test_extracts_single_key(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({"cmd": "ls -la"}) == ["ls -la"]

    def test_extracts_multiple_keys(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd", "command"))
        result = inspector.extract_content({"cmd": "ls", "command": "pwd"})
        assert result == ["ls", "pwd"]

    def test_first_matching_key_only(self) -> None:
        """When only one key matches, only that value is returned."""
        inspector = KeyBasedInspector(keys=("cmd", "command"))
        assert inspector.extract_content({"command": "ls"}) == ["ls"]

    def test_empty_when_no_matching_key(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({"other": "value"}) == []

    def test_skips_non_string_values(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({"cmd": 42}) == []

    def test_skips_empty_strings(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({"cmd": ""}) == []

    def test_skips_none_values(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({"cmd": None}) == []

    def test_empty_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            KeyBasedInspector(keys=())

    def test_keys_property(self) -> None:
        inspector = KeyBasedInspector(keys=("a", "b"))
        assert inspector.keys == ("a", "b")

    def test_skips_bool_values(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({"cmd": True}) == []

    def test_skips_float_values(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({"cmd": 3.14}) == []

    def test_preserves_key_order(self) -> None:
        """Results appear in the order keys are defined, not dict order."""
        inspector = KeyBasedInspector(keys=("b", "a"))
        result = inspector.extract_content({"a": "second", "b": "first"})
        assert result == ["first", "second"]

    def test_empty_arguments_dict(self) -> None:
        inspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({}) == []


# ── Factory Functions ───────────────────────────────────────────────


class TestBashInspector:
    """Tests for the bash_inspector factory function."""

    def test_extracts_cmd(self) -> None:
        inspector = bash_inspector()
        assert inspector.extract_content({"cmd": "ls -la"}) == ["ls -la"]

    def test_extracts_command_key(self) -> None:
        inspector = bash_inspector()
        assert inspector.extract_content({"command": "ls -la"}) == ["ls -la"]

    def test_both_keys_present(self) -> None:
        inspector = bash_inspector()
        result = inspector.extract_content({"cmd": "ls", "command": "pwd"})
        assert result == ["ls", "pwd"]

    def test_empty_arguments(self) -> None:
        inspector = bash_inspector()
        assert inspector.extract_content({}) == []

    def test_keys_are_cmd_and_command(self) -> None:
        inspector = bash_inspector()
        assert inspector.keys == ("cmd", "command")


class TestPythonInspector:
    """Tests for the python_inspector factory function."""

    def test_extracts_code(self) -> None:
        inspector = python_inspector()
        assert inspector.extract_content({"code": "print('hi')"}) == ["print('hi')"]

    def test_keys_are_code(self) -> None:
        inspector = python_inspector()
        assert inspector.keys == ("code",)


# ── FallbackInspector ───────────────────────────────────────────────


class TestFallbackInspector:
    """Tests for the FallbackInspector class."""

    def test_extracts_all_string_values(self) -> None:
        inspector = FallbackInspector()
        result = inspector.extract_content({"a": "hello", "b": 42, "c": "world"})
        assert set(result) == {"hello", "world"}

    def test_skips_non_string_values(self) -> None:
        inspector = FallbackInspector()
        assert inspector.extract_content({"a": 42, "b": True, "c": None}) == []

    def test_skips_empty_strings(self) -> None:
        inspector = FallbackInspector()
        assert inspector.extract_content({"a": "", "b": "hello"}) == ["hello"]

    def test_empty_dict(self) -> None:
        inspector = FallbackInspector()
        assert inspector.extract_content({}) == []

    def test_all_strings(self) -> None:
        inspector = FallbackInspector()
        result = inspector.extract_content({"x": "one", "y": "two", "z": "three"})
        assert len(result) == 3

    def test_skips_float_values(self) -> None:
        inspector = FallbackInspector()
        assert inspector.extract_content({"a": 1.5}) == []


# ── ToolInspectorRegistry ──────────────────────────────────────────


class TestToolInspectorRegistry:
    """Tests for the ToolInspectorRegistry class."""

    def test_bash_registered_by_default(self) -> None:
        registry = ToolInspectorRegistry()
        inspector = registry.get("bash")
        assert isinstance(inspector, KeyBasedInspector)
        assert inspector.keys == ("cmd", "command")

    def test_python_registered_by_default(self) -> None:
        registry = ToolInspectorRegistry()
        inspector = registry.get("python")
        assert isinstance(inspector, KeyBasedInspector)
        assert inspector.keys == ("code",)

    def test_unknown_tool_gets_fallback(self) -> None:
        registry = ToolInspectorRegistry()
        inspector = registry.get("unknown_tool")
        assert isinstance(inspector, FallbackInspector)

    def test_custom_registration(self) -> None:
        registry = ToolInspectorRegistry()
        custom = KeyBasedInspector(keys=("kql_query", "query"))
        registry.register("kql", custom)
        assert registry.get("kql") is custom

    def test_register_from_config(self) -> None:
        config = ToolInspectorConfig(tool_name="kql", argument_keys=("query", "kql_query"))
        registry = ToolInspectorRegistry()
        registry.register_from_config(config)
        inspector = registry.get("kql")
        assert isinstance(inspector, KeyBasedInspector)
        content = inspector.extract_content({"kql_query": "traces | take 5"})
        assert content == ["traces | take 5"]

    def test_register_from_config_overrides_existing(self) -> None:
        config = ToolInspectorConfig(tool_name="bash", argument_keys=("script",))
        registry = ToolInspectorRegistry()
        registry.register_from_config(config)
        inspector = registry.get("bash")
        assert isinstance(inspector, KeyBasedInspector)
        assert inspector.keys == ("script",)

    def test_registered_tools_property(self) -> None:
        registry = ToolInspectorRegistry()
        assert "bash" in registry.registered_tools
        assert "python" in registry.registered_tools

    def test_registered_tools_includes_custom(self) -> None:
        registry = ToolInspectorRegistry()
        registry.register("kql", KeyBasedInspector(keys=("query",)))
        assert "kql" in registry.registered_tools

    def test_registered_tools_returns_frozenset(self) -> None:
        registry = ToolInspectorRegistry()
        assert isinstance(registry.registered_tools, frozenset)

    def test_fallback_is_shared_instance(self) -> None:
        """All unknown tools return the same FallbackInspector instance."""
        registry = ToolInspectorRegistry()
        a = registry.get("tool_a")
        b = registry.get("tool_b")
        assert a is b


# ── ToolArgumentKey ────────────────────────────────────────────────


class TestToolArgumentKey:
    """Tests for the ToolArgumentKey enum."""

    def test_cmd_value(self) -> None:
        assert ToolArgumentKey.CMD == "cmd"

    def test_code_value(self) -> None:
        assert ToolArgumentKey.CODE == "code"

    def test_command_value(self) -> None:
        assert ToolArgumentKey.COMMAND == "command"

    def test_input_value(self) -> None:
        assert ToolArgumentKey.INPUT == "input"


# ── ToolInspectorConfig ───────────────────────────────────────────


class TestToolInspectorConfig:
    """Tests for the ToolInspectorConfig model."""

    def test_construction(self) -> None:
        config = ToolInspectorConfig(tool_name="kql", argument_keys=("query",))
        assert config.tool_name == "kql"
        assert config.argument_keys == ("query",)

    def test_frozen(self) -> None:
        config = ToolInspectorConfig(tool_name="kql", argument_keys=("query",))
        with pytest.raises((TypeError, ValueError)):
            config.tool_name = "other"  # type: ignore[misc]

    def test_empty_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolInspectorConfig(tool_name="kql", argument_keys=())


# ── Protocol Compliance ────────────────────────────────────────────


class TestProtocolCompliance:
    """Verify both inspector types satisfy the ToolInspector protocol."""

    def test_key_based_satisfies_protocol(self) -> None:
        inspector: ToolInspector = KeyBasedInspector(keys=("cmd",))
        assert inspector.extract_content({"cmd": "ls"}) == ["ls"]

    def test_fallback_satisfies_protocol(self) -> None:
        inspector: ToolInspector = FallbackInspector()
        assert inspector.extract_content({"cmd": "ls"}) == ["ls"]
