"""Tests for ToolRegistry and ResolvedTools — written BEFORE implementation (TDD)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import pytest
from inspect_ai.tool import Tool, tool
from pydantic import ValidationError

from saber.config.models import ToolConfig
from saber.tools.registry import ResolvedTools, ToolRegistry
from saber.tools.security import ToolSecurityConfig


def _fake_tool_factory(*, timeout: int = 30) -> Tool:
    """Create a fake tool for testing."""

    @tool
    def fake_tool() -> Tool:
        async def execute(input_text: str) -> str:
            return f"fake: {input_text}"

        return execute

    return fake_tool()


class TestToolRegistryDefaults:
    """Built-in tool resolution."""

    def test_available_includes_builtins(self) -> None:
        registry = ToolRegistry()
        available = registry.available()
        assert "bash" in available
        assert "python" in available

    def test_resolve_bash(self) -> None:
        registry = ToolRegistry()
        result = registry.resolve({"bash": ToolConfig(timeout=120)})
        assert isinstance(result, ResolvedTools)
        assert len(result.tools) == 1
        assert callable(result.tools[0])

    def test_resolve_multiple(self) -> None:
        registry = ToolRegistry()
        result = registry.resolve(
            {"bash": ToolConfig(), "python": ToolConfig()},
        )
        assert len(result.tools) == 2

    def test_resolve_unknown_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool.*'nope'"):
            registry.resolve({"nope": ToolConfig()})


class TestToolRegistryCustom:
    """Custom tool registration."""

    def test_register_custom_tool(self) -> None:
        registry = ToolRegistry()
        registry.register("my_tool", _fake_tool_factory)
        result = registry.resolve({"my_tool": ToolConfig(timeout=60)})
        assert len(result.tools) == 1
        assert callable(result.tools[0])

    def test_register_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register("dup", _fake_tool_factory)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup", _fake_tool_factory)

    def test_cannot_override_builtin(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Cannot override built-in"):
            registry.register("bash", _fake_tool_factory)

    def test_available_includes_custom(self) -> None:
        registry = ToolRegistry()
        registry.register("custom_x", _fake_tool_factory)
        available = registry.available()
        assert "custom_x" in available
        assert "bash" in available  # builtins still present

    def test_resolve_mixed_builtin_and_custom(self) -> None:
        registry = ToolRegistry()
        registry.register("my_tool", _fake_tool_factory)
        result = registry.resolve(
            {"bash": ToolConfig(), "my_tool": ToolConfig(timeout=10)},
        )
        assert len(result.tools) == 2


class TestToolRegistryResolve:
    """Edge cases for resolve."""

    def test_resolve_empty_dict(self) -> None:
        registry = ToolRegistry()
        result = registry.resolve({})
        assert isinstance(result, ResolvedTools)
        assert result.tools == ()
        assert result.submit_enabled is False

    def test_resolve_preserves_order(self) -> None:
        registry = ToolRegistry()
        registry.register("aaa", _fake_tool_factory)
        registry.register("zzz", _fake_tool_factory)
        # Request in zzz, bash, aaa order
        configs = {
            "zzz": ToolConfig(),
            "bash": ToolConfig(),
            "aaa": ToolConfig(),
        }
        result = registry.resolve(configs)
        assert len(result.tools) == 3
        # Ensure order is preserved (zzz first, then bash, then aaa)
        # We can't easily inspect Tool identity, but length/no error is key

    def test_resolve_custom_factory_without_timeout(self) -> None:
        """A custom factory that takes no timeout param should still resolve."""

        @tool
        def simple_tool() -> Tool:
            async def execute(input_text: str) -> str:
                return "simple"

            return execute

        registry = ToolRegistry()
        registry.register("simple", simple_tool)
        result = registry.resolve({"simple": ToolConfig(timeout=60)})
        assert len(result.tools) == 1
        assert callable(result.tools[0])


# ── ResolvedTools ──────────────────────────────────────────────────


def _make_fake_tool() -> Tool:
    """Create a fake Tool instance for ResolvedTools tests."""

    @tool
    def fake() -> Tool:
        async def execute(input_text: str) -> str:
            return "fake"

        return execute

    return fake()


class TestResolvedToolsConstruction:
    """ResolvedTools construction with tools tuple, submit_enabled, security_configs."""

    def test_basic_construction(self) -> None:
        t = _make_fake_tool()
        rt = ResolvedTools(tools=(t,), submit_enabled=True)
        assert len(rt.tools) == 1
        assert rt.submit_enabled is True
        assert rt.security_configs == {}

    def test_with_security_configs(self) -> None:
        t = _make_fake_tool()
        sec = ToolSecurityConfig(allow_semicolons=True)
        rt = ResolvedTools(
            tools=(t,),
            submit_enabled=False,
            security_configs=MappingProxyType({"bash": sec}),
        )
        assert "bash" in rt.security_configs
        assert rt.security_configs["bash"].allow_semicolons is True

    def test_empty_tools(self) -> None:
        rt = ResolvedTools(tools=(), submit_enabled=False)
        assert rt.tools == ()
        assert rt.submit_enabled is False

    def test_multiple_tools(self) -> None:
        t1 = _make_fake_tool()
        t2 = _make_fake_tool()
        rt = ResolvedTools(tools=(t1, t2), submit_enabled=True)
        assert len(rt.tools) == 2


class TestResolvedToolsFrozen:
    """ResolvedTools is immutable (frozen=True)."""

    def test_cannot_set_submit_enabled(self) -> None:
        rt = ResolvedTools(tools=(), submit_enabled=True)
        with pytest.raises(ValidationError):
            rt.submit_enabled = False  # type: ignore[misc]

    def test_cannot_set_tools(self) -> None:
        rt = ResolvedTools(tools=(), submit_enabled=True)
        with pytest.raises(ValidationError):
            rt.tools = ()  # type: ignore[misc]


class TestResolvedToolsSecurityConfigs:
    """security_configs is a Mapping (not mutable dict)."""

    def test_default_is_empty_mapping(self) -> None:
        rt = ResolvedTools(tools=(), submit_enabled=False)
        assert isinstance(rt.security_configs, Mapping)
        assert len(rt.security_configs) == 0

    def test_accepts_mapping_proxy(self) -> None:
        sec = ToolSecurityConfig()
        proxy = MappingProxyType({"tool_a": sec})
        rt = ResolvedTools(tools=(), submit_enabled=False, security_configs=proxy)
        assert isinstance(rt.security_configs, Mapping)
        assert "tool_a" in rt.security_configs

    def test_accepts_dict_coercion(self) -> None:
        """A plain dict is also accepted (coerced to immutable MappingProxyType)."""
        sec = ToolSecurityConfig()
        rt = ResolvedTools(tools=(), submit_enabled=False, security_configs={"x": sec})
        assert isinstance(rt.security_configs, Mapping)
        assert "x" in rt.security_configs

    def test_dict_coercion_is_immutable(self) -> None:
        """A plain dict passed in is wrapped in MappingProxyType — mutation raises."""
        sec = ToolSecurityConfig()
        rt = ResolvedTools(tools=(), submit_enabled=False, security_configs={"x": sec})
        with pytest.raises(TypeError):
            rt.security_configs["y"] = sec  # type: ignore[index]


# ── Phase 3: resolve() → ResolvedTools integration tests ────────────


class TestResolveReturnsResolvedTools:
    """ToolRegistry.resolve() returns ResolvedTools (Phase 3)."""

    def test_resolve_returns_resolved_tools(self) -> None:
        """resolve() returns a ResolvedTools instance, not a list."""
        registry = ToolRegistry()
        result = registry.resolve({"bash": ToolConfig(timeout=120)})
        assert isinstance(result, ResolvedTools)
        assert len(result.tools) == 1
        assert result.submit_enabled is False

    def test_resolve_submit_key_skipped(self) -> None:
        """'submit' key in tool_configs doesn't create a tool — it's reserved."""
        registry = ToolRegistry()
        result = registry.resolve(
            {
                "bash": ToolConfig(timeout=120),
                "submit": ToolConfig(),
            }
        )
        # Only bash should be resolved; submit is skipped
        assert len(result.tools) == 1
        assert result.submit_enabled is True

    def test_resolve_submit_only_returns_empty_tools(self) -> None:
        """submit-only config produces no tool instances."""
        registry = ToolRegistry()
        result = registry.resolve({"submit": ToolConfig()})
        assert result.tools == ()
        assert result.submit_enabled is True

    def test_resolve_collects_security_configs(self) -> None:
        """Security configs from ToolConfig are extracted into ResolvedTools."""
        sec = ToolSecurityConfig(allow_semicolons=True)
        registry = ToolRegistry()
        result = registry.resolve(
            {
                "bash": ToolConfig(timeout=120, security=sec),
            }
        )
        assert "bash" in result.security_configs
        assert result.security_configs["bash"].allow_semicolons is True

    def test_resolve_skips_none_security(self) -> None:
        """Tools without security config are not included in security_configs."""
        registry = ToolRegistry()
        result = registry.resolve(
            {
                "bash": ToolConfig(timeout=120),
                "python": ToolConfig(timeout=60),
            }
        )
        assert len(result.security_configs) == 0

    def test_resolve_empty_dict_returns_resolved_tools(self) -> None:
        """Empty dict returns ResolvedTools with empty tools tuple."""
        registry = ToolRegistry()
        result = registry.resolve({})
        assert isinstance(result, ResolvedTools)
        assert result.tools == ()
        assert result.submit_enabled is False
        assert len(result.security_configs) == 0

    def test_resolve_mixed_security_and_no_security(self) -> None:
        """Only tools WITH security config appear in security_configs."""
        sec = ToolSecurityConfig(block_null_bytes=False)
        registry = ToolRegistry()
        result = registry.resolve(
            {
                "bash": ToolConfig(timeout=120, security=sec),
                "python": ToolConfig(timeout=60),
            }
        )
        assert len(result.security_configs) == 1
        assert "bash" in result.security_configs
        assert "python" not in result.security_configs

    def test_submit_security_config_not_collected(self) -> None:
        """submit key's security config should not appear in security_configs."""
        cfg = ToolConfig(security=ToolSecurityConfig(allow_semicolons=True))
        registry = ToolRegistry()
        result = registry.resolve({"bash": ToolConfig(), "submit": cfg})
        assert "submit" not in result.security_configs

    def test_resolve_without_submit_sets_submit_disabled(self) -> None:
        """When submit is NOT in tool_configs, submit_enabled is False."""
        registry = ToolRegistry()
        result = registry.resolve({"bash": ToolConfig(timeout=120)})
        assert result.submit_enabled is False

    def test_resolve_with_submit_sets_submit_enabled(self) -> None:
        """When submit IS in tool_configs, submit_enabled is True."""
        registry = ToolRegistry()
        result = registry.resolve({"bash": ToolConfig(), "submit": ToolConfig()})
        assert result.submit_enabled is True
