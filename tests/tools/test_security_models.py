# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for ToolSecurityConfig — written BEFORE implementation (TDD)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saber.tools.security import ToolSecurityConfig


class TestToolSecurityConfigDefaults:
    """ToolSecurityConfig constructed with all defaults."""

    def test_default_construction(self) -> None:
        cfg = ToolSecurityConfig()
        assert cfg.blocked_commands == frozenset()
        assert cfg.allowed_commands == frozenset()
        assert cfg.dangerous_patterns == ()
        assert cfg.semicolon_patterns == ()
        assert cfg.max_content_length == 4096
        assert cfg.allow_semicolons is False
        assert cfg.block_null_bytes is True
        assert cfg.block_control_chars is True
        assert cfg.block_path_traversal is True

    def test_default_sensitive_directories(self) -> None:
        cfg = ToolSecurityConfig()
        assert cfg.sensitive_directories == ("/etc", "/root", "/home/root", "/var/lib", "/sys", "/proc")

    def test_default_safe_path_prefixes(self) -> None:
        cfg = ToolSecurityConfig()
        assert cfg.safe_path_prefixes == ("/tmp", "/var/tmp", "/home", "/workspace")

    def test_extend_defaults_true_by_default(self) -> None:
        cfg = ToolSecurityConfig()
        assert cfg.extend_defaults is True


class TestToolSecurityConfigFrozen:
    """ToolSecurityConfig is immutable (frozen=True)."""

    def test_cannot_set_allow_semicolons(self) -> None:
        cfg = ToolSecurityConfig()
        with pytest.raises(ValidationError):
            cfg.allow_semicolons = True  # type: ignore[misc]

    def test_cannot_set_blocked_commands(self) -> None:
        cfg = ToolSecurityConfig()
        with pytest.raises(ValidationError):
            cfg.blocked_commands = frozenset({"rm"})  # type: ignore[misc]

    def test_cannot_set_max_content_length(self) -> None:
        cfg = ToolSecurityConfig()
        with pytest.raises(ValidationError):
            cfg.max_content_length = 999  # type: ignore[misc]


class TestToolSecurityConfigFrozensetCoercion:
    """frozenset fields accept lists (e.g. from YAML) and coerce to frozenset."""

    def test_blocked_commands_from_list(self) -> None:
        cfg = ToolSecurityConfig(blocked_commands=["rm", "dd"])  # type: ignore[arg-type]
        assert isinstance(cfg.blocked_commands, frozenset)
        assert cfg.blocked_commands == frozenset({"rm", "dd"})

    def test_allowed_commands_from_list(self) -> None:
        cfg = ToolSecurityConfig(allowed_commands=["ls", "cat"])  # type: ignore[arg-type]
        assert isinstance(cfg.allowed_commands, frozenset)
        assert cfg.allowed_commands == frozenset({"ls", "cat"})

    def test_blocked_commands_from_set(self) -> None:
        cfg = ToolSecurityConfig(blocked_commands={"rm", "dd"})  # type: ignore[arg-type]
        assert isinstance(cfg.blocked_commands, frozenset)
        assert cfg.blocked_commands == frozenset({"rm", "dd"})


class TestToolSecurityConfigTupleCoercion:
    """tuple fields accept lists (e.g. from YAML) and coerce to tuples."""

    def test_dangerous_patterns_from_list(self) -> None:
        cfg = ToolSecurityConfig(dangerous_patterns=["rm -rf", "dd if="])  # type: ignore[arg-type]
        assert isinstance(cfg.dangerous_patterns, tuple)
        assert cfg.dangerous_patterns == ("rm -rf", "dd if=")

    def test_semicolon_patterns_from_list(self) -> None:
        cfg = ToolSecurityConfig(semicolon_patterns=[";", "&&"])  # type: ignore[arg-type]
        assert isinstance(cfg.semicolon_patterns, tuple)
        assert cfg.semicolon_patterns == (";", "&&")

    def test_sensitive_directories_from_list(self) -> None:
        cfg = ToolSecurityConfig(sensitive_directories=["/custom"])  # type: ignore[arg-type]
        assert isinstance(cfg.sensitive_directories, tuple)
        assert cfg.sensitive_directories == ("/custom",)

    def test_safe_path_prefixes_from_list(self) -> None:
        cfg = ToolSecurityConfig(safe_path_prefixes=["/app"])  # type: ignore[arg-type]
        assert isinstance(cfg.safe_path_prefixes, tuple)
        assert cfg.safe_path_prefixes == ("/app",)


class TestToolSecurityConfigCustomValues:
    """Custom values override defaults."""

    def test_custom_max_content_length(self) -> None:
        cfg = ToolSecurityConfig(max_content_length=8192)
        assert cfg.max_content_length == 8192

    def test_allow_semicolons_enabled(self) -> None:
        cfg = ToolSecurityConfig(allow_semicolons=True)
        assert cfg.allow_semicolons is True

    def test_block_null_bytes_disabled(self) -> None:
        cfg = ToolSecurityConfig(block_null_bytes=False)
        assert cfg.block_null_bytes is False

    def test_block_path_traversal_disabled(self) -> None:
        cfg = ToolSecurityConfig(block_path_traversal=False)
        assert cfg.block_path_traversal is False

    def test_extend_defaults_false(self) -> None:
        cfg = ToolSecurityConfig(extend_defaults=False)
        assert cfg.extend_defaults is False

    def test_custom_sensitive_directories(self) -> None:
        cfg = ToolSecurityConfig(sensitive_directories=("/custom/dir",))
        assert cfg.sensitive_directories == ("/custom/dir",)

    def test_custom_safe_path_prefixes(self) -> None:
        cfg = ToolSecurityConfig(safe_path_prefixes=("/app", "/data"))
        assert cfg.safe_path_prefixes == ("/app", "/data")


class TestToolSecurityConfigValidation:
    """Validation constraints."""

    def test_max_content_length_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="max_content_length"):
            ToolSecurityConfig(max_content_length=0)

    def test_max_content_length_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="max_content_length"):
            ToolSecurityConfig(max_content_length=-1)


class TestToolSecurityConfigSerialization:
    """Serialization round-trip."""

    def test_round_trip_defaults(self) -> None:
        cfg = ToolSecurityConfig()
        dumped = cfg.model_dump()
        restored = ToolSecurityConfig(**dumped)
        assert restored == cfg

    def test_round_trip_custom(self) -> None:
        cfg = ToolSecurityConfig(
            blocked_commands=frozenset({"rm", "dd"}),
            allowed_commands=frozenset({"ls"}),
            dangerous_patterns=("rm -rf",),
            semicolon_patterns=(";",),
            max_content_length=2048,
            allow_semicolons=True,
            block_null_bytes=False,
            block_control_chars=False,
            block_path_traversal=False,
            sensitive_directories=("/custom",),
            safe_path_prefixes=("/app",),
            extend_defaults=False,
        )
        dumped = cfg.model_dump()
        restored = ToolSecurityConfig(**dumped)
        assert restored == cfg
