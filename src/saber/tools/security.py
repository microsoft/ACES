# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Per-tool security configuration, validation, and approval.

Provides :class:`ToolSecurityConfig` (frozen Pydantic model),
:class:`ValidationResult`, :class:`CommandSecurityValidator`,
default security constants, :func:`default_security_config`,
and :func:`build_tool_approval`.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path, PurePosixPath

from inspect_ai.approval import Approval, ApprovalPolicy, Approver, approver
from inspect_ai.model import ChatMessage
from inspect_ai.tool import ToolCall, ToolCallView
from pydantic import BaseModel, ConfigDict, Field

from saber.logging import get_logger
from saber.tools.inspectors import ToolInspectorRegistry

logger = get_logger(__name__)

# ── ToolSecurityConfig ──────────────────────────────────────────────


class ToolSecurityConfig(BaseModel):
    """Per-tool security configuration.

    Security is OFF by default — tools without a ``security:`` block
    are unvalidated. An explicit (even empty) ``security: {}`` block
    enables validation with all defaults.
    """

    model_config = ConfigDict(frozen=True)

    blocked_commands: frozenset[str] = Field(default_factory=frozenset)
    allowed_commands: frozenset[str] = Field(default_factory=frozenset)
    dangerous_patterns: tuple[str, ...] = Field(default_factory=tuple)
    semicolon_patterns: tuple[str, ...] = Field(default_factory=tuple)
    max_content_length: int = Field(default=4096, gt=0)
    allow_semicolons: bool = False
    block_null_bytes: bool = True
    block_control_chars: bool = True
    block_path_traversal: bool = True
    sensitive_directories: tuple[str, ...] = (
        "/etc",
        "/root",
        "/home/root",
        "/var/lib",
        "/sys",
        "/proc",
    )
    safe_path_prefixes: tuple[str, ...] = ("/tmp", "/var/tmp", "/home", "/workspace")
    extend_defaults: bool = True


# ── ValidationResult ────────────────────────────────────────────────


class ValidationResult(BaseModel):
    """Result of validating a command/content string against a security config."""

    model_config = ConfigDict(frozen=True)

    is_safe: bool
    reason: str | None = None
    matched_pattern: str | None = None
    blocked_command: str | None = None


# ── Private Constants ───────────────────────────────────────────────

# Characters < 32 that are permitted in content.
_ALLOWED_CONTROL_CHARS: frozenset[str] = frozenset({"\t", "\n", "\r"})

# Commands that wrap/proxy other commands — the validator scans past these
# to also check the real command against the blocklist.
# NOTE: ``sudo`` also appears in ``_DEFAULT_BLOCKED_COMMANDS``. When
# present at position 0 it is blocked immediately (step 4a).  The proxy
# entry here only matters when ``sudo`` is allowed by ``allowed_commands``
# and appears as a wrapper around another command.
_COMMAND_PROXIES: frozenset[str] = frozenset(
    {
        "env",
        "nice",
        "nohup",
        "timeout",
        "stdbuf",
        "command",
        "xargs",
        "sudo",
        "doas",
        "su",
    }
)

# Matches numeric arguments optionally followed by a time suffix (s/m/h/d).
_NUMERIC_ARG_RE: re.Pattern[str] = re.compile(r"^\d+(\.\d+)?[smhd]?$")

# Flags for proxy commands that take a separate argument value.
_PROXY_FLAGS_WITH_VALUE: frozenset[str] = frozenset(
    {
        "-s",
        "--signal",
        "-k",
        "--kill-after",
        "-u",
        "--unset",
        "-n",
        "--adjustment",
        "-i",
        "--input",
        "-o",
        "--output",
        "-e",
        "--error",
        "-g",
        "--group",
        "-C",
        "--close-from",
        # xargs flags that take a value
        "-I",
        "--replace",
        "-P",
        "--max-procs",
    }
)

# Maximum allowed length for a single regex pattern string.
_MAX_PATTERN_LENGTH: int = 500


# ── CommandSecurityValidator ────────────────────────────────────────


class CommandSecurityValidator:
    """Validates command/content strings against a :class:`ToolSecurityConfig`.

    The validation pipeline runs these checks in order:

    1. Content length
    2. Null bytes
    3. Control characters
    4. Base-command blocklist (via ``shlex.split``)
    5. Dangerous-pattern regexes
    6. Semicolon injection
    7. Argument-level path-traversal + shell-metacharacter check
    """

    __slots__ = (
        "_config",
        "_compiled_patterns",
        "_compiled_semicolons",
        "_shell_metachar_pattern",
        "_effective_blocked",
    )

    def __init__(self, config: ToolSecurityConfig) -> None:
        self._config: ToolSecurityConfig = config
        self._compiled_patterns: tuple[re.Pattern[str], ...] = self._compile_patterns(config.dangerous_patterns)
        self._compiled_semicolons: tuple[re.Pattern[str], ...] = self._compile_patterns(config.semicolon_patterns)
        self._shell_metachar_pattern: re.Pattern[str] = re.compile(r"[&|`$]")
        self._effective_blocked: frozenset[str] = config.blocked_commands - config.allowed_commands

    @staticmethod
    def _compile_patterns(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
        """Compile regex patterns with length validation for ReDoS protection.

        Raises:
            ValueError: If any pattern exceeds ``_MAX_PATTERN_LENGTH``.
            re.error: If any pattern is not valid regex.
        """
        compiled: list[re.Pattern[str]] = []
        for p in patterns:
            if len(p) > _MAX_PATTERN_LENGTH:
                msg = f"Pattern too long ({len(p)} chars, max {_MAX_PATTERN_LENGTH}): {p[:50]}..."
                raise ValueError(msg)
            compiled.append(re.compile(p))
        return tuple(compiled)

    # ── public API ──────────────────────────────────────────────────

    def validate(self, content: str) -> ValidationResult:
        """Validate a content string against the security config.

        Args:
            content: The command or code string to validate.

        Returns:
            A :class:`ValidationResult` with ``is_safe=True`` if *content*
            passes every check, or ``is_safe=False`` with a reason otherwise.
        """
        # 1. Length check
        if len(content) > self._config.max_content_length:
            return ValidationResult(
                is_safe=False,
                reason=(f"Content exceeds {self._config.max_content_length} character limit"),
            )

        # 2. Null-byte check
        if self._config.block_null_bytes and "\x00" in content:
            return ValidationResult(is_safe=False, reason="Null bytes not allowed")

        # 3. Control-character check
        if self._config.block_control_chars:
            for ch in content:
                if ord(ch) < 32 and ch not in _ALLOWED_CONTROL_CHARS:
                    return ValidationResult(
                        is_safe=False,
                        reason="Content contains control characters",
                    )

        # 4. Base-command blocklist
        try:
            parts = shlex.split(content)
        except ValueError:
            return ValidationResult(
                is_safe=False,
                reason="Malformed command (unparseable shell syntax)",
            )

        if parts:
            base_cmd = Path(parts[0]).name
            if base_cmd in self._effective_blocked:
                return ValidationResult(
                    is_safe=False,
                    reason=f"Command '{base_cmd}' is blocked",
                    blocked_command=base_cmd,
                )

            # 4b. Scan past command proxies to check the real command.
            real_cmd = self._resolve_through_proxies(parts)
            if real_cmd is not None and real_cmd in self._effective_blocked:
                return ValidationResult(
                    is_safe=False,
                    reason=f"Command '{real_cmd}' is blocked (via proxy)",
                    blocked_command=real_cmd,
                )

        # 5. Dangerous-pattern check
        for compiled, raw in zip(self._compiled_patterns, self._config.dangerous_patterns, strict=True):
            if compiled.search(content):
                return ValidationResult(
                    is_safe=False,
                    reason="Content matches dangerous pattern",
                    matched_pattern=raw,
                )

        # 6. Semicolon injection
        if not self._config.allow_semicolons:
            if ";" in content:
                return ValidationResult(is_safe=False, reason="Semicolons not allowed")
            for compiled, raw in zip(self._compiled_semicolons, self._config.semicolon_patterns, strict=True):
                if compiled.search(content):
                    return ValidationResult(
                        is_safe=False,
                        reason="Dangerous semicolon pattern",
                        matched_pattern=raw,
                    )

        # 7. Path-traversal + argument metacharacter check
        if self._config.block_path_traversal and parts:
            for arg in parts[1:]:
                if ".." in arg or arg.startswith("/"):
                    if not self._is_safe_path(arg):
                        return ValidationResult(
                            is_safe=False,
                            reason=f"Unsafe path in argument: {arg}",
                        )
                if self._shell_metachar_pattern.search(arg):
                    return ValidationResult(
                        is_safe=False,
                        reason=f"Shell metacharacters in argument: {arg}",
                    )

        # All checks passed
        return ValidationResult(is_safe=True)

    # ── private helpers ─────────────────────────────────────────────

    def _is_safe_path(self, path_str: str) -> bool:
        """Return ``True`` if *path_str* is considered safe.

        Uses pure string-based normalization (no filesystem access)
        because commands execute inside Docker containers, not on the host.
        """
        parts = PurePosixPath(path_str).parts
        if ".." in parts:
            return False
        # Normalize without filesystem access
        normalized = str(PurePosixPath(path_str))
        for sensitive in self._config.sensitive_directories:
            if normalized == sensitive or normalized.startswith(sensitive + "/"):
                return False
        if PurePosixPath(path_str).is_absolute():
            if not any(normalized == p or normalized.startswith(p + "/") for p in self._config.safe_path_prefixes):
                return False
        return True

    @staticmethod
    def _resolve_through_proxies(parts: list[str]) -> str | None:
        """Skip past known command proxies to find the real command."""
        idx = 0
        while idx < len(parts):
            name = Path(parts[idx]).name
            if name not in _COMMAND_PROXIES:
                return name if idx > 0 else None
            idx += 1
            while idx < len(parts) and parts[idx].startswith("-"):
                flag = parts[idx]
                idx += 1
                if flag in _PROXY_FLAGS_WITH_VALUE and idx < len(parts) and not parts[idx].startswith("-"):
                    idx += 1
            if idx < len(parts) and _NUMERIC_ARG_RE.match(parts[idx]):
                idx += 1
            while idx < len(parts) and "=" in parts[idx] and not parts[idx].startswith("-"):
                idx += 1
        return None


# ── Default Security Constants ──────────────────────────────────────

_DEFAULT_BLOCKED_COMMANDS: frozenset[str] = frozenset(
    {
        # Privilege escalation
        "sudo",
        "su",
        "doas",
        # User/group management
        "passwd",
        "chpasswd",
        "adduser",
        "useradd",
        "userdel",
        "usermod",
        "groupadd",
        "groupdel",
        "groupmod",
        # File permission/ownership
        "chmod",
        "chown",
        "chgrp",
        # Destructive file operations
        "rm",
        "mv",
        "cp",
        # Filesystem mount
        "mount",
        "umount",
        # Disk management
        "fdisk",
        "parted",
        "mkfs",
        # Firewall / networking
        "iptables",
        "ufw",
        "firewall-cmd",
        # Service management
        "systemctl",
        "service",
        "chkconfig",
        # Scheduled tasks
        "crontab",
        "at",
        "batch",
        # Email
        "mail",
        "sendmail",
        # Network utilities
        "nc",
        "netcat",
        "socat",
        "ssh",
        "scp",
        "rsync",
        "curl",
        "wget",
        "lynx",
        # Scripting / interpreters
        "python",
        "python3",
        "perl",
        "ruby",
        "node",
        "php",
        "java",
        # Shells
        "bash",
        "sh",
        "zsh",
        "csh",
        "tcsh",
        "ksh",
        "dash",
        "fish",
        # Compilers / build tools
        "gcc",
        "g++",
        "make",
        "cmake",
        # Container / orchestration
        "docker",
        "podman",
        "kubectl",
        # VCS
        "git",
        # Process control
        "kill",
        "killall",
        "pkill",
        # Kernel modules
        "insmod",
        "rmmod",
        "modprobe",
        # Tracing / debugging
        "strace",
        "ltrace",
        "gdb",
        # Package managers
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "pip",
        "pip3",
        # Disk / swap
        "dd",
        "mkswap",
        "swapon",
        "swapoff",
        # Shutdown
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
    }
)

_DEFAULT_DANGEROUS_PATTERNS: tuple[str, ...] = (
    # Redirect to / from root paths
    r">\s*/",
    r"<\s*/",
    # Pipe to shell
    r"\|\s*sh\b",
    r"\|\s*bash\b",
    r"\|\s*zsh\b",
    r"\|\s*csh\b",
    # Download-and-execute
    r"curl\s+.*\|\s*sh",
    r"wget\s+.*\|\s*sh",
    # Reverse shells / exfiltration
    r"nc\s+.*-e",
    r"socat\s+.*exec",
    # Destructive commands with flags
    r"rm\s+.*-rf",
    r"mv\s+.*\s+/",
    r"cp\s+.*\s+/",
    # Dangerous permission changes
    r"chmod\s+777",
    r"chown\s+.*root",
    # Privilege escalation in-line
    r"sudo\s+",
    r"su\s+",
    # User management
    r"passwd\s+",
    r"adduser\s+",
    r"useradd\s+",
    r"usermod\s+",
    # Process killing
    r"kill\s+-9",
    r"killall\s+",
    r"pkill\s+",
    # Sensitive file access
    r"/etc/passwd",
    r"/etc/shadow",
    r"/root/",
    r"~root/",
    # Null byte injection
    r"\x00",
    # Path traversal
    r"\.\./\.\.",
    # Additional attack vectors
    r"eval\s+",
    r"exec\s+",
    r"source\s+/",
    r"\bbase64\s+(-d|--decode)",
    r"/dev/(tcp|udp)/",
    r"mkfifo\s+",
    r"nohup\s+",
    r"xargs\s+.*sh",
    r"find\s+.*-exec",
    r"awk\s+.*system",
    r"python[23]?\s+-c",
    r"perl\s+-e",
    r"ruby\s+-e",
    # Catch-all: any shell metacharacter. Intentionally aggressive —
    # blocks legitimate uses of pipes, redirects, etc. to prevent
    # injection.  Specific patterns above catch named attack vectors;
    # this line is the defense-in-depth backstop.
    r"[&|`$<>]",
    r"\$\(",
    r"`[^`]*`",
)

_DEFAULT_SEMICOLON_PATTERNS: tuple[str, ...] = (
    r";\s*rm\s+",
    r";\s*sudo\s+",
    r";\s*su\s+",
    r";\s*chmod\s+",
    r";\s*chown\s+",
    r";\s*kill\s+",
    r";\s*/bin/",
    r";\s*/usr/bin/",
    r";\s*\$\(",
    r";\s*`",
)


def default_security_config() -> ToolSecurityConfig:
    """Return a ToolSecurityConfig populated with all default security rules.

    Returns:
        A ToolSecurityConfig with 60+ blocked commands, 33+ dangerous patterns,
        and 10 semicolon patterns covering common attack vectors.
    """
    return ToolSecurityConfig(
        blocked_commands=_DEFAULT_BLOCKED_COMMANDS,
        dangerous_patterns=_DEFAULT_DANGEROUS_PATTERNS,
        semicolon_patterns=_DEFAULT_SEMICOLON_PATTERNS,
    )


# ── build_tool_approval ────────────────────────────────────────────

# Tools that contain no executable content — always approved.
_AUTO_APPROVE_TOOLS: frozenset[str] = frozenset({"think"})


def _merge_with_defaults(config: ToolSecurityConfig) -> ToolSecurityConfig:
    """Merge a ToolSecurityConfig with the default security constants.

    Combines blocked_commands (union), dangerous_patterns (union, dedup),
    and semicolon_patterns (union, dedup). Other fields come from the config.

    Args:
        config: The per-tool security config to merge.

    Returns:
        A new ToolSecurityConfig with defaults merged in.
    """
    defaults = default_security_config()
    return ToolSecurityConfig(
        blocked_commands=defaults.blocked_commands | config.blocked_commands,
        allowed_commands=defaults.allowed_commands | config.allowed_commands,
        dangerous_patterns=tuple(dict.fromkeys(defaults.dangerous_patterns + config.dangerous_patterns)),
        semicolon_patterns=tuple(dict.fromkeys(defaults.semicolon_patterns + config.semicolon_patterns)),
        max_content_length=config.max_content_length,
        allow_semicolons=config.allow_semicolons,
        block_null_bytes=config.block_null_bytes,
        block_control_chars=config.block_control_chars,
        block_path_traversal=config.block_path_traversal,
        sensitive_directories=tuple(dict.fromkeys(defaults.sensitive_directories + config.sensitive_directories)),
        safe_path_prefixes=tuple(dict.fromkeys(defaults.safe_path_prefixes + config.safe_path_prefixes)),
        extend_defaults=config.extend_defaults,
    )


def build_tool_approval(
    security_configs: dict[str, ToolSecurityConfig],
) -> list[ApprovalPolicy]:
    """Build inspect_ai ApprovalPolicy list from per-tool security configs.

    Creates an async approver closure that:
    - For each tool call, looks up the tool name in *security_configs*.
    - If no config exists for that tool → approve (security OFF by default).
    - If config exists → use :class:`CommandSecurityValidator` to validate.
    - Uses :class:`ToolInspectorRegistry` for content extraction.
    - Auto-approves tools in ``_AUTO_APPROVE_TOOLS`` (currently: ``think``).
    - If ``ToolSecurityConfig.extend_defaults`` is True, merges with defaults.

    Args:
        security_configs: Mapping of tool name → security config.

    Returns:
        A list containing a single :class:`ApprovalPolicy` wiring all tools
        through the security approver.
    """
    registry = ToolInspectorRegistry()

    # Build per-tool validators, applying extend_defaults merging
    tool_validators: dict[str, CommandSecurityValidator] = {}
    for tool_name, config in security_configs.items():
        effective = _merge_with_defaults(config) if config.extend_defaults else config
        tool_validators[tool_name] = CommandSecurityValidator(effective)

    @approver(name="saber_security")  # type: ignore[misc]
    def _make_approver() -> Approver:
        async def approve(
            message: str,
            call: ToolCall,
            view: ToolCallView,
            history: list[ChatMessage],
        ) -> Approval:
            # Auto-approve tools with no executable content
            if call.function in _AUTO_APPROVE_TOOLS:
                return Approval(decision="approve")

            # No config for this tool → security OFF (approve)
            if call.function not in tool_validators:
                return Approval(decision="approve")

            # Extract content to validate from the tool call
            inspector = registry.get(call.function)
            contents = inspector.extract_content(call.arguments)

            # If no inspectable content found, approve
            if not contents:
                return Approval(decision="approve")

            validator = tool_validators[call.function]

            # Validate each content string
            for content in contents:
                result = validator.validate(content)
                if not result.is_safe:
                    explanation_parts: list[str] = [f"Blocked: {result.reason}"]
                    if result.blocked_command:
                        explanation_parts.append(f"Command: {result.blocked_command}")
                    if result.matched_pattern:
                        explanation_parts.append(f"Pattern: {result.matched_pattern}")
                    logger.warning(
                        "Rejected tool call '%s' (id=%s): %s",
                        call.function,
                        call.id,
                        " | ".join(explanation_parts),
                    )
                    return Approval(
                        decision="reject",
                        explanation=" | ".join(explanation_parts),
                    )

            logger.debug("Approved tool call '%s' (id=%s)", call.function, call.id)
            return Approval(decision="approve")

        return approve

    return [ApprovalPolicy(approver=_make_approver(), tools="*")]
