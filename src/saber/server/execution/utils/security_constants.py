"""
Security constants for tool executors.

This module contains security-related constants used throughout the tool execution
framework to validate and block potentially dangerous commands and patterns.
"""

# Dangerous command patterns that should be blocked
DANGEROUS_PATTERNS = [
    # Command injection patterns
    r"[;&|`$()<>]",  # Shell metacharacters (including < and >)
    r"\$\(",  # Command substitution
    r"`[^`]*`",  # Backtick command substitution
    r">\s*/",  # Redirect to filesystem root
    r"<\s*/",  # Redirect from filesystem root
    r"\|\s*sh",  # Pipe to shell
    r"\|\s*bash",  # Pipe to bash
    r"\|\s*zsh",  # Pipe to zsh
    r"\|\s*csh",  # Pipe to csh
    # Network/remote execution
    r"curl\s+.*\|\s*sh",  # Curl pipe to shell
    r"wget\s+.*\|\s*sh",  # Wget pipe to shell
    r"nc\s+.*-e",  # Netcat with execute
    r"socat\s+.*exec",  # Socat with exec
    # File system manipulation
    r"rm\s+.*-rf",  # Recursive force delete
    r"mv\s+.*\s+/",  # Move to root
    r"cp\s+.*\s+/",  # Copy to root
    r"chmod\s+777",  # Dangerous permissions
    r"chown\s+.*root",  # Change to root ownership
    # System manipulation
    r"sudo\s+",  # Sudo commands
    r"su\s+",  # Switch user
    r"passwd\s+",  # Password changes
    r"adduser\s+",  # Add user
    r"useradd\s+",  # Add user
    r"usermod\s+",  # Modify user
    # Process manipulation
    r"kill\s+-9",  # Force kill
    r"killall\s+",  # Kill all processes
    r"pkill\s+",  # Process kill
    # System information/enumeration
    r"/etc/passwd",  # Password file
    r"/etc/shadow",  # Shadow file
    r"/root/",  # Root directory access
    r"~root/",  # Root home access
]

# Commands that should never be allowed
BLOCKED_COMMANDS = {
    "sudo",
    "su",
    "doas",  # Privilege escalation
    "passwd",
    "chpasswd",  # Password changes
    "adduser",
    "useradd",
    "userdel",
    "usermod",  # User management
    "groupadd",
    "groupdel",
    "groupmod",  # Group management
    "chmod",
    "chown",
    "chgrp",  # Permission changes
    "rm",
    "mv",
    "cp",  # Dangerous file operations
    "mount",
    "umount",  # Filesystem mounting
    "fdisk",
    "parted",
    "mkfs",  # Disk operations
    "iptables",
    "ufw",
    "firewall-cmd",  # Firewall changes
    "systemctl",
    "service",
    "chkconfig",  # Service management
    "crontab",  # Scheduled tasks
    "at",
    "batch",  # Job scheduling
    "mail",
    "sendmail",  # Email sending
    "nc",
    "netcat",
    "socat",  # Network tools
    "ssh",
    "scp",
    "rsync",  # Remote access
    "curl",
    "wget",
    "lynx",  # HTTP clients (can be dangerous)
    "python",
    "python3",
    "perl",
    "ruby",  # Interpreters
    "node",
    "php",
    "java",  # More interpreters
    "gcc",
    "g++",
    "make",
    "cmake",  # Compilers
    "docker",
    "podman",
    "kubectl",  # Container tools
    "git",  # Version control (can execute hooks)
}

# Sensitive directories that should be protected
SENSITIVE_DIRECTORIES = ["/etc", "/root", "/home/root", "/var/lib", "/sys", "/proc"]

# Suspicious file extensions that should trigger warnings
SUSPICIOUS_EXTENSIONS = [".sh", ".py", ".pl", ".rb", ".exe", ".bat"]

# Default security settings (still used by SecurityValidator)
DEFAULT_SECURITY_SETTINGS = {
    "timeout": 30.0,  # Default timeout in seconds (fallback)
    "max_command_length": 4096,  # Maximum command string length
}

# Control characters that are allowed in commands
ALLOWED_CONTROL_CHARS = {"\t", "\n", "\r"}

# Shell metacharacters pattern for argument validation
SHELL_METACHARACTERS_PATTERN = r"[;&|`$()]"

# Minimum character code for control character detection
MIN_CONTROL_CHAR_CODE = 32

# Null byte character for security validation
NULL_BYTE = "\x00"
