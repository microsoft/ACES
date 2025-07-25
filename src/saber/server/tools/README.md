# SABER Execution Manager and Security Framework

This module provides a CLI-only tool execution system with comprehensive security validation, designed to safely execute command-line tools through MCP integration while preventing security vulnerabilities.

## Architecture

```
src/saber/server/tools/
├── execution_manager.py       # Main ExecutionManager with CLIConfiguration
├── base.py                 # result types
├── exceptions.py           # Tool related exceptions
├── executors/              # Tool execution frameworks
│   ├── base_executors.py   # ToolExecutor
│   └── cli.py              # CLIExecutor tool implementation
└── utils/                  # Security utilities
    └── security_validator.py # Comprehensive security validation
    └── security_constants.py # Security validation patterns and limits
```

## Key Components

### ExecutionManager (`execution_manager.py`)
Main execution manager for CLI tool execution with MCP integration:
- **Single CLI Tool**: Manages one CLIExecutor instance for flexible command execution
- **Security Validation**: Performs security validation at execution manager level before execution
- **Concurrency Control**: Built-in semaphore for limiting concurrent executions
- **MCP Integration**: Direct conversion to Model Context Protocol format
- **Configuration Management**: Uses CLIConfiguration for settings management

### CLIConfiguration
Configuration management for CLI tool execution:
- **YAML Support**: Load configuration from YAML files
- **Execution Settings**: Timeout and concurrency configuration
- **Security Settings**: Allowed commands and validation limits
- **CLI Options**: CLI-specific configuration options

### CLIExecutor Tool (`executors/cli.py`)
Secure command-line interface directly inheriting from ToolExecutor:
- **Command Building**: Parses command strings into safe execution arguments
- **Parameter Support**: Accepts command string and shell mode parameters
- **Configuration Integration**: Accepts CLI configuration for parameter defaults
- **Output Parsing**: Structured parsing of command output and errors
- **Security Restrictions**: Built-in subprocess security restrictions and resource limits
- **MCP Schema Generation**: Generates own parameter schema for MCP integration

### ToolExecutor Hierarchy
- **ToolExecutor**: Base class with common functionality, parameter management, and MCP schema generation
- **CLIExecutor**: Consolidated command-line tool implementation with integrated security features

## Security Features

### Command Validation (ExecutionManager Level)
- **Pre-execution validation**: SecurityValidator validates commands before CLIExecutor execution
- **Global command blacklist**: Dangerous commands are blocked system-wide
- **Optional whitelist override**: Configuration-based `allowed_commands` can override blocked commands
- **Pattern detection**: Advanced regex-based detection of dangerous constructs
- **Argument sanitization**: Protection against injection attacks using shlex parsing

### Sandboxed Execution (CLIExecutor Level)
- **Resource limits**: CPU, memory, file size, and process constraints applied during execution
- **Environment isolation**: Restricted environment variables
- **Working directory**: Execution in controlled sandbox directory

### Input Sanitization
- **Shell metacharacter detection**: Prevent command injection
- **Path traversal prevention**: Block unauthorized file access
- **Command length limits**: Prevent buffer overflow attacks

## Configuration

CLI tool execution is managed through YAML configuration:

```yaml
# Execution settings for CLI tool
execution:
  timeout: 300                    # Default timeout for commands (seconds)
  max_concurrent: 10              # Maximum concurrent CLI executions

# Security configuration
security:
  max_command_length: 4096        # Maximum command string length
  allowed_commands:      # Optional whitelist for command execution
    - "file"                      # Overrides blocked commands when specified
    - "strings"
    - "hexdump"
    - "python3"

# CLI-specific settings
cli:
  default_shell_mode: false       # Default shell mode for CLI tool
  max_output_size: 1048576        # Maximum output size (1MB)
```

## Usage Workflow

### 1. Initialize ExecutionManager

Create an ExecutionManager instance with configuration:

```python
from saber.server.tools.execution_manager import ExecutionManager

# Initialize with configuration file
registry = ExecutionManager(config_file="config.yaml")

# Or initialize with configuration dict
config = {
    "execution": {"timeout": 300, "max_concurrent": 10},
    "security": {"allowed_commands": ["ls", "cat", "grep"]},
    "cli": {"default_shell_mode": False}
}
registry = ExecutionManager(config=config)
```

### 2. Execute Commands

Execute shell commands through the CLI tool:

```python
# Execute a simple command
result = await registry.execute_command({
    "command": "ls -la /tmp",
    "shell": False
})

# Execute with shell features
result = await registry.execute_command({
    "command": "ls -la | grep .txt",
    "shell": True
})

if result.success:
    print("Command output:", result.data["stdout"])
else:
    print("Command failed:", result.error)
```

### 3. Validate Commands

Validate commands before execution:

```python
validation = registry.validate_command("rm -rf /")
if not validation.valid:
    print("Command blocked:", validation.errors)
```

### 4. MCP Integration

Convert to MCP format for language model integration:

```python
mcp_tools = registry.to_mcp_tools()
# Returns list with single CLI tool definition
```

## CLI Tool Parameters

The CLI tool accepts the following parameters:

```python
{
    "command": {
        "type": "string",
        "description": "Command string to execute (will be validated for security)",
        "required": True
    },
    "shell": {
        "type": "boolean", 
        "description": "Whether to execute command through shell (enables pipes, redirections, etc.)",
        "required": False,
        "default": False  # Configurable via cli.default_shell_mode in configuration
    }
}
```

### Parameter Usage Examples

```python
# Simple command without shell
await registry.execute_command({
    "command": "ls -la"
})

# Complex command with shell features
await registry.execute_command({
    "command": "ps aux | grep python | wc -l",
    "shell": True
})

# File operations
await registry.execute_command({
    "command": "cat /etc/passwd | head -5",
    "shell": True
})
```

## Security Validation

The security framework includes multiple layers of protection:

### SecurityValidator
- **Base command validation**: Validates commands against blocked lists with optional whitelist override
- **Pattern detection**: Identifies dangerous shell constructs
- **Argument sanitization**: Validates all command arguments using shlex parsing
- **Full command validation**: Comprehensive validation of complete command structures

### SecurityConstants
Comprehensive security configuration:
- **Blocked commands**: Commands that are never allowed
- **Dangerous patterns**: Regex patterns for malicious constructs
- **Resource limits**: Default limits for execution
- **Restricted environment**: Safe environment variables

## Testing

The module includes comprehensive testing for the CLI-only architecture:

```bash
# Run all tool-related tests
uv run pytest tests/tools/ -v

# Run specific test suites
uv run pytest tests/tools/test_execution_manager.py -v        # Execution manager and configuration  
uv run pytest tests/tools/test_security_validator.py -v  # Security validation
uv run pytest tests/tools/test_cli_executor.py -v        # CLIExecutor tool execution
uv run pytest tests/tools/test_integration.py -v         # Integration tests

# Test coverage: 143+ tests across 6 test files
```

## Integration with MCP

The CLIExecutor tool automatically generates its MCP schema from its parameter definitions:

```python
mcp_tools = registry.to_mcp_tools()
# ExecutionManager delegates to CLIExecutor's to_mcp_schema() method
# Uses CLIExecutor metadata and parameter definitions
# Returns:
# [{
#     "name": "cli",
#     "description": "Execute validated shell commands in a secure environment",
#     "inputSchema": {
#         "type": "object",
#         "properties": {
#             "command": {"type": "string", "description": "..."},
#             "shell": {"type": "boolean", "description": "...", "default": false}
#         },
#         "required": ["command"]
#     }
# }]
```

The default value for the `shell` parameter comes from the CLI configuration's `default_shell_mode` setting.

This enables seamless integration with MCP-compatible language models and agents for secure command execution.
