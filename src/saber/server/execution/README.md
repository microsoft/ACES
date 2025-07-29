# SABER Docker Sandbox Execution Manager and Security Framework

This module provides a Docker container-based command execution system with comprehensive security validation, designed to safely execute command-line commands through MCP integration while providing complete isolation and preventing security vulnerabilities.

## Architecture

```
src/saber/server/tools/
├── execution_manager.py       # Main ExecutionManager with ExecutionConfiguration
├── base.py                    # result types
├── exceptions.py              # Tool related exceptions
├── executors/                 # Command execution frameworks
│   ├── base_executors.py      # ToolExecutor
│   └── cli.py                 # DockerCLIExecutor tool implementation
├── sandbox/                   # Docker container management
│   ├── sandbox_manager.py     # SandboxManager for container lifecycle
│   └── docker_environment.py  # DockerExecutionEnvironment implementation
└── utils/                     # Security utilities
    ├── security_validator.py  # Comprehensive security validation
    └── security_constants.py  # Security validation patterns and limits
```

## Key Components

### ExecutionManager (`execution_manager.py`)
Main execution manager for Docker-based CLI command execution with MCP integration:
- **Single Docker CLI Tool**: Manages one DockerCLIExecutor instance for containerized command execution
- **Sandbox Manager**: Integrates SandboxManager for Docker container lifecycle management
- **Security Validation**: Performs security validation at execution manager level before execution
- **Concurrency Control**: Built-in semaphore for limiting concurrent executions
- **MCP Integration**: Direct conversion to Model Context Protocol format
- **Configuration Management**: Uses ExecutionConfiguration for settings management including sandbox config

### ExecutionConfiguration
Configuration management for Docker-based CLI command execution:
- **YAML Support**: Load configuration from YAML files
- **Execution Settings**: Timeout and concurrency configuration
- **Security Settings**: Allowed commands and validation limits
- **CLI Options**: CLI-specific configuration options
- **Sandbox Configuration**: Docker container settings and security options

### DockerCLIExecutor Tool (`executors/cli.py`)
Secure command-line interface executing in Docker containers:
- **Docker Container Execution**: All commands executed in isolated Docker containers
- **Session-based Containers**: Each session gets dedicated container environment
- **Command Building**: Parses command strings into safe execution arguments
- **Parameter Support**: Accepts command string and shell mode parameters
- **Configuration Integration**: Accepts CLI configuration for parameter defaults
- **Output Parsing**: Structured parsing of command output and errors
- **MCP Schema Generation**: Generates own parameter schema for MCP integration
- **Sandbox Manager Integration**: Requires SandboxManager for container management

### SandboxManager (`sandbox/sandbox_manager.py`)
Docker container lifecycle management:
- **Session-based Containers**: Creates and manages containers per session
- **Container Health Monitoring**: Tracks container status and health
- **Automatic Cleanup**: Handles container destruction when sessions end
- **Configuration Management**: Manages Docker container security settings

### DockerExecutionEnvironment (`sandbox/docker_environment.py`)
Individual Docker container management:
- **Container Lifecycle**: Start, stop, and manage individual containers
- **Command Execution**: Execute commands within container with result capture
- **File Operations**: Copy files to/from containers
- **Health Checks**: Monitor container health and availability

### ToolExecutor Hierarchy
- **ToolExecutor**: Base class with common functionality, parameter management, and MCP schema generation
- **DockerCLIExecutor**: Docker-based command-line tool implementation with container isolation

## Security Features

### Docker Container Isolation
- **Complete Environment Isolation**: All commands execute in isolated Docker containers
- **Session-based Containers**: Each session gets its own dedicated container environment
- **Network Isolation**: Containers run with restricted network access (configurable)
- **Read-only Root Filesystem**: Container root filesystem is read-only by default
- **Non-root User**: Commands execute as non-privileged user within container
- **Resource Limits**: Container-level CPU, memory, and process constraints

### Command Validation (ExecutionManager Level)
- **Pre-execution validation**: SecurityValidator validates commands before DockerCLIExecutor execution
- **Global command blacklist**: Dangerous commands are blocked system-wide
- **Optional whitelist override**: Configuration-based `allowed_commands` can override blocked commands
- **Pattern detection**: Advanced regex-based detection of dangerous constructs
- **Argument sanitization**: Protection against injection attacks using shlex parsing

### Container Security (SandboxManager Level)
- **Container lifecycle management**: Automatic container creation, monitoring, and cleanup
- **Health monitoring**: Continuous health checks on container environments
- **Session isolation**: Each session's container is completely isolated from others
- **Automatic cleanup**: Containers are destroyed when sessions end

### Input Sanitization
- **Shell metacharacter detection**: Prevent command injection
- **Path traversal prevention**: Block unauthorized file access
- **Command length limits**: Prevent buffer overflow attacks

## Configuration

Docker-based CLI command execution is managed through YAML configuration:

```yaml
# Execution settings for CLI tool
execution:
  timeout: 300                    # Default timeout for commands (seconds)
  max_concurrent: 10              # Maximum concurrent CLI executions

# Docker sandbox configuration
sandbox:
  image: "saber/base-sandbox:latest"  # Docker image for command execution
  network_mode: "none"                # Container network isolation
  read_only_root: true                # Read-only root filesystem
  user: "tooluser"                    # Non-root user for command execution

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
uv run pytest tests/execution/test_cli_executor.py -v        # DockerCLIExecutor command execution
uv run pytest tests/tools/test_integration.py -v         # Integration tests

# Test coverage: 143+ tests across 6 test files
```

## Integration with MCP

The DockerCLIExecutor tool automatically generates its MCP schema from its parameter definitions:

```python
mcp_tools = registry.to_mcp_tools()
# ExecutionManager delegates to DockerCLIExecutor's to_mcp_schema() method
# Uses DockerCLIExecutor metadata and parameter definitions
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
