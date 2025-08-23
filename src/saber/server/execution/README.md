# SABER Execution Framework

This module provides a Docker container-based command execution system with comprehensive security validation and a scalable executor hierarchy, designed to safely execute multiple types of commands (CLI and Python) through MCP integration while providing complete isolation and preventing security vulnerabilities.

## Architecture

```
src/saber/server/execution/
├── execution_manager.py       # Main ExecutionManager with ExecutorFactory
├── base.py                    # Result types and base classes
├── exceptions.py              # Execution related exceptions
├── executors/                 # Hierarchical executor framework
│   ├── base.py               # CommandExecutor base class
│   ├── docker_executor.py   # DockerExecutor abstract base for Docker-based executors
│   ├── cli.py               # CLIExecutor for shell command execution
│   ├── python_executor.py   # PythonExecutor for Python script execution
│   └── factory.py           # ExecutorFactory for scalable executor management
├── sandbox/                  # Docker container management
│   ├── sandbox_manager.py   # SandboxManager for container lifecycle
│   └── docker_environment.py # DockerExecutionEnvironment implementation
└── utils/                    # Security utilities
    ├── security_validator.py # Comprehensive security validation
    └── security_constants.py # Security validation patterns and limits
```

## Key Components

### ExecutionManager (`execution_manager.py`)
Main execution manager using factory pattern for scalable executor management:
- **Multiple Executor Types**: Supports CLI and Python execution with easy extensibility
- **Sandbox Manager**: Integrates SandboxManager for Docker container lifecycle management
- **Modular Configuration**: Uses ExecutionConfiguration with generic configuration delegation
- **Security**: Security validation handled by individual executors for separation of concerns
- **Concurrency Control**: Built-in semaphore for limiting concurrent executions

### ExecutorFactory (`executors/factory.py`)
Factory pattern implementation for scalable executor management:
- **Registry System**: Maintains registry of available executor types for easy extensibility
- **Command Analysis**: Intelligent routing of commands to appropriate executor types
- **MCP Aggregation**: Combines MCP schemas from all registered executors
- **Scalable Design**: Supports adding many executor types through registration
- **Executor Filtering**: Supports configuring a subset of available executors through `allowed_executors` parameter, enabling task-specific executor restrictions for security or functionality requirements

### Executor Hierarchy

#### CommandExecutor (`executors/base.py`)
Base class for all command executors:
- **Parameter Management**: Common parameter handling and validation framework
- **MCP Schema Generation**: Abstract method for generating MCP tool schemas
- **Timeout Management**: Configurable execution timeouts
- **Result Standardization**: Consistent CommandResult return types

#### DockerExecutor (`executors/docker_executor.py`)
Abstract base class for Docker-based executors:
- **Docker Integration**: Shared Docker container management functionality
- **Session Environment**: Session-based container environment management
- **Container Health**: Container readiness and health checking
- **Security Info**: Docker-specific security information reporting

#### CLIExecutor (`executors/cli.py`)
Docker-based shell command executor:
- **Shell Command Execution**: Executes shell commands in isolated Docker containers
- **Integrated Security Validation**: Each executor manages its own SecurityValidator for separation of concerns
- **Modular Configuration**: Extracts CLI-specific settings using `configuration.get_section("cli")`
- **Command Building**: Parses command strings into safe execution arguments
- **Session-based Containers**: Each session gets dedicated container environment
- **Shell Mode Support**: Configurable shell vs direct command execution
- **Output Parsing**: Structured parsing of command output and errors

#### PythonExecutor (`executors/python_executor.py`)
Docker-based Python script executor:
- **Python Script Execution**: Executes Python code in isolated Docker containers using uv
- **Modular Configuration**: Extracts Python-specific settings using `configuration.get_section("python")`
- **Dependency Management**: Automatic installation of Python packages via pip/uv
- **Code Validation**: Security validation of Python code before execution
- **Script Templates**: Template-based script generation for consistent execution environment
- **Module Restrictions**: Configurable allowed modules for security

### ExecutionConfiguration
**Modular configuration management for multi-executor command execution:**
- **Generic Section Access**: `get_section(name)` provides any configuration section for executor-specific configs
- **Executor Independence**: Each executor extracts its own configuration independently using `get_section()`
- **Scalable Design**: No coupling between configuration class and specific executor types
- **Manager-level Settings**: Execution timeout, concurrency, and security settings for manager use
- **YAML Support**: Load configuration from YAML files
- **Dynamic Extensibility**: Supports adding new executor types without modifying configuration class

The modular configuration system allows unlimited executor types to be added without changing the core configuration management. Each executor calls `configuration.get_section("executor_name")` to extract its specific settings.

### SandboxManager (`sandbox/sandbox_manager.py`)
Docker container lifecycle management:
- **Session-based Containers**: Creates and manages containers per session
- **Container Health Monitoring**: Tracks container status and health
- **Automatic Cleanup**: Handles container destruction when sessions end
- **Configuration Management**: Manages Docker container security settings
- **Orchestrator Integration**: Coordinates with external orchestrator for container cleanup during failures
- **Dynamic Image Building**: Builds required Docker images on server initialization if missing

### DockerExecutionEnvironment (`sandbox/docker_environment.py`)
Individual Docker container management:
- **Container Lifecycle**: Start, stop, and manage individual containers
- **Command Execution**: Execute commands within container with result capture
- **File Operations**: Copy files to/from containers
- **Health Checks**: Monitor container health and availability

### Executor Hierarchy Overview
- **CommandExecutor**: Base class with common functionality, parameter management, and MCP schema generation
- **DockerExecutor**: Abstract base for Docker-based executors with shared container management
- **CLIExecutor**: Docker-based shell command execution implementation
- **PythonExecutor**: Docker-based Python script execution implementation

## Security Features

### Docker Container Isolation
- **Complete Environment Isolation**: All commands execute in isolated Docker containers
- **Session-based Containers**: Each session gets its own dedicated container environment
- **Network Isolation**: Containers run with restricted network access (configurable)
- **Read-only Root Filesystem**: Container root filesystem is read-only by default
- **Non-root User**: Commands execute as non-privileged user within container
- **Resource Limits**: Container-level CPU, memory, and process constraints

### Command Validation (ExecutionManager Level)
- **Pre-execution validation**: SecurityValidator validates commands before executor execution
- **Global command blacklist**: Dangerous commands are blocked system-wide
- **Optional whitelist override**: Configuration-based `allowed_commands` can override blocked commands
- **Pattern detection**: Advanced regex-based detection of dangerous constructs
- **Argument sanitization**: Protection against injection attacks using shlex parsing
- **Executor-specific validation**: Each executor type performs additional validation for its domain

### Container Security (SandboxManager Level)
- **Container lifecycle management**: Automatic container creation, monitoring, and cleanup
- **Health monitoring**: Continuous health checks on container environments
- **Session isolation**: Each session's container is completely isolated from others
- **Automatic cleanup**: Containers are destroyed when sessions end

## Configuration

Multi-executor command execution uses a **modular configuration system** where each executor independently extracts its configuration section:

```yaml
# Manager-level execution settings (accessed via get_execution_timeout(), etc.)
execution:
  timeout: 300                    # Default timeout for commands (seconds)
  max_concurrent: 10              # Maximum concurrent executions across all executor types

# Manager-level security settings (accessed via get_allowed_commands(), etc.)
security:
  max_command_length: 4096        # Maximum command string length
  allowed_commands:               # Optional whitelist for command execution
    - "file"                      # Overrides blocked commands when specified
    - "strings"
    - "hexdump"
    - "python3"

# Manager-level sandbox configuration (accessed via get_sandbox_config())
sandbox:
  image: "saber/base-sandbox:latest"  # Docker image for command execution
  network_mode: "none"                # Container network isolation
  read_only_root: true                # Read-only root filesystem
  user: "tooluser"                    # Non-root user for command execution

# Executor-specific configurations (accessed via get_section("executor_name"))
cli:                              # CLIExecutor extracts via get_section("cli")
  default_shell_mode: false       # Default shell mode for CLI executor
  max_output_size: 1048576        # Maximum output size (1MB)

python:                           # PythonExecutor extracts via get_section("python")
  allowed_modules:                # Optional whitelist of allowed Python modules
    - "os"
    - "sys"
    - "json"
    - "requests"
  max_script_size: 1048576       # Maximum Python script size (1MB)
  default_requirements: []        # Default packages to install

# Future executors can add their own sections without modifying core classes
database:                         # Example future DatabaseExecutor section
  connection_timeout: 30
  max_connections: 10

file_ops:                         # Example future FileOperationsExecutor section
  allowed_extensions: [".txt", ".json", ".yaml"]
  max_file_size: 1000000
```

### Modular Configuration Benefits

- **Scalability**: Add unlimited executor types without changing ExecutionConfiguration
- **Separation of Concerns**: Each executor manages its own configuration needs
- **No Coupling**: Core configuration class remains generic and executor-agnostic
- **Easy Extension**: New executors simply call `configuration.get_section("my_executor_name")`

## Usage Workflow

### 1. Initialize ExecutionManager

Create an ExecutionManager instance with configuration (uses ExecutorFactory internally):

```python
from saber.server.execution.execution_manager import ExecutionManager

# Initialize with configuration file
registry = ExecutionManager(config_file="config.yaml")

# Or initialize with configuration dict
config = {
    "execution": {"timeout": 300, "max_concurrent": 10},
    "security": {"allowed_commands": ["ls", "cat", "grep"]},
    "cli": {"default_shell_mode": False},
    "python": {"allowed_modules": ["os", "sys", "json"]}
}
registry = ExecutionManager(config=config)

# Check available executor types
available_executors = registry._executor_factory.get_available_executors()
print(f"Available executors: {available_executors}")  # ['cli', 'python']
```

### 2. Execute Commands

Execute commands through different executor types:

```python
# Execute shell commands via CLI executor
cli_action = Action(tool_name="cli", command="ls -la /tmp")
context = {"session_id": "my_session"}
result = await registry.step(cli_action, context)

# Execute Python scripts via Python executor
python_action = Action(
    tool_name="python",
    parameters={
        "code": "print('Hello from Python!')\nprint(f'Current directory: {os.getcwd()}')",
        "requirements": ["requests"]
    }
)
result = await registry.step(python_action, context)

if result.success:
    print("Command output:", result.output)
else:
    print("Command failed:", result.error)
```

### 3. Validate Commands

Validate commands before execution:

```python
# ExecutionManager automatically determines executor type and validates
cli_action = Action(tool_name="cli", command="rm -rf /")
validation = registry.validate_command(cli_action)
if not validation.valid:
    print("Command blocked:", validation.errors)

# Python code validation
python_action = Action(tool_name="python", parameters={"code": "import subprocess; subprocess.run(['rm', '-rf', '/'])"})
validation = registry.validate_command(python_action)
if not validation.valid:
    print("Python code blocked:", validation.errors)
```

### 4. MCP Integration

Convert to MCP format for language model integration:

```python
mcp_tools = registry.to_mcp_tools()
# Returns list with both CLI and Python tool definitions
print(f"Available MCP tools: {len(mcp_tools)}")  # 2 tools

# Each executor contributes its own MCP schema
for tool in mcp_tools:
    print(f"Tool: {tool['name']}")
```

## Executor Tool Parameters

### CLI Executor Parameters

The CLI executor accepts the following parameters:

```python
{
    "command": {
        "type": "string",
        "description": "Shell command to execute (will be validated for security)",
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

### Python Executor Parameters

The Python executor accepts the following parameters:

```python
{
    "code": {
        "type": "string",
        "description": "Python code to execute (will be validated for security)",
        "required": True
    },
    "requirements": {
        "type": "array",
        "description": "List of Python packages to install before execution",
        "required": False,
        "default": []
    }
}
```

### Parameter Usage Examples

```python
# CLI executor examples
cli_action = Action(tool_name="cli", command="ls -la")
cli_action = Action(tool_name="cli", command="ps aux | grep python", parameters={"shell": True})

# Python executor examples
python_action = Action(
    tool_name="python", 
    parameters={
        "code": "import sys; print(sys.version)",
        "requirements": []
    }
)

python_action = Action(
    tool_name="python",
    parameters={
        "code": "import requests; response = requests.get('https://httpbin.org/json'); print(response.json())",
        "requirements": ["requests"]
    }
)
```

## Security Validation

The security framework uses **distributed security validation** where each executor manages its own security validation for separation of concerns:

### Distributed Security Architecture
- **Executor-Level Validation**: Each executor owns a SecurityValidator instance for its specific security needs
- **Separation of Concerns**: CLIExecutor handles shell command security, PythonExecutor handles Python code security
- **Validation Before Execution**: Each executor validates parameters before any command execution
- **Independent Configuration**: Each executor can be configured with its own allowed commands and restrictions

### SecurityValidator
- **Base command validation**: Validates commands against blocked lists with optional whitelist override
- **Pattern detection**: Identifies dangerous shell constructs and Python code patterns
- **Argument sanitization**: Validates all command arguments using shlex parsing
- **Full command validation**: Comprehensive validation of complete command structures
- **Executor-agnostic**: Works with any executor type through common validation interface

### Executor-specific Security
- **CLI Executor**: Each CLIExecutor instance has its own SecurityValidator for shell command validation, metacharacter detection, command injection prevention
- **Python Executor**: Python code analysis, import restrictions, dangerous function detection (can be extended with its own SecurityValidator)
- **Docker Isolation**: All executors benefit from Docker container isolation regardless of type

### SecurityConstants
Comprehensive security configuration:
- **Blocked commands**: Commands that are never allowed
- **Dangerous patterns**: Regex patterns for malicious constructs
- **Resource limits**: Default limits for execution
- **Restricted environment**: Safe environment variables

## Integration with MCP

The ExecutorFactory automatically aggregates MCP schemas from all registered executors:

```python
mcp_tools = registry.to_mcp_tools()
# ExecutionManager delegates to ExecutorFactory's to_mcp_tools() method
# ExecutorFactory aggregates schemas from all registered executors
# Returns:
# [
#   {
#     "name": "cli_shell_command",
#     "description": "Execute validated shell commands in Docker containers",
#     "inputSchema": {
#       "type": "object",
#       "properties": {
#         "command": {"type": "string", "description": "Shell command to execute"},
#         "shell": {"type": "boolean", "description": "Enable shell features", "default": false}
#       },
#       "required": ["command"]
#     }
#   },
#   {
#     "name": "python_python_script", 
#     "description": "Execute Python scripts with dependency management in Docker containers",
#     "inputSchema": {
#       "type": "object",
#       "properties": {
#         "code": {"type": "string", "description": "Python code to execute"},
#         "requirements": {"type": "array", "description": "Python packages to install", "default": []}
#       },
#       "required": ["code"]
#     }
#   }
# ]
```

### Adding New Executor Types

The factory pattern makes it easy to add new executor types:

```python
# 1. Create new executor inheriting from CommandExecutor or DockerExecutor
class JavaExecutor(DockerExecutor):
    def execute(self, action: Action) -> CommandResult:
        # Implementation here
        pass
    
    def to_mcp_schema(self) -> Dict[str, Any]:
        # Return MCP schema for Java execution
        pass

# 2. Register with factory (done automatically in __init__.py)
factory.register_executor("java", JavaExecutor)

# 3. New executor becomes available immediately
available = factory.get_available_executors()  # ['cli', 'python', 'java']
mcp_tools = registry.to_mcp_tools()  # Now includes Java tool
```

### Custom Executor Registration

Users can define and register custom executors from external Python files without modifying the SABER framework. When SABER initializes, it automatically loads any `*_executor.py` files from the configuration directory (e.g., `pentest_demo/server/`) and registers the custom executors they contain. External executor files should import the registration hook and register their executors:

```python
# In pentest_demo/server/nmap_executor.py
from saber.server.execution.executors.executor_registry import register_executor
from saber.server.execution.executors.docker_executor import DockerExecutor

class NmapExecutor(DockerExecutor):
    # Implement required methods: setup_parameters, execute, to_mcp_schema
    pass

# Register the executor - SABER will discover it automatically
register_executor("nmap", NmapExecutor)
```

This approach allows users to extend SABER's capabilities for specific domains (penetration testing, malware analysis, etc.) by simply placing custom executor files in their project directories. The custom executors become available through the same MCP interface as built-in executors.

The default values for executor parameters come from their respective configuration sections (cli, python, etc.).

This enables seamless integration with MCP-compatible language models and agents for secure multi-type command execution with easy extensibility for future executor types.
