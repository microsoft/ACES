# SABER Execution Framework

This module provides a Docker container-based command execution system with comprehensive security validation and a scalable executor hierarchy, designed to safely execute multiple types of commands (CLI and Python) through MCP integration while providing complete isolation and preventing security vulnerabilities.

## Architecture

```
src/saber/server/execution/
├── execution_manager.py       # Main ExecutionManager with ExecutorFactory
├── base.py                    # Result types and base classes
├── exceptions.py              # Execution related exceptions
├── environment_loader.py      # Environment specification loading and resolution
├── executors/                 # Hierarchical executor framework
│   ├── base_executors.py     # CommandExecutor base class and abstractions
│   ├── docker_executor.py    # DockerExecutor abstract base for Docker-based executors
│   ├── executor_factory.py   # ExecutorFactory for scalable executor management
│   ├── executor_registry.py  # Dynamic executor registration system
│   └── standard_registry/    # Built-in executor implementations
│       ├── bash_executor.py   # BashExecutor for shell command execution
│       └── python_executor.py # PythonExecutor for Python script execution
├── sandbox/                  # Docker container management
│   ├── sandbox_manager.py   # SandboxManager for container lifecycle
│   ├── docker_sandbox_environment.py # DockerSandboxEnvironment implementation
│   └── environment_spec.py   # Environment specification classes
└── utils/                    # Security utilities
    ├── security_validator.py # Comprehensive security validation
    └── security_constants.py # Security validation patterns and limits
```

## Key Components

### ExecutionManager (`execution_manager.py`)
Main execution manager using factory pattern for scalable executor management:
- **Multiple Executor Types**: Supports CLI and Python execution with easy extensibility
- **Sandbox Manager**: Integrates SandboxManager for Docker container lifecycle management
- **Dynamic Configuration**: Uses dictionary-based configuration with task-specific settings
- **Security**: Security validation handled by individual executors for separation of concerns
- **Concurrency Control**: Built-in semaphore for limiting concurrent executions

### ExecutorFactory (`executors/executor_factory.py`)
Factory pattern implementation for scalable executor management:
- **Registry System**: Maintains registry of available executor types for easy extensibility
- **Command Analysis**: Intelligent routing of commands to appropriate executor types
- **MCP Aggregation**: Combines MCP schemas from all registered executors
- **Scalable Design**: Supports adding many executor types through registration
- **Executor Filtering**: Supports configuring a subset of available executors through `allowed_executors` parameter, enabling task-specific executor restrictions for security or functionality requirements

### ExecutorRegistry (`executors/executor_registry.py`)
Dynamic executor registration system for runtime extensibility:
- **Runtime Registration**: Allows registering executors dynamically without code modification
- **External Executor Support**: Enables loading custom executors from external files
- **Discovery System**: Automatically discovers and loads `*_executor.py` files from project directories
- **Isolation**: Provides clean separation between built-in and custom executors

### Executor Hierarchy

#### CommandExecutor (`executors/base_executors.py`)
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

#### BashExecutor (`executors/standard_registry/bash_executor.py`)
Docker-based shell command executor:
- **Shell Command Execution**: Executes shell commands in isolated Docker containers
- **Integrated Security Validation**: Each executor manages its own SecurityValidator for separation of concerns
- **Modular Configuration**: Extracts CLI-specific settings using `configuration.get_section("cli")`
- **Command Building**: Parses command strings into safe execution arguments
- **Session-based Containers**: Each session gets dedicated container environment
- **Shell Mode Support**: Configurable shell vs direct command execution
- **Output Parsing**: Structured parsing of command output and errors

#### PythonExecutor (`executors/standard_registry/python_executor.py`)
Docker-based Python script executor:
- **Python Script Execution**: Executes Python code in isolated Docker containers using uv
- **Modular Configuration**: Extracts Python-specific settings using `configuration.get_section("python")`
- **Dependency Management**: Automatic installation of Python packages via pip/uv
- **Code Validation**: Security validation of Python code before execution
- **Script Templates**: Template-based script generation for consistent execution environment
- **Module Restrictions**: Configurable allowed modules for security

### EnvironmentLoader (`environment_loader.py`)
Environment specification loading and resolution system:
- **Flexible Environment Specs**: Supports string templates, granular dict configs, and hybrid approaches
- **Template Resolution**: Resolves environment template references to actual configurations
- **Configuration Validation**: Validates environment specifications before container creation
- **Docker Compose Integration**: Handles complex multi-container environment setups
- **Dynamic Loading**: Loads environment specifications from various sources (YAML, dict, etc.)

### SandboxManager (`sandbox/sandbox_manager.py`)
Docker container lifecycle management:
- **Session-based Containers**: Creates and manages containers per session
- **Container Health Monitoring**: Tracks container status and health
- **Automatic Cleanup**: Handles container destruction when sessions end
- **Configuration Management**: Manages Docker container security settings
- **Dynamic Image Building**: Builds required Docker images on server initialization if missing
- **Multi-Strategy Cleanup**: Implements fallback cleanup strategies for robust container termination

### DockerSandboxEnvironment (`sandbox/docker_sandbox_environment.py`)
Individual Docker container management:
- **Container Lifecycle**: Start, stop, and manage individual containers
- **Command Execution**: Execute commands within container with result capture
- **File Operations**: Copy files to/from containers
- **Health Checks**: Monitor container health and availability

### EnvironmentSpec (`sandbox/environment_spec.py`)
Environment specification classes for flexible container configuration:
- **Specification Models**: Defines data models for different environment types
- **Validation Logic**: Ensures environment specifications are valid and complete
- **Type Safety**: Provides type-safe environment configuration handling
- **Extensibility**: Supports multiple environment specification formats

### Executor Hierarchy Overview
- **CommandExecutor**: Base class with common functionality, parameter management, and MCP schema generation
- **DockerExecutor**: Abstract base for Docker-based executors with shared container management
- **BashExecutor**: Docker-based shell command execution implementation
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
- **Executor-specific validation**: Each executor type performs additional validation for its domain

### Container Security (SandboxManager Level)
- **Container lifecycle management**: Automatic container creation, monitoring, and cleanup
- **Health monitoring**: Continuous health checks on container environments
- **Session isolation**: Each session's container is completely isolated from others
- **Automatic cleanup**: Containers are destroyed when sessions end

### Episode Termination & Container Cleanup
The system provides robust container cleanup across all episode termination scenarios:

#### Episode Termination Conditions
1. **Normal Completion**:
   - Agent calls `end_episode` MCP tool
   - Episode step indicates completion (`step.done = True`)
   - Maximum steps reached (configurable, default: 20)

2. **Session-Level Termination**:
   - Manual session termination via REST API (`DELETE /session/{id}`)
   - Session timeout due to inactivity (configurable timeout)
   - Server shutdown (graceful termination of all sessions)

3. **Error-Triggered Termination**:
   - Command execution failures (any exception during action execution)
   - Episode tracking errors (failures in episode management)
   - Container health check failures

#### Container Cleanup Strategies
SABER implements multiple fallback cleanup strategies:

1. **Graceful Stop**: `docker compose stop` with configurable timeout (default: 30s)
2. **Force Cleanup**: `docker compose down --remove-orphans --volumes --timeout 10`
3. **Orphaned Resource Cleanup**: Find containers by SABER session labels and force remove individually

This design ensures containers are properly cleaned up even when:
- Session cleanup fails due to errors
- Docker daemon experiences problems
- Manual intervention is required

## Configuration

TBD
