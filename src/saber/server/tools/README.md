# SABER Tool Registry and Security Framework

This module provides a comprehensive system for managing and executing security tools with a security-first approach, designed to safely execute command-line tools while preventing security vulnerabilities.

## Architecture

```
src/saber/server/tools/
├── tool_registry.py        # Main orchestration layer for tool management
├── base.py                 # SecurityTool and ToolExecutor abstractions
├── security_constants.py   # Security validation patterns and limits
├── registry/               # Modular registry components
│   ├── registrar.py        # Tool registration and analytics
│   ├── discoverer.py       # Auto-discovery of tools from domains
│   ├── execution_manager.py # Secure tool execution coordination
│   └── configuration.py    # YAML configuration management
├── executors/              # Tool execution frameworks
│   └── base_executors.py   # CommandLineToolExecutor with security validation
├── utils/                  # Security utilities
│   └── security_validator.py # Comprehensive security validation
└── domains/                # Domain-organized tools
    ├── malware/            # Malware analysis tools
    └── threat_investigation/ # Threat intel tools
```

## Key Components

### ToolRegistry (`tool_registry.py`)
Main orchestration layer that coordinates specialized components for managing security tools:
- **Modular Architecture**: Delegates to focused components for separation of concerns
- **Unified Interface**: Provides single entry point for all tool operations
- **Configuration-Driven**: YAML-based initialization with auto-loading
- **Component Access**: Exposes individual components for advanced usage

#### Core Components:

**ToolRegistrar** (`registry/registrar.py`):
- Tool registration and removal with validation
- Registry analytics and statistics
- Thread-safe tool storage with domain filtering
- Tool discovery coordination

**ToolDiscoverer** (`registry/discoverer.py`):
- Auto-discovery of tools from domain packages
- Metadata extraction from tool classes
- Recursive domain scanning with error handling
- Integration with registrar for seamless loading

**ToolExecutionManager** (`registry/execution_manager.py`):
- Secure tool execution with concurrency control
- Timeout management and resource limiting
- Background execution support with progress tracking
- Security validation integration

**RegistryConfiguration** (`registry/configuration.py`):
- YAML configuration file management
- Thread-safe configuration access
- Domain-specific settings
- Security and execution policy configuration

#### Architecture Benefits:

**Separation of Concerns**: Each component has a single, well-defined responsibility
- Registration logic isolated in ToolRegistrar
- Discovery logic separated in ToolDiscoverer  
- Execution coordination centralized in ToolExecutionManager
- Configuration management abstracted in RegistryConfiguration

**Extensibility**: Components can be extended or replaced independently
- Custom registrars for specialized storage backends
- Alternative discovery mechanisms for different tool formats
- Pluggable execution managers for different security models
- Multiple configuration sources (YAML, JSON, database)

**Testability**: Each component can be unit tested in isolation
- Mock dependencies easily for focused testing
- Component interfaces clearly defined
- Independent component lifecycle management

**Maintainability**: Smaller, focused codebases are easier to understand and modify
- Each component under 200 lines vs. original 592-line monolith
- Clear interfaces between components
- Reduced coupling and increased cohesion

### SecurityTool (`base.py`)
Represents a security tool with metadata and executor:
- **Validation**: Comprehensive tool configuration validation
- **MCP Integration**: Convert to Model Context Protocol format
- **Parameter Management**: Rich parameter system with validation

### ToolExecutor Hierarchy
- **ToolExecutor**: Abstract base class defining execution contract
- **BaseToolExecutor**: Common functionality and parameter management
- **CommandLineToolExecutor**: Secure command-line tool execution

## Security Features

### Command Validation
- **Tool-declared commands**: Each tool explicitly declares what commands it needs
- **Global command blacklist**: Dangerous commands are blocked system-wide
- **Pattern detection**: Advanced regex-based detection of dangerous constructs
- **Argument sanitization**: Protection against injection attacks

### Sandboxed Execution
- **Path restrictions**: Optional sandbox directory limitations
- **Resource limits**: CPU, memory, file size, and process constraints
- **Environment isolation**: Restricted environment variables

### Input Sanitization
- **Shell metacharacter detection**: Prevent command injection
- **Path traversal prevention**: Block unauthorized file access
- **Command length limits**: Prevent buffer overflow attacks

## Configuration

Tools and security policies are managed through YAML configuration:

```yaml
# Tool execution configuration
tools:
  # Load entire domains (all tools in these domains)
  domains:
    - malware
    - threat_investigation
  
  # Load specific tools only (individual tool selection)
  specific_tools:
    - "malware.static_analysis.File"
    - "malware.static_analysis.Strings"
  
  execution:
    timeout: 300
    max_concurrent: 10

# Security configuration
security:
  sandbox_path: "/tmp/sandbox"
  max_command_length: 4096

# Domain-specific settings
domain_settings:
  malware:
    default_timeout: 60
    quarantine_path: "/tmp/quarantine"
```

## Tool Development Workflow

### 1. Create Tool Executor

Extend `CommandLineToolExecutor` with security validation:

```python
class FileAnalyzer(CommandLineToolExecutor):
    _security_tool_metadata = {
        "domain": "malware",
        "name": "file_analyzer",
        "description": "Analyze file type and properties",
        "version": "1.0.0",
        "author": "SABER Team"
    }
    
    def __init__(self):
        super().__init__(command="file", allowed_commands=["file"])
        self.add_parameter(Parameter(
            name="file_path",
            type=ParameterType.STRING,
            description="Path to file to analyze",
            required=True
        ))
    
    def build_command(self, parameters, context):
        return ["file", "-b", parameters["file_path"]]
    
    def parse_output(self, stdout, stderr, return_code):
        if return_code != 0:
            return ToolResult.error_result(f"Analysis failed: {stderr}")
        return ToolResult.success_result({"file_type": stdout.strip()})
```

### 2. Define Metadata

Add `_security_tool_metadata` to your executor class for auto-discovery.

### 3. Register Tool

Use ToolRegistry API or place in appropriate domain directory for auto-discovery:

```python
from saber.server.tools.tool_registry import ToolRegistry

registry = ToolRegistry(domain="malware")
registry.register_tool(SecurityTool(
    name="file_analyzer",
    domain="malware",
    description="Analyze file type and properties",
    author="SABER Team",
    parameters={"file_path": Parameter(...)},
    executor=FileAnalyzer()
))
```

#### Advanced Component Access

Access individual components for specialized operations:

```python
# Access the registrar for direct tool management
registrar = registry.get_registrar()
registrar.add_tool(tool)
stats = registrar.get_tool_statistics()

# Access the discoverer for custom tool discovery
discoverer = registry.get_discoverer()
discovered_tools = discoverer.discover_tools_in_domain("custom_domain")

# Access the execution manager for advanced execution control
exec_manager = registry.get_execution_manager()
exec_manager.update_execution_settings(timeout=60, max_concurrent=5)

# Access configuration for runtime settings
config = registry.get_configuration()
security_settings = config.get_security_config()
domain_settings = config.get_domain_settings("malware")
```

### 4. Configure Security

Tools automatically declare their required commands. No additional configuration needed for command permissions.

Set sandbox restrictions and limits in your configuration file:

```yaml
security:
  sandbox_path: "/tmp/sandbox"
  max_command_length: 4096
```

### 5. Test Execution

Use the comprehensive test suite to validate security controls:

```bash
uv run pytest tests/test_cli_security.py
uv run pytest tests/test_tool_configuration.py
```

## Parameter System

Rich parameter validation with type checking and constraints:

```python
Parameter(
    name="file_path",
    type=ParameterType.STRING,
    description="Path to file to analyze",
    required=True,
    pattern=r"^[a-zA-Z0-9/_.-]+$"  # Path validation regex
)

Parameter(
    name="timeout",
    type=ParameterType.INTEGER,
    description="Analysis timeout in seconds",
    required=False,
    default=60,
    min_value=1,
    max_value=300
)
```

## Security Validation

The security framework includes multiple layers of protection:

### SecurityValidator
- **Base command validation**: Ensures only whitelisted commands
- **Pattern detection**: Identifies dangerous shell constructs
- **Argument sanitization**: Validates all command arguments
- **Path safety**: Prevents directory traversal attacks

### SecurityConstants
Comprehensive security configuration:
- **Blocked commands**: Commands that are never allowed
- **Dangerous patterns**: Regex patterns for malicious constructs
- **Resource limits**: Default limits for execution
- **Restricted environment**: Safe environment variables

## Testing

The module includes extensive testing for the modular architecture:

```bash
# Run all tool-related tests
uv run pytest tests/ -v

# Run specific test suites
uv run pytest tests/test_tool_configuration.py -v  # Registry components
uv run pytest tests/test_cli_security.py -v       # Security validation

# Test individual components
uv run pytest tests/test_tool_configuration.py::test_tool_registrar -v
uv run pytest tests/test_tool_configuration.py::test_tool_discoverer -v
uv run pytest tests/test_tool_configuration.py::test_execution_manager -v
```

## Integration with MCP

Tools are automatically converted to Model Context Protocol format:

```python
mcp_tool = security_tool.to_mcp_tool()
# Returns MCP-compatible tool definition with schema
```

This enables seamless integration with MCP-compatible language models and agents.
