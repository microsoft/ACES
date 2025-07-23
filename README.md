# SABER - Security Agent Benchmarking and Evaluation Research

A distributed system for benchmarking agentic workflows in cybersecurity domains. SABER provides a server-client architecture where security domains are hosted as dedicated servers, and customer agents operate as independent clients to execute complex multi-step security tasks.

## Features

- **Domain-Specific Tasks**: Complex multi-step security workflows (malware analysis, threat investigation, etc.)
- **Task Management**: YAML-driven task definitions with dependency management and context propagation
- **Session Management**: Stateful execution tracking with progress monitoring
- **Secure Tool Execution**: Security-first tool registry with command validation and sandboxing
- **Flexible Configuration**: YAML-based configuration for tools, security policies, and execution limits
- **MCP Integration**: Model Context Protocol support for tool exposure
- **Evaluation Framework**: Action tracking and trajectory analysis for agent benchmarking

## Task Manager Architecture

The SABER TaskManager provides a robust framework for managing complex multi-step security tasks. Tasks are defined in YAML files and executed through stateful sessions that maintain context between subtasks.

### Directory Structure

```
src/saber/server/tasks/
├── __init__.py              # Task management exports
├── task_manager.py          # Main TaskManager orchestrator
├── domain_task.py           # High-level security task representation
├── subtask.py               # Individual task steps with dependencies
├── task_session.py          # Stateful session management
├── enums.py                 # Task and session state enums
└── exceptions.py            # Task management exceptions
```

### Example Task Definition

```yaml
domain: "malware_classification"
tasks:
  - task_id: "malware_family_analysis"
    title: "Malware Family Classification and Analysis"
    description: "Analyze malware sample to determine family, capabilities, and threat level"
    initial_context:
      sample_path: "/data/samples/unknown_sample.exe"
      analysis_timeout: 300
    subtasks:
      - subtask_id: "static_analysis"
        title: "Static Analysis"
        description: "Perform static analysis of the malware sample"
        objective: "Extract basic file properties, strings, and structural information"
        required_tools: ["file_analyzer", "string_extractor", "pe_parser"]
        success_criteria:
          - "File type and architecture identified"
          - "Suspicious strings extracted"
          - "PE structure analyzed (if applicable)"
        context_dependencies: []
        depends_on: []
      
      - subtask_id: "dynamic_analysis"
        title: "Dynamic Analysis"
        description: "Execute sample in sandboxed environment"
        objective: "Observe runtime behavior and system interactions"
        required_tools: ["sandbox_executor", "behavior_monitor", "network_monitor"]
        success_criteria:
          - "Sample executed successfully"
          - "System calls captured"
          - "Network activity logged"
        context_dependencies: ["static_analysis.file_type"]
        depends_on: ["static_analysis"]
```

## Tool Registry and Security Framework

SABER includes a robust tool execution framework built around security-first principles, designed to safely execute command-line tools while preventing security vulnerabilities.

### Tool Registry Architecture

The ToolRegistry provides a comprehensive system for managing and executing security tools:

```
src/saber/server/
├── mcp/
│   └── tool_registry.py     # Main ToolRegistry class with thread-safe operations
├── tools/
│   ├── base.py             # SecurityTool and ToolExecutor abstractions
│   ├── security_constants.py # Security validation patterns and limits
│   ├── executors/          # Tool execution frameworks
│   │   └── base_executors.py # CommandLineToolExecutor with security validation
│   ├── utils/              # Security utilities
│   │   └── security_validator.py # Comprehensive security validation
│   └── domains/            # Domain-organized tools
│       ├── malware/        # Malware analysis tools
│       └── threat_investigation/ # Threat intel tools
```

### Security Features

- **Command Validation**: Whitelist-based command execution with dangerous pattern detection
- **Sandboxed Execution**: Optional path-based sandboxing for tool isolation
- **Resource Limits**: Configurable timeouts, memory limits, and output size restrictions
- **Input Sanitization**: Protection against command injection and shell metacharacter attacks
- **Audit Logging**: Complete logging of all tool executions with security metadata

### Configuration-Driven Tool Management

Tools and security policies are managed through YAML configuration:

```yaml
# Tool execution configuration
tools:
  domains:
    - malware
    - threat_investigation
  execution:
    timeout: 300
    max_concurrent: 10

# Security configuration
security:
  allowed_commands:
    - "echo"
    - "ls"
    - "cat"
    - "file"
  sandbox_path: "/tmp/sandbox"
  max_command_length: 4096

# Domain-specific settings
domain_settings:
  malware:
    default_timeout: 60
    quarantine_path: "/tmp/quarantine"
```

### Tool Development Workflow

1. **Create Tool Executor**: Extend `CommandLineToolExecutor` with security validation
2. **Define Metadata**: Add tool metadata for auto-discovery
3. **Register Tool**: Use ToolRegistry API or auto-discovery
4. **Configure Security**: Set allowed commands and sandbox restrictions
5. **Test Execution**: Comprehensive test suite validates security controls

Example tool implementation:

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

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SABER
   ```

2. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env  # Add uv to PATH
   ```

3. **Install dependencies:**
   ```bash
   # Install core dependencies
   uv sync
   
   # Install with all optional dependencies (dev, server, client)
   uv sync --all-extras
   ```

4. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   uv run pre-commit install
   ```

### Quick Start

1. **Check code quality:**
   ```bash
   uv run pre-commit run --all-files
   ```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tool configuration tests specifically
uv run pytest tests/test_tool_configuration.py

# Run security validation tests
uv run pytest tests/test_cli_security.py

# Run with coverage
uv run pytest --cov=src
```

### Code Quality

This project uses several tools to maintain code quality:

```bash
# Format code
uv run black src/

# Sort imports
uv run isort src/

# Lint code
uv run flake8 src/

# Type checking
uv run mypy src/

# Run all quality checks
uv run pre-commit run --all-files
```

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies:**
   ```bash
   uv sync --all-extras
   uv run pre-commit install
   ```
4. **Make your changes and ensure tests pass:**
   ```bash
   uv run pytest
   uv run pre-commit run --all-files
   ```
5. **Submit a pull request**

### Development Guidelines

- Follow PEP 8 style guidelines (enforced by `black` and `flake8`)
- Add type hints for all functions and methods
- Write tests for new functionality
- Update documentation as needed
- Ensure all pre-commit hooks pass

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Architecture

For detailed architecture documentation, see [docs/README.md](docs/README.md) and the implementation plan at [docs/TASK_MANAGER_IMPLEMENTATION_PLAN.md](docs/TASK_MANAGER_IMPLEMENTATION_PLAN.md).
