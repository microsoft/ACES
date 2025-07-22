# SABER - Security Agent Benchmarking and Evaluation Research

A distributed system for benchmarking agentic workflows in cybersecurity domains. SABER provides a server-client architecture where security domains are hosted as dedicated servers, and customer agents operate as independent clients to execute complex multi-step security tasks.

## Features

- **Domain-Specific Tasks**: Complex multi-step security workflows (malware analysis, threat investigation, etc.)
- **Task Management**: YAML-driven task definitions with dependency management and context propagation
- **Session Management**: Stateful execution tracking with progress monitoring
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
