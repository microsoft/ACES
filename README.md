# SABER - Security Agent Benchmarking and Evaluation Research

A distributed system for benchmarking agentic workflows in cybersecurity domains. SABER provides a server-client architecture where security domains are hosted as dedicated servers, and customer agents operate as independent clients to execute complex multi-step security tasks.

## Features

- **Domain-Specific Tasks**: Complex multi-step security workflows (malware analysis, threat investigation, etc.)
- **Benchmark Management**: YAML-driven benchmark definitions with multiple episode orchestration and pass@k evaluation
- **Session Management**: Stateful execution tracking with progress monitoring
- **Secure Command Execution**: Security-first command registry with command validation and sandboxing
- **Flexible Configuration**: YAML-based configuration for commands, security policies, and execution limits
- **MCP Integration**: Model Context Protocol support for command exposure
- **Evaluation Framework**: Action tracking and trajectory analysis for agent benchmarking

## Architecture Overview

SABER implements a distributed server-client architecture designed for security domain benchmarking:

- **Server Side**: Domain servers host security capabilities, tasks, and commands
- **Client Side**: Security agents connect to execute multi-step security workflows  
- **Communication**: Real-time bidirectional communication via HTTP/SSE
- **Evaluation**: Comprehensive action tracking and trajectory analysis

### Key Components

- **SessionManager**: Unified endpoint for all client interactions
- **BenchmarkManager**: Handles complex multi-step security workflows and benchmark orchestration
- **Command Registry**: Secure execution of domain-specific security commands
- **Evaluation Framework**: Action tracking and performance analysis

For detailed architecture documentation, see [docs/README.md](docs/README.md).

## Module Documentation

- **[Benchmark Management](src/saber/server/benchmarks/README.md)**: Multi-step security benchmark orchestration
- **[Command Registry](src/saber/server/execution/README.md)**: Secure command execution framework
- **[System Architecture](docs/README.md)**: Detailed architecture documentation

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
