# SABER

**Security Agent Benchmarking and Evaluation Research**

A distributed system for benchmarking agentic workflows in cybersecurity domains using **inspect_ai** integration with **Model Context Protocol (MCP)**.

## Overview

SABER provides a modern architecture for evaluating security agents through a dual-protocol client-server design:

- **Client Side**: `inspect_ai` integration with async orchestration, agent management, and MCP client for tool access
- **Server Side**: FastAPI REST API + FastMCP server managing sessions, benchmarks, and Docker-sandboxed execution
- **Domain Management**: CLI-based orchestration via `uv run saber-domain` for domain lifecycle management

The system emphasizes **fail-fast validation**, **type safety with Pydantic**, and **async context managers** for reliable resource management.

## Key Components

### Client Components
- **`run_saber_eval_async`**: Main entry point for inspect_ai-compatible evaluations
- **`SABEREvaluationOrchestrator`**: Async context manager coordinating evaluation lifecycle
- **`ClientSessionManager`**: Unified interface for REST + MCP communication with server
- **`AgentManager`**: Agent discovery and lifecycle management
- **`DatasetManager`**: Dataset creation from server benchmark tasks

### Server Components
- **`SessionManager`**: Central orchestrator for multi-session server management
- **`SessionRestAPI`**: FastAPI endpoints for session and episode management
- **`SessionMCPAPI`**: FastMCP server for tool discovery and execution
- **`BenchmarkManager`**: YAML-based task and benchmark configuration
- **`ExecutionManager`**: Docker sandbox orchestration for secure command execution

### Domain CLI (`saber-domain`)
Command-line interface for domain lifecycle management:
- **`list`**: Show available security domains
- **`validate`**: Verify domain configuration and structure
- **`build`**: Build Docker images for domain
- **`start`**: Launch domain server with REST + MCP APIs
- **`stop`**: Shutdown domain services
- **`test`**: Run full evaluation cycle with automated server management

## Quick Start

### Prerequisites
- Python 3.11-3.12 (managed by uv via `.python-version`)
- Docker
- Git

### Installation

1. **Install uv package manager:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and install:**
   ```bash
   git clone <repository-url>
   cd saber
   uv sync --all-extras
   ```

3. **Verify installation:**
   ```bash
   uv run python -c "from saber.client.inspect_ai import run_saber_eval_async; print('✅ SABER installed')"
   ```

4. **Configure environment variables:**
   ```bash
   # Copy the template
   cp .env.template .env
   
   # Edit .env and add your Azure OpenAI credentials
   # Required variables:
   #   AZUREAI_OPENAI_API_KEY=your-api-key-here
   #   AZUREAI_OPENAI_BASE_URL=https://your-resource.openai.azure.com
   #   AZUREAI_OPENAI_API_VERSION=2024-12-01-preview
   ```
   
   **Note**: The `.env` file is gitignored and will never be committed. Domain-specific variables (ports, image names, etc.) are auto-configured by the `saber-domain` CLI and don't require manual editing.

### Running Domains

**Run from SABER Context**
```
cd /path/to/SABER
```

**Help menu**
```bash
uv run saber-domain --help
```

**List available domains:**
```bash
uv run saber-domain list
```

**Build domain images:**

The build system supports three mutually exclusive modes:

```bash
# 1. INCREMENTAL BUILD (default): Build only missing images
#    - Fastest option for development
#    - Skips images that already exist
uv run saber-domain build cybench

# 2. COMPLETE REBUILD: Remove and rebuild all images
#    - Use when you need a clean slate
#    - Removes ALL existing domain images and rebuilds from scratch
uv run saber-domain build cybench --rebuild-all

# 3. SELECTIVE REBUILD: Remove and rebuild specific images by prefix
#    - Rebuild only server image
uv run saber-domain build cybench --rebuild server
#    - Rebuild all cookie challenge images (cookie_*)
uv run saber-domain build cybench --rebuild cookie
#    - Rebuild sandbox image
uv run saber-domain build cybench --rebuild sandbox
```

**Important**: These three options (`--build` is implicit default, `--rebuild-all`, `--rebuild <prefix>`) are mutually exclusive and cannot be combined.

**Start a domain server:**

The `start` command also supports all three build modes:

```bash
# Start without building (server must already have images)
uv run saber-domain start cybench
# helpful --dry-run flag here will provide diagnostics for the run
# Server endpoints:
#   REST API: http://localhost:8000
#   MCP Server: http://localhost:8001

# Build missing images before starting (INCREMENTAL):
uv run saber-domain start cybench --build

# Rebuild all images before starting (COMPLETE REBUILD):
uv run saber-domain start cybench --rebuild-all

# Rebuild specific images before starting (SELECTIVE REBUILD):
uv run saber-domain start cybench --rebuild server    # Only rebuild server
uv run saber-domain start cybench --rebuild cookie    # Only rebuild cookie_* images
```

**Note**: The `--rebuild` and `--rebuild-all` flags will stop the server first if it's running, rebuild the images, then start the server with the new images.

**Run evaluation tests:**

The `test` command automatically manages the server lifecycle and supports all build modes:

```bash
# Basic test (server kept running after completion for faster re-runs)
uv run saber-domain test cybench

# Test with INCREMENTAL BUILD (builds only missing images):
uv run saber-domain test cybench --build --stop-after

# Test with COMPLETE REBUILD (removes and rebuilds all images):
uv run saber-domain test cybench --rebuild-all --stop-after

# Test with SELECTIVE REBUILD (rebuild specific images):
uv run saber-domain test cybench --rebuild server --stop-after

# Additional helpful flags:
# --stop-after: Stop server after test completion (default keeps running)
# --dry-run: Show what would happen without executing
# --no-ui: Disable TUI and use console output instead
```

**How it works**:
- Automatically starts server if not running
- Runs evaluation against the domain
- By default, keeps server running for faster subsequent tests
- Use `--stop-after` to clean up after completion

**Stop domain:**
```bash
uv run saber-domain stop cybench
```

### Running Evaluations Programmatically

```python
from saber.client.inspect_ai import run_saber_eval_async
from saber.client.models import SABERConfig

config = SABERConfig(
    server_url="http://localhost:8000",
    mcp_url="http://localhost:8001",
    agent_name="my_agent",
    max_episodes=10
)

eval_log = await run_saber_eval_async(config)
```

## Development and Contributing

### Development Setup

```bash
# Install with development dependencies
uv sync --all-extras

# Run unit test suite
uv run pytest

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality

```bash
# Run all quality checks
uv run pre-commit run --all-files
```


### Example: Type-Safe Configuration

```python
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Fail-fast configuration with strict validation."""
    name: str = Field(..., description="Agent identifier")
    timeout: int = Field(default=300, ge=1)
    
    class Config:
        extra = "forbid"  # Fail on unknown fields
```


1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes with type hints and validation
3. Run tests: `uv run pytest tests/your_test.py -v`
4. Validate: `uv run pre-commit run --all-files`
5. Submit pull request with clear description

### Dependency Management

**Standard installation (default):**
```bash
uv sync --all-extras
# Uses inspect_ai from MSEC ADO repository
```

**Local development with inspect_ai:**
```bash
# 1. Edit pyproject.toml [tool.uv.sources]:
#    Comment out: inspect-ai = { git = "https://..." }
#    Uncomment: inspect-ai = { path = "./external/inspect_ai" }

# 2. Initialize submodule
git submodule update --init --recursive

# 3. Install
uv sync --all-extras
```

---

**Documentation**: [docs/README.md](docs/README.md) | **Domains**: [domains/](domains/) | **Architecture**: [docs/system/](docs/system/)
