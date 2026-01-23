# SABER

**Security Agent Benchmarking and Evaluation Research**

A distributed system for benchmarking agentic workflows in cybersecurity domains using **inspect_ai** integration with **Model Context Protocol (MCP)**.

## Overview

SABER provides a modern architecture for evaluating security agents through a dual-protocol client-server design:

- **Client Side**: `inspect_ai` integration with async orchestration, agent management, and MCP client for tool access
- **Server Side**: FastAPI REST API + FastMCP server managing sessions, benchmarks, and Docker-sandboxed execution

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
   uv run python -c "from saber.inspect_ai import SABERSandboxEnvironment; print('✅ SABER installed')"
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
   
   **Note**: The `.env` file is gitignored and will never be committed. Domain-specific variables (ports, image names, etc.) are configured via task parameters (`-T` flags) and don't require environment variables.

### Running Domains

All domain operations use `inspect eval` commands with task parameters (`-T`) to control SABER behavior. The SABER server is automatically managed - started on first evaluation and kept running for faster subsequent runs.

**Available domains** in `domains/`:
- `excytin_demo` - Database forensics and incident response demonstrations

**List available tasks:**
```bash
uv run inspect list tasks
```

**Basic evaluation:**
```bash
# Evaluate the excytin_demo domain (server auto-starts and stays running after)
uv run inspect eval domains/excytin_demo --model openai/gpt-4
```

**Enable detailed logging for debugging:**

The `INSPECT_LOG_LEVEL` environment variable is extremely helpful for debugging SABER evaluations:

```bash
# Enable detailed logging to see server communication, tool calls, and execution details
INSPECT_LOG_LEVEL=info uv run inspect eval domains/excytin_demo --model openai/gpt-4

# Combine with other options for comprehensive debugging
INSPECT_LOG_LEVEL=info uv run inspect eval domains/excytin_demo \
    --model openai/azure/gpt-4.1 \
    -T max_concurrent_episodes=12 \
    -T task_filter="incident_*"
```

**Note:** This is not a default option but provides valuable insights into:
- Server startup and health checks
- MCP tool discovery and invocation
- Docker sandbox execution
- Episode lifecycle events
- Task loading and filtering

**Server endpoints** (when running):
- REST API: `http://localhost:8000`
- MCP Server: `http://localhost:8001`

**Build options:**

The build system supports three mutually exclusive modes:

```bash
# 1. INCREMENTAL BUILD: Build only missing images
#    - Fastest option for development
#    - Only builds images that don't exist
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T build=true

# 2. COMPLETE REBUILD: Remove and rebuild all images
#    - Use when you need a clean slate
#    - Removes ALL existing domain images and rebuilds from scratch
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T rebuild_all=true

# 3. SELECTIVE REBUILD: Remove and rebuild specific images by prefix
#    - Rebuild only server image
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T rebuild=server
```

**Important**: These three options (`build`, `rebuild_all`, `rebuild=<prefix>`) are mutually exclusive and cannot be combined.

**Server lifecycle management:**

```bash
# Keep server running after evaluation (default for faster re-runs)
uv run inspect eval domains/excytin_demo --model openai/gpt-4

# Stop server after evaluation completes (clean shutdown)
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T stop_saber_after=true

# Rebuild all images AND stop server after
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T rebuild_all=true -T stop_saber_after=true
```

**Task filtering:**

```bash
# Filter to specific tasks using exact match
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T task_filter=incident_5_task_1

# Filter using glob patterns
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T task_filter="incident_*"

# Filter to multiple patterns (OR logic)
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T task_filter="incident_5_*,incident_6_*"
```

**Custom ports:**

```bash
# Use non-default ports (useful for running multiple domains)
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T rest_port=9000 -T mcp_port=9001
```

**Concurrency control:**

```bash
# Inspect AI provides two key concurrency controls:
# --max-connections: Limits concurrent API calls to the LLM (default 10)
#                    Use this to avoid rate limiting from your model provider
# --max-samples:     Limits how many samples/episodes run in parallel (default 8)
#                    Each sample is an independent evaluation episode with its own sandbox

# Reduce LLM API concurrency (useful for rate-limited endpoints)
uv run inspect eval domains/excytin_demo --model openai/gpt-4 --max-connections 5

# Run samples sequentially (one at a time) - useful for debugging
uv run inspect eval domains/excytin_demo --model openai/gpt-4 --max-samples 1

# Run 4 samples in parallel with 20 concurrent LLM connections
uv run inspect eval domains/excytin_demo --model openai/gpt-4 \
  --max-samples 4 \
  --max-connections 20

# Combine with --limit to control total samples evaluated
uv run inspect eval domains/excytin_demo --model openai/gpt-4 \
  --limit 10 \
  --max-samples 4
```

**Preflight validation:**

```bash
# Run preflight health checks on all environments before evaluation
# Validates all compose files are properly configured
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T run_preflight=true

# Useful when making changes to domain configurations
uv run inspect eval domains/excytin_demo \
    --model openai/gpt-4 \
    -T rebuild_all=true \
    -T run_preflight=true
```

**Combined example:**

```bash
# Rebuild server images, filter to incident tasks, preflight check, and stop server after
uv run inspect eval domains/excytin_demo \
    --model openai/gpt-4 \
    -T rebuild=server \
    -T task_filter="incident_*" \
    -T run_preflight=true \
    -T stop_saber_after=true
```

**Manual server cleanup:**

If the server doesn't shut down properly, you can manually stop containers:

```bash
# View running containers
docker ps

# Stop domain containers
docker stop $(docker ps -q --filter "name=<domain_slug>")
```

### Development Setup

```bash
# Install with development dependencies
uv sync --all-extras

# Run unit test suite
uv run pytest

# Install pre-commit hooks
uv run pre-commit install
```

### Copilot Agent (Optional)

SABER supports using GitHub Copilot as an agent backend. This requires additional setup:

**Prerequisites:**
- Node.js 22+ (required by the Copilot CLI)
- GitHub Copilot subscription

**Installation:**

```bash
# 1. Install Node.js 22+ using fnm (fast node manager)
curl -fsSL https://fnm.vercel.app/install | bash
source ~/.bashrc  # or restart your shell
fnm install 22
fnm use 22

# 2. Install the GitHub Copilot CLI globally
npm install -g @github/copilot

# 3. Verify installation
copilot --version  # Should show 0.0.384 or later

# 4. Install SABER with copilot extras
uv sync --extra copilot
```

**Usage:**

```bash
# Run evaluation with the Copilot agent
uv run inspect eval domains/excytin_demo --model openai/azure/gpt-4o -T agent=copilot
```

### Claude Code Agent (Optional)

SABER supports using Claude Code as an agent backend. This uses Anthropic's Claude Code SDK for agentic tool use.

**Prerequisites:**
- Node.js 18+ (required by the Claude Code CLI)
- Anthropic API key with Claude Code access

**Installation:**

```bash
# 1. Install Node.js 18+ using fnm (fast node manager) if not already installed
curl -fsSL https://fnm.vercel.app/install | bash
source ~/.bashrc  # or restart your shell
fnm install 22
fnm use 22

# 2. Install the Claude Code CLI globally
npm install -g @anthropic-ai/claude-code

# 3. Verify CLI installation
claude --version  # Should show 2.x.x (Claude Code)

# 4. Install SABER with claude-code extras
uv sync --extra claude-code

# 5. Set your Anthropic API key
export ANTHROPIC_API_KEY=your-api-key-here
# Or add to your .env file:
# ANTHROPIC_API_KEY=your-api-key-here

# 6. Verify Python SDK installation
uv run python -c "from claude_code_sdk import query; print('✅ Claude Code SDK installed')"
```

### Testing and Coverage

**Run unit tests:**

```bash
# Setup environment
uv sync --all-extras

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/server/test_session_manager.py -v

# Run specific test class or function
uv run pytest tests/server/test_session_manager.py::TestSessionManagerBasics::test_create_session -v

# Run tests quietly (less verbose)
uv run pytest tests/server/ -q
```

**Measure code coverage:**

```bash
# Run tests with coverage collection
uv run coverage run -m pytest tests/server/ -q

# View coverage report in terminal
uv run coverage report

# View coverage for specific file
uv run coverage report --include="src/saber/server/session_manager.py"

# Generate detailed HTML coverage report
uv run coverage html
# Open htmlcov/index.html in browser to view line-by-line coverage
```

**Analyze coverage gaps:**

```bash
# Generate JSON coverage data for programmatic analysis
uv run coverage json

# View missing lines as annotated source
uv run coverage annotate src/saber/server/session_manager.py
# Creates session_manager.py,cover with ! marking uncovered lines

# Quick coverage summary with line numbers
uv run coverage report --show-missing --include="src/saber/server/session_manager.py"
```

**Coverage targets:**

- Overall project: Aim for >80% coverage
- Core modules (session_manager, execution_manager): Aim for >85% coverage
- Critical paths (security, evaluation): Aim for >90% coverage

**Example workflow:**

```bash
# 1. Run tests with coverage
uv run coverage run -m pytest tests/server/ -q

# 2. Check overall coverage
uv run coverage report

# 3. Identify gaps in specific module
uv run coverage report --show-missing --include="src/saber/server/session_manager.py"

# 4. Generate HTML report for detailed analysis
uv run coverage html

# 5. Open htmlcov/index.html to see which lines need tests
```

### Code Quality

```bash
# Run all quality checks
uv run pre-commit run --all-files
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
# Uses inspect_ai from MSEC ADO repository (dev/saber_integration branch)
```

---

**Documentation**: [docs/README.md](docs/README.md) | **Domains**: [domains/](domains/) | **Architecture**: [docs/system/](docs/system/)
