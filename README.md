# SABER — Security Agent Benchmarking and Evaluation Research

**Agent Capability Evaluation Suite (ACES) / Security Agent Benchmarking and Evaluation Research (SABER)**

> **Naming:** The external name for this project is **ACES**. **SABER** is the internal Microsoft codename. The Python package, CLI commands, and code all use the name `saber`. Both names refer to the same system.
>
> **Dual repositories:**
> - **GitHub (external):** [ACESEvals](https://github.com/microsoft/ACESEvals) (benchmarks) + [ACES](https://github.com/microsoft/ACES) (this library)
> - **Azure DevOps (internal):** [oss_saber](https://dev.azure.com/MSECAIModels/Benchmarking/_git/oss_saber) (benchmarks) + [SABER](https://dev.azure.com/MSECAIModels/Benchmarking/_git/SABER) (this library)

A thin Python library (~5,200 LOC) that lets you define cybersecurity benchmarks using YAML files and run them through inspect_ai's native evaluation engine. No server, no client, no REST API.

```
YAML task configs  →  saber  →  inspect_ai Task  →  inspect eval
     (data)          (library)     (native engine)      (CLI)
```

## What saber does

1. **Loads YAML** — task definitions with 3-level config inheritance (global → shared → task)
2. **Renders prompts** — Jinja2 templates → agent instructions, judge prompts
3. **Creates scorers** — atomic, factory-created scorers with unified `ScoringContext`
4. **Provides tools** — `@tool`-decorated wrappers (SQL, KQL, etc.) around `sandbox().exec()`
5. **Switches agents** — CLI-driven agent selection via `-T agent=<name>`
6. **Manages environments** — Docker Compose lifecycle for permanent services

## What saber does NOT do

- Run a server
- Manage sessions or episodes
- Orchestrate Docker Compose directly (inspect_ai does this)
- Implement its own agent loop (wraps `react()` and other agents)
- Implement its own MCP server (uses `@tool` directly)

## Quick Start

### Prerequisites

- Python 3.11–3.12
- Docker (with Docker Compose v2)
- uv package manager

### Installation

```bash
uv sync --all-extras

# Verify
uv run python -c "from saber.task import create_task; print('✅ saber installed')"
```

### Running an Evaluation

```bash
# List available tasks
uv run inspect list tasks

# Run with default react agent (uses the domain's default dataset)
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1

# Select a specific dataset
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T dataset=legacy_test_set

# Dataset + task filter
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T dataset=latest_test_set -T task_filter="incident_5*"

# Use copilot agent
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=copilot

# Use claude_code agent
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=claude_code
```

## Package Structure

```
src/saber/
├── task.py                         # create_task() factory
├── sandbox.py                      # SaberSandboxEnvironment (DockerSandboxEnvironment subclass)
├── hooks.py                        # SetupHook protocol for pre-eval domain setup
├── logging.py                      # configure_logging() for saber.* loggers
├── agents/
│   ├── __init__.py                 # AgentRegistry + auto-discovery
│   ├── models.py                   # AgentPromptKwargs, AgentCapabilities
│   ├── resolver.py                 # Agent name → factory lookup
│   ├── solver_factory.py           # Agent factory → Solver bridge
│   ├── bridge_utils.py             # Shared sandbox_agent_bridge utilities
│   └── registry/                   # Convention-based agent modules
│       ├── react.py                # Default react agent
│       ├── copilot/                # GitHub Copilot SDK agent
│       └── claude_code/            # Claude Code CLI agent
├── config/
│   ├── models.py                   # Pydantic v2 models (all frozen)
│   ├── loader.py                   # YAML loading + deep merge + inheritance
│   └── converter.py                # TaskConfig → inspect_ai Sample
├── scoring/
│   ├── factory.py                  # ScorerFactory — YAML → @scorer list
│   ├── context.py                  # ScoringContext, ToolStep models
│   ├── strategies.py               # 6 built-in scoring strategies
│   ├── batch.py                    # Batched LLM judge scoring
│   ├── parsing.py                  # Score response parsing
│   ├── trajectory.py               # Tool step extraction from messages
│   ├── aggregation.py              # Score aggregation + checkpoint summary
│   ├── templates.py                # Jinja2 judge template rendering
│   ├── registry.py                 # Pluggable strategy registry
│   └── metrics.py                  # Custom @metric functions
├── prompts/
│   └── renderer.py                 # Jinja2 → prompt rendering
├── tools/
│   ├── registry.py                 # Tool name → Tool resolver
│   └── sql.py                      # sql_query @tool
├── approval/
│   ├── models.py                   # SecurityPolicy, ToolInspectorConfig
│   ├── security.py                 # CommandSecurityValidator
│   ├── inspectors.py               # Tool inspector factory functions
│   └── approver.py                 # @approver saber_security
├── environments/
│   ├── __init__.py                 # resolve_sandbox_spec() factory
│   ├── images.py                   # Docker image build management
│   └── preflight.py                # Compose file validation
└── cli/
    ├── app.py                      # saber CLI (build, teardown, start)
    ├── discovery.py                # Domain discovery utilities
    └── output.py                   # Rich console output
```

## Agents

Three agents are available via `-T agent=<name>`:

### React (Default)

Wraps inspect_ai's native `react()` solver. Runs in-process, no special setup.

```bash
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1
```

### GitHub Copilot

Runs the Copilot SDK Python client inside the Docker sandbox via a bridge proxy that routes LLM traffic back through inspect_ai.

**Prerequisite:** The `copilot` Python package must be installed inside the sandbox Docker image.

```bash
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=copilot

# With persona and skills
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=copilot \
  -T persona_file=path/to/persona.md \
  -T skills_dir=path/to/skills/ \
  -T timeout=600 \
  -T max_steps=100
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persona_file` | None | Agent persona markdown file (YAML frontmatter + body) |
| `skills_dir` | None | Skills directory — uploaded to sandbox at `.github/skills/` |
| `timeout` | 300 | Runner timeout (seconds) |
| `max_steps` | 50 | Max tool calls before forced completion |
| `port_base` | 3000 | Bridge proxy starting port |

### Claude Code

Runs the Claude Code CLI binary inside the Docker sandbox via a bridge proxy.

**Prerequisite:** The `claude` CLI must be installed inside the sandbox Docker image.

```bash
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=claude_code

# With persona, skills, and tool restrictions
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=claude_code \
  -T persona_file=path/to/persona.md \
  -T skills_dir=path/to/skills/ \
  -T disallowed_tools="WebFetch,NotebookEdit"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persona_file` | None | Persona content passed as `--append-system-prompt` |
| `skills_dir` | None | Skills directory — uploaded to sandbox at `.claude/skills/` |
| `version` | `"auto"` | Claude binary path or `"auto"` to search PATH |
| `disallowed_tools` | `[]` | Tools to disallow via `--disallowed-tools` |
| `timeout` | 300 | Execution timeout (seconds) |
| `max_steps` | 50 | Max tool calls before forced completion |

### Bridge Architecture

Both copilot and claude_code use a sandbox bridge pattern:

```
┌─────────────────────┐     ┌─────────────────────┐
│  Docker Sandbox     │     │  Host (inspect_ai)   │
│                     │     │                      │
│  Agent CLI/SDK      │────▶│  Bridge Proxy        │
│  (copilot/claude)   │     │  (localhost:port)    │
│                     │     │         │            │
│  bash, python       │     │         ▼            │
│  (native tools)     │     │  inspect_ai Model    │
│                     │     │         │            │
│  SABER MCP tools    │◀───│  MCP Tool Server     │
│  (bridged)          │     │  (sql, kql, etc.)   │
└─────────────────────┘     └─────────────────────┘
```

## CLI Usage

### Running Evaluations (`uv run inspect eval`)

```bash
uv run inspect eval domains/<domain> --model <model> [-T key=value ...]
```

All `-T` flags are passed as task parameters to the domain's `@task` function and then to `create_task()`. Parameters are type-coerced by inspect_ai (e.g., `true` → Python `bool`).

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_filter` | `str` | `None` | Glob or comma-separated task name filter |
| `dataset` | `str` | From `global.yaml` | Named task group selector |
| `agent` | `str` | `"react"` | Agent implementation: `react`, `copilot`, `claude_code` |
| `rebuild` | `str\|bool` | `None` | `true` → rebuild all images; `"name1,name2"` → specific images |
| `run_preflight` | `bool` | `false` | Validate compose files before evaluation |
| `keep_permanent` | `bool` | `false` | Keep permanent Docker services alive after eval |

#### Docker / Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sandbox_compose` | `str` | `compose/sandbox.compose.yml` | Relative path to sandbox compose file |
| `permanent_compose` | `str` | `None` | Relative path to permanent services compose file |
| `permanent_project` | `str` | `saber-permanent` | Docker Compose project name for permanent services |

#### Agent-Specific Parameters

These are forwarded through `**kwargs` to the agent factory.

**Copilot / Claude Code (shared):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `persona_file` | `str` | `None` | Path to agent persona markdown file |
| `skills_dir` | `str` | `None` | Path to skills directory (uploaded into sandbox) |
| `timeout` | `int` | `300` | Agent execution timeout in seconds |
| `max_steps` | `int` | `50` | Max tool calls before forced completion |

**Copilot only:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port_base` | `int` | `3000` | Bridge proxy starting port |

**Claude Code only:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | `str` | `"auto"` | Claude binary path or `"auto"` to search PATH |
| `disallowed_tools` | `str` | `""` | Comma-separated tools to disallow via `--disallowed-tools` |

#### CRSBench Domain Parameters

These parameters are consumed by CRSBench's setup hooks and do **not** apply to other domains.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str` | `None` | Dataset selector: `competition`, `sanity`, `lite`, `all` |
| `build` | `str` | `None` | `"true"` → force-build Docker images for benchmarks |
| `rebuild_images` | `str` | `None` | Prefix filter for rebuilding specific benchmark images |
| `data_dir` | `str` | `None` | External directory for benchmark data (symlinked to `<domain>/data`) |
| `include_ground_truth` | `bool` | `true` | Include ground-truth patches in downloaded data |
| `force_download` | `bool` | `false` | Re-download data even if it already exists |
| `force_build` | `bool` | `false` | Force rebuild all benchmark Docker images |

#### Examples

```bash
# Basic evaluation
uv run inspect eval domains/excytin_demo --model openai/azure/gpt-4.1

# Filter to specific tasks
uv run inspect eval domains/excytin_demo --model openai/azure/gpt-4.1 \
  -T task_filter="incident_5*"

# Rebuild all Docker images before eval
uv run inspect eval domains/excytin_demo --model openai/azure/gpt-4.1 \
  -T rebuild=true

# Rebuild a specific image
uv run inspect eval domains/excytin_demo --model openai/azure/gpt-4.1 \
  -T rebuild=sandbox

# Use copilot agent with persona
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=copilot \
  -T persona_file=personas/ir_expert.md

# Use claude_code agent
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=claude_code \
  -T timeout=600

# CRSBench: run lite dataset
uv run inspect eval domains/crsbench --model openai/azure/gpt-4.1 \
  -T dataset=lite

# CRSBench: force rebuild images, store data externally
uv run inspect eval domains/crsbench --model openai/azure/gpt-4.1 \
  -T force_build=true \
  -T data_dir=/mnt/crsbench_data

# Select a specific dataset and keep permanent services alive
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T dataset=latest_test_set \
  -T keep_permanent=true

# Validate compose files before running
uv run inspect eval domains/excytin_demo --model openai/azure/gpt-4.1 \
  -T run_preflight=true

# Concurrency control (inspect_ai flags, not -T)
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  --max-samples 4 --max-connections 20
```

> **`dataset` vs `task_filter`:** Each domain defines a `default_dataset` in its `global.yaml`. Running without `-T dataset` uses that default. Use `-T dataset=<name>` to switch between known task groups. Use `-T task_filter` for ad-hoc name-pattern filtering. Both compose: dataset filters first, then task_filter narrows further. For domains with setup hooks (e.g., CRSBench), `dataset` also scopes data downloads to only the selected group.

### SABER CLI (`uv run saber`)

Operational commands for managing Docker environments and images outside of evaluations.

#### `saber build` — Build Docker images

```bash
# Build images for all domains
uv run saber build

# Build images for a specific domain
uv run saber build excytin_demo

# Force rebuild all images for a domain
uv run saber build excytin_demo --rebuild

# Rebuild only a specific image
uv run saber build excytin_demo --rebuild --image sandbox
```

| Argument / Option | Type | Description |
|-------------------|------|-------------|
| `DOMAIN` | `str` (optional) | Domain slug. Omit to build all discovered domains |
| `--rebuild`, `-r` | flag | Force rebuild even if images already exist |
| `--image`, `-i` | `str` | Build only this specific image name |

Shows a Rich progress bar with per-image build status when stderr is a TTY.

#### `saber start` — Start permanent environment

```bash
# Start permanent services (databases, etc.)
uv run saber start excytin

# Start permanent services + a specific task sandbox
uv run saber start excytin --task incident_5_task_1
```

| Argument / Option | Type | Description |
|-------------------|------|-------------|
| `DOMAIN` | `str` (required) | Domain slug |
| `--task`, `-t` | `str` | Also start sandbox for this task ID |

#### `saber teardown` — Tear down Docker resources

```bash
# List and tear down all SABER compose projects
uv run saber teardown

# Tear down only a specific domain's projects
uv run saber teardown excytin_demo

# Skip confirmation prompt
uv run saber teardown --yes
```

| Argument / Option | Type | Description |
|-------------------|------|-------------|
| `DOMAIN` | `str` (optional) | Domain slug. Omit to tear down all SABER projects |
| `--yes`, `-y` | flag | Skip confirmation prompt |

Matches projects containing "saber", `{slug}-databases`, or `inspect-{slug}-*` patterns.

## Scoring

Atomic factory-created scorers with unified `ScoringContext`. Each scoring unit is an independent `@scorer` visible in eval logs.

**Built-in strategies:** `static`, `llm_judge`, `tool_call`, `tool_call_count`, `static_jaccard`, `none`

**Aggregation:** `average` (default), `weighted_sum`, `max` — configurable in YAML or via `-T score_aggregation=...`

**Domain-extensible:** Register custom strategies via `ScoringStrategyRegistry` without modifying core code.

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/

# Pre-commit
uv run pre-commit run --all-files

# Coverage
uv run coverage run -m pytest tests/ && uv run coverage report
```

## Design Documentation

| Document | Purpose |
|----------|---------|
| [Design Index](../../docs/design/refactor/README.md) | Full architecture docs |
| [Data Models](../../docs/design/refactor/01-data-models-and-config.md) | YAML config pipeline |
| [Scoring Engine](../../docs/design/refactor/02-scoring-engine.md) | Atomic scorer pipeline |
| [Domain Extensibility](../../docs/design/refactor/05-domain-extensibility.md) | Custom tools & strategies |
| [Environments](../../docs/design/refactor/06-environments.md) | Docker Compose conventions |
| [Approval System](../../docs/design/refactor/09-approval-system.md) | Security validation |
