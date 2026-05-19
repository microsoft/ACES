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

## Dual Repository Setup

This project is maintained in two repositories. Use whichever you have access to — the content is the same:

| | GitHub (external) | Azure DevOps (Microsoft internal) |
|---|---|---|
| **Benchmarks** | [ACESEvals](https://github.com/microsoft/ACESEvals) | [oss_saber](https://dev.azure.com/MSECAIModels/Benchmarking/_git/oss_saber) |
| **Library** (this repo) | [ACES](https://github.com/microsoft/ACES) | [SABER](https://dev.azure.com/MSECAIModels/Benchmarking/_git/SABER) |

The `pyproject.toml` has labeled source blocks for `inspect-ai` — uncomment the matching block for your environment. The GitHub source is active by default.

> **⚠️ Azure DevOps (Microsoft internal) users — required setup step:**
>
> The `pyproject.toml` defaults to **GitHub** sources for `inspect-ai`. If you cloned from Azure DevOps (`SABER`), you **must** switch to the ADO source before running `uv sync`:
>
> 1. Open `pyproject.toml` and find the `[tool.uv.sources]` section
> 2. Comment the GitHub line, uncomment the ADO line:
>    ```toml
>    # inspect-ai = { git = "https://github.com/microsoft/ACESEvals", branch = "inspect-ai/dev/aces_integration" }
>    inspect-ai = { git = "https://MSECAIModels@dev.azure.com/MSECAIModels/Benchmarking/_git/inspect_ai", branch = "dev/aces_integration" }
>    ```
> 3. Run `uv sync --all-extras`
>
> **Without this step, `uv sync` will fail** because GitHub sources may not be accessible from internal networks.

> **💡 Local development with inspect-ai:**
>
> If you have a local clone of inspect_ai and want to iterate on it, you can also use the local path source:
> ```toml
> inspect-ai = { path = "../inspect_ai", editable = true }
> ```

---

## Quick Start

### Prerequisites

- Python 3.11–3.12
- Docker (with Docker Compose v2)
- uv package manager

### Installation

```bash
# ⚠️ ADO users: switch inspect-ai source in pyproject.toml first (see above)
uv sync --all-extras

# Verify
uv run python -c "from saber.task import create_task; print('✅ saber installed')"
```

### Creating a New Evaluation Workspace

Use `saber new-eval-workspace <directory>` to scaffold a fresh uv-managed SABER
benchmark workspace. By default, the command creates a minimal `starter_demo`
domain so you can immediately inspect a valid domain layout and run a first eval.

If you already have this repo checked out locally:

```bash
uv run saber new-eval-workspace my-eval-workspace
```

If you want to run it directly from the GitHub-hosted remote without cloning:

```bash
uvx --from git+https://github.com/microsoft/ACES.git saber new-eval-workspace my-eval-workspace
```

To create the workspace without the starter demo domain:

```bash
uv run saber new-eval-workspace my-eval-workspace --no-demo-domain
```

Or via `uvx`:

```bash
uvx --from git+https://github.com/microsoft/ACES.git saber new-eval-workspace my-eval-workspace --no-demo-domain
```

To run the command from a specific Git ref:

```bash
uvx --from git+https://github.com/microsoft/ACES.git@main saber new-eval-workspace my-eval-workspace
```

After creating the workspace:

```bash
cd my-eval-workspace
uv sync

# If you created the default starter scaffold
uv run inspect list tasks | grep starter_demo
uv run inspect eval domains/starter_demo --model openai/gpt-4.1-mini
```

#### Default generated structure

```text
my-eval-workspace/
├── README.md
├── pyproject.toml
└── domains/
    └── starter_demo/
        ├── starter_demo.py
        ├── eval.yaml
        ├── prompts/
        │   ├── assistants/
        │   │   └── starter_assistant.j2
        │   └── instructions/
        │       └── starter_demo.j2
        └── tasks/
            ├── global.yaml
            └── starter_task.yaml
```

What the starter scaffold gives you:

- `pyproject.toml` — a uv-managed project pinned to the same `saber` source or version used to generate the workspace
- `README.md` — workspace-local instructions for creating new domains and tasks
- `domains/starter_demo/` — a minimal, runnable example domain
- `starter_demo.py` — Inspect AI entrypoint that calls `saber.task.create_task()`
- `eval.yaml` — top-level domain metadata
- `tasks/global.yaml` — shared defaults such as prompts and max steps
- `tasks/starter_task.yaml` — one simple static-scored example task
- `prompts/` — local instruction and assistant prompt templates used by the starter task

#### Structure with `--no-demo-domain`

If you pass `--no-demo-domain`, the command creates the workspace shell and leaves
`domains/` empty so you can start from scratch:

```text
my-eval-workspace/
├── README.md
├── pyproject.toml
└── domains/
```

The generated workspace `README.md` includes copy/pasteable examples for:

- the domain entrypoint module
- `eval.yaml`
- `tasks/global.yaml`
- a minimal task YAML file

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

# Use a runtime agent bundle with nested/subagent definitions
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=copilot \
  -T agent_bundle=/path/to/agent-bundle \
  -T main_agent="Recon Agent" \
  -T nested_agents=bridge
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
    ├── app.py                      # saber CLI (build, teardown, start, new-eval-workspace)
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

**Prerequisite:** `github-copilot-sdk>=0.3.0` must be installed inside the sandbox Docker image.

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

# With a complete agent bundle, including sibling subagents and .mcp.json
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=copilot \
  -T agent_bundle=/path/to/recon-agent \
  -T main_agent="Recon Agent" \
  -T nested_agents=bridge
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persona_file` | None | Agent persona markdown file (YAML frontmatter + body) |
| `skills_dir` | None | Skills directory — uploaded to sandbox at `.github/skills/` |
| `agent_bundle` | None | Runtime bundle root containing `*.agent.md`, optional `manifest.json`, `skills/`, and `.mcp.json` |
| `agents_dir` | None | Directory of agent definitions when it differs from `agent_bundle` |
| `main_agent` | None | Agent name or filename selector for the top-level agent |
| `mcp_config` | None | Additional MCP config JSON to merge with SABER's bridge MCP tools |
| `nested_agents` | `"bridge"` | Copilot nested-agent mode. `"bridge"` installs a SABER-owned `agent` delegation tool that runs sibling custom agents through the same Inspect model bridge; `"native"` uses SDK-native custom-agent registration only; `"disabled"` skips registration |
| `nested_agent_timeout` | 600 | Per-delegation timeout in seconds for `nested_agents=bridge` |
| `nested_agent_quiet_timeout` | 5 | Seconds of quiet after a nested assistant response with no in-flight tools before treating the delegation as complete |
| `nested_agent_max_calls` | 16 | Maximum bridge-backed nested agent calls per sample |
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

# With a complete agent bundle, including .claude/agents staging
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=claude_code \
  -T agent_bundle=/path/to/recon-agent \
  -T main_agent="Recon Agent"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persona_file` | None | Persona content passed as `--append-system-prompt` |
| `skills_dir` | None | Skills directory — uploaded to sandbox at `.claude/skills/` |
| `agent_bundle` | None | Runtime bundle root containing `*.agent.md`, optional `manifest.json`, `skills/`, and `.mcp.json` |
| `agents_dir` | None | Directory of agent definitions when it differs from `agent_bundle` |
| `main_agent` | None | Agent name or filename selector for the top-level agent |
| `mcp_config` | None | Additional MCP config JSON to merge with SABER's bridge MCP tools |
| `version` | `"auto"` | Claude binary path or `"auto"` to search PATH |
| `disallowed_tools` | `[]` | Tools to disallow via `--disallowed-tools` |
| `timeout` | 300 | Execution timeout (seconds) |
| `max_steps` | 50 | Max tool calls before forced completion |

### Runtime agent bundles

`agent_bundle`, `agents_dir`, `main_agent`, `skills_dir`, and `mcp_config` are runtime evaluation choices. They are intentionally not part of scenario/task export formats. A scenario bundle should stay reusable across `react`, `copilot`, `claude_code`, custom harnesses, and future solvers; the eval operator selects the harness, model, persona, skills, and subagent bundle at `inspect eval` time.

A generic bundle can look like:

```text
agent-bundle/
├── manifest.json          # optional: {"agent": "recon.agent.md"}
├── recon.agent.md         # top-level persona; may allow the agent/task tool
├── fingerprint.agent.md   # sibling subagent
├── report.agent.md        # sibling subagent
├── skills/                # optional, staged into the harness-specific skills dir
└── .mcp.json              # optional, merged with SABER bridge MCP tools
```

Copilot stages definitions under `.github/agents` and registers SDK `custom_agents`/`agent` session configuration. By default, `nested_agents=bridge` also exposes a SABER-owned `agent` tool so parent agents can delegate to sibling custom agents without requiring GitHub OAuth/HMAC in the sandbox; each nested session reuses the same Inspect bridge provider, merged MCP config, and skills. Claude Code stages definitions under `.claude/agents` and keeps `CLAUDE_CODE_SUBAGENT_MODEL=inspect` so subagent model calls continue to route through the bridge.

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
| `agent_bundle` | `str` | `None` | Runtime bundle root containing agent definitions, optional skills, and optional `.mcp.json` |
| `agents_dir` | `str` | `None` | Agent definition directory when separate from `agent_bundle` |
| `main_agent` | `str` | `None` | Top-level agent selector by name or filename |
| `mcp_config` | `str` | `None` | Additional MCP config JSON file |
| `timeout` | `int` | `300` | Agent execution timeout in seconds |
| `max_steps` | `int` | `50` | Max tool calls before forced completion |

**Copilot only:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port_base` | `int` | `3000` | Bridge proxy starting port |
| `nested_agents` | `str` | `"bridge"` | Use SABER bridge-backed delegation by default; use `"native"` for SDK-only custom-agent registration or `"disabled"` to skip registration |
| `nested_agent_timeout` | `int` | `600` | Per-delegation timeout in seconds for `nested_agents=bridge` |
| `nested_agent_quiet_timeout` | `int` | `5` | Quiet period after a nested assistant response before considering the delegated call complete |
| `nested_agent_max_calls` | `int` | `16` | Maximum bridge-backed nested calls per sample |

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

# Use a Copilot-compatible runtime agent bundle
uv run inspect eval domains/excytin --model openai/azure/gpt-4.1 \
  -T agent=copilot \
  -T agent_bundle=/home/me/agents/recon-agent \
  -T main_agent="Recon Agent"

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

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
