---
name: inspect-eval-execution
description: Guide for running SABER inspect_ai evaluations locally. Use this when asked to run, re-run, or configure an inspect eval for any SABER domain.
---

To run SABER inspect_ai evaluations locally, follow this process:

## 1. Understand the Domain Structure

Each domain lives in `domains/<domain>/` and follows this layout:

```text
domains/<domain>/
  <domain>.py          # @task entry point — calls create_task()
  eval.yaml            # Domain metadata, Docker image definitions
  tasks/
    global.yaml        # Default prompts, tools, max_steps, aggregation
    <task_group>/
      <task>.yaml      # Individual task definitions
  compose/
    sandbox.compose.yml  # Docker Compose for sandbox containers
  scoring/             # Domain-specific scoring strategies
  tools/               # Domain-specific MCP tools
  prompts/             # Jinja2 prompt templates (instructions/, assistants/, etc.)
  docker/              # Dockerfiles for sandbox images
```

The `<domain>.py` file is minimal — it just calls `saber.task.create_task(**kwargs)` which wires up config loading, prompt rendering, scoring, tools, and the agent.

## 2. Build Docker Images First

Before running evals, ensure Docker images are built:

```bash
# Build all images for a domain (if saber CLI available)
uv run saber build <domain>

# Or auto-build during eval
uv run inspect eval domains/<domain> --model <model> -T build=true

# Force rebuild all images
uv run inspect eval domains/<domain> --model <model> -T rebuild_all=true
```

## 3. Run an Evaluation

```bash
# Basic evaluation (all tasks)
uv run inspect eval domains/<domain> --model <model>

# Run a single sample for quick iteration
uv run inspect eval domains/<domain> --model <model> --limit 1

# Filter to specific tasks
uv run inspect eval domains/<domain> --model <model> -T task_filter="sanity_*"

# Multiple task patterns (comma-separated)
uv run inspect eval domains/<domain> --model <model> -T task_filter="sanity_*,advanced_*"
```

## 4. Key `-T` Parameters

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `task_filter` | Glob pattern to select tasks | `-T task_filter="sanity_*"` |
| `build=true` | Build missing Docker images | `-T build=true` |
| `rebuild_all=true` | Rebuild ALL Docker images | `-T rebuild_all=true` |
| `rebuild=<prefix>` | Rebuild images matching prefix | `-T rebuild=sandbox` |
| `agent=<name>` | Agent implementation | `-T agent=react` |
| `run_preflight=true` | Health check before eval | `-T run_preflight=true` |
