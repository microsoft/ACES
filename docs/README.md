# SABER Architecture Documentation

PlantUML diagrams documenting SABER's architecture, components, and workflows.

## Diagram Index

### Overview & Structure

| # | Diagram | Description |
|---|---------|-------------|
| 01 | [High-Level Architecture](diagrams/01-high-level-architecture.puml) | What is SABER? Thin library positioning between YAML tasks, inspect_ai, and Docker. |
| 02 | [Module Dependencies](diagrams/02-module-dependencies.puml) | Internal package dependency graph with directed import arrows. |
| 15 | [Domain Structure](diagrams/15-domain-structure.puml) | Domain directory layout — what goes where and which SABER component consumes each file. |

### Core Workflows

| # | Diagram | Description |
|---|---------|-------------|
| 03 | [create_task() Sequence](diagrams/03-create-task-sequence.puml) | Step-by-step sequence of the single entry point all domains call. |
| 04 | [Runtime Lifecycle](diagrams/04-runtime-lifecycle.puml) | Full lifecycle: inspect eval → task_init → sample loop → scoring → cleanup. |
| 05 | [YAML Config Cascade](diagrams/05-yaml-config-cascade.puml) | 3-level inheritance: global.yaml → shared.yaml → task.yaml with deep merge rules. |
| 14 | [Sample Conversion Flow](diagrams/14-sample-conversion-flow.puml) | TaskConfig → inspect_ai Sample transformation with metadata mapping. |

### Configuration & Data Models

| # | Diagram | Description |
|---|---------|-------------|
| 06 | [Config Model Hierarchy](diagrams/06-config-model-hierarchy.puml) | Pydantic model class diagram — TaskConfig, ScorerConfig, ToolConfig, and all criteria types. |
| 09 | [Scoring Context Models](diagrams/09-scoring-context-models.puml) | ScoringContext, ToolStep, JudgeTemplateContext — data available to scorers. |

### Scoring System

| # | Diagram | Description |
|---|---------|-------------|
| 07 | [Scoring Pipeline](diagrams/07-scoring-pipeline.puml) | End-to-end scoring: saber_overall runs first, caches LLM results, per-task scorers read cache. |
| 08 | [Scoring Strategies](diagrams/08-scoring-strategies.puml) | SaberScoringStrategy protocol and 6 built-in implementations with registry extension point. |

### Agent System

| # | Diagram | Description |
|---|---------|-------------|
| 10 | [Agent Registry & Solver](diagrams/10-agent-registry-and-solver.puml) | Auto-discovery, registration, two-level factory pattern, per-sample prompt injection. |
| 11 | [Sandbox Agent Bridge](diagrams/11-sandbox-agent-bridge.puml) | How copilot/claude_code agents interact with sandbox via MCP bridge. |

### Tools & Security

| # | Diagram | Description |
|---|---------|-------------|
| 12 | [Tool Resolution & Security](diagrams/12-tool-resolution-and-security.puml) | ToolRegistry resolution pipeline and CommandSecurityValidator's 7-step validation. |

### Sandbox Environment

| # | Diagram | Description |
|---|---------|-------------|
| 13 | [Sandbox Environment Lifecycle](diagrams/13-sandbox-environment-lifecycle.puml) | SaberSandboxEnvironment state machine — preflight, permanent services, sample loop. |

## Rendering Diagrams

### Using PlantUML CLI

```bash
# Install PlantUML (requires Java)
brew install plantuml  # macOS
sudo apt install plantuml  # Ubuntu

# Render all diagrams to PNG
plantuml docs/diagrams/*.puml

# Render to SVG
plantuml -tsvg docs/diagrams/*.puml
```

### Using VS Code

Install the "PlantUML" extension (jebbs.plantuml) for live preview.

### Using Online Server

Paste `.puml` contents into [plantuml.com/plantuml](https://www.plantuml.com/plantuml/uml/).

## Reading Order for New Contributors

1. **[01 — High-Level Architecture](diagrams/01-high-level-architecture.puml)** — Start here. Understand what SABER is.
2. **[15 — Domain Structure](diagrams/15-domain-structure.puml)** — See how domains are organized on disk.
3. **[05 — YAML Config Cascade](diagrams/05-yaml-config-cascade.puml)** — Understand the configuration inheritance model.
4. **[06 — Config Model Hierarchy](diagrams/06-config-model-hierarchy.puml)** — See the Pydantic data models.
5. **[03 — create_task() Sequence](diagrams/03-create-task-sequence.puml)** — Follow the wiring logic.
6. **[04 — Runtime Lifecycle](diagrams/04-runtime-lifecycle.puml)** — Understand the evaluation flow.
7. **[07 — Scoring Pipeline](diagrams/07-scoring-pipeline.puml)** — See how scoring works end-to-end.
8. Browse remaining diagrams based on your area of interest.
