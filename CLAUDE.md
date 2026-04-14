# Claude Code Instructions

This is the ACES (Agent Capability Evaluation Suite) repository — the SABER framework library for benchmarking AI security agents using inspect_ai.

## Running Commands

This project uses [uv](https://docs.astral.sh/uv/) (by Astral) for package management. **Do not use `pip install`, `python -m venv`, `source .venv/bin/activate`, or bare `python`/`pytest` commands.** Always use `uv` to run commands:

- `uv sync --all-extras` — install/sync dependencies
- `uv run pytest ...` — run tests (not `pytest` or `python -m pytest`)
- `uv run python ...` — run Python scripts (not `python` or `python3`)
- `uv run inspect eval ...` — run evaluations
- `uv run ruff ...` — run the linter

Note: `uv` is Astral's Python package manager. It is not related to `uvicorn` (an ASGI web server) — do not confuse them.

## Repository Structure

| Path | Purpose |
|------|---------|
| `src/saber/` | SABER framework library — config, scoring, agents, tools, sandbox |
| `src/saber/agents/` | Agent registry and implementations (react, copilot, claude_code) |
| `src/saber/config/` | YAML config loading and Pydantic models |
| `src/saber/scoring/` | Scoring engine — strategies, factories, context |
| `src/saber/tools/` | Tool wrappers (SQL, KQL, etc.) |
| `src/saber/environments/` | Docker sandbox environment |
| `src/saber/prompts/` | Jinja2 prompt rendering |
| `domains/` | Benchmark domains (excytin_demo) |
| `tests/` | Framework test suite |
| `docs/` | Architecture documentation and diagrams |
| `.github/agents/` | Custom Copilot agent definitions |
| `.github/skills/` | Specialized workflow skills |

## Contributing

For development setup, contribution guidelines, and PR process, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Coding Style

- **Type hints**: Required on all public functions
- **Docstrings**: Google style with Args/Returns/Raises sections for public APIs
- **Pydantic v2**: Use `ConfigDict(frozen=True)` for immutable data models
- **No `Any` types**: Use proper Pydantic models instead of `dict` or `Any`
- **Small focused functions**: DRY, meaningful names, proper error handling
- **Comments**: Explain *why*, not *what*. Don't narrate changes. Base on code state, not history.

## Workflows

For common workflows (domain review, PR preparation, eval reporting, trajectory analysis), see [AGENTS.md](AGENTS.md).

## Pull Requests

When creating a pull request, always read `.github/PULL_REQUEST_TEMPLATE.md` and include its contents in the PR body. Fill in the checklist items and add your summary above them.

## How to Work

Understand before acting. Read the code, map the dependencies, and understand why things are the way they are before proposing changes. Present your analysis and tradeoffs to the user before implementing — let them decide what's worth changing. Don't start editing files based on assumptions or descriptions you haven't verified.

## Running Evaluations

```bash
# Run eval with demo domain
uv run inspect eval domains/excytin_demo --model openai/gpt-4

# Run specific task
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T task_filter="<pattern>"

# With logging
INSPECT_LOG_LEVEL=info uv run inspect eval domains/excytin_demo --model openai/gpt-4
```

## Testing

```bash
# Run all framework tests
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/scoring/test_strategies.py -v

# Run with coverage
uv run coverage run -m pytest && uv run coverage report
```
