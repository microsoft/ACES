# Contributing to ACES / SABER

## What We're Looking For

ACES contributions include both framework improvements (`src/saber/`) and new benchmark domains (`domains/`).

**Framework contributions:**
- New scoring strategies, agent implementations, tool wrappers
- Config pipeline improvements, sandbox environment enhancements
- Bug fixes, performance improvements, test coverage

**Domain contributions:**
- New security benchmark domains (incident response, CTI, vulnerability patching, etc.)
- Well-scoped, challenging, verifiable, and reproducible evaluations

## Development Setup

### Prerequisites

- Python 3.11+ (managed by `uv` and `.python-version`)
- Docker and Docker Compose v2
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ACES

# Install dependencies
uv sync --all-extras

# Verify installation
uv run pytest --co -q  # List all available tests
```

## Creating a New Domain

### Quick Start

1. Create the domain directory structure:

   ```text
   domains/<domain>/
   ├── __init__.py
   ├── <domain>.py          # @task entry point
   ├── eval.yaml            # Domain metadata
   ├── compose/
   │   └── sandbox.compose.yml
   ├── docker/
   │   └── Dockerfile.*
   ├── prompts/
   │   └── *.md
   ├── scoring/             # If custom scorers
   ├── tasks/
   │   ├── global.yaml
   │   └── <task_group>/
   ├── tests/
   │   └── test_*.py
   └── tools/               # If custom tools
   ```

2. Implement the `@task` entry point using `create_task()`
3. Add task YAML configurations
4. Add Docker sandbox setup
5. Add scorers and tests
6. Run the `/prepare-submission-workflow` skill to verify everything

## Code Quality Standards

### Required

- **Type hints** on all public functions
- **No `Any` types** in public APIs — use Pydantic models
- **Pydantic v2** with `ConfigDict(frozen=True)` for immutable data
- **Lint clean**: `uv run ruff check` and `uv run ruff format --check`
- **Tests pass**: `uv run pytest tests/` (framework) or `uv run pytest domains/<domain>/tests/` (domain)

### Style Guidelines

- Small, focused functions
- Meaningful names (no single-letter variables outside loops)
- DRY — extract shared logic into utilities
- YAGNI — don't build for hypothetical future requirements
- Comments explain *why*, not *what*
- Google-style docstrings for public APIs

## Testing Standards

### Required Tests

- **Unit tests**: Scorers, tools, config pipeline, utility functions
  - Cover edge cases and error conditions
  - Test with known-good AND known-bad inputs
- **Setup tests**: Docker build validation, compose config checks (for domains)
- **Integration tests**: Mark with `@pytest.mark.integration` (requires Docker)

### Running Tests

```bash
# Framework tests
uv run pytest tests/ -v

# Domain-specific tests
uv run pytest domains/<domain>/tests/ -v

# All tests
uv run pytest -v
```

## PR Process

### Before Submitting

1. Run the `/prepare-submission-workflow` skill, or manually:
   - `uv run pytest tests/ -v` — framework tests pass
   - `uv run ruff check` — lint passes
   - `uv run ruff format --check` — formatting passes
   - Type hints on all public functions

2. Create a draft PR using the [PR template](.github/PULL_REQUEST_TEMPLATE.md)

### Review Process

- PRs receive both automated (Claude) and human review
- Automated review triggers on PR creation or with `/review` comment
- ALL production code must be human-reviewed before merge
- Human reviewers address human comments; AI addresses AI-labeled comments

### Agent/LLM Usage

We encourage "centaur mode" — human + AI collaboration:

- Use AI to draft code, write tests, debug issues
- Human must review ALL code before submitting PR
- If AI generates review comments, label them: "Comment written by [AI name]"
- Human reviewers should not need to reply to AI-generated comments

## Running Evaluations

```bash
# Run evaluation with demo domain
uv run inspect eval domains/excytin_demo --model openai/gpt-4

# Run specific task
uv run inspect eval domains/excytin_demo --model openai/gpt-4 -T task_filter="<pattern>" --limit 1
```
