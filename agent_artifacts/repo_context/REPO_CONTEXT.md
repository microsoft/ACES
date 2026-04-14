# REPO_CONTEXT.md

> Auto-generated institutional knowledge. Last updated: 2026-04-14.
> Source: Initial setup.

## Rules & Conventions

- **Package manager**: Always use `uv`, never `pip` or bare `python`
- **Type safety**: No `Any` types in public APIs; use Pydantic v2 models with `ConfigDict(frozen=True)`
- **Tests**: Framework tests in `tests/`, domain tests in `domains/<domain>/tests/` (if present)
- **Pre-commit**: Ruff linting/formatting enforced via `.pre-commit-config.yaml`

## Testing Recipe

- Test scorers with known-good AND known-bad inputs
- Mark Docker-dependent tests with `@pytest.mark.integration`
- Use `--limit 1` and `task_filter` for rapid iteration during debugging
- Parse `.eval` ZIP files for detailed analysis (see inspect-eval-log-analysis skill)

## Known Tech Debt

- (None documented yet — will be populated as PRs are reviewed)

## CI/Tooling

- Lint with `uv run ruff check`
- Format with `uv run ruff format`
- Pre-commit hooks configured at repo root
- GitHub Actions workflows for Claude review, test fixing, and issue solving

## Domain Patterns

- Domain entry points use `create_task()` factory pattern from `saber.task`
- Task YAML configs live under `domains/<domain>/tasks/`
- `global.yaml` provides defaults; task-specific YAML overrides
- Prompts are Jinja2 templates in `domains/<domain>/prompts/`
