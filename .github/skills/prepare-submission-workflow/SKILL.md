---
name: prepare-submission-workflow
description: Prepare a domain for PR submission — add tests, linting, eval.yaml, README updates. Use when user asks to prepare a domain for submission or finalize a PR. Trigger when the user asks you to run the "Prepare For Submission" workflow.
---

# Prepare Domain For Submission

Follow these steps in order to prepare a domain for PR submission.

## Prerequisites

- Domain code exists in `domains/<domain>/`
- Domain has been tested locally at least once

## Steps

### 1. Verify Domain Structure

Confirm the domain has the required directory structure:

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
│   └── <task_group>/<task>.yaml
├── tests/
│   └── test_*.py
└── tools/               # If custom tools
```

### 2. Add/Verify Tests

Ensure the domain has tests in `domains/<domain>/tests/`:

- **Unit tests**: Custom scorers, tools, utility functions
- **Setup tests**: Docker image build verification, compose config validation
- **Integration tests**: Mark with `@pytest.mark.integration` (requires Docker)

Run tests to verify:

```bash
uv run pytest domains/<domain>/tests/ -v
```

### 3. Verify eval.yaml

Ensure `domains/<domain>/eval.yaml` exists with required fields:

```yaml
title: "Domain Display Name"
description: "Brief description of what this domain evaluates"
group: "security"
contributors:
  - name: "Your Name"
    email: "your.email@example.com"
tasks:
  - name: "task_name"
    description: "What this task tests"
metadata:
  sandbox: true
  requires_internet: false
```

### 4. Run Linting

```bash
uv run ruff check domains/<domain>/
uv run ruff format --check domains/<domain>/
```

Fix any issues found.

### 5. Run a Smoke Test

Run at least one task with `--limit 1` to verify end-to-end:

```bash
uv run inspect eval domains/<domain> --model <model> --limit 1 -T task_filter="<simple_task>"
```

### 6. Final Checks

- [ ] All tests pass
- [ ] Lint passes
- [ ] eval.yaml is complete
- [ ] At least one successful eval run
- [ ] No `Any` types in public APIs
- [ ] Type hints on all public functions
- [ ] Docker images build successfully
