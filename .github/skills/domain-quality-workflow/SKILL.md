---
name: domain-quality-workflow
description: Fix or review a single domain against quality standards. Use "fix" mode to refactor a domain into compliance, or "review" mode to assess compliance without making changes. Use when user asks to fix, review, or check a domain's quality. Trigger when the user asks you to run the "Fix A Domain" or "Review A Domain" workflow.
---

# Domain Quality Workflow

This skill reviews or fixes a single SABER domain against quality standards.

## Determine Mode

Ask the user (or infer from context):
- **Fix mode**: Make changes to bring the domain into compliance
- **Review mode**: Assess compliance and write a report without changing code

## Setup Working Directory

```bash
mkdir -p agent_artifacts/<domain>/<mode>/
```

Where `<mode>` is `fix` or `review`.

## Checklist Items

Go through each item below. For each:
- Assess confidence: **High** (certain), **Medium** (likely but verify), **Low** (uncertain)
- In Fix mode: fix High-confidence items, note Medium/Low in UNCERTAINTIES.md
- In Review mode: document all findings without changes

### 1. Domain Structure

- [ ] Has standard directory layout (tasks/, scoring/, tools/, prompts/, compose/, docker/, tests/)
- [ ] Has `__init__.py` with proper exports
- [ ] Has `<domain>.py` with `@task` entry point or `create_task()` wrapper
- [ ] Has `eval.yaml` with metadata

### 2. Task Configuration

- [ ] Has `tasks/global.yaml` with defaults
- [ ] Task YAML files have required fields (task_id, description, evaluation_config)
- [ ] Prompts referenced in YAML exist in `prompts/` directory
- [ ] No hardcoded values that should be in YAML config

### 3. Docker & Sandbox

- [ ] Docker Compose config is valid (`docker compose config` passes)
- [ ] Dockerfiles follow best practices (multi-stage, no unnecessary packages)
- [ ] Images build without errors
- [ ] Sandbox has proper security constraints

### 4. Scoring

- [ ] Scorers test for actual task completion, not proxies
- [ ] Score explanations are informative
- [ ] Edge cases handled (empty submission, malformed input)
- [ ] Scores are deterministic for the same input

### 5. Code Quality

- [ ] Type hints on all public functions
- [ ] No `Any` types in public APIs
- [ ] Pydantic models use `ConfigDict(frozen=True)` where appropriate
- [ ] Clean, focused functions with meaningful names
- [ ] DRY — no unnecessary duplication

### 6. Testing

- [ ] Has unit tests for custom scorers and tools
- [ ] Has setup/validation tests
- [ ] Tests pass: `uv run pytest domains/<domain>/tests/ -v`

### 7. Documentation

- [ ] eval.yaml is complete with title, description, contributors
- [ ] Task descriptions are clear
- [ ] Complex logic has comments explaining *why*

## Output Files

Create these in `agent_artifacts/<domain>/<mode>/`:

### CHECKLIST.md

```markdown
# Domain Quality Checklist: <domain>

| # | Item | Status | Confidence | Notes |
|---|------|--------|------------|-------|
| 1.1 | Standard directory layout | ✅/❌ | High/Med/Low | ... |
...
```

### NOTES.md

Working notes, observations, and reasoning during the review.

### UNCERTAINTIES.md (Fix mode only)

Items with Medium/Low confidence that need human review.

### SUMMARY.md

```markdown
# Domain Quality Summary: <domain>

## Overall Assessment
[2-3 sentence summary]

## Stats
- Items checked: N
- Passing: N
- Failing: N
- Uncertain: N

## Key Findings
1. [Finding with impact]
2. ...

## Recommendations
1. [Actionable recommendation]
2. ...
```
