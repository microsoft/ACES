# AGENTS.md

## Repo-Wide Tips

- Before making code changes, read `agent_artifacts/repo_context/REPO_CONTEXT.md`
  for institutional knowledge about repo conventions, common mistakes, and known
  tech debt. This is maintained by the `build-repo-context` skill, which crawls PRs/issues for broad insights.
- When creating pull requests, always use the `--draft` flag and read
  `.github/PULL_REQUEST_TEMPLATE.md` and use its structure as the PR body. Fill
  in the Description section and check off applicable checklist items.
- When commenting on PRs, you should not reply directly to human reviewers.
  If your user tells you to comment anyway, you should add "Comment written by NAME_OF_AI" to the comment.
- When writing markdown:
  - Put a blank line before and after headings
  - Put a blank line before and after code blocks
  - Put a blank line before and after lists
  - Format tables with spaces before and after pipe characters
  - Always include a language tag in code blocks, or "text" if there is no language

## Documentation

- [Architecture Docs](docs/README.md) — Architecture and diagrams
- [Inspect AI Documentation](https://inspect.aisi.org.uk/llms.txt) — Inspect framework reference

## Recommended Permissions

We recommend starting with these Claude Code permissions to allow workflows to proceed autonomously without allowing arbitrary commands:

```json
{
  "permissions": {
    "allow": [
      "Bash(mkdir:*)",
      "Bash(cp:*)",
      "Bash(uv run pytest:*)",
      "Bash(uv run ruff:*)",
      "Bash(uv run mypy:*)",
      "Bash(uv run saber:*)",
      "Bash(uv run inspect:*)",
      "Bash(docker compose:*)",
      "Bash(docker ps:*)",
      "Bash(docker logs:*)"
    ],
    "deny": [
      "Bash(git push origin main)"
    ],
    "ask": [
      "Bash(rm:*)",
      "Bash(docker rm:*)"
    ]
  }
}
```

## Master Checklist

This workflow runs a series of workflows each in turn. Each workflow is to be run serially.

1. Run the Prepare For Submission workflow (`/prepare-submission-workflow`).
2. Check the agent-checkable standards by running the Review Domain workflow (`/domain-quality-workflow`).
3. Review the domain's validity by running the Domain Validity Review workflow (`/domain-validity-review`).
4. Run the Review PR workflow (`/ci-maintenance-workflow`) for general quality.
5. Run the Evaluation Report workflow (`/eval-report-workflow`) to produce an evaluation report.
6. Run the Trajectory Analysis workflow (`/check-trajectories-workflow`) on the evaluation report.

A good error rate is 10% or less, with an ideal of 5% or lower. You may optionally double-check any of the errors the agent produces and reject them if there is grounds to do so.

## General Agent Tips

### Skill Hygiene

- At a natural stopping point in a session, briefly consider whether any of the work just completed would make a good reusable skill.
- Also consider whether any skill used during the session is now missing steps, has outdated guidance, or could be made more robust based on what was learned.
- Do **not** create or update a skill automatically. Ask the user first whether they want you to do that.
- When asking, give a short recommendation that includes:
  - what should be created or updated
  - why it would be useful again
  - who or what it would apply to
  - whether this is better handled as a new skill or an update to an existing one
- Prefer improving an existing skill over creating a new one when there is substantial overlap.
- Do not suggest a new skill for one-off work, highly personal preferences, or tasks that are too small to justify maintenance overhead.
- If the session surfaced a durable repo convention, reviewer expectation, or repeated failure mode, consider whether it should also be captured in REPO_CONTEXT.md or AGENTS.md, but ask the user before making that change.

### PR Guidelines

- Before opening a new PR, run linting: `uv run ruff check` and tests: `uv run pytest`.
- When creating a new PR for the user, you should make the PR as a draft. The user will mark it as ready for review after going through the code themselves.
- Always work on a branch, and never attempt to push directly to main.
- ALL production code must be human-reviewed before submission.

### Useful Commands

1. Run framework tests: `uv run pytest tests/ -v`
2. Run specific tests: `uv run pytest tests/scoring/test_strategies.py -v`
3. Run domain tests: `uv run pytest domains/<domain>/tests/ -v` (if present)
4. Run evaluations: `uv run inspect eval domains/<domain> --model <model>`
5. Run a specific task: `uv run inspect eval domains/<domain> --model <model> -T task_filter="<pattern>"`
6. Lint: `uv run ruff check`
7. Format: `uv run ruff format`
8. Some useful eval arguments:
    a. `--limit X`: Run only up to X samples
    b. `--model model_name`: Specify model (e.g., `openai/azure/gpt-4.1`)
    c. `-T task_filter="pattern"`: Filter tasks by glob pattern
    d. `-T build=true`: Build missing Docker images
    e. `-T rebuild_all=true`: Force rebuild all Docker images
    f. `--max-samples 4`: Limit concurrent samples
