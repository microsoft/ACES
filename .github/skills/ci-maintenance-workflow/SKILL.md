---
name: ci-maintenance-workflow
description: CI and maintenance workflows — fix a failing test, review a PR against agent-checkable standards. Use when user asks to fix a failing test or review a PR. Trigger when the user asks you to run the "Fix A Failing Test" or "Review PR" workflow.
---

# CI Maintenance Workflow

This skill contains sub-workflows for CI maintenance tasks.

## Sub-Workflow 1: Fix A Failing Test

### Inputs

- Failing test name or file path
- Error message or CI output (optional)

### Steps

1. **Checkout and update**:

   ```bash
   git checkout main
   git pull
   ```

2. **Create a fix branch**:

   ```bash
   git checkout -b fix/<descriptive-name>
   ```

3. **Reproduce the failure locally**:

   ```bash
   uv run pytest tests/<module>/<test_file>.py::<test_name> -v
   ```

4. **Analyze the failure**:
   - Read the test code and understand what it expects
   - Read the implementation code it tests (in `src/saber/` or `domains/`)
   - Identify the root cause (test bug vs implementation bug)

5. **Implement the fix**:
   - If test is wrong: fix the test
   - If implementation is wrong: fix the implementation
   - Prefer minimal, targeted fixes

6. **Verify the fix**:

   ```bash
   uv run pytest tests/ -v
   ```

7. **Run linting**:

   ```bash
   uv run ruff check
   ```

8. **Commit and push**:

   ```bash
   git add -A
   git commit -m "fix: <descriptive message>"
   git push -u origin fix/<descriptive-name>
   ```

9. **Create a draft PR** with this structure:
   - **The bug**: What was failing and why
   - **Error message**: The actual error output
   - **Root cause**: What caused the failure
   - **Fix**: What was changed and why
   - **Validation**: How the fix was verified
   - **Uncertainty**: Anything the human should double-check

## Sub-Workflow 2: Review PR

Review a PR against agent-checkable standards from this repository.

### Steps

1. **Gather context**:
   - Read the PR diff
   - Identify affected modules (`src/saber/`, `tests/`, `domains/`)
   - Read relevant code

2. **Check standards**:
   - [ ] Lint passes (`uv run ruff check`)
   - [ ] Tests pass (`uv run pytest tests/`)
   - [ ] Type hints on public functions
   - [ ] No `Any` types in public APIs
   - [ ] Pydantic models use `ConfigDict(frozen=True)` where appropriate
   - [ ] Docker images build if Dockerfiles changed
   - [ ] eval.yaml updated if needed (domain changes)
   - [ ] No hardcoded secrets or credentials
   - [ ] No unnecessary file additions

3. **Write summary** to `/tmp/SUMMARY.md`:

   ```markdown
   # PR Review Summary

   ## Overall: 🟢 READY / 🟡 NEEDS WORK / 🔴 MAJOR ISSUES

   ## Files Changed
   - file1.py: [brief description of change]
   - ...

   ## Checks
   | Check | Status | Notes |
   |-------|--------|-------|
   | Lint | ✅/❌ | ... |
   | Tests | ✅/❌ | ... |
   | Types | ✅/❌ | ... |
   | Docker | ✅/❌/N/A | ... |

   ## Issues Found
   1. **[Severity]**: Description — File:Line
      - Fix: recommendation

   ## Positive Observations
   - ...

   ---
   *This is an automatic review performed by an AI agent. A human reviewer should verify these findings.*
   ```
