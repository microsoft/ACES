---
name: build-repo-context
description: Crawl repository PRs, issues, and review comments to distill institutional knowledge into a shared knowledge base. Run periodically to maintain agent_artifacts/repo_context/REPO_CONTEXT.md. Trigger only on specific request.
---

# Build Repo Context

Crawl PRs, issues, and review comments to distill institutional knowledge into REPO_CONTEXT.md.

## Purpose

Maintain a living document of lessons learned, decisions made, and conventions established through PR reviews and issue discussions. This helps agents avoid repeating past mistakes and follow established patterns.

## Workflow

### Step 1: Identify What's New

Check the last crawl date in REPO_CONTEXT.md header. Identify PRs/issues since that date.

```bash
# List recent merged PRs
gh pr list --state merged --limit 50 --json number,title,mergedAt,body

# List recent closed issues
gh issue list --state closed --limit 20 --json number,title,closedAt,body
```

### Step 2: Triage

**Skip:**
- Dependency updates (renovate, dependabot)
- Changelog-only PRs
- Bot-generated PRs
- PRs with no review comments

**Prioritize:**
- PRs with substantive review comments (corrections, conventions established)
- Issues that revealed bugs or design problems
- PRs that introduced new patterns or changed established ones

### Step 3: Extract

For each relevant PR/issue, extract:
- PR body (decision rationale)
- Review comments (corrections, guidance)
- Issue discussion (problem patterns, root causes)

### Step 4: Distill

Extract actionable insights. Each insight should:
- **Cite its source** (PR #N, Issue #N)
- **Be actionable** (tells you what to do or avoid)
- **Add nuance** (not just "write tests" but "test scorers with known-good AND known-bad inputs")
- **Focus on reversals** (things that were wrong and corrected)
- **Be broadly applicable** (not one-off fixes)

Categories:
- **Rules & Conventions**: Coding standards, naming, structure
- **Testing Recipe**: How to test specific component types
- **Known Tech Debt**: Acknowledged issues not to fix without discussion
- **CI/Tooling**: Build and CI specifics
- **Domain Patterns**: Domain-specific patterns and gotchas

### Step 5: Merge into REPO_CONTEXT.md

Update `agent_artifacts/repo_context/REPO_CONTEXT.md`:

```markdown
# REPO_CONTEXT.md

> Auto-generated institutional knowledge. Last updated: YYYY-MM-DD.
> Source: PRs #X–#Y, Issues #A–#B.

## Rules & Conventions
- [Convention] (Source: PR #N)
- ...

## Testing Recipe
- [Pattern] (Source: PR #N)
- ...

## Known Tech Debt
- [Issue] (Source: Issue #N)
- ...

## CI/Tooling
- [Note] (Source: PR #N)
- ...

## Domain Patterns
- [Pattern] (Source: PR #N)
- ...
```

### Step 6: Consolidate

- Deduplicate: exactly one location per insight
- Remove insights that are no longer accurate
- Keep the file between 200-500 lines

## Bounds

- Process at most 50 PRs per run
- Follow at most 3 hops of linked references
- Target 200-500 lines in REPO_CONTEXT.md
