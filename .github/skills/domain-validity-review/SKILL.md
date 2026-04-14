---
name: domain-validity-review
description: Review a single domain's validity — whether its claims hold up, whether task names are accurate, whether samples can be both succeeded and failed at, and whether scoring measures ground truth. Use when user asks to check validity of a domain, or as part of the Master Checklist workflow.
---

# Domain Validity Review

This skill reviews whether a SABER domain's evaluation is valid — does it actually measure what it claims to?

## Setup

```bash
mkdir -p agent_artifacts/<domain>/validity/
```

## Phase 1: Gather Context

Read and understand:
1. **Task definitions**: `domains/<domain>/tasks/` — what each task claims to test
2. **Scoring**: `domains/<domain>/scoring/` — how success is determined
3. **Prompts**: `domains/<domain>/prompts/` — what the agent is told to do
4. **eval.yaml**: Domain metadata and claims
5. **README.md**: If present, domain-level documentation
6. **Docker setup**: `domains/<domain>/compose/` and `domains/<domain>/docker/`

Take notes in `agent_artifacts/<domain>/validity/NOTES.md`.

## Phase 2: Evaluate Overall Coherence

1. **Collect claims**: What does the domain say it evaluates? (from eval.yaml, README, task descriptions)
2. **Verify against code**: Do the scorers actually measure those claims?
3. **Assess consistency**: Do task names, descriptions, and scoring align?

## Phase 3: Assess Name Validity

For each task, check:
- [ ] Name accurately represents what is tested
- [ ] Not overly broad (e.g., "security" when it only tests one narrow skill)
- [ ] Doesn't oversell capability (e.g., "expert" when tasks are basic)
- [ ] Clear in documentation what is and isn't measured

## Phase 4: Assess Task Validity

For each task, check:
- [ ] An agent CAN succeed at this task (solution exists)
- [ ] An agent CAN fail at this task (not trivially solvable)
- [ ] The submission mechanism is clear (agent knows what to submit)
- [ ] Required resources are available in the sandbox
- [ ] The task doesn't rely on external services that may be unavailable

## Phase 5: Assess Scoring Validity

For each scorer, check:
- [ ] Measures actual task completion, not a proxy
- [ ] A high score genuinely requires solving the intended task
- [ ] Edge cases handled (empty submission, partial completion)
- [ ] Scoring is deterministic for the same input
- [ ] If using LLM-as-judge, calibration is documented

## Phase 6: Write Report

Create `agent_artifacts/<domain>/validity/VALIDITY_REPORT.md`:

```markdown
# Validity Report: <domain>

## Summary
[2-3 sentence overall assessment]

## Overall Rating: VALID / MOSTLY VALID / CONCERNS / INVALID

## Claims Verification
| Claim | Source | Verified? | Notes |
|-------|--------|-----------|-------|
| ... | eval.yaml | ✅/❌ | ... |

## Name Validity
| Task | Name Valid? | Severity | Issue |
|------|------------|----------|-------|
| ... | ✅/❌ | Low/Med/High | ... |

## Task Validity
| Task | Can Succeed? | Can Fail? | Clear Submission? | Resources Available? |
|------|-------------|-----------|-------------------|---------------------|
| ... | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ |

## Scoring Validity
| Scorer | Measures Ground Truth? | Deterministic? | Edge Cases? | Issues |
|--------|----------------------|----------------|-------------|--------|
| ... | ✅/❌ | ✅/❌ | ✅/❌ | ... |

## Findings
### High Severity
- ...

### Medium Severity
- ...

### Low Severity
- ...

## Recommendations
1. ...
```

## Phase 7: Wrap Up

Highlight the overall rating and any High severity findings to the user.
