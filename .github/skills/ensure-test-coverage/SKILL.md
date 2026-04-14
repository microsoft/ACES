---
name: ensure-test-coverage
description: Ensure test coverage for a module or domain - both reviewing existing tests and creating missing ones. Analyzes testable components, reports coverage gaps, and creates or improves tests. Use when user asks to check/review/create/add/ensure tests for a module or domain.
---

# Ensure Test Coverage

Analyze and ensure test coverage for a SABER module or domain.

## Phase 1: Discover Testable Components

Scan the target for testable code:

```bash
# Framework modules
find src/saber/<module>/ -name "*.py" -not -name "__init__.py"

# Domain code
find domains/<domain>/ -name "*.py" -not -path "*/tests/*" -not -name "__init__.py"
```

Categorize components:

| Component Type | Location | Priority |
|----------------|----------|----------|
| Scorers | `src/saber/scoring/` or `domains/<domain>/scoring/` | **High** — directly affects results |
| Tools | `src/saber/tools/` or `domains/<domain>/tools/` | **High** — agent-facing functionality |
| Config | `src/saber/config/` | **High** — data pipeline correctness |
| Agents | `src/saber/agents/` | **Medium** — agent registration and factory |
| Setup/Build | `setup.py`, `build_images.py` | **Medium** — infrastructure |
| Utilities | Any helper modules | **Low** — internal logic |

## Phase 2: Check Existing Tests

```bash
# Framework tests
find tests/ -name "test_*.py"

# Domain tests
find domains/<domain>/tests/ -name "test_*.py"

# Run existing tests
uv run pytest tests/ -v --tb=short
```

For each test file, check:
- What component does it test?
- Does it cover edge cases?
- Does it cover error paths?
- Are assertions meaningful (not just "doesn't crash")?

## Phase 3: Identify Coverage Gaps

Create a coverage matrix:

| Component | Has Tests? | Unit Tests | Edge Cases | Error Paths | Notes |
|-----------|-----------|------------|------------|-------------|-------|
| scorer_X | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ... |
| tool_Y | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ... |

Priority for new tests:
1. **P1**: Untested scorers (directly affect evaluation correctness)
2. **P2**: Untested tools (agent-facing, affect behavior)
3. **P3**: Missing edge case coverage on existing tested components
4. **P4**: Untested utilities and helpers

## Phase 4: Create/Fix Tests (if requested)

### Test Patterns

**Scorer tests:**

```python
import pytest

class TestMyScorer:
    """Tests for the scorer."""

    def test_correct_answer_scores_high(self):
        """Verify a correct submission gets a high score."""

    def test_incorrect_answer_scores_zero(self):
        """Verify an incorrect submission gets zero."""

    def test_empty_submission_handled(self):
        """Verify empty/missing submission doesn't crash."""

    def test_malformed_submission_handled(self):
        """Verify malformed input doesn't crash."""
```

**Tool tests:**

```python
class TestMyTool:
    """Tests for the tool."""

    def test_happy_path(self):
        """Verify tool works with valid input."""

    def test_invalid_input_returns_error(self):
        """Verify tool handles bad input gracefully."""
```

**Pytest markers:**

```python
@pytest.mark.integration  # Requires Docker
@pytest.mark.slow         # Takes >10 seconds
```

### Running Tests After Creation

```bash
uv run pytest tests/ -v        # Framework
uv run pytest domains/<domain>/tests/ -v  # Domain
```

## Phase 5: Present Results

Report:
1. Component inventory
2. Existing test coverage
3. Coverage gaps found
4. Tests created (if fix mode)
5. Remaining uncovered components
