# SABER Multi-Episode Orchestration Architecture

This directory contains PlantUML diagrams documenting the multi-episode orchestration feature added to SABER for coordinated dual-role and multi-agent benchmarking.

## Overview

SABER's orchestration system enables complex benchmark scenarios where multiple AI agents must collaborate or compete in coordinated episodes. The canonical example is **CTI-REALM**, where a red team agent creates exploits and a blue team agent must detect and mitigate them.

## Architecture Documents

### Workflow Diagrams

1. **[single-episode-workflow.puml](single-episode-workflow.puml)**
   - Traditional SABER execution flow
   - One task → one episode → one container
   - Shows semaphore acquisition, episode lifecycle, and cleanup
   - **Use case:** CyBench, standard cybersecurity tasks

2. **[orchestrated-episode-workflow.puml](orchestrated-episode-workflow.puml)**
   - Multi-episode coordinated execution
   - One orchestration → multiple episodes → shared context
   - Shows sequential episode creation with dependency ordering
   - **Use case:** CTI-REALM red/blue team scenarios

### Class Diagrams

3. **[handler-class-diagram.puml](handler-class-diagram.puml)**
   - Polymorphic handler pattern
   - `BenchmarkTaskHandler` abstract base class
   - `SingleEpisodeTaskHandler` vs `OrchestratedTaskHandler`
   - Factory function for polymorphic dispatch

4. **[role-configuration.puml](role-configuration.puml)**
   - Role-based agent/model assignment
   - `RoleAgentConfig` and `RoleBasedConfig` classes
   - Configuration loading and merging logic
   - CLI override patterns

### Design Decisions

5. **[semaphore-management.puml](semaphore-management.puml)**
   - Critical design: **1 semaphore slot per orchestration** (not per episode)
   - Prevents resource exhaustion from paired episodes
   - Shows capacity planning scenarios

6. **[dependency-validation.puml](dependency-validation.puml)**
   - `DependencyGraph` class validation logic
   - Valid patterns: simple pairs, one-to-many, independent groups
   - Invalid patterns: circular dependencies, nested orchestrations, orphan dependents
   - Prevents deadlocks at task creation time

## Key Design Principles

### 1. **Polymorphic Task Types**
```python
BenchmarkTask (abstract)
├── SingleEpisodeTask  # Legacy compatibility
└── OrchestratedTask   # Multi-episode coordination
```

**Rationale:** Type-safe dispatch, clear separation of concerns, extensible for future task types.

### 2. **Handler Pattern**
- **Abstract base class:** Shares semaphore tracking, error counting, leak detection
- **Concrete handlers:** Implement episode creation/cleanup for each task type
- **Factory function:** `get_benchmark_task_handler(task)` provides polymorphic dispatch

**Rationale:** Prevents code duplication (200+ lines of semaphore management), enables independent testing.

### 3. **Semaphore Accounting**
- **Single episode:** 1 task = 1 semaphore slot
- **Orchestrated task:** 1 orchestration (N episodes) = 1 semaphore slot

**Rationale:** Prevents resource exhaustion. A dual-role orchestration should consume the same concurrency quota as a single task, since episodes run sequentially.

### 4. **Dependency Validation**
- Built at task creation time (fail-fast)
- Uses `graphlib.TopologicalSorter` for cycle detection
- Explicitly forbids nested orchestrations (simplifies semaphore logic)

**Rationale:** Catch configuration errors before execution, prevent runtime deadlocks.

### 5. **Role-Based Configuration**
- Per-role agent/model assignment
- Defaults + overrides pattern
- CLI integration for runtime customization

**Rationale:** Enables different models for different roles (e.g., GPT-4 for attacker, Claude for defender).

## Viewing the Diagrams

### Option 1: VS Code (Recommended)
1. Install the **PlantUML** extension
2. Open any `.puml` file
3. Press `Alt+D` to preview

### Option 2: Command Line
```bash
# Install PlantUML
sudo apt install plantuml  # Ubuntu/Debian
brew install plantuml      # macOS

# Generate PNG
plantuml single-episode-workflow.puml

# Generate SVG (better for docs)
plantuml -tsvg single-episode-workflow.puml
```

### Option 3: Online
Paste the `.puml` content into [PlantText](https://www.planttext.com/) or [PlantUML Online](http://www.plantuml.com/plantuml/uml/)

## Common Scenarios

### Scenario 1: Single Agent Task (CyBench)
```
Task: "Find and exploit SQL injection vulnerability"

Flow:
1. Create SingleEpisodeTask
2. SingleEpisodeTaskHandler acquires semaphore
3. Create episode → agent executes → end episode
4. Release semaphore

Diagram: single-episode-workflow.puml
```

### Scenario 2: Dual-Role Task (CTI-REALM)
```
Task: "Red creates exploit, Blue detects and mitigates"

Flow:
1. Create OrchestratedTask (2 sub-tasks: red, blue)
2. OrchestratedTaskHandler acquires 1 semaphore slot
3. Create red episode → wait ready → create blue episode
4. Both agents execute (blue can access red's artifacts)
5. End both episodes → release 1 semaphore slot

Diagram: orchestrated-episode-workflow.puml
```

### Scenario 3: Mixed Workload
```
Workload:
- 3 CyBench tasks (single episodes)
- 2 CTI-REALM tasks (dual-role orchestrations = 4 episodes)

Semaphore usage: 5 slots (not 7)
- 3 single tasks = 3 slots
- 2 orchestrations = 2 slots

Diagram: semaphore-management.puml
```

## Testing

All workflows are covered by unit and integration tests:

- **Handler tests:** `tests/inspect_ai/test_task_handlers.py` (693 lines)
- **Role config tests:** `tests/client/test_role_config.py` (506 lines)
- **Retry/cleanup tests:** `tests/inspect_ai/test_eval_retry_support.puml` (484 lines)
- **Dataset tests:** `tests/client/inspect_ai_tests/test_saber_dataset.py` (398 lines)

## Implementation Files

| Component | File | Lines |
|-----------|------|-------|
| Task models | `src/saber/models/benchmark_task.py` | 449 |
| Task handlers | `src/saber/inspect_ai/task_handlers.py` | 687 |
| Role config | `src/saber/client/models.py` | 595 |
| Config loader | `src/saber/client/config_loader.py` | 254 |
| **Total** | | **1,985** |

## Related Documentation

- **[DOMAIN_DEVELOPMENT.md](../DOMAIN_DEVELOPMENT.md)** - Domain creation guide
- **[INSPECT_AI_DOMAIN_TASKS.md](../INSPECT_AI_DOMAIN_TASKS.md)** - Task definition format
- **[dependent_episode_scheduling_summary.md](../dependent_episode_scheduling_summary.md)** - Original design doc

## Design Rationale

This architecture was reviewed for complexity reduction on 2025-11-23. Key findings:

- **Test-to-code ratio:** 1.2:1 (not 6:1 as initially reported)
- **Role classes:** 2 core classes (not 6)
- **Handler pattern:** Prevents 200+ lines of duplication
- **Polymorphic tasks:** Provides type safety vs composition alternative

**Conclusion:** Complexity is proportional to problem space. No significant simplification opportunities identified without sacrificing functionality or type safety.

---

**Last Updated:** November 23, 2025  
**Diagrams Version:** 1.0  
**Feature Status:** Production-ready ✅
