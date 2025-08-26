# SABER Benchmark Configuration Format

This document describes the enhanced YAML configuration format for SABER benchmarks, which supports multiple episode attempts for pass@k evaluation.

## Configuration Structure

```yaml
# Domain-level configuration
domain: "security_domain_name"

# Benchmark configuration at domain level (optional)
benchmark_config:
  episode_attempts: 5  # Default number of episode attempts for pass@k evaluation

# Executor configuration (optional)
executors:
  - "cli"
  - "python"

# Task definitions
tasks:
  - task_id: "task_name"
    title: "Task Title"
    description: "Task description"

    # Environment specification
    environment: "environment_template_name"

    # Execution configuration
    execution_config:
      timeout: 120
      allowed_executors: ["cli", "python"]

    # Episode configuration
    episode_config:
      max_steps: 50

    # Task-specific benchmark configuration (optional)
    benchmark_config:
      episode_attempts: 3  # Override domain default for this task

    # Subtasks and other existing configuration...
    subtasks: []
```

## Benchmark Configuration

### Domain-Level Configuration

The `benchmark_config` section at the domain level provides default settings for all tasks in the domain:

- `episode_attempts`: Default number of episode attempts for pass@k evaluation (default: 1)

### Task-Level Configuration

Individual tasks can override domain-level benchmark settings:

```yaml
tasks:
  - task_id: "complex_task"
    # ... other task configuration ...

    # Override domain default
    benchmark_config:
      episode_attempts: 10  # This task needs more attempts
```

## Backward Compatibility

- Existing YAML files without `benchmark_config` will continue to work
- Missing `episode_attempts` defaults to 1 (single episode execution)
- All existing task configuration remains unchanged

## Example Configurations

### Simple Domain Configuration
```yaml
domain: "malware_analysis"

benchmark_config:
  episode_attempts: 5

tasks:
  - task_id: "basic_analysis"
    # This task will use 5 episode attempts
```

### Mixed Configuration
```yaml
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 3  # Domain default

tasks:
  - task_id: "simple_xss"
    # Uses domain default: 3 attempts

  - task_id: "complex_exploit"
    benchmark_config:
      episode_attempts: 7  # Override for difficult task
```

## Integration with BenchmarkManager

The BenchmarkManager uses this configuration to:

1. Load domain-level benchmark settings
2. Create multiple episodes per task based on `episode_attempts`
3. Coordinate with SessionManager for episode execution
4. Enable pass@k evaluation through EvaluationManager
