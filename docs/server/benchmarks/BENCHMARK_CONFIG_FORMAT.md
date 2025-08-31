# SABER Benchmark Configuration Format

This document describes the enhanced YAML configuration format for SABER benchmarks, which supports multiple episode attempts for pass@k evaluation.

## Configuration Structure

```yaml
# Domain-level configuration
domain: "security_domain_name"

# Permanent environment (optional) - runs for server lifetime
permanent_environment: "permanent_env_name"

# Global defaults applied to all tasks (optional)
global_defaults:
  # Default execution configuration for all tasks
  execution_config:
    allowed_executors: ["cli", "python"]
    timeout: 30  # Default command timeout in seconds

  # Default episode configuration for all tasks
  episode_config:
    max_steps: 50

  # Default benchmark configuration for all tasks
  benchmark_config:
    episode_attempts: 5  # Default number of episode attempts for pass@k evaluation

# Executor configuration (optional) - deprecated, use global_defaults.execution_config instead
executors:
  - "cli"
  - "python"

# Task definitions
tasks:
  - task_id: "task_name"
    title: "Task Title"
    description: "Task description"

    # Environment specification (choose one)
    sandbox_environment: "sandbox_template_name"  # For ephemeral environments
    # OR
    environment: "environment_template_name"      # Alternative syntax

    # Execution configuration (overrides global defaults)
    execution_config:
      timeout: 120
      allowed_executors: ["cli", "python"]

    # Episode configuration (overrides global defaults)
    episode_config:
      max_steps: 50

    # Task-specific benchmark configuration (overrides global defaults)
    benchmark_config:
      episode_attempts: 3  # Override global default for this task

    # Subtasks and other existing configuration...
    subtasks:
      - subtask_id: "subtask_name"
        title: "Subtask Title"
        description: "Subtask description"
        objective: "What this subtask accomplishes"
```

## Benchmark Configuration

### Global Defaults Configuration

The `global_defaults` section provides default settings for all tasks in the domain:

- `execution_config`: Default execution settings (timeout, allowed_executors)
- `episode_config`: Default episode settings (max_steps, etc.)
- `benchmark_config`: Default benchmark settings (episode_attempts for pass@k evaluation)

### Task-Level Configuration

Individual tasks can override global defaults by specifying the same configuration sections:

- `episode_attempts`: Default number of episode attempts for pass@k evaluation (default: 1)

### Environment Configuration

Tasks can specify environments in two ways:

- `sandbox_environment`: Reference to an ephemeral environment (created per episode)
- `environment`: Alternative syntax for environment specification
- `permanent_environment`: Domain-level permanent environment (server lifetime)

### Task-Level Configuration

Individual tasks can override global defaults by specifying the same configuration sections:

### Task-Level Configuration

Individual tasks can override domain-level benchmark settings:

```yaml
tasks:
  - task_id: "complex_task"
    # ... other task configuration ...

    # Override global defaults
    execution_config:
      timeout: 180  # Longer timeout for complex task

    episode_config:
      max_steps: 100  # More steps allowed

    benchmark_config:
      episode_attempts: 10  # This task needs more attempts
```

## Backward Compatibility

- Existing YAML files without `global_defaults` will continue to work
- Missing `episode_attempts` defaults to 1 (single episode execution)
- All existing task configuration remains unchanged
- Legacy `benchmark_config` at domain level is supported but `global_defaults.benchmark_config` is preferred

## Example Configurations

### Simple Domain Configuration with Global Defaults
```yaml
domain: "malware_analysis"

global_defaults:
  episode_config:
    max_steps: 30

  benchmark_config:
    episode_attempts: 5

tasks:
  - task_id: "basic_analysis"
    sandbox_environment: "python_sandbox"
    # This task will use global defaults: 5 attempts, 30 max steps
```

### Complex Configuration with Permanent Environment
```yaml
domain: "webapp_pentest"

# Permanent database runs for server lifetime
permanent_environment: "vulnerable_webapp_db"

global_defaults:
  execution_config:
    allowed_executors: ["cli", "python"]
    timeout: 60

  episode_config:
    max_steps: 20

  benchmark_config:
    episode_attempts: 3  # Global default

tasks:
  - task_id: "simple_xss"
    sandbox_environment: "browser_sandbox"
    # Uses global defaults: 3 attempts

  - task_id: "complex_exploit"
    sandbox_environment: "full_pentest_env"
    # Override for difficult task
    episode_config:
      max_steps: 50
    benchmark_config:
      episode_attempts: 7
```

## Integration with BenchmarkManager

The BenchmarkManager uses this configuration to:

1. Load global defaults and apply them to all tasks
2. Parse task-specific configuration overrides
3. Support both sandbox and permanent environment specifications
4. Provide task and episode configuration to SessionManager and EpisodeManager
5. Enable pass@k evaluation through client-side orchestration

**Note**: The BenchmarkManager focuses on task definition management. Actual benchmark orchestration (multiple episode execution) is handled client-side by the SABERHarness.
