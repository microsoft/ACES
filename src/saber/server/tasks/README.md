# SABER Task Management System

The SABER TaskManager provides a robust framework for managing complex multi-step security tasks with RL-friendly episode-based execution. The system has been refactored into specialized components following single responsibility principles for better maintainability and testability.

## Architecture

```
src/saber/server/tasks/
├── __init__.py                      # Task management exports
├── task_manager.py                  # Simplified orchestrator (308 lines, down from 655)
├── exceptions.py                    # Task management exceptions
├── core/                            # Core task definitions and specialized components
│   ├── __init__.py                  # Core exports
│   ├── task.py                      # High-level security task with embedded progression logic
│   ├── subtask.py                   # Internal checkpoints with automatic progression
│   └── task_config_loader.py        # YAML parsing and task definition loading
└── episodes/                        # Episode management components
    ├── __init__.py                  # Episode exports
    ├── episode.py                   # RL episode data structures
    └── episode_manager.py           # Episode lifecycle management and State enum
```

## Refactored Components

### TaskManager (Simplified Orchestrator)
- **Task Storage & Retrieval**: Core task and subtask access methods
- **Episode Lifecycle Coordination**: Delegates to EpisodeManager for episode operations
- **RL Gym Interface**: Provides step() and reset() methods for reinforcement learning

### TaskConfigLoader (Component)
- **YAML Parsing**: Loads and validates task definitions from YAML files
- **Task Creation**: Converts YAML data into Task and SubTask objects

### EpisodeManager
- **Episode Lifecycle**: Start, step, end operations with unified step() method
- **Step Creation**: Returns Step objects directly from step() method

### Task
- **Task Definition**: YAML-based task specifications

### SubTask
- **SubTask Definition**: YAML-based task specifications

### Episode
Complete task attempt representation:
- **Action History**: Full sequence of actions and responses
- **State Tracking**: Checkpoint progression and completion status

## Task Definition Format

Tasks are defined in YAML files with the new checkpoint-based format:

```yaml
domain: "malware_classification"
tasks:
  - task_id: "malware_family_analysis"
    title: "Malware Family Classification and Analysis"
    description: "Analyze malware sample to determine family, capabilities, and threat level"
    initial_context:
      sample_path: "/data/samples/unknown_sample.exe"
      analysis_timeout: 300
    subtasks:
      - subtask_id: "static_analysis"
        title: "Static Analysis"
        description: "Perform static analysis of the malware sample"
        objective: "Extract basic file properties, strings, and structural information"
        completion_conditions:  # New: exact commands that must be executed
          - "file ${sample_path}"
          - "strings ${sample_path}"
          - "objdump -h ${sample_path}"
        depends_on: []  # No dependencies - entry point
      
      - subtask_id: "dynamic_analysis"
        title: "Dynamic Analysis"
        description: "Execute sample in sandboxed environment"
        objective: "Observe runtime behavior and system interactions"
        completion_conditions:  # New: exact commands for completion
          - "sandbox_execute ${sample_path}"
          - "monitor_behavior ${sample_path}"
          - "capture_network ${sample_path}"
        depends_on: ["static_analysis"]  # Must complete static_analysis first
```

## Episode States

Episodes progress through RL-compatible states:

```python
class EpisodeState(Enum):
    CREATED = "created"       # Episode created, not started
    ACTIVE = "active"         # Currently executing
    COMPLETED = "completed"   # Episode completed successfully
    FAILED = "failed"         # Episode failed
    TIMEOUT = "timeout"       # Episode timed out
    RESET = "reset"           # Episode was reset
```

## RL Episode Execution Flow

1. **Episode Start**: Create new Episode for task attempt
2. **Action Execution**: Agent executes actions (tool calls)
3. **Step Recording**: Record action-response pairs as Steps
5. **Episode End**: Complete episode when all checkpoints satisfied or failure occurs
6. **Episode Reset**: Optionally reset for new attempt

## Integration with SessionManager

The TaskManager integrates with SessionManager:

```python
# SessionManager delegates to TaskManager components
session_manager = SessionManager()
task_manager = session_manager.task_manager

# Start new episode
episode = task_manager.start_episode(session_id="client_001", task_id="malware_analysis")

# Execute tool and record as episode step
action = Action(tool_name="docker_cli_executor", parameters={"command": "strings /sample.exe"})
tool_result = session_manager.execute_tool("docker_cli_executor", {"command": "strings /sample.exe"})
step_result = task_manager.step(session_id="client_001", action=action, tool_result=tool_result)
```
