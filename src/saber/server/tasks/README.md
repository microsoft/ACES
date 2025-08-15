# SABER Task Management System

The SABER TaskManager provides task definition management for security benchmarking scenarios. TaskManager loads and provides access to task definitions while EpisodeManager handles episode lifecycle operations.

## Architecture

```
src/saber/server/tasks/
├── __init__.py                      # Task management exports
├── task_manager.py                  # Task definition manager
├── task.py                          # Task definition class
├── subtask.py                       # SubTask definition class
├── task_config_loader.py            # YAML parsing and task loading
└── exceptions.py                    # Task management exceptions

src/saber/server/episodes/
├── __init__.py                      # Episode management exports
├── episode_manager.py               # Episode lifecycle management
└── exceptions.py                    # Episode management exceptions
```

## Components

### TaskManager
- **Task Storage & Retrieval**: Core task and subtask access methods
- **YAML Configuration Loading**: Loads task definitions from configuration files

### TaskConfigLoader
- **YAML Parsing**: Loads and validates task definitions from YAML files
- **Task Creation**: Converts YAML data into Task and SubTask objects

### Task
- **Task Definition**: Contains task metadata, environment specification, and execution configuration
- **Subtask Management**: Contains list of subtasks for informational purposes

### SubTask
- **SubTask Definition**: Individual task components with objectives and descriptions

### EpisodeManager (separate component)
- **Episode Lifecycle**: Start, step, and end operations for RL-compatible episodes
- **Step Creation**: Returns Step objects from episode progression

## Task Definition Format

Tasks are defined in YAML files with the new checkpoint-based format:

```yaml
domain: "malware_classification"

# Optional: Executor filtering configuration
# Restricts which command executors are available for this domain's tasks
# If omitted, all available executors (cli, python, curl, sql, file_io) are enabled
executors:
  - "cli"
  - "python"
  # Note: any other executor is disabled for this domain

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
