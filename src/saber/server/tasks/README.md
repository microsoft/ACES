# SABER Task Management System

The SABER TaskManager provides a robust framework for managing complex multi-step security tasks with RL-friendly episode-based execution. Tasks are defined in YAML files and executed through stateful episodes that track complete action-response sequences.

## Architecture

```
src/saber/server/tasks/
├── __init__.py              # Task management exports
├── task_manager.py          # Main TaskManager orchestrator
├── task.py                  # High-level security task representation (renamed from domain_task.py)
├── subtask.py               # Internal checkpoints with automatic progression
├── task_session.py          # Stateful session management
├── episode.py               # RL episode data structures
├── episode_manager.py       # RL episode lifecycle management
├── enums.py                 # Task, session, and episode state enums
└── exceptions.py            # Task management exceptions
```

## Key Components

### TaskManager
Main orchestrator for task execution:
- **Task Management**: Load and manage domain-specific tasks
- **Session Creation**: Create and track task execution sessions
- **Episode Management**: RL-style episode lifecycle with gym compatibility
- **Automatic Progression**: Handle checkpoint progression based on command execution

### Task (formerly DomainTask)
High-level security task representation:
- **Task Definition**: YAML-based task specifications
- **Checkpoint Organization**: Subtasks used as internal checkpoints
- **Progression Logic**: Automatic checkpoint advancement based on episode state

### SubTask (Refactored for Checkpoints)
Internal checkpoints with automatic progression:
- **Completion Conditions**: Exact commands that must be executed
- **Entry/Exit Criteria**: Dependency-based checkpoint validation
- **Command Matching**: Template-based command execution tracking

### EpisodeManager
RL-friendly episode management:
- **Episode Lifecycle**: Start, step, end, reset operations
- **Action-Response Tracking**: Complete history of agent interactions
- **Replay Capability**: Episode replay for analysis and training

### Episode
Complete task attempt representation:
- **Action History**: Full sequence of actions and responses
- **State Tracking**: Checkpoint progression and completion status
- **Metadata**: Episode timing, attempt numbers, and context

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
4. **Checkpoint Progression**: Automatically advance checkpoints based on command execution
5. **Episode End**: Complete episode when all checkpoints satisfied or failure occurs
6. **Episode Reset**: Optionally reset for new attempt

## Automatic Checkpoint Progression

Checkpoints advance automatically based on command execution:

```python
# Example: static_analysis checkpoint completes when these commands are executed:
completion_conditions = [
    "file ${sample_path}",      # File type identification
    "strings ${sample_path}",   # String extraction  
    "objdump -h ${sample_path}" # PE header analysis
]

# Context updated after each subtask
updated_context = {
    "sample_path": "/data/samples/unknown_sample.exe",
    "analysis_timeout": 300,
    "static_analysis": {
        "file_type": "PE32 executable",
        "strings": ["suspicious_string1", "suspicious_string2"],
        "pe_info": {...}
    }
}
```

## Dependency Management

Subtasks can depend on:
- **Execution Dependencies**: Other subtasks that must complete first
- **Context Dependencies**: Specific context data from previous subtasks

```yaml
subtask:
  depends_on: ["static_analysis"]  # Must complete first
  context_dependencies: ["static_analysis.file_type"]  # Needs this data
```

## Integration with SessionManager

The TaskManager integrates with SessionManager for client communication:

```python
# SessionManager delegates to TaskManager
session_manager = SessionManager()
task_manager = session_manager.task_manager

# Get current subtask for client
current_subtask = task_manager.get_current_subtask(session_id)

# Complete subtask and get next
next_subtask = task_manager.complete_subtask(session_id, results)
```

## Testing

Test the task management system:

```bash
# Run task manager tests
uv run pytest tests/test_task_manager.py -v

# Test specific functionality
uv run pytest tests/test_task_manager.py::test_task_loading -v
uv run pytest tests/test_task_manager.py::test_session_management -v
```

## Example Usage

```python
from saber.server.tasks import TaskManager

# Initialize task manager
task_manager = TaskManager()

# Load tasks from YAML file
task_manager.load_tasks_from_file("malware_classification/tasks.yaml")

# Create session for client
session = task_manager.create_task_session(
    task_id="malware_family_analysis",
    client_id="agent_001"
)

# Get first subtask
current_subtask = task_manager.get_current_subtask(session.session_id)

# Complete subtask and advance
results = {"file_type": "PE32 executable", "strings": [...]}
next_subtask = task_manager.complete_subtask(session.session_id, results)
```
