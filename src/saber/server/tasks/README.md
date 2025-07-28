# SABER Task Management System (Refactored)

The SABER TaskManager provides a robust framework for managing complex multi-step security tasks with RL-friendly episode-based execution. The system has been refactored into specialized components following single responsibility principles for better maintainability and testability.

## Refactored Architecture

```
src/saber/server/tasks/
├── __init__.py                      # Task management exports
├── task_manager.py                  # Simplified orchestrator (308 lines, down from 655)
├── base.py                          # Task and episode state enums, Observation class
├── exceptions.py                    # Task management exceptions
├── core/                            # Core task definitions and specialized components
│   ├── __init__.py                  # Core exports
│   ├── task.py                      # High-level security task representation
│   ├── subtask.py                   # Internal checkpoints with automatic progression
│   ├── task_config_loader.py        # YAML parsing and task definition loading (NEW)
│   └── subtask_progression_engine.py # DAG progression logic and state management (NEW)
└── episodes/                        # Episode management components
    ├── __init__.py                  # Episode exports
    ├── episode.py                   # RL episode data structures
    └── episode_manager.py           # Enhanced episode lifecycle management
```

## Refactored Components

### TaskManager (Simplified Orchestrator)
**Reduced from 655 to 308 lines** - now focuses on coordination:
- **Task Storage & Retrieval**: Core task and subtask access methods
- **Episode Lifecycle Coordination**: Delegates to EpisodeManager for episode operations
- **RL Gym Interface**: Provides step() and reset() methods for reinforcement learning
- **Component Coordination**: Orchestrates TaskConfigLoader, SubTaskProgressionEngine, and EpisodeManager
- **Session Management**: Links with SessionManager for client interactions

### TaskConfigLoader (New Component)
**Specialized YAML configuration management** (~150 lines):
- **YAML Parsing**: Loads and validates task definitions from YAML files
- **Task Creation**: Converts YAML data into Task and SubTask objects
- **Dependency Validation**: Ensures all subtask dependencies are valid and acyclic
- **Domain Consistency**: Validates task definitions match expected domain
- **Error Handling**: Comprehensive validation with detailed error messages

### SubTaskProgressionEngine (New Component)
**DAG-based progression logic** (~150 lines):
- **Automatic Progression**: Handles subtask state transitions based on command execution
- **Entry/Exit Conditions**: Validates when subtasks can be started or completed
- **Command Matching**: Tracks executed commands against completion conditions
- **State Management**: Updates completed/in_progress/not_visited subtask sets
- **Dependency Resolution**: Ensures DAG constraints are respected during progression

### EpisodeManager
Enhanced episode management for RL workflows:
- **Episode Lifecycle**: Start, step, end operations with unified step() method
- **Command Extraction**: Extract commands from DockerCLIExecutor for completion matching
- **RL Integration**: Provides gym-compatible interfaces and Observation building
- **Simplified Operation**: No episode history tracking, focus on current state

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

### Episode
Complete task attempt representation:
- **Action History**: Full sequence of actions and responses
- **State Tracking**: Checkpoint progression and completion status
- **Current Episode Only**: Simplified to active episode per session

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

## Refactoring Benefits

### Improved Maintainability
- **Single Responsibility**: Each component has one clear purpose
- **Reduced Complexity**: TaskManager reduced from 655 to 308 lines
- **Better Testing**: Specialized components can be tested independently
- **Clearer Dependencies**: Component relationships are explicit and minimal

### Enhanced Modularity
- **TaskConfigLoader**: Can be swapped for different configuration sources
- **SubTaskProgressionEngine**: Progression logic can be enhanced without affecting other components
- **EpisodeManager**: Episode handling is self-contained and feature-complete

### Better Separation of Concerns
- **Configuration**: TaskConfigLoader handles all YAML parsing
- **Business Logic**: SubTaskProgressionEngine manages DAG progression
- **State Management**: EpisodeManager handles episode lifecycle and observations
- **Orchestration**: TaskManager coordinates between components

## RL Gym Compatibility

The refactored system provides standard RL gym interfaces:

```python
# Standard RL gym pattern
task_manager = TaskManager(domain="malware_classification", tasks_file="tasks.yaml")

# Reset environment for new episode
episode = task_manager.reset(session_id="agent_001", task_id="malware_analysis")

# Execute actions and get observations
action = Action(tool_name="file", parameters={"path": "/sample.exe"})
tool_result = execute_tool_somehow()  # Tool execution happens first
step_result = task_manager.step(session_id="agent_001", action=action, tool_result=tool_result)

# Check if episode is complete
if step_result.done:
    print(f"Episode completed!")
```

## Integration with SessionManager

The refactored TaskManager integrates seamlessly with SessionManager:

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

## Testing

Test the refactored task management system:

```bash
# Run all task manager tests
uv run pytest tests/tasks/ -v

# Test specific components
uv run pytest tests/tasks/test_task_manager.py -v        # Core orchestration
uv run pytest tests/tasks/test_config_loader.py -v      # YAML parsing (if exists)
uv run pytest tests/tasks/test_progression_engine.py -v # DAG logic (if exists)
uv run pytest tests/tasks/test_episode_manager.py -v    # Episode management

# Test specific functionality
uv run pytest tests/test_task_manager.py::test_task_loading -v
uv run pytest tests/test_task_manager.py::test_session_management -v
```

## Example Usage (Refactored API)

```python
from saber.server.tasks import TaskManager
from saber.server.tasks.episodes.episode import Action

# Initialize task manager with domain and tasks file
task_manager = TaskManager(
    domain="malware_classification",
    tasks_file_path="malware_classification/tasks.yaml"
)

# RL-style usage pattern
session_id = "agent_001"
task_id = "malware_family_analysis"

# Reset environment for new episode
episode = task_manager.reset(session_id=session_id, task_id=task_id)
print(f"Started episode: {episode.episode_id}")

# Execute actions in RL gym style
while not episode.is_complete:
    # Create action for DockerCLIExecutor
    action = Action(
        tool_name="docker_cli_executor",
        parameters={"command": "file /data/samples/unknown_sample.exe"}
    )
    
    # Execute tool (handled by SessionManager) and record step
    tool_result = session_manager.execute_tool("docker_cli_executor", 
                                             {"command": "file /data/samples/unknown_sample.exe"})
    step_result = task_manager.step(session_id=session_id, action=action, tool_result=tool_result)
    
    # Check completion
    if step_result.done:
        print(f"Episode completed!")
        break

# Traditional task management usage
task = task_manager.get_task(task_id="malware_family_analysis")
subtask = task_manager.get_subtask(task_id="malware_family_analysis", subtask_id="static_analysis")

# Episode management
episode_info = task_manager.get_episode_info(session_id=session_id)
all_tasks = task_manager.list_tasks()
```

### Direct Component Usage

```python
from saber.server.tasks.core import TaskConfigLoader, SubTaskProgressionEngine
from saber.server.tasks.episodes import EpisodeManager

# Use components independently for testing or specialized needs
config_loader = TaskConfigLoader(domain="malware_classification")
tasks = config_loader.load_tasks_from_file("tasks.yaml")

progression_engine = SubTaskProgressionEngine()
episode_manager = EpisodeManager()

# Manual progression control (advanced usage)
task = tasks["malware_analysis"]
episode = episode_manager.start_episode("session_001", "malware_analysis")
progression_engine.initialize_episode_subtasks(task, episode)
```
