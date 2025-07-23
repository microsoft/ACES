# SABER Task Management System

The SABER TaskManager provides a robust framework for managing complex multi-step security tasks. Tasks are defined in YAML files and executed through stateful sessions that maintain context between subtasks.

## Architecture

```
src/saber/server/tasks/
├── __init__.py              # Task management exports
├── task_manager.py          # Main TaskManager orchestrator
├── domain_task.py           # High-level security task representation
├── subtask.py               # Individual task steps with dependencies
├── task_session.py          # Stateful session management
├── enums.py                 # Task and session state enums
└── exceptions.py            # Task management exceptions
```

## Key Components

### TaskManager
Main orchestrator for task execution:
- **Task Management**: Load and manage domain-specific tasks
- **Session Creation**: Create and track task execution sessions
- **Subtask Orchestration**: Handle subtask dependencies and progression
- **Context Management**: Maintain state between subtasks

### DomainTask
High-level security task representation:
- **Task Definition**: YAML-based task specifications
- **Subtask Organization**: Ordered list of subtasks with dependencies
- **Initial Context**: Starting context for task execution

### SubTask
Individual task steps with specific objectives:
- **Requirements**: Required tools and success criteria
- **Dependencies**: Task dependencies and context requirements
- **Validation**: Success criteria and completion checking

### TaskSession
Stateful session management:
- **Session State**: Track execution progress and status
- **Context Propagation**: Maintain context between subtasks
- **Progress Tracking**: Monitor subtask completion and results

## Task Definition Format

Tasks are defined in YAML files with the following structure:

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
        required_tools: ["file_analyzer", "string_extractor", "pe_parser"]
        success_criteria:
          - "File type and architecture identified"
          - "Suspicious strings extracted"
          - "PE structure analyzed (if applicable)"
        context_dependencies: []
        depends_on: []
      
      - subtask_id: "dynamic_analysis"
        title: "Dynamic Analysis"
        description: "Execute sample in sandboxed environment"
        objective: "Observe runtime behavior and system interactions"
        required_tools: ["sandbox_executor", "behavior_monitor", "network_monitor"]
        success_criteria:
          - "Sample executed successfully"
          - "System calls captured"
          - "Network activity logged"
        context_dependencies: ["static_analysis.file_type"]
        depends_on: ["static_analysis"]
```

## Session States

Task sessions progress through defined states:

```python
class SessionState(Enum):
    PENDING = "pending"       # Session created, not started
    ACTIVE = "active"         # Currently executing
    PAUSED = "paused"         # Temporarily paused
    COMPLETED = "completed"   # All subtasks completed
    FAILED = "failed"         # Execution failed
```

## Task Execution Flow

1. **Task Loading**: Load task definition from YAML
2. **Session Creation**: Create new TaskSession for client
3. **Initial Subtask**: Assign first subtask to client
4. **Subtask Execution**: Client executes subtask using required tools
5. **Context Update**: Update session context with results
6. **Dependency Check**: Verify dependencies for next subtask
7. **Progress**: Continue until all subtasks completed

## Context Management

Context is maintained throughout task execution:

```python
# Initial context from task definition
initial_context = {
    "sample_path": "/data/samples/unknown_sample.exe",
    "analysis_timeout": 300
}

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
