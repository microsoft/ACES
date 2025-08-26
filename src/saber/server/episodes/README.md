# SABER Episodes Framework

This module manages episode lifecycle and coordination with the execution framework for container cleanup.

## Architecture

```
src/saber/server/episodes/
├── episode_manager.py      # Episode lifecycle management
└── exceptions.py          # Episode-specific exceptions
```

## Key Components

### EpisodeManager (`episode_manager.py`)
Episode lifecycle management:
- **Episode Lifecycle**: Manages episode creation, execution, and termination
- **Session Coordination**: Coordinates with SessionManager for episode state
- **Failure Recovery**: Automatically triggers cleanup when episodes fail or are terminated
- **Error Handling**: Removes episode tracking immediately on errors to trigger cleanup

## Episode Workflow

1. **Episode Creation**: EpisodeManager creates new episode for session
2. **Episode Execution**: Episodes track steps and completion status
3. **Normal Completion**: Episode ends normally, triggering session cleanup
4. **Error Handling**: Episode fails, tracking removed to trigger immediate cleanup
5. **Session Cleanup**: SessionManager coordinates with ExecutionManager for container cleanup

This design ensures robust episode management with clear separation of concerns between episode tracking and container lifecycle.
