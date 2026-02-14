# SABER Episodes Framework

This module manages episode lifecycle, transcript coordination, and WebSocket-based push-only protocol.

## Architecture

```
src/saber/server/episodes/
├── __init__.py
├── connection_manager.py   # WebSocket connection tracking
├── constants.py            # Episode constants and metadata keys
├── episode_manager.py      # Episode lifecycle management
├── exceptions.py           # Episode-specific exceptions
├── protocols.py            # Protocol interfaces
└── transcript/
    ├── __init__.py
    ├── auto_continue_manager.py  # Auto-continue injection
    ├── coordinator.py            # Central transcript orchestrator
    ├── repository.py             # Redis-backed transcript storage
    ├── state_machine.py          # Transcript state detection
    └── stuck_state_monitor.py    # Stuck episode detection
```

## Key Components

### EpisodeManager (`episode_manager.py`)
Episode lifecycle management:
- **Episode Lifecycle**: Manages episode creation, execution, and termination
- **Session Coordination**: Coordinates with SessionManager for episode state
- **Failure Recovery**: Automatically triggers cleanup when episodes fail or are terminated
- **Error Handling**: Removes episode tracking immediately on errors to trigger cleanup

### TranscriptCoordinator (`transcript/coordinator.py`)
Central orchestrator for transcript operations:
- **Push Processing**: Handles push_message requests from clients
- **State Broadcasting**: Broadcasts state events via WebSocket after mutations
- **Red Team Injection**: Supports cross-episode message injection
- **Restart Operations**: Resets transcript to initial checkpoint

### TranscriptStateMachine (`transcript/state_machine.py`)
Detects transcript state from message history:
- **WAITING_FOR_USER**: Last message is from assistant (no pending tool_calls)
- **WAITING_FOR_ASSISTANT**: Last message is from user or all tools responded
- **WAITING_FOR_TOOLS**: Assistant has pending tool_calls without responses

## Episode Workflow

1. **Episode Creation**: EpisodeManager creates new episode for session
2. **WebSocket Connection**: Client connects, receives initial state event
3. **Push-Only Protocol**: Client pushes messages, server broadcasts state events
4. **Normal Completion**: Episode ends normally, triggering session cleanup
5. **Error Handling**: Episode fails, tracking removed to trigger immediate cleanup
6. **Session Cleanup**: SessionManager coordinates with ExecutionManager for container cleanup

This design ensures robust episode management with clear separation of concerns between episode tracking, transcript coordination, and container lifecycle.
