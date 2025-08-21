# SABER Episodes Framework

This module manages episode lifecycle and orchestrates container cleanup to prevent orphaned Docker containers during failures.

## Architecture

```
src/saber/server/episodes/
├── episode_manager.py      # Episode lifecycle and cleanup token coordination
├── episode_orchestrator.py # External container orchestrator for cleanup
└── exceptions.py          # Episode-specific exceptions
```

## Key Components

### EpisodeManager (`episode_manager.py`)
Episode lifecycle management with cleanup coordination:
- **Episode Lifecycle**: Manages episode creation, execution, and termination
- **Cleanup Tokens**: Generates secure tokens for orchestrator authentication during cleanup
- **Failure Recovery**: Automatically triggers cleanup when episodes fail or are terminated
- **Error Coordination**: Passes cleanup tokens to execution layer for orchestrator communication

### EpisodeContainerOrchestrator (`episode_orchestrator.py`)
External container orchestrator for robust cleanup:
- **Token Authentication**: Validates cleanup tokens from episode manager
- **Container Monitoring**: Tracks all session-related Docker containers
- **Orphan Prevention**: Ensures containers are cleaned up even when episodes fail unexpectedly
- **Background Operation**: Runs as separate container to survive session manager failures
- **Graceful Cleanup**: Handles both graceful termination and forced cleanup scenarios

## Cleanup Workflow

1. **Episode Creation**: EpisodeManager generates cleanup token and starts episode
2. **Token Distribution**: Cleanup token passed to ExecutionManager for orchestrator coordination
3. **Container Creation**: Sandbox containers created with orchestrator monitoring
4. **Normal Cleanup**: Episode ends normally, orchestrator removes containers via token
5. **Failure Cleanup**: Episode fails, cleanup token triggers orchestrator cleanup automatically
6. **Orphan Recovery**: Orchestrator detects and removes orphaned containers from failed sessions

This design ensures robust cleanup across all failure modes while maintaining security through token-based authentication.
