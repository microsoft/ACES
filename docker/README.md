# SABER Episode Container Orchestrator

This directory contains the SABER Episode Container Orchestrator - a robust solution for managing Docker container lifecycles during SABER episodes.

## Problem Solved

During SABER episode execution, Docker containers are dynamically created for:
- Execution environments (where agent commands run)
- Target services (vulnerable web apps, databases, etc.)
- Support services (monitoring, logging, etc.)

When episodes fail or are terminated due to errors, these containers can become "orphaned" - continuing to run and consume resources even though the episode is no longer active.

## Solution: External Container Orchestrator

Instead of modifying individual container entrypoints, we use a dedicated **Episode Container Orchestrator** that:

1. **Monitors Episode Status**: Polls the SABER server to check if the episode is still active
2. **Manages All Containers**: Uses Docker Compose to orchestrate the entire episode environment
3. **Performs Cleanup**: When episode becomes inactive, gracefully shuts down all containers
4. **Handles Failures**: Multiple fallback strategies ensure cleanup even in failure scenarios

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SABER Server  │────│  Orchestrator    │────│ Episode         │
│                 │    │  Container       │    │ Containers      │
│ - Episode Mgmt  │    │                  │    │                 │
│ - Status API    │    │ - Polls Status   │    │ - Execution     │
│ - Error Removal │    │ - Manages All    │    │ - Web App       │
│                 │    │ - Cleanup on End │    │ - Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Files

- **`episode_orchestrator.py`**: Main orchestrator script that monitors and manages containers
- **`Dockerfile.orchestrator`**: Dockerfile for building the orchestrator container
- **`docker-compose.orchestrator-example.yml`**: Example showing integration pattern
- **`test_cleanup_integration.py`**: Test script for the episode cleanup coordination

## How It Works

### 1. Episode Creation
When a SABER episode starts:
```python
# EpisodeManager creates episode with cleanup token
episode, cleanup_token = episode_manager.start_episode(session_id, task_id)

# Token is passed to Docker environment
environment_vars = {
    "SABER_SESSION_ID": session_id,
    "SABER_CLEANUP_TOKEN": cleanup_token,
    "SABER_COMPOSE_PROJECT": f"saber-{session_id}"
}
```

### 2. Container Orchestration
Docker Compose starts all containers including the orchestrator:
```yaml
services:
  saber-orchestrator:
    # Monitors episode status
    environment:
      - SABER_SESSION_ID=${SABER_SESSION_ID}
      - SABER_CLEANUP_TOKEN=${SABER_CLEANUP_TOKEN}
  
  execution:
    # Your execution container
    labels:
      - "saber.session_id=${SABER_SESSION_ID}"
  
  webapp:
    # Your target service
    labels:
      - "saber.session_id=${SABER_SESSION_ID}"
```

### 3. Status Monitoring
The orchestrator continuously polls:
```bash
GET /internal/episode-status/{session_id}?token={cleanup_token}
# Returns: {"active": true/false}
```

### 4. Error-Triggered Cleanup
When any error occurs in SABER:
```python
# SessionManager catches any error and immediately removes episode
try:
    result = await execution_manager.step(action, context)
except Exception as e:
    # KEY INSIGHT: Remove episode tracking immediately
    episode_manager.remove_episode_on_error(session_id, e)
    # Orchestrator will detect this and cleanup containers
```

### 5. Container Cleanup Strategies
The orchestrator uses multiple fallback strategies:

1. **Graceful Stop**: `docker compose stop` with timeout
2. **Force Cleanup**: `docker compose down --remove-orphans --volumes`
3. **Nuclear Option**: Find containers by label and force remove

## Benefits

### ✅ **Non-Intrusive**
- Doesn't modify existing container entrypoints
- Works with any container image (web apps, databases, etc.)
- No changes needed to your application containers

### ✅ **Robust Cleanup**
- Multiple fallback strategies ensure cleanup
- Handles network partitions and failures
- Recovers from partial cleanup states

### ✅ **Immediate Response**
- Episode removal on errors triggers immediate cleanup
- No waiting for polling intervals during error conditions

### ✅ **Complete Orchestration**
- Manages all episode resources (containers, networks, volumes)
- Uses Docker Compose for coordinated lifecycle management
- Proper dependency ordering with `depends_on`

## Usage

### Building the Orchestrator
```bash
cd docker/
docker build -f Dockerfile.orchestrator -t saber-orchestrator:latest .
```

### Integration in Your Environment Templates
Add the orchestrator to your environment specifications:

```yaml
services:
  saber-orchestrator:
    image: saber-orchestrator:latest
    environment:
      - SABER_SESSION_ID=${SABER_SESSION_ID}
      - SABER_CLEANUP_TOKEN=${SABER_CLEANUP_TOKEN}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    
  # Your existing services...
  execution:
    image: your-execution-image
    labels:
      - "saber.session_id=${SABER_SESSION_ID}"
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SABER_SESSION_ID` | Yes | - | Session identifier |
| `SABER_CLEANUP_TOKEN` | Yes | - | Authentication token |
| `SABER_HOST_URL` | No | `http://host.docker.internal:8000` | SABER server URL |
| `SABER_POLL_INTERVAL` | No | `30` | Polling interval in seconds |
| `SABER_COMPOSE_PROJECT` | Yes | - | Docker Compose project name |
| `SABER_GRACEFUL_TIMEOUT` | No | `30` | Graceful shutdown timeout |

## Testing

Run the integration test:
```bash
cd docker/
python test_cleanup_integration.py
```

This verifies:
- Episode creation with cleanup tokens
- Status polling endpoint
- Error-triggered episode removal
- Container coordination flow

## Security Notes

- Orchestrator needs access to Docker socket (`/var/run/docker.sock`)
- Cleanup tokens provide authentication for status checks
- All containers are labeled with session ID for identification
- Nuclear cleanup only affects containers with correct session labels
