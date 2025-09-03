# Excytin Demo Domain

## Overview

The `excytin_demo` domain is designed to demonstrate and test SABER's container-based execution capabilities. This domain showcases how SABER:

1. **Docker Compose configurations** used to spin up environments
2. **Container networking** and communication between services  
3. **Container lifecycle events** (start, stop, health checks, failures)

## What We're Testing

- **Container Management**: SABER's container orchestration and lifecycle management
- **Environment Creation Flow**: Episodes should create Docker environments with MySQL containers
- **Volume Mounts**: Host-mounted logs directory (`./server/logs:/app/logs:rw`) for persistent log storage
- **Multi-container Orchestration**: Testing database + execution container setups

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Host System                                     │
│  ┌─────────────────────────────────────────────┐│
│  │ excytin_demo/                               ││
│  │  ├── server/                                ││
│  │  │   ├── config/                            ││
│  │  │   │   ├── tasks.yaml                     ││
│  │  │   │   └── environments.yaml              ││
│  │  │   ├── data/                              ││
│  │  │   │   └── sql_files/                     ││
│  │  │   │       └── incident_5.sql             ││
│  │  │   └── logs/                              ││
│  │  │       └── [container logs appear here]   ││
│  │  ├── client/                                ││
│  │  │   └── demo_client.py                     ││
│  │  └── docker-compose.yml                     ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
              │ Docker-in-Docker │
        ┌─────────────────────────────────┐
        │ SABER Server Container          │
        │  /app/config/ (mounted)         │
        │  /app/data/ (mounted)           │
        │  /app/logs/ (mounted)           │
        │  /app/src/ (mounted)            │
        │                                 │
        │  Creates episodes that spawn:   │
        │  ┌─────────────────────────────┐│
        │  │ Episode Container           ││
        │  │  - MySQL Database           ││
        │  │  - Application Services     ││
        │  │  - [Logs captured to host]  ││
        │  └─────────────────────────────┘│
        └─────────────────────────────────┘
```

## Cold Start Instructions

### 1. Prerequisites

Ensure you have:
- Docker and Docker Compose installed
- Python with `uv` package manager
- Access to the SABER repository

### 2. Build Images

From the repository root:

```bash
cd /path/to/SABER/domains/excytin_demo
./build-images.sh
```

This builds:
- `saber/excytin-server:latest` - SABER server for excytin_demo domain
- `saber/excytin-client:latest` - Demo client for testing
- `saber/excytin-sandbox:latest`  - Sandbox execution environment with mysql client"
- `saber/excytin-incident-5:latest`   - Custom MySQL with SQL data (Docker-in-Docker workaround)"

### 3. Start Environment

```bash
cd /path/to/SABER/domains/excytin_demo
docker-compose up -d
```

This starts:
- `saber-excytin-server` - Main SABER server (ports 8000/8001)
- `saber-excytin-incident-5` - Permanent database container that is tied to the server lifecycle
 - NOTE: This is docker composed from WITHIN the server, not in the domain docker compose. Look in server/logs/comopse-configs/permanent-environments for the latest compose setup
- `saber-excytin-client` - Demo client container

### 4. Verify Server is Running

```bash
# Check server logs
docker logs saber-excytin-server --tail 20

# Should see:
# - "SessionManager initialized successfully"
# - "Starting SABER server..."
# - "Uvicorn running on http://0.0.0.0:8000"
```

### 5. Run Demo Client

The excytin demo now uses the unified SABER client entry point with enhanced UI integration:

#### Quick Start (Recommended)
```bash
cd /home/ms_test/repos/saber_vibin/domains/excytin_demo/client
./run_demo.sh
```

#### Manual Execution
```bash
# Plain UI with clean logging (default)
docker exec -it saber-excytin-client uv run python -m saber.client 
  --agent /app/client/demo_agent.py 
  --tasks excytin_demo 
  --ui plain 
  --quiet-logs

# Rich UI with progress bars
docker exec -it saber-excytin-client uv run python -m saber.client 
  --agent /app/client/demo_agent.py 
  --tasks excytin_demo 
  --ui rich 
  --quiet-logs

# Full interactive textual UI
docker exec -it saber-excytin-client uv run python -m saber.client 
  --agent /app/client/demo_agent.py 
  --tasks excytin_demo 
  --ui textual 
  --quiet-logs
```

#### Script Options
```bash
./run_demo.sh --help                    # Show help
./run_demo.sh --ui rich                 # Rich UI mode
./run_demo.sh --ui textual --verbose    # Full TUI with debug logs
./run_demo.sh --console-logs            # Show logs on console
```

## Expected Behavior

### Successful Flow:
1. **Session Creation**: Demo client creates a session
2. **Benchmark Start**: Triggers `excytin_demo` task  
3. **Episode Creation**: Episode created for the task
4. **Environment Setup**: MySQL container (`saber-excytin-incident-5`) should be created
5. **Container Communication**: 
   - Docker Compose config saved to `./server/logs/`
   - Lifecycle events logged
6. **Verification**: Check `./server/logs/` for captured logs

### Current Issue (As of Investigation):
- ✅ Sessions create successfully
- ✅ Episodes create successfully  
- ❌ **Environments are NOT being created**
- ❌ No Docker containers spawned for episodes
- ❌ No container logs captured (because no containers exist)

## Debugging

### Check Server Logs
```bash
docker logs saber-excytin-server | grep -E "(DEBUG|ERROR|environment|episode|container)"
```

### Check Episode Status
Look for log messages like:
- `"Created episode '...' for session '...'"` ✅
- `"ExecutionManager configured for session..."` ✅  
- `"Created sandbox environment for session..."` ❌ (Missing!)

### Check Container Creation
```bash
# Should see episode-specific containers
docker ps -a | grep saber-session-

# Should see episode containers like:
# saber-session-[session-id]-mysql...
```

### Check Logs Directory
```bash
# Find the latest session logs
ls -la ./client/logs/

# Check unified logging structure  
ls -la ./client/logs/{timestamp}/
# Should contain:
#   harness-execution/     - Harness execution logs
#   container-logs/        - All container logs
#   client-logs/          - Client application logs  
#   container-events/     - Container lifecycle events

# Check client logs
tail -f ./client/logs/{timestamp}/client-logs/saber_client.log

# Check agent execution logs
ls ./client/logs/{timestamp}/container-logs/agent-containers/

# Check container events
cat ./client/logs/{timestamp}/container-events/container-events-*.jsonl
```

## Configuration Files

### `server/config/tasks.yaml`
Defines the `excytin_demo` task that:
- Uses environment `excytin_incident_5`
- Has 3 subtasks for container interaction
- Allows `cli` and `python` executors

### `server/config/environments.yaml`  
Defines the `excytin_incident_5` environment:
- MySQL 8.0 container
- Initializes with `incident_5.sql` data
- Exposes port 3306
- Creates `env_monitor_db` database

### `server/data/sql_files/incident_5.sql`
Sample database schema with:
- `container_logs` table for logging demo
- `environment_status` table for health monitoring
- Sample data for testing

## Troubleshooting

### No Containers Created
If episodes are created but no Docker containers appear:
1. Check environment resolution in server logs
2. Verify `environments.yaml` syntax
3. Check Docker-in-Docker permissions
4. Verify volume mounts in docker-compose.yml

### Volume Mount Issues
Ensure paths exist and are accessible:
```bash
ls -la ./server/config/  # Should contain tasks.yaml, environments.yaml
ls -la ./server/data/    # Should contain sql_files/
ls -la ./server/logs/    # Should exist (may be empty initially)
```

### Permission Issues
```bash
# Fix log directory permissions if needed
chmod 755 ./server/logs/
```

## What Success Looks Like

When everything works correctly:

1. **Episode Creates Environment**: 
   ```
   Created sandbox environment for session [session-id]
   ```

2. **Docker Containers Spawn**:
   ```bash
   docker ps
   # Shows: saber-session-[id]-mysql, etc.
   ```

3. **Unified Logs Are Captured**:
   ```bash
   ls ./logs/{timestamp}/
   # Shows unified logging structure:
   #   ├── harness-execution/        # Harness execution logs
   #   ├── container-logs/
   #   │   ├── agent-containers/     # Agent execution containers  
   #   │   └── sidecar-containers/   # MCP sidecar containers
   #   ├── client-logs/              # Client application logs
   #   ├── container-events/         # Container lifecycle events
   #   ├── system.log               # System logs
   #   └── meta.json                # Session metadata
   ```

4. **Container Networking Works**: Host filesystem contains all debugging information needed to diagnose container issues.

5. **Clean UI Output**: Detailed logs saved to files, console shows only essential UI messages and progress.

## Docker-in-Docker Limitations

**Issue**: MySQL container fails with "Can't initialize batch_readline" when mounting SQL files from SABER server container.

**Root Cause**: Docker-in-Docker can't mount files from parent container filesystem to child containers.

**Current Solution**: Custom MySQL image with SQL files baked in during build.
- **Pros**: Reliable, standard Docker pattern, works in all environments
- **Cons**: Increases image size, requires rebuild when data changes

**Production Fix**: Use proper volume mounting from host filesystem or external database services.

## Next Steps

Once this domain works correctly, the Excytin capabilities can be applied to other domains like `webapp_pentest` for more complex multi-container pentesting scenarios.
