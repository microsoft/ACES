# Excytin Demo Domain - Enhanced Container Logging

## Overview

The `excytin_demo` domain is designed to demonstrate and test SABER's enhanced container logging capabilities. This domain showcases how SABER captures:

1. **Docker Compose configurations** used to spin up environments
2. **Container logs** from all services in the environment  
3. **Container lifecycle events** (start, stop, health checks, failures)

The logging system helps developers debug crashed containers by providing comprehensive visibility into what happened during execution.

## What We're Testing

- **Container Logging Manager**: Captures docker-compose configs and container logs
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
cd /home/ms_test/repos/saber_chillin/domains/excytin_demo
./build-images.sh
```

This builds:
- `saber-excytin-server:latest` - SABER server for excytin_demo domain
- `saber-excytin-client:latest` - Demo client for testing

### 3. Start Environment

```bash
cd /home/ms_test/repos/saber_chillin/domains/excytin_demo
docker-compose up -d
```

This starts:
- `saber-excytin-server` - Main SABER server (ports 8000/8001)
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

```bash
cd /home/ms_test/repos/saber_chillin/domains/excytin_demo
docker exec -it saber-excytin-client uv run demo_client.py --verbose
```

## Expected Behavior

### Successful Flow:
1. **Session Creation**: Demo client creates a session
2. **Benchmark Start**: Triggers `basic_logging_demo` task  
3. **Episode Creation**: Episode created for the task
4. **Environment Setup**: MySQL container (`saber-excytin-incident-5`) should be created
5. **Container Logging**: 
   - Docker Compose config saved to `./server/logs/`
   - Container logs captured and saved
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
ls -la ./server/logs/
# Should contain subdirectories per session with container logs
```

## Configuration Files

### `server/config/tasks.yaml`
Defines the `basic_logging_demo` task that:
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

3. **Logs Are Captured**:
   ```bash
   ls ./server/logs/[session-id]/
   # Shows: docker-compose.yml, container-logs/, lifecycle-events.json
   ```

4. **Enhanced Logging Works**: Host filesystem contains all debugging information needed to diagnose container issues.

## Docker-in-Docker Limitations

**Issue**: MySQL container fails with "Can't initialize batch_readline" when mounting SQL files from SABER server container.

**Root Cause**: Docker-in-Docker can't mount files from parent container filesystem to child containers.

**Current Solution**: Custom MySQL image with SQL files baked in during build.
- **Pros**: Reliable, standard Docker pattern, works in all environments
- **Cons**: Increases image size, requires rebuild when data changes

**Production Fix**: Use proper volume mounting from host filesystem or external database services.

## Next Steps

Once this domain works correctly, the enhanced logging capabilities can be applied to other domains like `webapp_pentest` for more complex multi-container pentesting scenarios.
