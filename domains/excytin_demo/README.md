# Excytin Demo Domain

## Overview

The `excytin_demo` domain demonstrates SABER's unified evaluation architecture using the new inspect_ai integration. This domain showcases how SABER:

1. **Unified Client Architecture**: Uses the new `python -m saber.client` entry point with YAML configuration
2. **Direct Agent Execution**: Agents run directly via inspect_ai without container overhead
3. **MCP Tool Integration**: Direct MCP client connections with per-episode session context
4. **Episode-Based Execution**: RL-style episodes with automatic termination and resource cleanup
5. **Evaluate-as-you-Code**: Real-time UI with progress bars and structured logging

## What We're Testing

- **Unified Client Architecture**: SABER's new inspect_ai-based evaluation framework
- **Episode Management**: Session/episode lifecycle with proper resource cleanup
- **MCP Tool Integration**: Direct agent-to-server communication via MCP tools (`cli`, `python`, `file_operations`)
- **Container Orchestration**: Sandbox environments created per episode with MySQL connectivity
- **Evaluation Orchestration**: Multi-sample dataset execution with fail-fast error handling
- **Structured Logging**: Timestamped logs with inspect_ai integration

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
│  │  │       ├── compose-configs/               ││
│  │  │       ├── container-logs/                ││
│  │  │       └── container-events-*.jsonl      ││
│  │  ├── client/                                ││
│  │  │   ├── saber.yaml                         ││
│  │  │   ├── run_demo.sh                        ││
│  │  │   └── logs/                              ││
│  │  │       ├── {timestamp}/                   ││
│  │  │       │   └── saber_client.log           ││
│  │  │       └── {timestamp}_task_*.eval        ││
│  │  └── docker-compose.yml                     ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
              │ SABER Unified Architecture │
        ┌─────────────────────────────────────────┐
        │ SABER Client Container                  │
        │  ┌───────────────────────────────────┐  │
        │  │ inspect_ai eval_async             │  │
        │  │  ├─ SABEREvaluationOrchestrator  │  │
        │  │  ├─ AgentManager (React Agent)   │  │
        │  │  ├─ DatasetManager               │  │
        │  │  └─ ClientSessionManager         │  │
        │  └───────────────────────────────────┘  │
        │              │ MCP/REST │               │
        └──────────────────────────────────────────┘
                       │
        ┌─────────────────────────────────────────┐
        │ SABER Server Container                  │
        │  ├─ SessionManager                      │
        │  ├─ EpisodeManager                      │
        │  ├─ ExecutionManager                    │
        │  └─ MCP Server (tools: cli, python, file)
        │                                         │
        │  Creates per-episode sandbox:           │
        │  ┌─────────────────────────────────────┐│
        │  │ Episode Sandbox Container           ││
        │  │  - excytin-sandbox image            ││
        │  │  - Connected to permanent MySQL     ││
        │  │  - Networked execution environment  ││
        │  │  - Logs captured automatically     ││
        │  └─────────────────────────────────────┘│
        └─────────────────────────────────────────┘
                       │
        ┌─────────────────────────────────────────┐
        │ Permanent MySQL Container               │
        │  - saber-excytin-incident-5             │
        │  - Runs for server lifetime             │
        │  - Shared across all episodes           │
        │  - Pre-loaded with incident_5.sql data │
        └─────────────────────────────────────────┘
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

**Option A: Quick Setup with Permissions Fix (Recommended)**
```bash
cd /path/to/SABER/domains/excytin_demo
./setup-permissions.sh
```

This automatically:
- Sets proper user permissions for log files
- Fixes existing log file ownership
- Starts containers with your host user ID
- Provides helpful usage examples

**Option B: Manual Setup**
```bash
cd /path/to/SABER/domains/excytin_demo

# Set environment variables for proper permissions
export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)

# Fix existing log permissions (optional)
sudo chown -R $DOCKER_UID:$DOCKER_GID ./client/logs/ ./server/logs/

# Start containers
docker compose up -d
```

**Important: Log File Permissions**
By default, Docker containers run as root and create log files with root ownership. The updated docker-compose.yml now includes user mapping (`user: "${DOCKER_UID:-1000}:${DOCKER_GID:-1000}"`) to ensure log files are created with your host user permissions, making them readable without sudo.

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

The excytin demo uses the unified SABER client with YAML configuration and inspect_ai integration:

#### Quick Start (Recommended)
```bash
cd /path/to/SABER/domains/excytin_demo/client
./run_demo.sh
```

#### Advanced Usage
```bash
# Verbose logging with file output
./run_demo.sh --verbose

# Console logging (no file)  
./run_demo.sh --console-logs

# Both verbose and console
./run_demo.sh --verbose --console-logs

# Manual execution
docker exec -it saber-excytin-client uv run python -m saber.client --config /app/client/saber.yaml
```

#### What the Demo Does
1. **Loads Configuration**: Parses `saber.yaml` for model, server, task, and agent settings
2. **Creates Session**: Establishes session with SABER server
3. **Executes eval_async**: Uses inspect_ai framework to run dataset samples
4. **Creates Episodes**: Each task attempt creates a new episode with sandbox environment
5. **Executes Agent**: React agent uses MCP tools (cli, python, file_operations) to complete tasks
6. **Captures Logs**: Structured logging to timestamped directories and inspect_ai eval files
7. **Cleanup**: Automatic resource cleanup when complete

## Expected Behavior

### Successful Flow:
1. **Session Creation**: Client creates session with SABER server
2. **Task Discovery**: Retrieves `excytin_demo` task configuration from server
3. **Dataset Creation**: Creates inspect_ai dataset with task samples (multiple attempts)
4. **Episode Execution**: For each sample:
   - Creates new episode with unique ID
   - Sets up MCP client connection
   - Creates sandbox environment (`excytin-sandbox` container)
   - Connects to permanent MySQL database (`saber-excytin-incident-5`)
   - Agent executes using available MCP tools
   - Episode automatically terminates on completion or max steps
5. **Logging**: Structured logs captured in timestamped directories
6. **Cleanup**: All resources automatically cleaned up

### Current Status:
- ✅ **Sessions create successfully**
- ✅ **Episodes create successfully** 
- ✅ **MCP client connections work**
- ✅ **Tool execution successful** (`cli`, `python`, `file_operations`)
- ✅ **Agent execution completes**
- ✅ **Structured logging works**
- ✅ **Resource cleanup automatic**
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
- `"Created sandbox environment for session..."` (should appear in server logs)

### Check Container Creation
```bash
# Should see episode-specific containers
docker ps -a | grep saber-session-

# Should see episode containers like:
# saber-session-[session-id]-excytin-sandbox...
```

### Check Logs Directory Structure
```bash
# Check client logs (timestamped structure)
ls -la ./client/logs/
# Shows:
#   saber_client_YYYYMMDD_HHMMSS/     - Client application logs
#   YYYY-MM-DDTHH-MM-SS_task_*.eval   - inspect_ai evaluation results

# Check client application logs
tail -f ./client/logs/saber_client_*/saber_client.log

# Check server logs  
ls -la ./server/logs/
# Shows:
#   compose-configs/          - Docker compose configurations
#   ├── permanent-environments/    - Permanent container configs
#   └── sandbox-environments/      - Episode sandbox configs
#   container-logs/           - Container execution logs
#   container-events-*.jsonl  - Container lifecycle events
```

## Configuration Files

### `server/config/tasks.yaml`
Defines the `excytin_demo` task that:
- Uses sandbox environment `excytin_sandbox` 
- Has 3 subtasks for container interaction testing
- Allows `cli` and `python` executors
- Configures episode attempts and step limits
- References permanent environment `excytin_incident_5`

### `server/config/environments.yaml`  
Defines two environments:

**Permanent Environment (`excytin_incident_5`)**:
- MySQL 8.0 container (`saber-excytin-incident-5`)
- Runs for server lifetime
- Pre-loaded with `incident_5.sql` data
- Exposes port 3306 for connectivity

**Sandbox Environment (`excytin_sandbox`)**:
- Lightweight execution container (`saber/excytin-sandbox:latest`)
- Created per episode
- Connected to shared network for MySQL access
- Resource limits: 512MB memory, 0.5 CPU

### `client/saber.yaml`
Main client configuration:
- **Model**: Azure OpenAI GPT-4.1 (inspect_ai format)
- **Agent**: React agent with max 50 steps
- **Server URLs**: REST (8000) and MCP (8001) endpoints
- **Tasks**: `["incident_5_task_1", "incident_5_task_2", "incident_5_task_3", "incident_5_task_4", "incident_5_task_5"]` (all incident tasks)
- **Logging**: Structured logging to `./logs` directory
- **Docker Commands**: Azure CLI credential mounting

### `server/data/sql_files/incident_5.sql`
Sample database schema with:
- `container_logs` table for logging demo
- `environment_status` table for health monitoring
- Sample data for testing

## Troubleshooting

### Configuration Issues
```bash
# Verify YAML syntax
docker exec -it saber-excytin-client python -c "
import yaml
with open('/app/client/saber.yaml') as f:
    print('✅ Client config valid')
    yaml.safe_load(f)
"

# Check server config
docker exec -it saber-excytin-server python -c "
import yaml
with open('/app/config/tasks.yaml') as f:
    yaml.safe_load(f)
with open('/app/config/environments.yaml') as f:
    yaml.safe_load(f)
print('✅ Server configs valid')
"
```

### Connection Issues
```bash
# Test server connectivity
curl http://localhost:8000/health
curl http://localhost:8001/  # MCP server

# Check server status
docker logs saber-excytin-server --tail 20

# Check client container status
docker exec -it saber-excytin-client echo "Client accessible"
```

### Episode/Environment Issues
```bash
# Check active sessions via REST API
curl http://localhost:8000/sessions

# Check permanent environment status
docker ps | grep saber-excytin-incident-5

# Check if sandbox environments are being created
docker ps -a | grep excytin-sandbox
```

### Log Analysis
```bash
# Check most recent client execution
ls -la ./client/logs/ | tail -2

# Check specific client run
tail -f ./client/logs/saber_client_*/saber_client.log

# Check for MCP connection issues
grep -i "mcp\|connection\|error" ./client/logs/saber_client_*/saber_client.log

# Check server-side episode creation
docker logs saber-excytin-server | grep -i "episode\|environment"
```

## What Success Looks Like

When everything works correctly:

1. **Session and Episode Creation**: 
   ```
   Created session: <session-id>
   Created episode: <episode-id>
   ```

2. **MCP Client Connection**:
   ```
   MCP client connected successfully
   Discovered 3 MCP tools
   ```

3. **Agent Execution**:
   ```
   Tool execution completed: cli
   Tool execution completed: python
   Tool execution completed: file_operations
   ```

4. **Structured Logging**:
   ```bash
   ls ./client/logs/
   # Shows timestamped client logs:
   #   ├── saber_client_YYYYMMDD_HHMMSS/
   #   │   └── saber_client.log          # Detailed execution logs
   #   └── YYYY-MM-DDTHH-MM-SS_task_*.eval  # inspect_ai evaluation results
   ```

5. **Server-Side Container Management**:
   ```bash
   docker ps
   # Shows permanent MySQL container and episode sandbox:
   #   saber-excytin-incident-5         # Permanent database
   #   saber-session-<id>-excytin-sandbox  # Episode sandbox (if active)
   ```

6. **Clean Resource Cleanup**:
   ```
   MCP client disconnected
   Session <session-id> terminated successfully
   ClientSessionManager cleanup completed
   ```

## Unified Architecture Benefits

This new architecture provides:

- **Direct Agent Execution**: No container overhead for agent runtime
- **Fail-Fast Error Handling**: Immediate failure alerts with clear error messages
- **Resource Management**: Automatic cleanup of sessions, episodes, and containers
- **Structured Logging**: Timestamped logs compatible with inspect_ai tooling
- **Real-time UI**: Progress bars and status updates during execution
- **Modular Design**: Clean separation between client orchestration and server execution
- **Tool Integration**: Direct MCP client connections for efficient tool usage

## Next Steps

This unified architecture serves as the foundation for more complex domains:
- **webapp_pentest**: Multi-container pentesting scenarios with target applications
- **malware_analysis**: Isolated sandbox environments for malware execution
- **red_team_ops**: Complex attack chain scenarios across multiple targets

The inspect_ai integration ensures consistent evaluation methodology across all domains while maintaining the flexibility for domain-specific customization.
