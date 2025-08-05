# SABER Network Investigation Domain Example

This example demonstrates a complete end-to-end deployment of the SABER (Security Agent Benchmarking System) with a network investigation domain. It includes both server and client components containerized and orchestrated with Docker Compose.

## Architecture Overview

- **SABER Server**: Hosts the network investigation domain with network analysis and process investigation tasks
- **SABER Client**: Contains a barebones security agent that connects to the server and executes commands
- **Docker-in-Docker**: Server uses Docker containers for isolated command execution
- **REST API**: Communication via HTTP REST API with Server-Sent Events for real-time updates

## Quick Start

1. **Prerequisites**:
   ```bash
   docker --version  # Ensure Docker is installed
   docker compose version  # Ensure Docker Compose is available
   ```

2. **Start the System**:
   ```bash
   docker compose up --build
   ```

3. **Monitor Logs**:
   ```bash
   # Server logs
   docker compose logs -f saber-server
   
   # Client logs  
   docker compose logs -f saber-client
   
   # All logs
   docker compose logs -f
   ```

4. **Access the API**:
   - Health Check: http://localhost:8000/health
   - API Documentation: http://localhost:8000/docs
   - Session Management: http://localhost:8000/sessions

## Example Security Domain

The network investigation domain includes:

### Tasks
- **Network Connection Analysis**: Analysis of active network connections and listening ports

### Available Commands
- `file` - Determine file type and properties
- `strings` - Extract printable strings from files
- `hexdump` - Display file contents in hexadecimal
- `netstat` - Display network connections
- `ps` - List running processes
- `ls` - List directory contents

## Development

### Server Configuration
- Configuration files: `server/src/network_investigation_server/config/`
- Tasks definitions: `server/src/network_investigation_server/config/tasks.yaml`
- Execution settings: `server/src/network_investigation_server/config/execution.yaml`

### Client Implementation
- Agent code: `client/src/network_investigation_client/`
- NetworkInvestigationAgent following SABER's zero-friction design
- Demonstrates SABER TestHarness integration and command selection strategy

### Testing
```bash
# TODO
```

## Architecture Components

### Server Side
- **SessionManager**: Central orchestrator managing client sessions
- **TaskManager**: Handles task assignment and episode management
- **ExecutionManager**: Manages Docker-based command execution
- **PolicyManager**: Provides domain-specific guidelines
- **EvaluationManager**: Tracks agent performance and trajectories

### Client Side
- **TestHarness**: SABER framework component for zero-friction agent testing
- **NetworkInvestigationAgent**: Security agent with process() and reset() methods
- **ServerClient**: Communication with server API using SABER client framework

See SABER for more details on implementation

## API Endpoints
see SABER session_api.py for endpoint definitions

## Cleanup

```bash
# Stop services
docker compose down

# Remove volumes and networks
docker compose down -v

# Clean up Docker images
docker system prune
```
