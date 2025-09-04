# Excytin Demo Client

This client demonstrates SABER's enhanced container logging capabilities by running episodes that create sandbox containers and generate logs.

## Setup

The client uses `uv` for Python package management. The dependencies are defined in `pyproject.toml`.

## Usage

### Basic Demo
```bash
# Run a single episode to test logging
uv run demo_client.py

# Run multiple episodes
uv run demo_client.py --episodes 3

# Run with verbose logging
uv run demo_client.py --verbose
```

### What the Demo Does

1. **Creates a session** with the SABER server
2. **Starts a benchmark** using the `basic_logging_demo` task
3. **Executes commands** that generate container activity:
   - MySQL database queries
   - System process listing
   - Python code execution
4. **Verifies logging** by checking for collected logs
5. **Cleans up** the session (triggers final log collection)

### Expected Results

After running the demo, you should see logs collected in `../server/logs/`:

```
server/logs/
├── compose-configs/
│   └── sandbox-environments/
│       └── YYYY-MM-DDTHH:MM:SS_sandbox_session-xyz.yml
├── container-logs/
│   └── sandbox-environments/
│       └── YYYY-MM-DDTHH:MM:SS_sandbox_session-xyz_saber-excytin-incident-5.log
└── container-events-YYYY-MM-DD.jsonl
```

### Prerequisites

- SABER server running on `localhost:8000`
- Enhanced container logging enabled in server configuration
- Docker available for sandbox container creation

### Troubleshooting

**No logs generated:**
- Check that `SABER_ENABLE_CONTAINER_LOGGING=true` in docker-compose.yml
- Verify the logs directory is mounted: `./server/logs:/app/logs:rw`
- Ensure the server has write permissions to the logs directory

**Connection errors:**
- Verify the SABER server is running: `curl http://localhost:8000/health`
- Check if containers are running: `docker-compose ps`

**Container creation fails:**
- Check Docker daemon is running
- Verify environment configurations in `../server/config/environments.yaml`

This demo validates that the Excytin system successfully captures:
- Docker compose configurations used to create containers
- Container execution and communication
- Container lifecycle events with timestamps and metadata
