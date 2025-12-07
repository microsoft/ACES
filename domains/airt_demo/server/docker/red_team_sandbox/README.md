# Red Team Sandbox Docker Context

This directory contains the Docker build context for the AI Red Team sandbox container.

## Contents

- `Dockerfile` - Docker image definition for the red team sandbox
- `websocket_daemon.py` - Persistent WebSocket daemon for prompt injection
- `inject_via_ipc.py` - CLI wrapper for sending injection commands via IPC

## Building

The image is built as part of the SABER domain orchestration. The scripts are copied into the container at build time and made available in `/usr/local/bin/`.

## WebSocket Daemon

The daemon maintains a persistent WebSocket connection to the SABER server and exposes an HTTP IPC interface on `localhost:9999` for injection commands.

### Environment Variables

- `SABER_WEBSOCKET_URL` - WebSocket URL (e.g., `ws://host.docker.internal:8000/api/v1/episodes/{id}/ws`)
- `EPISODE_ID` - Red team episode ID
- `IPC_PORT` - Port for IPC server (default: 9999)

### Starting the Daemon

```bash
python /usr/local/bin/websocket_daemon.py
```

### Using the CLI

```bash
# Basic injection
python /usr/local/bin/inject_via_ipc.py "Your injection message" --target blue-episode-id

# Rewind injection
python /usr/local/bin/inject_via_ipc.py "New context" --target blue-episode-id --strategy rewind --rewind-count 2

# Insert at position
python /usr/local/bin/inject_via_ipc.py "System override" --target blue-episode-id --strategy insert --insert-position 0
```

## Dependencies

- Python 3.11+
- websockets
- aiohttp
- requests
