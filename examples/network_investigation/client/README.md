# SABER Network Investigation Client

A specialized security agent for network investigation tasks in the SABER framework.

## Features

- Network connection analysis
- Process investigation
- Security assessment and reporting
- Rich console output with progress tracking

## Usage

This client connects to a SABER server to execute network investigation tasks.

```bash
python -m network_investigation_client.main
```

## Configuration

The client reads configuration from environment variables:
- `SABER_SERVER_URL`: Server endpoint (default: http://localhost:8000)
- `SABER_SESSION_ID`: Session identifier (auto-generated if not provided)
