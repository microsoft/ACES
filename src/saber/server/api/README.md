# SABER Server API Layer

This directory contains the API layer components that handle external communication protocols for the SABER domain server.

## Overview

The SABER server implements a dual-protocol architecture to support different client interaction patterns:

- **REST API**: For session management, episode control, and system administration
- **MCP API**: For tool discovery and execution by AI agents using the Model Context Protocol

## Components

### SessionRestAPI (`session_rest_api.py`)

FastAPI-based REST server that handles HTTP requests for session and episode management.

**Responsibilities:**
- Session lifecycle management (create, list, terminate)
- Episode management (start, end, status)
- Policy document access
- Health checks and system status
- Server-Sent Events for real-time updates
- Administrative operations

**Key Endpoints:**
- `POST /session` - Create new session
- `DELETE /session/{id}` - Terminate session
- `POST /session/{id}/start-episode` - Start episode with task
- `GET /session/{id}/current-task` - Get current task information
- `GET /session/{id}/policy` - Get domain policy document
- `GET /session/{id}/events` - SSE stream for real-time updates
- `GET /health` - Health check
- `GET /sessions` - List active sessions

### SessionMCPAPI (`session_mcp_api.py`)

Model Context Protocol server that provides MCP-compliant tool interfaces for AI agents.

**Responsibilities:**
- Tool discovery (`list_tools`)
- Tool execution (`call_tool`)
- Connection-based session mapping
- MCP protocol compliance
- Session context management via connection headers

**MCP Compliance Features:**
- ✅ Clean tool schemas without session context
- ✅ Semantic parameter names (`code`, `command` vs `arguments`)
- ✅ Connection-based session mapping
- ✅ Standard MCP client compatibility
- ✅ Pure functional tool interfaces

**Available Tools:**
- `python(code: str)` - Execute Python code in sandbox
- `cli(command: str)` - Execute shell commands in sandbox  
- `end_episode(submission?: str)` - End episode with optional result

**Connection Headers:**
- `X-SABER-Session-ID` - Use existing session (optional)
- `X-SABER-Task-ID` - Auto-start episode with task (optional)
- `X-SABER-Client-ID` - Client identifier for logging

## Architecture Patterns

### Dual Protocol Design

```
Client Requests
├── REST API (Session Management)
│   ├── Session CRUD operations
│   ├── Episode lifecycle
│   ├── System administration
│   └── Real-time events (SSE)
└── MCP API (Tool Execution)
    ├── Tool discovery
    ├── Command execution
    └── Session context via headers
```

### Session Context Handling

**REST API**: Explicit session_id in URL paths and request bodies
```http
POST /session/abc123/start-episode
```

**MCP API**: Implicit session context via connection mapping
```javascript
// Connection establishment with headers
await mcp_client.connect({
  headers: {
    'X-SABER-Session-ID': 'abc123',
    'X-SABER-Task-ID': 'webapp_pentest_1'
  }
});

// Clean tool calls without session context
await mcp_client.call_tool('python', {
  code: 'print("hello world")'
});
```

### Connection-Based Session Mapping

The MCP API implements connection-scoped session management:

1. **Connection**: Client connects with session info in headers
2. **Mapping**: Server maps connection_id → session_id
3. **Execution**: Tool calls automatically use mapped session
4. **Cleanup**: Session cleaned up on connection disconnect

This approach ensures:
- MCP compliance (no session_id in tool schemas)
- Clean tool interfaces for optimal LLM context usage
- Automatic session management
- Standard MCP client compatibility

## Usage Examples

### REST API Client
```python
import httpx

# Create session
response = await client.post("/session", json={"client_id": "test_client"})
session_id = response.json()["session_id"]

# Start episode
await client.post(f"/session/{session_id}/start-episode", 
                  json={"task_id": "webapp_pentest_1"})

# Monitor via SSE
async with client.stream("GET", f"/session/{session_id}/events") as stream:
    async for line in stream.aiter_lines():
        event = json.loads(line)
        print(f"Event: {event}")
```

### MCP Client
```python
from mcp import Client

# Connect with session headers
client = await Client.connect_sse(
    "http://localhost:8001",
    headers={
        "X-SABER-Task-ID": "webapp_pentest_1",
        "X-SABER-Client-ID": "security_agent"
    }
)

# Discover tools
tools = await client.list_tools()

# Execute clean tools
result = await client.call_tool("python", {
    "code": "import requests; print(requests.__version__)"
})

result = await client.call_tool("cli", {
    "command": "nmap -sV target.local"
})

# End episode
await client.call_tool("end_episode", {
    "submission": "flag{sql_injection_found}"
})
```

## Error Handling

Both APIs implement comprehensive error handling:

- **REST API**: Standard HTTP status codes with JSON error responses
- **MCP API**: MCP-compliant error format with detailed error messages

## Configuration

API servers are configured through the SessionManager:

```python
session_manager = SessionManager(
    domain_name="webapp_pentest",
    rest_config={"host": "0.0.0.0", "port": 8000},
    mcp_config={"host": "0.0.0.0", "port": 8001}
)

await session_manager.start_server()
```

## See Also

- [MCP Integration Sequence Diagram](../../../docs/server/api/mcp_integration_sequence.puml)
- [Server Architecture Overview](../../../docs/server/session/server_detailed_architecture.puml)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/docs/)
