# SABER Client Architecture Documentation

This directory contains PlantUML diagrams documenting the SABER client-side architecture, which uses container-based execution with industry-standard MCP compatibility.

## Architecture Overview

The SABER client implements a **container-based execution architecture** that provides:

- **Container Isolation**: Agents run in isolated Docker containers with resource limits
- **MCP Sidecar**: Shared HTTP proxy service for standard MCP protocol communication  
- **Agent Runtime**: Universal adapter system supporting various agent interfaces
- **Logging Infrastructure**: Comprehensive container log collection and management

## Diagrams

### 1. [Client Architecture](client_architecture.puml)
**High-level client architecture overview**

Shows the complete client infrastructure including:
- `SABERHarness` - Main client orchestrator for container execution
- `ContainerEpisodeExecutor` - Container lifecycle management and episode orchestration
- `SidecarManager` - Shared MCP sidecar container management
- `AgentManager` - Individual agent container execution and monitoring
- `ContainerFactory` - Agent packaging and container creation
- Container logging infrastructure for comprehensive execution tracking

**Key Features:**
- Container isolation with Docker networking
- Standard MCP protocol compatibility via HTTP proxy
- Universal agent adapter system
- Comprehensive logging and monitoring

### 2. [Container Execution Architecture](container_execution_architecture.puml)
**Detailed container-based execution system**

Deep dive into the container execution infrastructure:
- Container lifecycle management (sidecar + agents)
- Docker network isolation (`saber-network`)
- Resource limits and security policies
- Multi-channel termination monitoring
- Session management and routing

**Container Details:**
- **MCP Sidecar**: FastAPI HTTP proxy for MCP protocol, shared across episodes
- **Agent Runtime**: Lightweight containers with standard MCP clients and universal adapters
- **Network Isolation**: Agent containers communicate only with sidecar, no direct SABER access
- **Logging System**: Real-time log streaming and file collection for debugging

### 3. [MCP Sidecar Architecture](mcp_sidecar_architecture.puml)
**MCP sidecar service architecture and session management**

Details the shared MCP sidecar service:
- FastAPI application with standard MCP HTTP endpoints
- Session registry for agent ↔ SABER session mapping
- MCP proxy with connection pooling and error handling
- Health monitoring and readiness checks
- Configuration and environment management

**HTTP API:**
- `GET /ready` - Process readiness (no upstream dependency)
- `GET /health` - Full health including upstream checks
- `POST /admin/sessions` - Session registration/cleanup
- `POST /mcp/list_tools` - MCP tool discovery
- `POST /mcp/call_tool` - MCP tool execution
- `GET /mcp/list_resources` - MCP resource listing

### 4. [Agent Runtime Architecture](agent_runtime_architecture.puml)
**Universal agent adapters and container runtime system**

Shows the agent runtime system running inside containers:
- `AgentExecutor` - Main entry point for container execution
- Universal agent adapters (class, function, async patterns)
- Standard MCP client integration
- Agent discovery and loading mechanisms
- Parameter mapping and interface detection

**Agent Compatibility:**
- **Zero code changes** required for existing agents
- **Auto-detection** of agent interfaces (class/function/async)
- **Standard MCP libraries** (anthropic/mcp-python) supported
- **Universal adapters** for different agent patterns

### 5. [Full E2E Benchmark Sequence](full_e2e_benchmark_sequence.puml)
**Complete end-to-end benchmark execution sequence**

Comprehensive sequence diagram showing:
- Benchmark initialization and infrastructure setup
- Container orchestration and parallel execution
- MCP protocol flow between agents and SABER server
- Session management and routing
- Results collection and cleanup
- Error handling and termination scenarios

**Execution Flow:**
1. **Initialization**: Harness → Sidecar → Agent containers
2. **Episode Execution**: Parallel agent execution with MCP tools
3. **Session Routing**: Agent sessions mapped to SABER sessions
4. **Tool Execution**: Commands run in SABER sandboxes (not agent containers)
5. **Results Collection**: Multi-channel result aggregation
6. **Cleanup**: Guaranteed container termination and resource cleanup

### 6. Container Logging Architecture
**Comprehensive logging infrastructure for debugging and monitoring**

Features:
- **Real-time Log Streaming**: Live capture from all containers during execution
- **File-based Collection**: Persistent logs saved to structured directory hierarchy
- **Multi-container Support**: Separate logs for sidecar and each agent container
- **Debug Integration**: Complete execution traces for troubleshooting failures
- **Session Organization**: Logs grouped by session timestamp for easy correlation

## Key Architectural Benefits

### For Agents
- ✅ **Zero code changes** required for MCP-compatible agents
- ✅ **Standard MCP libraries** (anthropic/mcp-python) work directly
- ✅ **Industry-standard patterns** for tool discovery and execution
- ✅ **No SABER-specific dependencies**

### for SABER
- ✅ **Robust termination** with guaranteed container cleanup
- ✅ **Resource isolation** and limits per agent container
- ✅ **Scalable architecture** with shared sidecar
- ✅ **Better observability** through container metrics

### For Operations
- ✅ **Predictable resource usage** with container limits
- ✅ **Clean process management** without race conditions
- ✅ **Parallel execution** with container isolation
- ✅ **Production-ready reliability**
- ✅ **Comprehensive logging** with real-time streaming and persistent collection
- ✅ **Debug capabilities** with complete execution traces

## Container Images

### MCP Sidecar (`saber/mcp-service:latest`)
- **Purpose**: Shared HTTP proxy for MCP protocol
- **Dependencies**: FastAPI, aiohttp, minimal Python runtime
- **Resources**: 256MB RAM, 0.5 CPU limit

### Agent Runtime (`saber/agent-runner:latest`)  
- **Purpose**: Lightweight agent execution environment
- **Dependencies**: Standard MCP clients, HTTP libraries, Python runtime
- **Resources**: 512MB RAM, 1.0 CPU limit (configurable)
- **Security**: Non-root user, read-only filesystem, isolated network

## Usage Example

```python
from saber.client import SABERHarness, SABERHarnessConfig

# Container-based execution
config = SABERHarnessConfig(
    parallelism=4,             # Run 4 agents concurrently
    server_url="http://server:8000",
    task_ids=["task1", "task2"],
    enable_container_logs=True  # Enable comprehensive logging
)

harness = SABERHarness(config)
await harness.initialize(my_agent)
results = await harness.run()
```

## Implementation Status

**✅ PRODUCTION READY**:
- Container infrastructure and orchestration
- MCP sidecar service with HTTP proxy
- Universal agent runtime and adapters
- Docker images and build system
- Comprehensive container logging
- Real-time log streaming and collection

**🎯 CURRENT ARCHITECTURE**:
The container architecture is the primary and only execution mode, providing robust isolation, logging, and MCP compatibility.

---

For implementation details and development history, see the SABER server documentation and source code.
