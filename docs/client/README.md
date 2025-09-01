# SABER Client Architecture Documentation

This directory contains PlantUML diagrams documenting the SABER client-side architecture, which has been transformed from an embedded MCP approach to a modern container-based execution system with industry-standard MCP compatibility.

## Architecture Overview

The SABER client implements a **dual execution mode architecture** that supports both:

- **Container Mode (Recommended)**: Modern containerized agent execution with MCP sidecar
- **Embedded Mode (Legacy)**: Process-based execution with embedded MCP client

## Diagrams

### 1. [Client Architecture](client_architecture.puml)
**High-level client architecture overview**

Shows the complete client infrastructure including:
- `SABERHarness` - Main client orchestrator with dual execution modes
- Container execution components (sidecar, agent manager, container factory)
- MCP sidecar service with FastAPI HTTP proxy
- Universal agent runtime with adapters
- Embedded mode components (legacy)

**Key Features:**
- Dual execution modes with seamless switching
- Container isolation and resource management
- Standard MCP protocol compatibility
- Universal agent adapter system

### 2. [Container Execution Architecture](container_execution_architecture.puml)
**Detailed container-based execution system**

Deep dive into the container execution infrastructure:
- Container lifecycle management (sidecar + agents)
- Docker network isolation (`saber-network`)
- Resource limits and security policies
- Multi-channel termination monitoring
- Session management and routing

**Container Details:**
- **MCP Sidecar**: 295MB, shared HTTP proxy for MCP protocol
- **Agent Runtime**: 458MB, lightweight with standard MCP clients
- **Network Isolation**: Agents ↔ Sidecar only, no direct SABER access

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

## Container Images

### MCP Sidecar (`saber-mcp-sidecar:latest`)
- **Size**: 295MB
- **Purpose**: Shared HTTP proxy for MCP protocol
- **Dependencies**: FastAPI, aiohttp, minimal Python runtime
- **Resources**: 256MB RAM, 0.5 CPU limit

### Agent Runtime (`saber-agent-runner:latest`)  
- **Size**: 458MB
- **Purpose**: Lightweight agent execution environment
- **Dependencies**: Standard MCP clients, HTTP libraries, Python runtime
- **Resources**: 512MB RAM, 1.0 CPU limit (configurable)
- **Security**: Non-root user, read-only filesystem, isolated network

## Usage Example

```python
from saber.client import SABERHarness, SABERHarnessConfig

# Container mode (recommended)
config = SABERHarnessConfig(
    use_containers=True,        # Enable container execution
    parallelism=4,             # Run 4 agents concurrently
    server_url="http://server:8000",
    task_ids=["task1", "task2"]
)

harness = SABERHarness(config)
await harness.initialize(my_agent)
results = await harness.run()
```

## Migration from Embedded Mode

The container architecture provides a seamless migration path:

1. **No Code Changes**: Existing agents work unchanged
2. **Opt-in**: Set `use_containers=True` in harness config  
3. **Gradual Migration**: Test with container mode, fallback to embedded
4. **Docker Required**: Ensure Docker daemon available on execution hosts

## Implementation Status

**✅ COMPLETED (Phase 3)**:
- Container infrastructure and orchestration
- MCP sidecar service with HTTP proxy
- Universal agent runtime and adapters
- Docker images and build system
- Dual execution mode harness integration

**🎯 READY FOR PRODUCTION**:
The container architecture is fully implemented and ready for production use with comprehensive testing and validation.

---

For implementation details and development history, see:
- [MCP Sidecar Implementation Plan](../.archive/mcp_sidecar_implementation_plan.md)
- [MCP Sidecar Implementation Summary](../.archive/mcp_sidecar_implementation_summary.md)
