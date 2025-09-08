# SABER Client Architecture Documentation

This directory contains PlantUML diagrams documenting the SABER client-side architecture, which uses **inspect_ai integration** with **Model Context Protocol (MCP)** for modern agent evaluation.

## Architecture Overview

The SABER client implements a **modern inspect_ai-integrated architecture** that provides:

- **inspect_ai Compatibility**: Native `eval_async` integration for seamless agent evaluation
- **MCP Protocol**: Industry-standard Model Context Protocol for tool communication
- **Type Safety**: Strict Pydantic models with fail-fast validation
- **Clean Architecture**: Separation of concerns with proper resource management
- **No Backwards Compatibility**: Modern API designed for maintainability and clarity

## Design Principles

Following SABER best practices:

- ✅ **FAIL FAST**: Upfront validation with clear error messages
- ✅ **NO BACKWARDS COMPATIBILITY**: Clean modern API without legacy baggage  
- ✅ **TYPE SAFETY**: Strict Pydantic models throughout
- ✅ **NO DEFENSIVE PROGRAMMING**: Hard failures instead of silent fallbacks
- ✅ **SEPARATION OF CONCERNS**: Clean component boundaries

## Diagrams

### 1. [Client Architecture](client_architecture.puml)
**Modern inspect_ai integration architecture**

Shows the complete client infrastructure including:
- `run_saber_eval_async` - Main inspect_ai compatible entry point
- `SABEREvaluationOrchestrator` - Component lifecycle management and resource cleanup
- `AgentManager` - Agent discovery, initialization, and lifecycle management
- `DatasetManager` - Task discovery and inspect_ai Dataset creation
- `ClientSessionManager` - Unified API layer for REST + MCP communication
- Type-safe configuration with fail-fast validation

**Key Features:**
- inspect_ai native compatibility via `eval_async`
- Async context manager lifecycle for guaranteed cleanup
- Strict type safety with Pydantic models
- Fail-fast configuration validation
- Clean separation of concerns

### 2. [SABER MCP Architecture](saber_mcp_architecture.puml)
**Model Context Protocol integration and communication flow**

Deep dive into the MCP-based communication system:
- Standard MCP protocol compliance for agent portability
- Dual protocol design (REST for session management, MCP for tool execution)
- Episode-scoped MCP client pooling
- Type-safe API models throughout
- Server-side FastMCP implementation with tool discovery

**MCP Communication Details:**
- **Standard Protocol**: Industry-standard Model Context Protocol
- **Tool Discovery**: Dynamic tool discovery via `list_tools`
- **Session Context**: Episode-scoped client instances with header-based context
- **Error Handling**: Comprehensive error handling and retry logic
- **Security**: Server-side validation and Docker sandbox execution

## Component Breakdown

### Core Components

#### run_saber_eval_async
**Main entry point for SABER evaluations**
- inspect_ai compatible interface for seamless integration
- Automatic resource management via async context managers
- Fail-fast configuration validation
- Type-safe configuration with SABERConfig

#### SABEREvaluationOrchestrator  
**Central orchestration component**
- Async context manager for component lifecycle
- Upfront validation and early failure detection
- Component initialization and cleanup
- Resource management and error recovery

#### ClientSessionManager
**Unified API layer for dual-protocol communication**
- REST API for session and episode management
- MCP API for tool execution and discovery
- Episode-scoped MCP client pooling
- Session context management

#### AgentManager & DatasetManager
**Separation of concerns for agent and dataset operations**
- AgentManager: Agent discovery, initialization, lifecycle
- DatasetManager: Task discovery, filtering, inspect_ai Dataset creation
- Shared ClientSessionManager for API operations
- Clean resource management

### Configuration Models

#### SABERConfig
**Type-safe configuration with fail-fast validation**
- Required fields: `model`, `session_config`, `agent_config`
- Strict validation in `__post_init__`
- No backwards compatibility
- Factory methods for common configurations

#### SessionManagerConfig & MCPConfig
**Protocol-specific configuration**
- SessionManagerConfig: Unified REST + MCP settings
- MCPConfig: MCP-specific timeout, retry, client settings
- Type conversion methods between configs
- Strict typing with `extra="forbid"`

## Integration Patterns

### inspect_ai Integration
```python
# Modern SABER evaluation
from saber.client.inspect_ai import run_saber_eval_async

config = SABERConfig.create(
    model="gpt-4o",
    rest_url="http://localhost:8000",
    mcp_url="http://localhost:8001",
    agent_id="my_security_agent"
)

eval_log = await run_saber_eval_async(
    config=config,
    task_ids=["webapp_pentest_1", "malware_analysis_1"]
)
```

### MCP Tool Usage
```python
# Standard MCP pattern in agents
async with MCPClient(config) as client:
    tools = await client.list_tools()
    result = await client.call_tool(MCPToolCallRequest(
        name="python",
        arguments={"code": "print('Security analysis')"}
    ))
```

## Error Handling

### Fail-Fast Principles
- Configuration validation at startup
- Early server connectivity testing
- Clear error messages with actionable guidance
- No silent failures or defensive fallbacks

### Resource Management
- Guaranteed cleanup via async context managers
- Automatic MCP client pool management
- Session lifecycle tracking
- Episode-scoped resource isolation

## Migration from Legacy Architecture

### Removed Components
- ❌ Container-based execution (replaced with inspect_ai)
- ❌ SABERHarness (replaced with run_saber_eval_async)
- ❌ Sidecar management (replaced with direct MCP clients)
- ❌ Agent adapters (replaced with standard MCP patterns)
- ❌ Container logging infrastructure

### New Components  
- ✅ inspect_ai integration via eval_async
- ✅ Direct MCP client communication
- ✅ Type-safe Pydantic models
- ✅ Async context manager lifecycle
- ✅ Fail-fast validation

## Benefits

### For Agents
- ✅ **Standard MCP compatibility** - agents work with any MCP-compliant system
- ✅ **inspect_ai integration** - seamless evaluation framework integration
- ✅ **Type safety** - clear contracts and early error detection
- ✅ **No SABER-specific dependencies** - portable agent implementations

### For SABER
- ✅ **Clean architecture** - clear separation of concerns
- ✅ **Modern patterns** - async context managers and fail-fast design
- ✅ **Type safety** - Pydantic models throughout
- ✅ **MCP compliance** - industry-standard protocol

### For Operations
- ✅ **Predictable behavior** - fail-fast validation eliminates runtime surprises
- ✅ **Clear error messages** - actionable guidance for troubleshooting
- ✅ **Resource management** - guaranteed cleanup via context managers
- ✅ **Debugging** - comprehensive logging with structured data

This directory contains PlantUML diagrams documenting the SABER client-side architecture, which uses **inspect_ai integration** with **Model Context Protocol (MCP)** for modern agent evaluation.

## Architecture Overview

The SABER client implements a **modern inspect_ai-integrated architecture** that provides:

- **inspect_ai Compatibility**: Native `eval_async` integration for seamless agent evaluation
- **MCP Protocol**: Industry-standard Model Context Protocol for tool communication
- **Type Safety**: Strict Pydantic models with fail-fast validation
- **Clean Architecture**: Separation of concerns with proper resource management
- **No Backwards Compatibility**: Modern API designed for maintainability and clarity

## Design Principles

Following SABER best practices:

- ✅ **FAIL FAST**: Upfront validation with clear error messages
- ✅ **NO BACKWARDS COMPATIBILITY**: Clean modern API without legacy baggage  
- ✅ **TYPE SAFETY**: Strict Pydantic models throughout
- ✅ **NO DEFENSIVE PROGRAMMING**: Hard failures instead of silent fallbacks
- ✅ **SEPARATION OF CONCERNS**: Clean component boundaries

## Diagrams

### 1. [Client Architecture](client_architecture.puml)
**Modern inspect_ai integration architecture**

Shows the complete client infrastructure including:
- `run_saber_eval_async` - Main inspect_ai compatible entry point
- `SABEREvaluationOrchestrator` - Component lifecycle management and resource cleanup
- `AgentManager` - Agent discovery, initialization, and lifecycle management
- `DatasetManager` - Task discovery and inspect_ai Dataset creation
- `ClientSessionManager` - Unified API layer for REST + MCP communication
- Type-safe configuration with fail-fast validation

**Key Features:**
- inspect_ai native compatibility via `eval_async`
- Async context manager lifecycle for guaranteed cleanup
- Strict type safety with Pydantic models
- Fail-fast configuration validation
- Clean separation of concerns

### 2. [SABER MCP Architecture](saber_mcp_architecture.puml)
**Model Context Protocol integration and communication flow**

Deep dive into the MCP-based communication system:
- Standard MCP protocol compliance for agent portability
- Dual protocol design (REST for session management, MCP for tool execution)
- Episode-scoped MCP client pooling
- Type-safe API models throughout
- Server-side FastMCP implementation with tool discovery
