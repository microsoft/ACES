# Security Agent Benchmarking System - Architecture Documentation

## Overview

This document describes the high-level architecture for a security agent benchmarking system designed to evaluate agentic workflows in the cybersecurity domain. The system employs a distributed server-client architecture where security domains are hosted as dedicated servers, and customer agents operate as independent clients.

## System Architecture

### Core Design Principles

1. **Domain Isolation**: Each security domain (malware classification, threat investigation, etc.) runs as a separate server instance
2. **Client Autonomy**: Customer agents run independently with their choice of LLM provider
3. **Stateful Task Execution**: Complex security tasks maintain context across multiple subtasks
4. **MCP-Based Tool Exposure**: All domain tools and resources exposed via Model Context Protocol
5. **Single-Client Constraint**: Only one client can interact with a domain server at a time (MVP)
6. **Persistent Evaluation**: All agent trajectories are logged for analysis and scoring

## System Components

### Server Side Architecture

#### DomainServer
The central orchestrator for each security domain, responsible for:
- Managing client connections and enforcing single-client constraint
- Coordinating between all server-side components
- Providing domain-specific information and capabilities

#### TaskManager
Handles complex multi-step security tasks:
- **DomainTask**: High-level security scenarios (e.g., "Investigate APT campaign")
- **SubTask**: Individual steps with specific objectives and success criteria
- **TaskSession**: Maintains state and context between subtasks for a client session
- **State Continuity**: Ensures agents can build upon previous subtask results

#### MCPServer (FastMCP Integration)
Exposes domain capabilities via Model Context Protocol:
- **Tool Endpoints**: Security tools (malware analysis, threat intel, forensics)
- **Policy Endpoint**: Domain-specific guidelines and context
- **Task Endpoint**: Available tasks and subtasks
- **Resource Management**: Handles tool execution and result delivery

#### ToolRegistry
Domain-specific security tool management:
- **SecurityTool**: Individual tools with validation and execution logic
- **Tool Categories**: Organized by security function (analysis, intel, forensics)
- **Extensible Executors**: Plugin architecture for different tool implementations

#### EvaluationManager
Tracks and evaluates agent performance:
- **ActionTracker**: Records all tool calls and decisions
- **TrajectoryStore**: Persistent storage for evaluation data
- **Scoring Engine**: MVP focuses on action sequence tracking
- **Future Extensions**: Rich trajectory analysis and automated scoring

#### PolicyManager
Manages domain-specific operational context:
- **PolicyDocument**: Guidelines, constraints, and available capabilities
- **Context Hints**: Domain-specific guidance for agents
- **Action Validation**: Ensures agent actions comply with domain policies

### Client Side Architecture

#### AgentClient
Main client orchestrator that:
- Establishes and maintains server connections
- Manages task execution lifecycle
- Handles server-client communication protocol

#### SecurityAgent
Customer-provided agent implementation featuring:
- **LLM Provider**: Configurable interface for any LLM service
- **MCPClient**: Communicates with server tools via MCP protocol
- **ReasoningEngine**: Agent's decision-making and planning capabilities
- **Tool Integration**: Seamless access to domain-specific security tools

#### Communication Layer
Real-time bidirectional communication:
- **HTTP/SSE**: Server-Sent Events for real-time updates
- **Structured Messages**: Type-safe protocol for different interaction types
- **Session Management**: Maintains connection state and handles failures

### Data Persistence Layer

#### TrajectoryStore
Persistent storage for evaluation data:
- **Pluggable Backends**: Support for filesystem, database, or cloud storage
- **Trajectory Logging**: Complete action sequences with timestamps
- **Query Interface**: Flexible data retrieval for analysis
- **Archival Support**: Long-term storage management

## Communication Protocol

### Message Types

1. **TaskAssignment**: Server assigns subtask to client
2. **ToolCallRequest**: Client requests tool execution
3. **ToolCallResponse**: Server returns tool results
4. **SubTaskCompletion**: Client reports subtask completion
5. **ContextUpdate**: State synchronization between subtasks

### Connection Flow

1. Client connects to domain server via HTTP
2. Server creates ClientSession and TaskSession
3. Establishes SSE connection for real-time communication
4. Server assigns initial subtask with context
5. Client executes subtask using MCP tools
6. Results logged and next subtask assigned
7. Process continues until task completion

## Security Domains

### Malware Classification
- **Tools**: Static analysis, dynamic analysis, ML classifiers
- **Tasks**: Family identification, behavior analysis, threat attribution
- **Context**: Sample metadata, analysis environments, reputation data

### Threat Investigation
- **Tools**: OSINT collection, IOC analysis, timeline construction
- **Tasks**: Campaign tracking, actor attribution, impact assessment
- **Context**: Threat intelligence feeds, historical data, correlation engines

### Digital Forensics
- **Tools**: Evidence collection, artifact analysis, timeline reconstruction
- **Tasks**: Incident response, data recovery, chain of custody
- **Context**: System images, log files, network captures

## Evaluation Framework

### MVP Capabilities
- **Action Sequence Tracking**: Record all tool calls and parameters
- **Trajectory Storage**: Persistent logging of agent decisions
- **Basic Metrics**: Task completion rates, tool usage patterns

### Future Enhancements
- **Automated Scoring**: ML-based evaluation of agent effectiveness
- **Comparative Analysis**: Benchmarking across different agent implementations
- **Rich Analytics**: Performance trends, error analysis, optimization insights

## Deployment Considerations

### Scalability
- **Horizontal Scaling**: Multiple domain server instances
- **Load Balancing**: Client distribution across server instances
- **Resource Management**: Tool execution isolation and resource limits

### Security
- **Authentication**: Client verification and authorization
- **Tool Sandboxing**: Isolated execution environments
- **Data Protection**: Secure storage and transmission of sensitive data

### Monitoring
- **Health Checks**: Server and client status monitoring
- **Performance Metrics**: Response times, success rates, resource usage
- **Audit Trails**: Complete logging of all system interactions

## Implementation Roadmap

### Phase 1 (MVP)
- Core server-client architecture
- Basic task and subtask management
- MCP tool exposure
- Simple action tracking
- Single domain implementation (malware classification)

### Phase 2 (Enhancement)
- Multiple domain support
- Advanced evaluation metrics
- Performance optimization
- Enhanced monitoring and logging

### Phase 3 (Scale)
- Concurrent client support
- Advanced analytics and reporting
- Integration with external security platforms
- Production hardening and security features

## API Specifications

### MCP Endpoints
```
GET /mcp/tools - List available tools
POST /mcp/tools/{tool_name}/execute - Execute tool
GET /mcp/policy - Get domain policy document
GET /mcp/tasks - List available tasks
GET /mcp/tasks/{task_id}/subtasks - Get task subtasks
```

### HTTP/SSE Endpoints
```
POST /session/start - Initialize client session
GET /session/events - SSE event stream
POST /session/message - Send message to server
DELETE /session/end - Terminate session
```

## Configuration

### Server Configuration
- Domain-specific tool configurations
- Policy document specifications
- Storage backend settings
- Concurrency and resource limits

### Client Configuration
- LLM provider settings
- Server connection parameters
- Agent-specific configurations
- Logging and monitoring preferences

---

*This architecture is designed to be extensible and scalable, supporting the evolution from MVP to a comprehensive security agent benchmarking platform.*
