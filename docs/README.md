# Security Agent Benchmarking System - Architecture Documentation

## Overview

This document describes the high-level architecture for a security agent benchmarking system designed to evaluate agentic workflows in the cybersecurity domain. The system employs a distributed server-client architecture where security domains are hosted as dedicated servers, and customer agents operate as independent clients.

### Package Management

This package is managed by uv for its python environments, use
```
uv run
```
whenever executing anything in python.

## Security-First Tool Execution

### Enhanced Security Framework
SABER implements a comprehensive security framework for tool execution, addressing the unique security challenges of running arbitrary security tools in a benchmarking environment.

#### Key Security Features
- **Command Whitelisting**: Only explicitly allowed commands can be executed
- **Pattern Detection**: Advanced regex-based detection of dangerous shell constructs
- **Argument Validation**: Comprehensive validation of all tool arguments
- **Path Restrictions**: Sandbox-based path limitation to prevent directory traversal
- **Resource Limits**: CPU, memory, file size, and process count restrictions
- **Shell Injection Prevention**: Multi-layer protection against command injection attacks

#### Security Configuration Example
```yaml
security:
  allowed_commands:
    - "file"
    - "strings" 
    - "hexdump"
    - "python3"
  sandbox_path: "/tmp/saber_sandbox"
  max_command_length: 10000
  
tools:
  execution:
    timeout: 300
    max_concurrent: 10
```

#### SecurityValidator Components
- **Base Command Validation**: Ensures only whitelisted commands are used
- **Pattern Detection**: Identifies dangerous shell metacharacters and constructs
- **Argument Sanitization**: Validates and sanitizes all command arguments
- **Path Safety**: Prevents directory traversal and unauthorized file access
- **Resource Monitoring**: Enforces execution limits and prevents resource exhaustion

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
- Hosting the SessionManager as the unified client endpoint
- Providing domain-specific information and capabilities

#### SessionManager
Unified endpoint for all client interactions:
- **Session Lifecycle**: Creates, tracks, and manages client sessions
- **Single-Client Enforcement**: Ensures only one client per domain server (MVP)
- **Unified API**: Coordinates all client requests across server components
- **Task Orchestration**: Delegates to TaskManager for workflow management
- **Tool Execution**: Delegates to MCPServer for all tool operations
- **Policy Retrieval**: Delegates to PolicyManager for domain guidelines
- **Action Logging**: Delegates to EvaluationManager for performance tracking

#### TaskManager
Handles complex multi-step security tasks:
- **DomainTask**: High-level security scenarios (e.g., "Investigate APT campaign")
- **SubTask**: Individual steps with specific objectives and success criteria
- **TaskSession**: Maintains state and context between subtasks for a client session
- **State Continuity**: Ensures agents can build upon previous subtask results

#### MCPServer (FastMCP Integration)
Complete tool execution layer:
- **Tool Registry**: Contains and manages all domain-specific security tools
- **Tool Execution**: Handles all tool operations with context awareness
- **MCP Endpoints**: Can expose standard MCP protocol for development/testing
- **Internal Service**: Used internally by SessionManager, not directly accessed by clients

#### ToolRegistry
Security-first domain tool management (contained within MCPServer):
- **SecurityTool**: Individual tools with comprehensive validation and execution logic
- **Security Executors**: CommandLineToolExecutor with security validation and sandboxing
- **Parameter Validation**: Enhanced parameter system with type, range, and pattern validation
- **Tool Categories**: Organized by security function (analysis, intel, forensics)
- **Thread Safety**: Thread-safe registration with RLock for concurrent access
- **Auto-Discovery**: Metadata-based tool discovery with module scanning
- **Security Validation**: Comprehensive command security with pattern detection
- **Concurrency Control**: Semaphore-based execution limiting
- **Configuration**: YAML-based tool and security configuration

#### Security Framework
Comprehensive security controls for tool execution:
- **SecurityValidator**: Pattern detection, argument validation, path safety
- **Command Whitelisting**: Only allowed commands can be executed
- **Sandbox Restrictions**: Optional path-based sandboxing for tool execution
- **Shell Injection Prevention**: Pattern matching for dangerous shell constructs
- **Resource Limits**: CPU, memory, file size, and process limits
- **Security Constants**: Extensive lists of dangerous patterns and blocked commands

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
- Establishes and maintains session connection to SessionManager
- Manages task execution lifecycle
- Handles unified communication protocol

#### SecurityAgent
Customer-provided agent implementation featuring:
- **LLM Provider**: Configurable interface for any LLM service
- **SessionClient**: Communicates with SessionManager unified API for all operations
- **ReasoningEngine**: Agent's decision-making and planning capabilities
- **Unified Integration**: Access to tasks, tools, and policies through single endpoint

#### Communication Layer
Real-time bidirectional communication:
- **HTTP/SSE**: Server-Sent Events for real-time updates via SessionManager
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

1. **TaskAssignment**: SessionManager assigns subtask to client
2. **ToolCallRequest**: Client requests tool execution via SessionManager
3. **ToolCallResponse**: SessionManager returns tool results from MCPServer
4. **SubTaskCompletion**: Client reports subtask completion to SessionManager
5. **ContextUpdate**: State synchronization between subtasks

### Connection Flow

1. Client connects to SessionManager via HTTP
2. SessionManager creates ClientSession and delegates TaskSession creation to TaskManager
3. Establishes SSE connection for real-time communication
4. SessionManager coordinates with TaskManager to assign initial subtask with context
5. Client executes subtask using tools via SessionManager unified API
6. Results logged via EvaluationManager and next subtask assigned via TaskManager
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
- **Resource Management**: Tool execution isolation and comprehensive resource limits
- **Concurrency Control**: Thread-safe operations with configurable execution limits

### Security
- **Authentication**: Client verification and authorization
- **Tool Sandboxing**: Isolated execution environments with comprehensive security validation
- **Command Security**: Whitelist-based command control with pattern detection
- **Shell Injection Prevention**: Comprehensive protection against command injection attacks
- **Resource Limits**: CPU, memory, file size, and process restrictions
- **Path Restrictions**: Sandbox-based path limitation for tool operations
- **Data Protection**: Secure storage and transmission of sensitive data

### Monitoring
- **Health Checks**: Server and client status monitoring
- **Performance Metrics**: Response times, success rates, resource usage
- **Audit Trails**: Complete logging of all system interactions

## Implementation Roadmap

### Phase 1 (MVP) - ✅ COMPLETED
- ✅ Core server-client architecture design
- ✅ Security-first tool registry with comprehensive validation
- ✅ CommandLineToolExecutor with security framework
- ✅ Parameter validation system with type checking
- ✅ Thread-safe tool registration and management
- ✅ Configuration-based security controls
- ✅ Auto-discovery of domain tools
- ✅ Basic task and subtask management framework
- ✅ Comprehensive security testing suite
- ✅ MCP tool conversion capability
- ✅ Simple action tracking foundation

### Phase 2 (Current) - 🔄 IN PROGRESS
- 🔄 Complete MCP server integration with FastMCP
- 🔄 Implement actual domain tools (beyond placeholders)
- 🔄 SessionManager unified API implementation
- 🔄 TaskManager integration with ToolRegistry
- 🔄 Client-side SecurityAgent implementation
- 🔄 Communication protocol implementation
- 🔄 Single domain implementation (malware classification)

### Phase 3 (Enhancement)
- Multiple domain support
- Advanced evaluation metrics and trajectory analysis
- Performance optimization and caching
- Enhanced monitoring and logging
- Cross-domain tool dependencies
- Advanced analytics and reporting

### Phase 4 (Scale)
- Concurrent client support
- Integration with external security platforms
- Production hardening and additional security features
- Real-time performance monitoring
- Advanced benchmarking capabilities

## API Specifications

### SessionManager Unified API
```
POST /session/start - Initialize client session
GET /session/events - SSE event stream
POST /session/message - Send message to server
DELETE /session/end - Terminate session

GET /session/current-task - Get current or next task
POST /session/complete-task - Report task completion
POST /session/execute-tool - Execute tool with context
GET /session/list-tools - List available tools
GET /session/policy - Get domain policy document
GET /session/context - Get current task context
```

### Optional MCP Endpoints (Development/Testing)
```
GET /mcp/tools - List available tools
POST /mcp/tools/{tool_name}/execute - Execute tool
GET /mcp/policy - Get domain policy document
GET /mcp/tasks - List available tasks
GET /mcp/tasks/{task_id}/subtasks - Get task subtasks
```

## Testing and Quality Assurance

### Comprehensive Security Testing
SABER includes extensive security testing to validate the robustness of the security framework:

- **Command Security Validation**: Tests for dangerous command patterns and shell injection attempts
- **Parameter Validation**: Comprehensive testing of parameter type checking and constraint validation
- **Tool Registration**: Thread-safety and duplicate detection testing
- **Configuration Management**: YAML configuration loading and validation
- **Sandbox Restrictions**: Path traversal and directory access limitation testing
- **Resource Limits**: CPU, memory, and process limit enforcement testing

### Test Coverage
- 400+ lines of security-focused test code
- Unit tests for each security component
- Integration tests for tool execution pipeline
- Mock tool execution with security validation
- Configuration-based testing scenarios

### Test Structure
```
tests/
├── test_cli_security.py          # Command-line security validation tests
├── test_tool_configuration.py    # Configuration and tool management tests  
├── test_task_manager.py          # Task management functionality tests
└── config/
    └── test_tool_config.yaml     # Test configuration files
```

## Configuration

### Server Configuration
- Domain-specific tool configurations with security controls
- Policy document specifications
- Storage backend settings
- Concurrency and resource limits
- Security validation settings (whitelists, sandbox paths, limits)
- Tool timeout and execution constraints

### Client Configuration
- LLM provider settings
- Server connection parameters
- Agent-specific configurations
- Logging and monitoring preferences

---
