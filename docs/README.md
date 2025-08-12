# Security Agent Benchmarking System - Architecture Documentation

## Overview

This document describes the high-level architecture for a security agent benchmarking system designed to evaluate agentic workflows in the cybersecurity domain. The system employs a distributed server-client architecture where security domains are hosted as dedicated servers, and customer agents operate as independent clients.

## Architecture Diagrams

The SABER system architecture is documented in several PlantUML diagrams:

- **[System Overview](system_overview_architecture.puml)**: High-level system components and relationships
- **[Server Detailed Architecture](server_detailed_architecture.puml)**: Main server architecture with simplified framework views
- **[Task Framework Architecture](task_framework_architecture.puml)**: Detailed Task Management system design
- **[Command Execution Architecture](command_execution_architecture.puml)**: Detailed Command Execution framework design
- **[Episode Workflow Sequence](episode_workflow_sequence.puml)**: Episode lifecycle and RL workflow

### Package Management

This package is managed by uv for its python environments, use
```
uv run
```
whenever executing anything in python.

## Docker Sandbox Command Execution

### Enhanced Security Framework with Container Isolation
SABER implements a comprehensive security framework for command execution using Docker container isolation, addressing the unique security challenges of running arbitrary security commands in a benchmarking environment.

#### Key Security Features
- **Docker Container Isolation**: All command execution happens in isolated Docker containers
- **Session-based Container Management**: Each session gets its own dedicated container environment
- **Command Whitelisting**: Optional whitelist to override blocked commands for specific use cases
- **Pattern Detection**: Advanced regex-based detection of dangerous shell constructs
- **Argument Validation**: Comprehensive validation of all command arguments
- **Resource Limits**: Container-level CPU, memory, and process restrictions
- **Network Isolation**: Containers run with restricted network access
- **Shell Injection Prevention**: Multi-layer protection against command injection attacks

#### Security Configuration Example
```yaml
security:
  allowed_commands:
    - "file"
    - "strings" 
    - "hexdump"
    - "python3"
  max_command_length: 10000

execution:
  timeout: 300
  max_concurrent: 10

sandbox:
  image: "saber/base-sandbox:latest"
  network_mode: "none"
  read_only_root: true
  user: "tooluser"

cli:
  default_shell_mode: false
```

#### SecurityValidator Components
- **Base Command Validation**: Validates commands against blocked lists with optional whitelist override
- **Pattern Detection**: Identifies dangerous shell metacharacters and constructs
- **Argument Sanitization**: Validates and sanitizes all command arguments
- **Container Isolation**: All execution happens in isolated Docker environments

## System Architecture

### Core Design Principles

1. **Domain Isolation**: Each security domain (malware classification, threat investigation, etc.) runs as a separate server instance
2. **Client Autonomy**: Customer agents run independently with their choice of LLM provider
3. **Stateful Task Execution**: Complex security tasks maintain context across multiple subtasks
4. **MCP-Based Command Exposure**: All domain commands and resources exposed via Model Context Protocol
5. **Single-Client Constraint**: Only one client can interact with a domain server at a time (MVP)
6. **Persistent Evaluation**: All agent trajectories are logged for analysis and scoring

## System Components

### Server Side Architecture

#### SessionManager (SABER Domain Server)
The central orchestrator for each security domain, responsible for:
- Hosting the domain and managing server lifecycle (start/shutdown)
- Managing multiple concurrent client sessions
- Providing domain-specific information and capabilities
- Coordinating all server components
- Built for horizontal scaling with multiple instances behind load balancing

**Core Responsibilities:**
- **Session Lifecycle**: Creates, tracks, and manages multiple client sessions
- **Multi-Client Support**: Handles concurrent client connections (scalable architecture)
- **Business Logic Coordination**: Coordinates all client requests across server components
- **Task Orchestration**: Delegates to TaskManager for workflow management
- **Command Execution**: Integrated ExecutionManager for Docker sandbox command execution
- **Policy Retrieval**: Delegates to PolicyManager for domain guidelines
- **Action Logging**: Delegates to EvaluationManager for performance tracking

#### SessionAPI
REST API layer that handles HTTP endpoints and delegates to SessionManager:
- **HTTP Endpoints**: FastAPI-based REST API for all client interactions
- **Server-Sent Events**: Real-time streaming updates for session events
- **Request/Response Handling**: Converts HTTP requests to SessionManager method calls
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

#### TaskManager
Simplified orchestrator for task management with specialized components:
- **Core Responsibilities**: Task storage/retrieval, episode lifecycle coordination, RL gym-style interfaces
- **TaskConfigLoader**: Handles YAML parsing and task definition loading (extracted for maintainability)
- **Simplified Task Framework**: Tasks contain basic information without progression logic
- **EpisodeManager**: Handles episode lifecycle and returns Step objects directly
- **Task**: High-level security scenarios with basic metadata
- **SubTask**: Informational elements returned at episode start (no progression tracking)
- **Episode**: Complete task attempts with action-response tracking for RL training
- **RL Compatibility**: Gym-style step() and reset() methods for reinforcement learning

##### TaskConfigLoader
Specialized YAML configuration management:
- **Task Definition Loading**: Parses YAML task configurations
- **Object Creation**: Creates Task and SubTask objects from YAML data
- **Basic Validation**: Ensures proper YAML structure and required fields

##### Simplified Task Framework
Each Task instance contains basic task information without complex progression logic:
- **Task Information**: Basic task metadata (ID, title, description)
- **SubTask References**: Informational subtasks returned at episode start
- **No State Tracking**: No progression logic, dependencies, or completion tracking
- **Episode Integration**: Returns task information when episodes are started

##### EpisodeManager
Simplified episode management for RL workflows:
- **Episode Lifecycle**: Start, step, end operations
- **Step Creation**: Returns Step objects directly from step() method
- **RL Integration**: Provides gym-compatible interfaces
- **Active Episodes Only**: No subtask state tracking for simplified operation

#### ExecutionManager Integration
Direct command execution with Docker sandbox isolation:
- **Docker CLI Command**: Single DockerCLIExecutor with comprehensive security validation  
- **Action-Based Interface**: Accepts Action objects containing command and parameters
- **ExecutionConfiguration**: Configuration management for execution, security, CLI-specific settings, and sandbox configuration
- **Docker Container Execution**: execute_command() method for CLI command execution in isolated Docker containers
- **SandboxManager Integration**: Built-in container lifecycle management with session-based isolation
- **Security Integration**: Built-in SecurityValidator with configurable allowed commands
- **Concurrency Control**: Semaphore-based execution limiting with configurable max_concurrent
- **REST API Integration**: Command listing and execution via SessionAPI REST endpoints
- **Configuration Management**: YAML-based configuration with dynamic CLI parameter defaults and sandbox settings
- **Parameter Schema Generation**: CLI command generates its own parameter schemas for API documentation
- **Security Info Access**: get_security_info() exposes SecurityValidator and sandbox configuration
- **Command Validation**: validate_command() for pre-execution security checks
- **Execution Statistics**: get_execution_stats() provides runtime metrics and configuration status
- **Session Management**: Automatic Docker container creation and cleanup per session

#### ExecutionManager
Docker sandbox execution manager for MCP integration:
- **Single Docker CLI Command**: Contains one DockerCLIExecutor command instance with comprehensive security validation  
- **ExecutionConfiguration**: Configuration management for execution, security, CLI-specific settings, and sandbox configuration
- **Docker Container Execution**: execute_command() method for CLI command execution in isolated Docker containers
- **SandboxManager Integration**: Built-in container lifecycle management with session-based isolation
- **Security Integration**: Built-in SecurityValidator with configurable allowed commands
- **Concurrency Control**: Semaphore-based execution limiting with configurable max_concurrent
- **MCP Command Conversion**: to_mcp_commands() generates MCP-compatible command definitions from CLI command metadata
- **Configuration Management**: YAML-based configuration with dynamic CLI parameter defaults and sandbox settings
- **Parameter Schema Generation**: CLI command generates its own MCP-compatible parameter schemas
- **Security Info Access**: get_security_info() exposes SecurityValidator and sandbox configuration
- **Command Validation**: validate_command() for pre-execution security checks
- **Execution Statistics**: get_execution_stats() provides runtime metrics and configuration status
- **Session Management**: Automatic Docker container creation and cleanup per session

#### Security Framework
Comprehensive security controls for command execution:
- **SecurityValidator**: Pattern detection, argument validation
- **Command Whitelisting**: Optional whitelist to override blocked command restrictions
- **Shell Injection Prevention**: Pattern matching for dangerous shell constructs
- **Resource Limits**: CPU, memory, file size, and process limits
- **Security Constants**: Extensive lists of dangerous patterns and blocked commands

#### EvaluationManager
Tracks and evaluates agent performance:
- **ActionTracker**: Records all command calls and decisions
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
- **Unified Integration**: Access to tasks, commands, and policies through single endpoint

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
2. **CommandCallRequest**: Client requests command execution via SessionManager
3. **CommandCallResponse**: SessionManager returns command results from ExecutionManager
4. **SubTaskCompletion**: Client reports subtask completion to SessionManager
5. **ContextUpdate**: State synchronization between subtasks

### Connection Flow

1. Client connects to SessionManager via HTTP
2. SessionManager creates ClientSession and delegates TaskSession creation to TaskManager
3. Establishes SSE connection for real-time communication
4. SessionManager coordinates with TaskManager to assign initial subtask with context
5. Client executes subtask using commands via SessionManager unified API
6. Results logged via EvaluationManager and next subtask assigned via TaskManager
7. Process continues until task completion

## Security Domains

### Malware Classification
- **Commands**: Static analysis, dynamic analysis, ML classifiers
- **Tasks**: Family identification, behavior analysis, threat attribution
- **Context**: Sample metadata, analysis environments, reputation data

### Threat Investigation
- **Commands**: OSINT collection, IOC analysis, timeline construction
- **Tasks**: Campaign tracking, actor attribution, impact assessment
- **Context**: Threat intelligence feeds, historical data, correlation engines

### Digital Forensics
- **Commands**: Evidence collection, artifact analysis, timeline reconstruction
- **Tasks**: Incident response, data recovery, chain of custody
- **Context**: System images, log files, network captures

## Evaluation Framework

### MVP Capabilities
- **Action Sequence Tracking**: Record all command calls and parameters
- **Trajectory Storage**: Persistent logging of agent decisions
- **Basic Metrics**: Task completion rates, command usage patterns

### Future Enhancements
- **Automated Scoring**: ML-based evaluation of agent effectiveness
- **Comparative Analysis**: Benchmarking across different agent implementations
- **Rich Analytics**: Performance trends, error analysis, optimization insights

## Deployment Considerations

### Scalability
- **Horizontal Scaling**: Multiple SessionManager instances (each handling multiple sessions)
- **Load Balancing**: Client distribution across SessionManager instances
- **Session Multiplexing**: Each SessionManager handles multiple concurrent client sessions
- **Resource Management**: Command execution isolation and comprehensive resource limits
- **Concurrency Control**: Thread-safe operations with configurable execution limits

### Security
- **Authentication**: Client verification and authorization
- **Command Security**: Isolated execution environments with comprehensive security validation
- **Command Security**: Blocked command lists with optional whitelist override
- **Shell Injection Prevention**: Comprehensive protection against command injection attacks
- **Resource Limits**: CPU, memory, file size, and process restrictions
- **Data Protection**: Secure storage and transmission of sensitive data

### Monitoring
- **Health Checks**: Server and client status monitoring
- **Performance Metrics**: Response times, success rates, resource usage
- **Audit Trails**: Complete logging of all system interactions

## Implementation Roadmap

### Phase 1 (MVP) - ✅ COMPLETED
- ✅ Core server-client architecture design
- ✅ CLI-only execution manager with comprehensive security validation
- ✅ CommandLineExecutor framework with security integration
- ✅ CommandExecutor base class with parameter management and MCP schema generation
- ✅ CLI command with configurable parameter defaults
- ✅ SecurityValidator with command whitelisting and pattern detection
- ✅ CLIConfiguration with YAML-based configuration management
- ✅ Parameter validation system with type, range, and pattern constraints
- ✅ Comprehensive security testing suite (142 tests)
- ✅ MCP command conversion with command-generated schemas
- ✅ Concurrency control with semaphore-based execution limiting
- ✅ Basic task and subtask management framework

### Phase 2 (Current) - 🔄 IN PROGRESS
- ✅ **TaskManager Refactoring**: Extracted TaskConfigLoader, embedded progression logic in Task, simplified EpisodeManager
- ✅ **RL Gym Integration**: Implemented step() and reset() methods for reinforcement learning compatibility
- ✅ **DAG Progression Logic**: Automated subtask progression based on command execution and dependencies
- ✅ **Architecture Cleanup**: Reduced TaskManager from 655 to 308 lines with single responsibility principle
- 🔄 ExecutionManager integration with SessionManager REST API
- 🔄 Implement actual domain commands (beyond placeholders)
- 🔄 SessionManager unified API implementation
- 🔄 TaskManager integration with ExecutionManager
- 🔄 Client-side SecurityAgent implementation
- 🔄 Communication protocol implementation
- 🔄 Single domain implementation (malware classification)

### Phase 3 (Enhancement)
- Multiple domain support
- Advanced evaluation metrics and trajectory analysis
- Performance optimization and caching
- Enhanced monitoring and logging
- Cross-domain command dependencies
- Advanced analytics and reporting

### Phase 4 (Scale)
- Concurrent client support
- Integration with external security platforms
- Production hardening and additional security features
- Real-time performance monitoring
- Advanced benchmarking capabilities

## API Specifications

### SessionManager REST API
```
POST /session/{session_id}/start-episode - Initialize episode for task
POST /session/{session_id}/step - Execute RL step with command
GET /session/{session_id}/current-task - Get current task information
GET /session/{session_id}/policy - Get domain policy document
DELETE /session/{session_id} - Terminate session
```

### RL Step API
```
POST /session/{session_id}/step
Request Body: {
  "command": "file malware.exe"
}

Response: {
  "success": true,
  "data": {
    "output": "malware.exe: PE32 executable..."
  },
  "step": {
    "done": false,
    "subtask_states": {...}
  }
}
```



## Testing and Quality Assurance

### Comprehensive Security Testing
SABER includes extensive security testing to validate the robustness of the security framework:

- **Command Security Validation**: Tests for dangerous command patterns and shell injection attempts
- **Parameter Validation**: Comprehensive testing of parameter type checking and constraint validation
- **Command Registration**: Thread-safety and duplicate detection testing
- **Configuration Management**: YAML configuration loading and validation
- **Resource Limits**: CPU, memory, and process limit enforcement testing

### Test Coverage
- 142 command framework tests across 6 test files
- Comprehensive security validation testing
- CLI command execution and parameter validation
- SecurityValidator pattern detection and command validation
- ExecutionManager configuration management and MCP schema generation
- Integration tests for complete command execution pipeline
- Mock execution environments with security validation

### Test Structure
```
tests/execution/
├── test_cli_executor.py              # CLI command execution tests (23 tests)
├── test_command_line_executor.py     # Base executor tests (15 tests)  
├── test_security_validator.py        # Security validation tests (32 tests)
├── test_security_constants.py        # Security constants tests (21 tests)
├── test_execution_manager.py         # Execution manager and configuration tests (39 tests)
├── test_integration.py               # Integration tests (15 tests)
└── config/
    └── test_command_config.yaml       # Test configuration files
```

## Configuration

### Server Configuration
- Domain-specific command configurations with security controls
- Policy document specifications
- Storage backend settings
- Concurrency and resource limits
- Security validation settings (whitelists, limits)
- Command timeout and execution constraints

### Client Configuration
- LLM provider settings
- Server connection parameters
- Agent-specific configurations
- Logging and monitoring preferences

---
