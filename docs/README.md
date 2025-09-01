# Security Agent Benchmarking System - Architecture Documentation

## Overview

This document describes the high-level architecture for a security agent benchmarking system designed to evaluate agentic workflows in the cybersecurity domain. The system employs a distributed server-client architecture where security domains are hosted as dedicated servers, and customer agents operate as independent clients.

## Architecture Diagrams

The SABER system architecture is documented in several PlantUML diagrams organized by module:

### System Overview
- **[System Overview](system/system_overview_architecture.puml)**: High-level system components and relationships

### Server Architecture
- **[Server Detailed Architecture](server/session/server_detailed_architecture.puml)**: Main server architecture with simplified framework views
- **[Benchmark Framework Architecture](server/benchmarks/benchmark_framework_architecture.puml)**: Detailed Benchmark Management system design
- **[Command Execution Architecture](server/execution/command_execution_architecture.puml)**: Detailed Command Execution framework design
- **[Episode Workflow Sequence](server/episodes/episode_workflow_sequence.puml)**: Episode lifecycle and RL workflow
- **[Enhanced Sandbox Sequence](server/execution/enhanced_sandbox_sequence.puml)**: Enhanced sandbox environment setup sequence
- **[MCP Integration Sequence](server/api/mcp_integration_sequence.puml)**: Model Context Protocol integration workflow and dual protocol communication

### Client Architecture
- **[Client Architecture](client/client_architecture.puml)**: High-level client architecture with dual execution modes (container vs embedded)
- **[Container Execution Architecture](client/container_execution_architecture.puml)**: Detailed container-based execution system with MCP sidecar
- **[MCP Sidecar Architecture](client/mcp_sidecar_architecture.puml)**: MCP sidecar service architecture and session management
- **[Agent Runtime Architecture](client/agent_runtime_architecture.puml)**: Universal agent adapters and container runtime system
- **[Full E2E Benchmark Sequence](client/full_e2e_benchmark_sequence.puml)**: Complete end-to-end benchmark execution sequence

### Package Management

This package is managed by uv for its python environments, use
```
uv run
```
whenever executing anything in python.

## System Architecture

### Core Design Principles

## System Components

### Server Side Architecture

#### SessionManager (SABER Domain Server)
The central orchestrator for each security domain, responsible for:
- **Multi-Session Management**: Creates, tracks, and manages multiple concurrent client sessions with lifecycle control
- **Component Coordination**: Orchestrates TaskManager, ExecutionManager, PolicyManager, and EvaluationManager
- **Domain Hosting**: Provides single-domain server instances designed for horizontal scaling behind load balancers
- **Server Lifecycle**: Handles startup, shutdown, and resource cleanup across all active sessions

#### SessionRestAPI
REST protocol handler that processes HTTP endpoints and delegates to SessionManager:
- **HTTP Endpoints**: FastAPI-based REST API with endpoints for session management, episode management, and status monitoring  
- **Request Delegation**: Converts HTTP requests to SessionManager method calls with proper error handling
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation for client integration

#### SessionMCPAPI
Model Context Protocol handler component managed by SessionManager:
- **MCP Server**: Hosts MCP server alongside REST API for agent tool execution only
- **Dynamic Tool Discovery**: Provides real-time tool schemas from ExecutionManager
- **Tool Execution**: Maps MCP tool calls to SessionManager command execution pipeline
- **Session Context**: Maintains session mapping between MCP clients and SessionManager sessions
- **Focused Scope**: ONLY handles tool discovery and execution via MCP protocol

#### BenchmarkManager
Enhanced benchmark orchestration and task management:
- **Task Configuration**: Loads and manages task definitions from YAML configuration files with support for both sandbox and permanent environments
- **Benchmark Orchestration**: Manages multiple episode attempts for pass@k evaluation
- **Task Lookup**: Provides task object retrieval by task ID with complete configuration including permanent_environment references
- **Episode Coordination**: Orchestrates multiple episode runs across all benchmark tasks
- **Configuration Containment**: Task objects now contain all execution parameters (sandbox_environment, permanent_environment, allowed_executors)
- **Benchmark Configuration**: Manages domain-level and task-specific benchmark settings

##### BenchmarkConfigLoader
Enhanced YAML configuration management:
- **Complete Task Loading**: Parses task definitions including execution configuration (allowed_executors, environment)
- **Benchmark Configuration**: Loads domain-level and task-specific benchmark settings (episode_attempts, etc.)
- **Task Validation**: Ensures proper YAML structure, required fields, and execution parameters
- **Object Creation**: Creates Task objects with complete configuration needed by other components
- **Single Source**: Only component that reads task configuration files

##### Benchmark Framework
Enhanced Task objects with complete execution and benchmark configuration:
- **Complete Configuration**: Tasks contain sandbox_environment, permanent_environment references, allowed_executors, and all execution parameters
- **Benchmark Settings**: Tasks include benchmark-specific configuration (episode_attempts, success_criteria)
- **Dual Environment Support**: Tasks can specify both ephemeral sandbox environments and references to permanent shared services
- **Self-Contained**: No need for separate configuration lookups - all parameters in Task object
- **Subtask Information**: Contains subtasks for informational purposes only
- **Component Integration**: Provides all configuration needed by ExecutionManager and environment managers

#### EpisodeManager
Moved to SessionManager for better separation of concerns:
- **Session-Level Management**: Now managed directly by SessionManager for better architectural separation
- **Episode Lifecycle**: Manages RL-style episode creation, progression, and termination
- **Step Coordination**: Handles individual action steps within episodes and tracks progression
- **RL Interface**: Provides episode state management and step creation for RL workflows

#### ExecutionManager
Docker sandbox execution manager for MCP integration:
- **Environment Management**: Orchestrates both ephemeral and permanent Docker environments for secure command processing
- **Dual Manager Architecture**: Uses SandboxEnvironmentManager for session-scoped containers and PermanentEnvironmentManager for server-scoped persistent services
- **Executor Factory**: Manages multiple executor types (CLI, Python) with dynamic selection based on command requirements
- **Security Validation**: Implements comprehensive security validation and command filtering before execution
- **Multi-Network Support**: Enables containers to connect to both isolated episode networks and permanent service networks
- **MCP Integration**: Provides MCP (Model Context Protocol) tool interfaces for agent command execution

#### EvaluationManager
Tracks and evaluates agent performance:
- **Action Logging**: Records all agent actions, commands, and execution results with timestamps
- **Trajectory Tracking**: Maintains complete session trajectories for performance analysis and debugging
- **Performance Metrics**: Tracks session and episode-level metrics for benchmarking evaluation
- **Storage Integration**: Provides pluggable storage backends for trajectory data persistence

#### PolicyManager
Manages domain-specific operational context:
- **Policy Documents**: Defines domain-specific guidelines, constraints, and available command sets
- **Action Validation**: Validates agent actions against domain policies and security constraints
- **Resource Management**: Provides policy information as MCP resources for agent context
- **Domain Configuration**: Manages domain-specific operational rules and behavioral guidelines

### Client Side Architecture

#### SABERHarness
Main client orchestrator with dual execution modes:
- **Dual Execution**: Supports both container-based (recommended) and embedded (legacy) execution modes
- **Agent Integration**: Provides universal agent compatibility with zero-code-change integration
- **Session Management**: Manages SABER server sessions and coordinates episode execution
- **Configuration**: Handles client configuration including parallelism, timeouts, and container settings

#### Container Execution System
Modern containerized agent execution infrastructure:
- **Container Orchestration**: Manages agent containers with resource isolation and parallel execution
- **MCP Sidecar**: Shared HTTP proxy service for standard MCP protocol communication
- **Universal Adapters**: Auto-detects and adapts arbitrary agent interfaces (class/function/async patterns)
- **Resource Management**: Enforces CPU/memory limits, security policies, and network isolation

#### MCP Integration
Industry-standard Model Context Protocol support:
- **Standard Compatibility**: Works with anthropic/mcp-python and other standard MCP libraries
- **HTTP Proxy**: FastAPI-based sidecar providing MCP-over-HTTP endpoints for agent containers
- **Session Routing**: Maps agent container sessions to SABER server sessions with header injection
- **Tool Execution**: Proxies MCP tool calls to SABER server while maintaining session context

#### Agent Runtime System
Lightweight container runtime for universal agent execution:
- **Agent Discovery**: Automatic agent loading from environment variables, modules, or files
- **Interface Adapters**: Universal adapters for different agent patterns with parameter mapping
- **MCP Client Factory**: Creates standard MCP clients for sidecar communication
- **Execution Environment**: Minimal 458MB container runtime with security hardening

### Connection Flow

1. **Client Initialization**: SABERHarness initializes with container or embedded execution mode
2. **Infrastructure Setup**: Container mode starts MCP sidecar and creates Docker network isolation
3. **Session Creation**: Client connects to SABER server via REST API to create session
4. **Agent Packaging**: Agent code packaged for container execution with universal adapters
5. **Episode Execution**: Agent containers execute episodes with MCP tool calls through sidecar
6. **Tool Routing**: MCP sidecar proxies tool calls to SABER server with session mapping
7. **Command Execution**: SABER server executes commands in ephemeral sandbox environments
8. **Results Collection**: Episode results collected from multiple channels (container, REST, MCP)
9. **Cleanup**: Guaranteed container termination and resource cleanup after completion
