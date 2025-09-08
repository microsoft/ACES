# SABER Architecture Documentation - Current Implementation

## Overview

This document describes the current architecture for SABER (Security Agent Benchmarking and Evaluation Research), a system designed to evaluate agentic workflows in cybersecurity domains using **Inspect AI integration**. 

**BREAKING CHANGE**: The system has moved from container-based execution to inspect_ai framework integration with async/await patterns throughout.

## Current Architecture Diagrams

### System Overview
- **[System Overview](system/system_overview_architecture.puml)**: Current system components with inspect_ai integration

### Server Architecture (Current)
- **[Server Architecture Current](server/server_architecture_current.puml)**: **NEW** - Current async FastAPI/FastMCP implementation
- **[Command Execution Architecture](server/execution/command_execution_architecture.puml)**: Docker sandbox execution framework (still current)

### Client Architecture (Current)
- **[Client Architecture](client/client_architecture.puml)**: **UPDATED** - Inspect AI integration architecture
- **[Inspect AI Integration Sequence](client/inspect_ai_integration_sequence.puml)**: **NEW** - Current client-server interaction flow

### Obsolete Documentation (Removed)
The following container-based architecture diagrams have been **removed** as they no longer reflect the current implementation:
- ❌ Container Execution Architecture
- ❌ MCP Sidecar Architecture  
- ❌ Agent Runtime Architecture
- ❌ Full E2E Benchmark Sequence

## Current Implementation Summary

### Key Changes from Container Architecture

1. **Inspect AI Integration**: Replaced container-based execution with inspect_ai framework
2. **Async Context Managers**: Proper resource management with async/await patterns
3. **Fail-Fast Design**: Upfront validation with structured error handling
4. **Session-Based MCP**: Per-episode MCP clients instead of sidecar containers
5. **Direct Agent Integration**: inspect_ai agent patterns instead of container adapters

### Core Components (Current)

#### Server Side
- **SessionManager**: Async FastAPI/FastMCP server with multi-session support
- **ExecutionManager**: Docker sandbox execution (unchanged from container architecture)
- **BenchmarkManager**: YAML-based task and benchmark configuration
- **EpisodeManager**: Multi-episode session support with termination handling
- **PolicyManager**: Dynamic policy and prompt generation

#### Client Side  
- **SABEREvaluationOrchestrator**: Main async context manager replacing function-based approach
- **ClientSessionManager**: HTTP session lifecycle with REST/MCP clients
- **AgentManager**: Agent integration with inspect_ai task creation
- **DatasetManager**: SABER dataset creation from server benchmark info
- **SABERReactAgent**: inspect_ai compatible agent with SABER MCP tool integration

## Package Management

This package uses **uv** for Python environment management:

```bash
# Server
uv run python -m saber.server --start --domain pentest_demo

# Client  
uv run python -m saber.client --agent examples/agents/my_agent.py
```

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
