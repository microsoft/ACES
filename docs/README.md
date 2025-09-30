# SABER Architecture Documentation - Current Implementation

## Overview

This document describes the current architecture for SABER (Security Agent Benchmarking and Evaluation Research), a modern distributed system designed to evaluate agentic workflows in cybersecurity domains using **Inspect AI integration** with **Model Context Protocol (MCP)**.

**ARCHITECTURE**: The system uses inspect_ai framework integration with async/await patterns throughout. Agents run directly via inspect_ai

## Current Architecture Diagrams

### System Overview
- **[System Overview](system/system_overview_architecture.puml)**: Current system components with inspect_ai integration

### Server Architecture (Current)
- **[Server Architecture Current](server/server_architecture_current.puml)**: Current async FastAPI/FastMCP implementation
- **[Command Execution Architecture](server/execution/command_execution_architecture.puml)**: Docker sandbox execution framework for server-side command isolation

### Client Architecture (Current)
- **[Client Architecture](client/client_architecture.puml)**: Inspect AI integration architecture
- **[Inspect AI Integration Sequence](client/inspect_ai_integration_sequence.puml)**: Current client-server interaction flow


## Current Implementation Summary

### Core Architecture Principles

1. **Direct Agent Execution**: Agents run in the same process as the evaluation orchestrator using inspect_ai framework
2. **Async Context Managers**: Proper resource management with async/await patterns throughout
3. **Fail-Fast Design**: Upfront validation with structured error handling
4. **Session-Based MCP**: Per-episode MCP clients for tool access via inspect_ai native integration
5. **Server-Side Sandboxing**: Docker containers used only on server side for secure command execution

### Core Components (Current)

#### Server Side
- **SessionManager**: Central orchestrator managing multi-session server with FastAPI/FastMCP
- **SessionRestAPI**: REST protocol handler for session/episode management
- **SessionMCPAPI**: MCP protocol handler for tool discovery and execution only
- **ExecutionManager**: Docker sandbox orchestration for secure command execution on server
- **BenchmarkManager**: YAML-based task and benchmark configuration
- **EpisodeManager**: Multi-episode session support with termination handling  
- **PolicyManager**: Dynamic policy and prompt generation
- **EvaluationManager**: Trajectory tracking and performance metrics

#### Client Side  
- **SABEREvaluationOrchestrator**: Main async context manager coordinating evaluation lifecycle
- **ClientSessionManager**: HTTP session lifecycle with REST/MCP clients
- **AgentManager**: Agent discovery, initialization, and lifecycle management
- **DatasetManager**: SABER dataset creation from server benchmark info
- **SABERReactAgent**: inspect_ai compatible agent with SABER MCP tool integration


## System Architecture

### Core Design Principles

1. **Separation of Concerns**: Clear boundaries between client orchestration and server execution
2. **Fail-Fast**: Upfront validation prevents runtime surprises
3. **Resource Management**: Async context managers ensure proper cleanup
4. **Security First**: Server-side Docker sandboxing for untrusted command execution
5. **Industry Standards**: inspect_ai for evaluation, MCP for tool protocol

## System Components

### Server Side Architecture

The server manages concurrent client sessions and provides secure command execution via Docker sandboxing.

#### SessionManager (SABER Domain Server)
The central orchestrator for each security domain, responsible for:
- **Multi-Session Management**: Creates, tracks, and manages multiple concurrent client sessions with lifecycle control
- **Component Coordination**: Orchestrates BenchmarkManager, ExecutionManager, PolicyManager, and EvaluationManager
- **Domain Hosting**: Provides single-domain server instances designed for horizontal scaling
- **Server Lifecycle**: Handles startup, shutdown, and resource cleanup across all active sessions
- **Episode Management**: Manages RL-style episode creation, progression, and termination per session

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
- **Focused Scope**: ONLY handles tool discovery and execution via MCP protocol - no session management

#### BenchmarkManager
Benchmark orchestration and task management:
- **Task Configuration**: Loads and manages task definitions from YAML configuration files
- **Benchmark Orchestration**: Manages multiple episode attempts for pass@k evaluation
- **Task Lookup**: Provides task object retrieval by task ID with complete configuration
- **Episode Coordination**: Orchestrates multiple episode runs across all benchmark tasks
- **Configuration Management**: Tasks contain all execution parameters (sandbox_environment, permanent_environment, allowed_executors)

##### BenchmarkConfigLoader
YAML configuration management:
- **Task Loading**: Parses task definitions including execution configuration
- **Benchmark Configuration**: Loads domain-level and task-specific benchmark settings
- **Task Validation**: Ensures proper YAML structure and required fields
- **Object Creation**: Creates Task objects with complete configuration
- **Single Source**: Only component that reads task configuration files

##### Benchmark Framework
Task objects with complete execution and benchmark configuration:
- **Complete Configuration**: Tasks contain sandbox_environment, permanent_environment references, allowed_executors
- **Benchmark Settings**: Tasks include episode_attempts, success_criteria
- **Dual Environment Support**: Both ephemeral sandbox and permanent shared services
- **Self-Contained**: All parameters in Task object - no separate lookups needed
- **Subtask Information**: Contains subtasks for informational purposes only

#### EpisodeManager
Managed directly by SessionManager for better separation of concerns:
- **Session-Level Management**: Per-session episode lifecycle management
- **Episode Lifecycle**: Manages RL-style episode creation, progression, and termination
- **Step Coordination**: Handles individual action steps within episodes
- **RL Interface**: Provides episode state management for RL workflows

#### ExecutionManager
Docker sandbox execution manager for server-side command isolation:
- **Environment Management**: Orchestrates ephemeral and permanent Docker environments for secure command processing
- **Dual Manager Architecture**: SandboxEnvironmentManager (session-scoped) and PermanentEnvironmentManager (server-scoped)
- **Executor Factory**: Manages multiple executor types (CLI, Python, SQL) with dynamic selection
- **Security Validation**: Comprehensive security validation and command filtering before execution
- **Multi-Network Support**: Containers connect to isolated episode networks and permanent service networks
- **MCP Tool Interface**: Provides MCP tool schemas for agent command execution

#### EvaluationManager
Tracks and evaluates agent performance:
- **Action Logging**: Records all agent actions, commands, and execution results with timestamps
- **Trajectory Tracking**: Maintains complete session trajectories for performance analysis
- **Performance Metrics**: Tracks session and episode-level metrics for benchmarking
- **Storage Integration**: Pluggable storage backends for trajectory data persistence

#### PolicyManager
Manages domain-specific operational context:
- **Policy Documents**: Domain-specific guidelines, constraints, and available command sets
- **Action Validation**: Validates agent actions against domain policies and security constraints
- **Resource Management**: Provides policy information as MCP resources for agent context
- **Domain Configuration**: Domain-specific operational rules and behavioral guidelines

### Client Side Architecture

The client orchestrates evaluations using inspect_ai framework with direct agent execution.

#### SABEREvaluationOrchestrator
Main async context manager orchestrating evaluation workflow:
- **Component Lifecycle**: Manages initialization and cleanup of all client components
- **Upfront Validation**: Validates configuration and server connectivity before execution
- **Resource Management**: Ensures proper cleanup via async context manager pattern
- **Evaluation Execution**: Coordinates dataset creation, agent initialization, and eval_async invocation
- **Error Handling**: Structured error propagation with detailed context

#### ClientSessionManager
HTTP session lifecycle manager for dual-protocol communication:
- **REST API Client**: Session and episode management via HTTP
- **MCP Client**: Per-episode MCP clients for tool discovery and execution
- **Connection Pooling**: Efficient HTTP session reuse
- **Error Handling**: Network error handling with retries and timeouts
- **Session Context**: Manages session IDs and episode contexts

#### AgentManager
Agent discovery and lifecycle management:
- **Agent Resolution**: Resolves agent assignments from config and dataset metadata
- **Direct Agent Creation**: Creates agents using SABERAgentFactory during task setup
- **Task Integration**: Integrates agents with inspect_ai Task objects
- **Lifecycle Management**: Handles agent initialization and cleanup
- **No Agent Caching**: Simplified architecture with on-demand agent creation

#### DatasetManager  
SABER dataset creation for inspect_ai integration:
- **Task Discovery**: Fetches available tasks from SABER server
- **Dataset Creation**: Converts SABER tasks to inspect_ai Sample objects
- **Metadata Handling**: Preserves task metadata for agent resolution
- **Episode Attempts**: Supports multiple attempts per task for pass@k evaluation
- **Clean Conversion**: Maintains separation between task description and agent prompt

#### SABERAgentFactory
Factory for creating agents from various sources:
- **Registry Creation**: Creates agents from SABERAgentRegistry by ID
- **Custom Agents**: Loads custom agent implementations from files
- **inspect_ai Integration**: Creates agents compatible with inspect_ai evaluation
- **MCP Tool Integration**: Integrates SABER MCP tools via inspect_ai native patterns
- **Session Context**: Passes session manager context to agents

#### SABERAgentRegistry
Registry system for agent discovery:
- **Agent Registration**: Decorator-based registration of agent factory functions
- **Agent Specs**: Maintains metadata about available agents
- **Built-in Agents**: Includes saber_react agent with MCP integration
- **Custom Registration**: Supports registration of custom agent implementations
- **Agent Discovery**: Provides list of available agents with metadata

#### SABERReactAgent
inspect_ai compatible agent with SABER MCP integration:
- **React Pattern**: Uses inspect_ai's react() agent for tool use loops
- **MCP Tools**: Accesses SABER server tools via inspect_ai's native MCP integration
- **Session Context**: Maintains session and episode context via inspect_ai store
- **Episode Management**: Handles episode creation and termination
- **Policy Integration**: Retrieves and uses domain-specific policies

### Connection Flow

1. **Client Initialization**: SABEREvaluationOrchestrator validates config and initializes components
2. **Server Connection**: ClientSessionManager establishes REST and MCP connections to SABER server
3. **Dataset Creation**: DatasetManager fetches tasks and creates inspect_ai dataset
4. **Agent Creation**: AgentManager creates agents using SABERAgentFactory for each task
5. **Evaluation Execution**: inspect_ai eval_async runs tasks with direct agent execution
6. **Tool Execution**: Agents use MCP tools via inspect_ai's native integration (no sidecar needed)
7. **Command Execution**: SABER server executes commands in Docker sandboxes and returns results
8. **Results Collection**: inspect_ai collects results and generates evaluation logs
9. **Cleanup**: Async context managers ensure proper resource cleanup
