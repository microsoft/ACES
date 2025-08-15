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
- **[Client Architecture](client_architecture.puml)**: Client side architecture responsible for attaching to agent and facilitating comms with server
- **[MCP Integration Sequence](mcp_integration_sequence.puml)**: Model Context Protocol integration workflow and dual protocol communication

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
- **Server-Sent Events**: Real-time streaming updates for session events and command execution progress  
- **Request Delegation**: Converts HTTP requests to SessionManager method calls with proper error handling
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation for client integration

#### SessionMCPAPI
Model Context Protocol handler component managed by SessionManager:
- **MCP Server**: Hosts MCP server alongside REST API for agent tool execution only
- **Dynamic Tool Discovery**: Provides real-time tool schemas from ExecutionManager
- **Tool Execution**: Maps MCP tool calls to SessionManager command execution pipeline
- **Session Context**: Maintains session mapping between MCP clients and SessionManager sessions
- **Focused Scope**: ONLY handles tool discovery and execution via MCP protocol

#### TaskManager
Simplified task definition management:
- **Task Configuration**: Loads and manages task definitions from YAML configuration files
- **Task Lookup**: Provides task object retrieval by task ID with complete configuration
- **Configuration Containment**: Task objects now contain all execution parameters (environment, allowed_executors)
- **Focused Responsibility**: Only handles task definition parsing and lookup - no episode management

##### TaskConfigLoader
Enhanced YAML configuration management:
- **Complete Task Loading**: Parses task definitions including execution configuration (allowed_executors, environment)
- **Task Validation**: Ensures proper YAML structure, required fields, and execution parameters
- **Object Creation**: Creates Task objects with complete configuration needed by other components
- **Single Source**: Only component that reads task configuration files

##### Task Framework
Enhanced Task objects with complete execution configuration:
- **Complete Configuration**: Tasks contain environment, allowed_executors, and all execution parameters
- **Self-Contained**: No need for separate configuration lookups - all parameters in Task object
- **Subtask Information**: Contains subtasks for informational purposes only
- **Component Integration**: Provides all configuration needed by ExecutionManager and other components

#### EpisodeManager
Moved to SessionManager for better separation of concerns:
- **Session-Level Management**: Now managed directly by SessionManager for better architectural separation
- **Episode Lifecycle**: Manages RL-style episode creation, progression, and termination
- **Step Coordination**: Handles individual action steps within episodes and tracks progression
- **RL Interface**: Provides episode state management and step creation for RL workflows

#### ExecutionManager
Docker sandbox execution manager for MCP integration:
- **Sandbox Management**: Orchestrates Docker-based isolated execution environments for secure command processing
- **Executor Factory**: Manages multiple executor types (CLI, Python) with dynamic selection based on command requirements
- **Security Validation**: Implements comprehensive security validation and command filtering before execution
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

#### ServerClient
REST API client for server communication:
- **HTTP Communication**: Handles all REST API communication with SABER server with proper error handling and timeouts
- **Session Management**: Manages client-side session state and maintains connection with server sessions
- **Rich UI Integration**: Provides console UI components for progress tracking, panels, and status displays
- **Event Streaming**: Supports Server-Sent Events for real-time updates from server operations

#### AgentWrapper
Automatic agent adaptation system:
- **Universal Compatibility**: Automatically detects and adapts arbitrary agent implementations to SABER interface
- **Method Detection**: Discovers agent capabilities (sync/async, method names, parameter conventions)
- **Interface Standardization**: Provides consistent async interface regardless of underlying agent implementation
- **Zero-Code Integration**: Enables existing agents to work with SABER without modification

#### TestHarness
Main test execution controller:
- **Orchestration**: Coordinates server client, agent wrapper, and prompt builder for complete test execution
- **Logging Framework**: Implements structured logging with JSON formatting and context tracking
- **Configuration Management**: Handles test configuration, timeouts, and execution parameters
- **Results Management**: Collects and processes test results with detailed execution tracking

#### PromptBuilder
Context-aware prompt generation:
- **Dynamic Prompts**: Generates context-specific prompts based on task state, episode progress, and domain information
- **Template System**: Uses configurable templates for consistent prompt structure across different scenarios
- **Context Integration**: Incorporates task context, policy information, and execution history into prompts
- **Adaptive Formatting**: Adjusts prompt format based on agent capabilities and task requirements

### Connection Flow

1. Client connects to SessionManager via HTTP for session management
2. SessionManager creates ClientSession and delegates TaskSession creation to TaskManager
3. Client establishes MCP connection to MCPSessionManager for tool access
4. Establishes SSE connection for real-time communication
5. SessionManager coordinates with TaskManager to assign initial subtask with context
6. Client discovers available tools via MCP protocol and executes commands through MCP tool calls
7. MCPSessionManager delegates tool execution to SessionManager's ExecutionManager
8. Results logged via EvaluationManager and next subtask assigned via TaskManager
9. Process continues until task completion with dual protocol communication
