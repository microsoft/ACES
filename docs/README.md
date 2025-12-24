# SABER Architecture Documentation

This directory contains comprehensive documentation for the SABER system architecture, including design documents and visual diagrams.

## Architecture Diagrams

All architecture diagrams are located in [assets/](assets/). Below is a complete inventory organized by subsystem.

### System Overview

| Diagram | Description |
|---------|-------------|
| [SABER System Overview](assets/system/SABER%20System%20Overview.png) | High-level architecture showing the complete SABER system with client-server separation, inspect_ai integration, and Docker sandbox execution. This is the best starting point for understanding how all components fit together. |

### Client-Side (Inspect AI Integration)

| Diagram | Description |
|---------|-------------|
| [SABER Inspect AI Integration Architecture](assets/client/SABER%20Inspect%20AI%20Integration%20Architecture.png) | Component diagram showing how SABER integrates with inspect_ai, including the SABERSandboxEnvironment, task factories, and agent registry. Illustrates the key abstractions that enable seamless evaluation workflows. |
| [SABER Inspect AI Integration Sequence](assets/client/SABER%20Inspect%20AI%20Integration%20Sequence.png) | Sequence diagram tracing a complete evaluation from task creation through sample execution. Shows the interactions between inspect_ai, SABER client components, and the server during an evaluation run. |
| [SABER MCP Architecture](assets/client/SABER%20MCP%20Architecture.png) | Architecture of the Model Context Protocol (MCP) client that provides tool access to agents. Details how tools are discovered, cached, and invoked through the MCP protocol. |

### Server-Side Architecture

| Diagram | Description |
|---------|-------------|
| [SABER Server Architecture - Current Implementation](assets/server/SABER%20Server%20Architecture%20-%20Current%20Implementation.png) | Overview of server-side components including the dual-protocol design with FastAPI REST endpoints and FastMCP server. Shows how SessionManager orchestrates all server operations. |
| [SABER Server Architecture - Detailed](assets/server/session/SABER%20Server%20Architecture%20-%20Detailed.png) | Detailed view of session management internals including lifecycle states, resource tracking, and cleanup strategies. Essential for understanding multi-session server behavior. |

#### API Layer

| Diagram | Description |
|---------|-------------|
| [MCP Integration Workflow](assets/server/api/MCP%20Integration%20Workflow.png) | End-to-end workflow for MCP tool operations from client request through execution and response. Shows how tool calls are validated, routed to the correct episode, and executed in Docker sandboxes. |

#### Benchmark Management

| Diagram | Description |
|---------|-------------|
| [SABER Benchmark Framework Architecture](assets/server/benchmarks/SABER%20Benchmark%20Framework%20Architecture%20-%20Current%20Implementation.png) | Architecture of the YAML-based benchmark and task configuration system. Explains how domain.yaml files are parsed, validated, and transformed into executable task definitions. |

#### Episode Management

| Diagram | Description |
|---------|-------------|
| [Episode Creation Flow with Semaphore Management](assets/server/episodes/Episode%20Creation%20Flow%20with%20Semaphore%20Management.png) | Detailed flow showing how episodes are created with proper concurrency control. Illustrates semaphore acquisition, Docker container setup, and resource allocation during episode initialization. |
| [Episode Management Workflow](assets/server/episodes/Episode%20Management%20Workflow.png) | Complete lifecycle of an episode from creation through execution to cleanup. Covers state transitions, error handling, and resource release patterns. |
| [Transcript Management Components](assets/server/episodes/Transcript%20Management%20Components.png) | Component diagram for the transcript subsystem that captures agent interactions. Shows TranscriptManager, storage backends, and integration points with episode execution. |
| [Transcript Management Flow](assets/server/episodes/Transcript%20Management%20Flow.png) | Data flow diagram showing how transcript entries are captured, buffered, and persisted. Includes handling of concurrent writes and flush strategies. |
| [Transcript State Machine](assets/server/episodes/Transcript%20State%20Machine.png) | State machine defining valid transcript lifecycle states and transitions. Documents the states (initializing, recording, finalizing, closed) and the events that trigger transitions. |

#### Execution Management

| Diagram | Description |
|---------|-------------|
| [Command Execution Architecture](assets/server/execution/Command%20Execution%20Architecture.png) | Architecture of the Docker sandbox execution system that runs agent commands securely. Details container management, command routing, output capture, and timeout handling. |
| [Sandbox Environment Setup Sequence](assets/server/execution/Sandbox%20Environment%20Setup%20Sequence.png) | Sequence diagram showing sandbox initialization from compose file parsing through container startup. Includes health checks, network configuration, and volume mounting. |

### Task Orchestration

| Diagram | Description |
|---------|-------------|
| [Dependency Graph Validation](assets/architecture/Dependency%20Graph%20Validation.png) | Diagram showing how task dependencies are validated for cycles and ordering. Explains the topological sort algorithm used to determine safe execution order. |
| [Orchestrated Multi-Episode Task Workflow](assets/architecture/Orchestrated%20Multi-Episode%20Task%20Workflow.png) | Workflow for running multiple evaluation episodes concurrently with proper resource management. Shows how max_concurrent_episodes is enforced and how failures are isolated. |
| [Role-Based Configuration System](assets/architecture/Role-Based%20Configuration%20System.png) | Configuration system for defining agent roles in multi-agent scenarios. Documents how roles are declared in YAML, validated, and applied to agent instances. |
| [Semaphore Management Strategy](assets/architecture/Semaphore%20Management%20Strategy.png) | System-wide concurrency control strategy using semaphores. Explains how resource limits are enforced across sessions, episodes, and Docker operations to prevent resource exhaustion. |
| [Single Episode Task Workflow](assets/architecture/Single%20Episode%20Task%20Workflow.png) | Complete workflow for a single evaluation episode from sample receipt through scoring. Shows the interaction between solver, tools, sandbox, and scoring components. |
| [Task Handler Class Diagram](assets/architecture/Task%20Handler%20Class%20Diagram.png) | Class diagram showing the TaskHandler hierarchy and related components. Documents the interfaces and implementations for different task types and execution strategies. |
