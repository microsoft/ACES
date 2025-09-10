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

#### SessionManagerConfig
**Unified configuration for REST and MCP communication**
- Combines both REST API and MCP server configuration 
- REST API settings: base_url, client_id, rest_timeout
- MCP server configuration: mcp_server_url for agent tools
- Factory method `from_urls()` for convenient creation
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

eval_log = await run_saber_eval_async(config=config)
```

### MCP Tool Usage
```python
# Native inspect_ai MCP integration (used internally by SABER agents)
from inspect_ai.tool import mcp_server_http

# SABER agents automatically get MCP tools via inspect_ai
# No manual MCP client management required
saber_tools = mcp_server_http(
    url=f"{config.session_config.mcp_server_url}/mcp",
    headers={"session_id": "...", "episode_id": "..."}
)
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
- ❌ **Container-based execution** (replaced with native inspect_ai agent execution)
- ❌ **SABERHarness** (replaced with `run_saber_eval_async`)
- ❌ **Sidecar management** (replaced with inspect_ai's native MCP integration)
- ❌ **Custom agent adapters** (replaced with standard inspect_ai agent patterns)
- ❌ **Container logging infrastructure** (replaced with inspect_ai logging)
- ❌ **Manual MCP client management** (handled natively by inspect_ai)

### New Components  
- ✅ **inspect_ai integration** via `run_saber_eval_async`
- ✅ **Native MCP integration** via inspect_ai's `mcp_server_http`
- ✅ **Type-safe Pydantic models** with fail-fast validation
- ✅ **Async context manager lifecycle** for resource management
- ✅ **Unified configuration** with `SABERConfig` and `SessionManagerConfig`
- ✅ **Agent registry system** with `InspectAIAgentFactory`
- ✅ **Enhanced CLI tools** for evaluation analysis

## CLI Tools

SABER provides a comprehensive CLI for interacting with the client and analyzing evaluation results. The CLI includes powerful tools for examining inspect-ai evaluation logs and launching the Inspect AI web viewer.

### Installation and Setup

After installing SABER, the CLI is available via:

```bash
# Via Python module
python -m saber.client

# Show all available commands
python -m saber.client --help
```

### Commands Overview

#### 1. Evaluation Execution
```bash
# Run SABER evaluation with default config (saber.yaml in current directory)
python -m saber.client run

# Run with specific config file
python -m saber.client run --config /path/to/saber.yaml

# Run with verbose logging
python -m saber.client run --verbose

# Run without file logging (console only)
python -m saber.client run --no-log-file
```

#### 2. Inspect AI Integration Commands
```bash
# Show inspect AI integration commands
python -m saber.client inspect --help
```

### Log Viewer (Web UI)

Launch Inspect AI's powerful web-based log viewer:

```bash
# Basic usage - launch viewer for logs directory
python -m saber.client inspect view --log-dir /path/to/logs

# Advanced options
python -m saber.client inspect view \
    --log-dir /path/to/logs \
    --host 0.0.0.0 \
    --port 7575 \
    --no-browser \
    --no-recursive
```

**Options:**
- `--log-dir`: Directory containing `.eval` or `.json` log files (required)
- `--host`: Host to bind the web server to (default: 127.0.0.1)
- `--port`: Port to bind the web server to (default: 7575)
- `--no-recursive`: Don't scan subdirectories for logs
- `--no-browser`: Don't automatically open browser

The web viewer provides:
- **Rich log visualization**: Interactive display of evaluation results
- **Sample-level analysis**: Detailed view of model conversations
- **Performance metrics**: Scoring and timing information
- **Real-time updates**: Live viewing of running evaluations
- **Export capabilities**: Download and share results

### Docker Usage

#### Running from Inside Containers

When running inside Docker containers (like the excytin demo), use `uv run`:

```bash
# From inside the excytin client container
docker exec -it saber-excytin-client uv run python -m saber.client --help

# Run evaluation from container
docker exec -it saber-excytin-client uv run python -m saber.client run --config /app/client/saber.yaml

# Launch log viewer from container (accessible from host at localhost:7575)
docker exec -d saber-excytin-client uv run python -m saber.client inspect view \
    --log-dir /app/logs \
    --no-browser \
    --host 0.0.0.0
```

#### Port Mapping for Log Viewer

The excytin demo Docker Compose configuration includes port mapping for the log viewer:

```yaml
# In docker-compose.yml
saber-excytin-client:
  ports:
    - "7575:7575"  # Inspect AI log viewer
```

This allows you to run the viewer inside the container but access it from your host browser at `http://localhost:7575`.

### Evaluation Log Analysis

The CLI provides powerful tools for analyzing inspect-ai evaluation logs (`.eval` files) generated by SABER runs.

#### Basic Usage

```bash
# Get help
python -m saber.client inspect eval --help

# Analyze an evaluation log
python -m saber.client inspect eval --log-file path/to/evaluation.eval
```

#### Output Formats

The CLI supports multiple output formats optimized for different use cases:

##### 1. Summary Format (Default)
**Quick overview of evaluation results**

```bash
python -m saber.client inspect eval --log-file evaluation.eval --format summary
```

Shows:
- 📊 **Evaluation Stats**: Total samples, duration, model usage
- 🎯 **Score Summary**: Aggregated results across all samples
- ⚠️ **Error Overview**: Any evaluation errors encountered

##### 2. Samples Format  
**Detailed agent conversation analysis**

```bash
python -m saber.client inspect eval --log-file evaluation.eval --format samples
```

Shows:
- 🤖 **Agent Conversations**: Complete interaction flow with thinking, actions, and tool calls
- 💭 **Agent Reasoning**: "Thought:" sections showing agent decision-making process
- 🛠️ **Tool Interactions**: Commands executed and their results
- 📊 **Sample Scores**: Per-sample evaluation metrics
- 🎨 **Rich Formatting**: Color-coded roles and emojis for easy reading

##### 3. Full Format
**Comprehensive evaluation analysis**

```bash
python -m saber.client inspect eval --log-file evaluation.eval --format full
```

Combines summary + samples + additional metadata:
- Everything from summary and samples formats
- 🛠️ **Plan Details**: Solver configuration and evaluation setup
- 📋 **Extended Metadata**: Full evaluation context

##### 4. JSON Format
**Machine-readable output for automation**

```bash
python -m saber.client inspect eval --log-file evaluation.eval --format json
```

Raw JSON output perfect for:
- 🔧 **Automation**: Parsing results in scripts
- 📈 **Analysis**: Integration with data analysis tools
- 🔍 **Debugging**: Detailed inspection of evaluation data

#### Advanced Options

```bash
# Control number of samples shown
python -m saber.client inspect eval --log-file evaluation.eval --format samples --max-samples 5

# Show specific sample by ID
python -m saber.client inspect eval --log-file evaluation.eval --format samples --sample-id "sample_123"

# Disable pretty formatting (plain text)
python -m saber.client inspect eval --log-file evaluation.eval --format samples --no-pretty

# JSON output without pretty printing
python -m saber.client inspect eval --log-file evaluation.eval --format json --no-pretty
```

#### Agent Conversation Analysis

The **samples format** provides detailed insight into agent behavior:

**🤖 Agent Thinking Process:**
```
🤖 Assistant:
  💭 To analyze command and control (C2) behavior through database investigation, 
     I need to start by exploring the database schema...
  ⚡ mysql -u root -e 'SHOW TABLES;' saber_security
  🛠️  cli: mysql -u root -e 'SHOW TABLES;' saber_security
```

**🔧 Tool Execution Results:**
```
🔧 Tool:
  ✅ Output: +------------------+
             | Tables_in_saber  |
             +------------------+
             | alerts           |
             | network_flows    |
  ❌ Error: Connection timeout after 30s
```

This format makes it easy to:
- **Debug Agent Behavior**: See exactly how agents reason and act
- **Identify Issues**: Spot where agents get stuck or make errors
- **Improve Prompts**: Understand agent decision-making patterns
- **Validate Performance**: Confirm agents follow expected workflows

### Example Workflows

#### Complete Workflow Example

```bash
# 1. Run an evaluation
python -m saber.client run --config saber.yaml --verbose

# 2. View the results in the web UI
python -m saber.client inspect view --log-dir ./logs

# 3. Analyze specific results from CLI
python -m saber.client inspect eval \
    --log-file ./logs/2025-09-10T02-52-40+00-00_task_T97zyZ8eu7ZQpQVFyHYJ4i.eval \
    --format summary
```

#### Docker Container Example

```bash
# 1. Start the containers
cd domains/excytin_demo
docker compose up -d

# 2. Run evaluation inside container
docker exec -it saber-excytin-client uv run python -m saber.client run \
    --config /app/client/saber.yaml

# 3. Launch viewer for results (detached, accessible at localhost:7575)
docker exec -d saber-excytin-client uv run python -m saber.client inspect view \
    --log-dir /app/logs \
    --no-browser \
    --host 0.0.0.0

# 4. Open browser to http://localhost:7575 to view results
```

#### Analysis Workflows

**Quick Health Check:**
```bash
# Get overview of recent evaluation
python -m saber.client inspect eval --log-file latest.eval --format summary
```

**Detailed Agent Analysis:**
```bash
# Examine agent conversations and tool usage
python -m saber.client inspect eval --log-file evaluation.eval --format samples --max-samples 3
```

**Automation Integration:**
```bash
# Extract scores for automated reporting
python -m saber.client inspect eval --log-file evaluation.eval --format json --no-pretty | jq '.samples[].scores'
```

**Debugging Failed Evaluations:**
```bash
# Full analysis to understand failures
python -m saber.client inspect eval --log-file failed.eval --format full
```

### Configuration

#### SABER Configuration File

The `run` command requires a SABER configuration file (YAML format). Example:

```yaml
# saber.yaml
agent_path: "./my_agent.py"
saber_rest_url: "http://localhost:8000"
saber_mcp_url: "http://localhost:8001"
log_dir: "./logs"
# ... other configuration options
```

#### Environment Variables

- `INSPECT_LOG_DIR`: Default directory for inspect-ai logs
- `SABER_CONFIG_DIR`: Default directory for SABER configurations

#### Debug Mode

For debugging, enable verbose logging:

```bash
python -m saber.client run --verbose --config your-config.yaml
```
