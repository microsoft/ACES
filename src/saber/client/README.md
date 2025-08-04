# SABER Client - Zero-Friction Agent Testing Framework

This module provides a generic client framework for testing security agents against the SABER benchmarking system. The client automatically adapts to any agent implementation without requiring code changes, interface implementations, or specific patterns.

## Architecture

```
src/saber/client/
├── __main__.py               # CLI entrypoint for zero-friction testing
├── cli.py                    # CLI module entry point
├── test_harness.py           # Main TestHarness orchestrator
├── server_client.py          # REST API client for SABER server
├── agent_wrapper.py          # Generic agent adaptation framework
└── prompt_builder.py         # Prompt formatting utilities
```

## Key Components

### CLI Interface (`__main__.py`, `cli.py`)
Zero-friction command-line interface for testing any agent:
- **No Code Changes Required**: Works with existing agent implementations
- **Automatic Agent Detection**: Finds and adapts agent methods automatically
- **Flexible Usage**: Supports files, modules, and specific classes
- **Simple Command**: `python -m saber.client --agent <path> --task <id> --episodes <n>`

### TestHarness (`test_harness.py`)
Main orchestrator for agent testing:
- **Generic Agent Support**: Accepts any object with callable methods
- **Session Management**: Handles server connections and episode lifecycle
- **Test Execution**: Coordinates agent-server interactions
- **Result Tracking**: Provides comprehensive test metrics and logging
- **Configuration**: Flexible configuration via TestHarnessConfig

### ServerClient (`server_client.py`)
Simple REST API client for SABER server communication:
- **Session Lifecycle**: Create, manage, and close test sessions
- **Episode Management**: Start episodes and execute test steps
- **Task Information**: Retrieve current task and policy details
- **Error Handling**: Robust error handling and connection management
- **Async Support**: Full async/await support for non-blocking operations

### AgentWrapper (`agent_wrapper.py`)
Generic adaptation framework for any agent implementation:
- **Method Detection**: Automatically finds agent processing methods
- **Parameter Adaptation**: Handles different parameter naming conventions
- **Sync/Async Support**: Works with both synchronous and asynchronous agents
- **Reset Handling**: Optional reset method detection and execution
- **No Interface Requirements**: Zero customer code changes needed

### AgentLoader (`agent_wrapper.py`)
Dynamic agent loading from various sources:
- **File Loading**: Load agents from Python files
- **Module Loading**: Import agents from installed packages
- **Class Detection**: Automatically find and instantiate agent classes
- **Function Support**: Works with function-based agents
- **Smart Selection**: Prefers classes with "agent" in the name

### PromptBuilder (`prompt_builder.py`)
Formats server data into clear prompts for agents:
- **Task Context**: Includes task descriptions and objectives
- **Command Information**: Lists available commands and constraints
- **Output Formatting**: Presents command results clearly
- **Error Handling**: Formats error messages for agent understanding
- **Progress Updates**: Shows completion status and next steps

## Zero-Friction Design

### Automatic Agent Detection
The client automatically detects and adapts to agents with these patterns:

#### Method Names (automatically detected)
- `process_prompt`, `process`, `generate`, `respond`, `answer`
- `call`, `__call__`, `run`, `execute`, `predict`, `inference`

#### Parameter Names (automatically adapted)
- `prompt`, `input`, `message`, `text`, `query`, `question`

#### Agent Types (all supported)
- **Class-based agents**: Instantiated automatically
- **Function-based agents**: Used directly
- **Async agents**: Handled with proper await
- **Sync agents**: Wrapped for async compatibility

#### Reset Methods (optional, automatically detected)
- `reset`, `clear`, `initialize`, `restart`, `new_session`

### Example Agent Patterns

#### Simple Class Agent
```python
class SecurityAgent:
    def process(self, prompt):
        return "nmap -sS target"
    
    def reset(self):
        pass
```

#### Async Agent
```python
class AsyncAgent:
    async def respond(self, message):
        return "file suspicious.exe"
    
    async def clear(self):
        pass
```

#### Function Agent
```python
def security_analyzer(input_text):
    return "strings malware.bin"
```

#### LLM Integration Agent
```python
class LLMSecurityAgent:
    def __init__(self):
        self.llm = SomeAIModel()
    
    def generate(self, prompt):
        return self.llm.complete(prompt)
```

## Usage Examples

### Basic Usage
```bash
# Test any agent file
python -m saber.client --agent my_agent.py

# Specify task and multiple episodes
python -m saber.client --agent my_agent.py --task malware_analysis --episodes 3

# Use specific class from file
python -m saber.client --agent agents/security_bot.py --agent-class SecurityBot

# Test installed package
python -m saber.client --agent my_package.MyAgent --server http://remote:8000
```

### Configuration Options
```bash
# Available CLI arguments
--agent AGENT              # Path to agent file or module name (required)
--agent-class CLASS         # Specific class name to load (optional)
--task TASK                 # Task ID to run (default: 'default_task')
--episodes N                # Number of episodes (default: 1)
--server URL                # SABER server URL (default: 'http://localhost:8000')
--log-level LEVEL           # Logging level: DEBUG, INFO, WARNING, ERROR
--log-file FILE             # Optional log file path
```

### Programmatic Usage
```python
from saber.client import TestHarness, TestHarnessConfig, AgentLoader

# Load any agent
agent = AgentLoader.load_from_path("my_agent.py")

# Configure test
config = TestHarnessConfig(
    server_url="http://localhost:8000",
    log_level="INFO"
)

# Run test
async with TestHarness(config) as harness:
    await harness.initialize(agent)
    results = await harness.run_test()
    print(f"Completed: {results['completed']}")
```

## Agent Requirements

### Minimal Requirements
Your agent needs **only one** of these patterns:
1. A callable method that accepts text input and returns text output
2. A callable function that accepts text input and returns text output

### Supported Patterns
- ✅ Any method name (process, generate, respond, etc.)
- ✅ Any parameter name (prompt, input, message, etc.)
- ✅ Sync or async methods
- ✅ Class or function-based
- ✅ Optional reset/clear methods
- ✅ Any return type (converted to string)

### What's NOT Required
- ❌ No specific interfaces to implement
- ❌ No inheritance requirements
- ❌ No method naming conventions
- ❌ No parameter naming conventions
- ❌ No async/await requirements
- ❌ No reset method requirements
- ❌ No code changes to existing agents

## Error Handling

### Agent Loading Errors
- **File not found**: Clear error message with file path
- **No agent detected**: Lists expected method names
- **Multiple agents**: Prefers classes with "agent" in name
- **Import errors**: Shows import traceback for debugging

### Runtime Errors
- **Server connection**: Retry logic and clear error messages
- **Agent errors**: Captured and logged without crashing
- **Network issues**: Timeout handling and graceful degradation
- **Session errors**: Automatic cleanup and resource management

### Debugging Support
- **Verbose logging**: Detailed execution traces
- **Method detection**: Shows which methods were found
- **Parameter mapping**: Logs parameter name detection
- **Step tracking**: Detailed step-by-step execution logs

## Integration Examples

### With Popular AI Frameworks

#### OpenAI Integration
```python
import openai

class OpenAISecurityAgent:
    def __init__(self):
        self.client = openai.OpenAI()
    
    def process(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

#### LangChain Integration
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class LangChainAgent:
    def __init__(self):
        self.llm = OpenAI()
        self.prompt = PromptTemplate.from_template(
            "Security analysis task: {input}"
        )
    
    def generate(self, input_text):
        formatted_prompt = self.prompt.format(input=input_text)
        return self.llm(formatted_prompt)
```

#### Custom Neural Network
```python
import torch

class CustomSecurityModel:
    def __init__(self):
        self.model = torch.load("security_model.pth")
    
    def predict(self, prompt):
        # Custom model inference
        result = self.model.generate(prompt)
        return result.decoded_text
```

## Performance Considerations

### Async Support
- Automatic detection of async methods
- Proper await handling for async agents
- Non-blocking server communication
- Concurrent episode execution support

### Memory Management
- Automatic agent cleanup between episodes
- Optional reset method calls
- Session-based resource management
- Configurable timeout handling

### Logging and Monitoring
- Structured logging with configurable levels
- Performance metrics tracking
- Step execution timing
- Error rate monitoring

## Testing and Development

### Testing Your Agent
```bash
# Quick test with minimal output
python -m saber.client --agent my_agent.py --episodes 1 --log-level WARNING

# Detailed debugging
python -m saber.client --agent my_agent.py --log-level DEBUG --log-file debug.log

# Multiple episodes for consistency
python -m saber.client --agent my_agent.py --episodes 10
```

### Development Workflow
1. **Create agent** with any pattern you prefer
2. **Test locally** using the CLI
3. **Debug issues** with verbose logging
4. **Scale testing** with multiple episodes
5. **Deploy** without any code changes

### Common Issues and Solutions

#### Agent Not Found
```bash
# Problem: "No agent found in module"
# Solution: Ensure your agent has a recognizable method name
# Good: process(), generate(), respond(), answer()
# Also good: Custom function names work too
```

#### Method Detection
```bash
# Problem: Wrong method detected
# Solution: Use --agent-class to specify exact class
python -m saber.client --agent my_agent.py --agent-class SpecificAgent
```

#### Parameter Mismatch
```bash
# Problem: Agent expects different parameter name
# Solution: The wrapper automatically detects parameter names
# Works with: prompt, input, message, text, query, question, etc.
```

This zero-friction design ensures that **any existing agent can be tested immediately** without requiring any code modifications or specific implementation patterns.
