# SABER Agent Registry System

This document describes the configurable agent system for SABER domains.

## Overview

SABER uses a three-tier agent discovery system that allows:
1. **Core SABER agents** - Reusable implementations shared across all domains
2. **Domain-local agents** - Custom agents specific to each domain
3. **CLI selection** - Choose agents via `-T agent=<name>` parameter

## Agent Discovery Order

When you specify an agent (or use the default), SABER searches in this order:

```
1. Domain-local agents → domains/{domain}/client/{agent}.py
2. Core SABER agents   → saber/inspect_ai/agents/{agent}.py
3. Error if not found
```

## Using Agents

### Use Default Agent (defined per domain)

```bash
# Uses default agent (typically "react")
inspect eval domains/cybench --model openai/gpt-4
```

### Use Core SABER Agent

```bash
# Use the core "react" agent
inspect eval domains/cybench --model openai/gpt-4 -T agent=react
```

### Use Domain-Local Agent

```bash
# Use custom agent from domains/cybench/client/custom_example.py
inspect eval domains/cybench --model openai/gpt-4 -T agent=custom_example
```

## Creating Core Agents

Core agents are maintained in `external/saber/src/saber/inspect_ai/agents/`.

**Example: `external/saber/src/saber/inspect_ai/agents/react.py`**

```python
"""React agent implementation for SABER."""

from inspect_ai.agent import react
from inspect_ai.agent._types import AgentPrompt
from ..tools import saber_tools


def create_agent(**kwargs):
    """Create a React agent with SABER integration.
    
    Returns:
        Callable that creates agent with prompts
    """
    def create_with_prompts(instruction_prompt: str, 
                           assistant_prompt: str, 
                           submit_prompt: str):
        return react(
            prompt=AgentPrompt(
                instructions=instruction_prompt,
                handoff_prompt=None,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            ),
            tools=[saber_tools()],
            **kwargs
        )
    
    return create_with_prompts


__all__ = ["create_agent"]
```

**Key Requirements:**
- File must be in `external/saber/src/saber/inspect_ai/agents/`
- Must have a `create_agent()` function
- Function returns a factory that accepts `(instruction_prompt, assistant_prompt, submit_prompt)`
- Auto-registered on import

## Creating Domain-Local Agents

Domain-local agents live in each domain's `client/` folder.

**Example: `domains/cybench/client/custom_example.py`**

```python
"""Custom CyBench agent."""

from inspect_ai.agent import react
from inspect_ai.agent._types import AgentPrompt
from saber.inspect_ai.tools import saber_tools


def create_agent(**kwargs):
    """Create custom agent for CyBench."""
    
    def create_with_prompts(instruction_prompt: str,
                           assistant_prompt: str,
                           submit_prompt: str):
        # Add domain-specific customizations
        custom_instruction = (
            f"{instruction_prompt}\n\n"
            "DOMAIN-SPECIFIC GUIDANCE:\n"
            "- Check for hidden files\n"
            "- Look in config directories"
        )
        
        return react(
            prompt=AgentPrompt(
                instructions=custom_instruction,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            ),
            tools=[saber_tools()],
            **kwargs
        )
    
    return create_with_prompts


__all__ = ["create_agent"]
```

**Key Requirements:**
- File must be in `domains/{domain}/client/`
- Must have a `create_agent()` function
- Function signature same as core agents
- Loaded dynamically at runtime

## Setting Domain Defaults

Each domain specifies its default agent in its main task file:

**`domains/cybench/cybench.py`**

```python
from pathlib import Path
from saber.inspect_ai import create_domain_task

_domains_root = Path(__file__).resolve().parent.parent

_cybench_factory = create_domain_task(
    domain_slug="cybench",
    domains_root=_domains_root,
    default_agent="react",  # <-- Default agent for this domain
)

@task
def cybench(**kwargs):
    return _cybench_factory(**kwargs)
```

## Agent Factory Pattern

All agents follow this pattern:

```python
def create_agent(**kwargs):
    """Outer function: receives configuration"""
    
    def create_with_prompts(instruction_prompt, assistant_prompt, submit_prompt):
        """Inner function: receives runtime prompts from SABER server"""
        
        # Create actual agent with prompts
        return some_agent(
            prompt=AgentPrompt(...),
            tools=[saber_tools()],
            **kwargs
        )
    
    return create_with_prompts
```

**Why this pattern?**
- Outer function: Called once during task setup, can receive config
- Inner function: Called per sample with SABER-provided prompts
- Allows prompt customization while maintaining SABER integration

## Error Handling

If an agent is not found, you'll see:

```
AgentNotFoundError: Agent 'my_agent' not found for domain 'cybench'.

Agent discovery order:
  1. Domain-local: /path/to/domains/cybench/client/my_agent.py
  2. Core SABER agents: react

To fix:
  - Use a core agent: -T agent=react
  - Create domain agent: /path/to/domains/cybench/client/my_agent.py with create_agent() function
  - Check spelling: 'my_agent'
```

## Available Core Agents

Current core agents in `saber/inspect_ai/agents/`:

- **react** - Standard ReAct agent with reasoning loop

## Roadmap

Future agent types to add:

- **chain** - Simple chain-of-thought agent
- **multiagent** - Coordinator for multiple specialized agents
- **planning** - Agent with explicit planning phase
- **reflexion** - Self-reflection and improvement loop

## Best Practices

1. **Start with core agents** - Use `react` for most tasks
2. **Domain customization** - Only create domain agents when needed
3. **Prompt engineering** - Customize prompts in agent, not server
4. **Tool selection** - All agents should use `saber_tools()`
5. **Logging** - Use `get_saber_logger()` for consistent logging
6. **Testing** - Test custom agents thoroughly before production use

## Examples

### Specialized Security Agent

```python
# domains/cybench/client/security_focused.py

def create_agent(**kwargs):
    def create_with_prompts(instruction_prompt, assistant_prompt, submit_prompt):
        security_instruction = (
            f"{instruction_prompt}\n\n"
            "SECURITY ANALYSIS PROTOCOL:\n"
            "1. Check file permissions and ownership\n"
            "2. Look for SUID/SGID binaries\n"
            "3. Examine environment variables\n"
            "4. Review running processes\n"
            "5. Check network connections"
        )
        
        return react(
            prompt=AgentPrompt(
                instructions=security_instruction,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            ),
            tools=[saber_tools()],
            attempts=3,  # Allow multiple attempts
            **kwargs
        )
    
    return create_with_prompts
```

Usage:
```bash
inspect eval domains/cybench --model openai/gpt-4 -T agent=security_focused
```

### Conservative Agent (fewer attempts)

```python
# domains/cybench/client/conservative.py

def create_agent(**kwargs):
    def create_with_prompts(instruction_prompt, assistant_prompt, submit_prompt):
        return react(
            prompt=AgentPrompt(
                instructions=instruction_prompt,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            ),
            tools=[saber_tools()],
            attempts=1,  # Single attempt only
            **kwargs
        )
    
    return create_with_prompts
```

## Debugging

Enable debug logging to see agent resolution:

```bash
# See which agent is loaded
export INSPECT_LOG_LEVEL=debug
inspect eval domains/cybench --model openai/gpt-4 -T agent=custom_example

# Look for these log entries:
# "Resolving agent 'custom_example' for domain 'cybench'"
# "Using domain-local agent 'custom_example' from cybench/client/"
```
