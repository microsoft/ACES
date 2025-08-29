#!/usr/bin/env python3
"""
Simple ReAct Agent for SABER

This file demonstrates how to use the ReAct agent from the SABER client library.
Can be used directly with the SABER CLI.

Usage:
    python -m saber.client --agent examples/simple_react_agent.py
"""

from saber.client.agents import ReActAgent

# Export the agent class for the CLI to discover
__all__ = ["ReActAgent"]

# The CLI will automatically detect and use ReActAgent class
# No additional code needed - the harness handles everything!
