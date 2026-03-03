"""SABER Inspect AI integration.

This package provides Inspect AI-specific integration code:
- agents/: Agent solver factory and registry
- integration/: Transcript sync, model wrapper, tool source

The old server-based pipeline (core/, server/, config/, saber.py) has been
archived to .archive/. Use saber.task.create_task for the new pipeline.
"""
