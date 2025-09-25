"""CLI entry point for SABER domain orchestration.

This module provides the main entry point for the domain CLI:
  python -m saber.domain [COMMAND] [OPTIONS]
"""

from .cli import cli

if __name__ == "__main__":
    cli()
