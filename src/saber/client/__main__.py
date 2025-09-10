#!/usr/bin/env python3
"""
SABER Client CLI Entry Point

Entry point that delegates to the Click CLI for all functionality.
"""

from .cli import cli


def main() -> None:
    """Main entry point that delegates to Click CLI."""
    cli()


if __name__ == "__main__":
    main()
