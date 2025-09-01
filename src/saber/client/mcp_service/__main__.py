"""
CLI module for MCP Service

Provides command-line interface for running and managing the MCP service.
"""

import asyncio
import logging
import sys
from typing import Any, Callable, Optional

import click


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the service."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
def cli() -> None:
    """SABER MCP Service CLI."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8002, help="Port to bind to")
@click.option("--saber-mcp-url", help="SABER MCP server URL")
@click.option("--timeout", default=30, help="HTTP timeout in seconds")
@click.option("--log-level", default="INFO", help="Log level")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@cli.command()
@click.option("--saber-mcp-url", default="http://localhost:8001", help="SABER MCP server URL")
@click.option("--timeout", default=5.0, help="Health check timeout")
async def health_check(saber_mcp_url: str, timeout: float) -> None:
    """Check health of MCP service and SABER server."""
    import aiohttp

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Check MCP service health
            async with session.get("http://localhost:8002/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    click.echo("✅ MCP Service: Healthy")
                    click.echo(f"   Active sessions: {health_data.get('active_sessions', 0)}")
                    click.echo(f"   Total sessions: {health_data.get('total_sessions', 0)}")

                    if health_data.get("saber_server_healthy"):
                        click.echo("✅ SABER Server: Healthy")
                    else:
                        click.echo("❌ SABER Server: Unhealthy")
                        sys.exit(1)
                else:
                    click.echo(f"❌ MCP Service: HTTP {response.status}")
                    sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Health check failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument("agent_id")
@click.argument("saber_session_id")
@click.option("--task-id", help="Optional task ID")
@click.option("--service-url", default="http://localhost:8002", help="MCP service URL")
async def register_session(agent_id: str, saber_session_id: str, task_id: Optional[str], service_url: str) -> None:
    """Register an agent session with the MCP service."""
    import aiohttp

    request_data = {"agent_id": agent_id, "saber_session_id": saber_session_id}

    if task_id:
        request_data["task_id"] = task_id

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{service_url}/admin/sessions", json=request_data) as response:

                if response.status == 200:
                    result = await response.json()
                    click.echo("✅ Session registered successfully")
                    click.echo(f"   Agent ID: {result['agent_id']}")
                    click.echo(f"   SABER Session: {result['saber_session_id']}")
                else:
                    error_text = await response.text()
                    click.echo(f"❌ Registration failed: HTTP {response.status}")
                    click.echo(f"   Error: {error_text}")
                    sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Registration failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--service-url", default="http://localhost:8002", help="MCP service URL")
async def list_sessions(service_url: str) -> None:
    """List all active sessions."""
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{service_url}/admin/sessions") as response:

                if response.status == 200:
                    result = await response.json()
                    sessions = result.get("sessions", {})

                    if not sessions:
                        click.echo("No active sessions")
                        return

                    click.echo(f"Active sessions ({len(sessions)}):")
                    for agent_id, session_info in sessions.items():
                        click.echo(f"  • {agent_id}")
                        click.echo(f"    SABER Session: {session_info['saber_session_id']}")
                        if session_info.get("task_id"):
                            click.echo(f"    Task: {session_info['task_id']}")
                        click.echo(f"    Last Activity: {session_info['last_activity']}")
                else:
                    error_text = await response.text()
                    click.echo(f"❌ Failed to list sessions: HTTP {response.status}")
                    click.echo(f"   Error: {error_text}")
                    sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Failed to list sessions: {e}")
        sys.exit(1)


# Async command support
def async_command(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to support async click commands."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))

    return wrapper


# Apply async decorator to async commands
health_check = async_command(health_check)
register_session = async_command(register_session)
list_sessions = async_command(list_sessions)


if __name__ == "__main__":
    cli()
