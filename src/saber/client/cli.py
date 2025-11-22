#!/usr/bin/env python3
"""
SABER Client CLI

Command-line interface for SABER client operations including inspect-ai log analysis.
"""

import json
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

from ..logging_config import LoggingConfig, init_logging


def load_environment_file(env_file_path: Optional[Path], verbose: bool = False) -> None:
    """
    Load environment variables from specified .env file.

    Args:
        env_file_path: Path to .env file, or None to skip loading
        verbose: Whether to print verbose loading information

    Raises:
        ClientConfigError: If .env file is specified but cannot be loaded
    """
    if not env_file_path:
        if verbose:
            click.echo("🔐 No environment file specified, using existing environment variables")
        return

    if not env_file_path.exists():
        raise ClientConfigError(f"Environment file not found: {env_file_path}")

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file_path, override=True, verbose=verbose)
        if verbose:
            click.echo(f"🔐 Loaded environment from {env_file_path}")
    except ImportError:
        raise ClientConfigError("python-dotenv is required for .env file loading. Install with: uv add python-dotenv")
    except Exception as e:
        raise ClientConfigError(f"Failed to load environment file {env_file_path}: {e}")


class ClientConfigError(Exception):
    """Exception raised for client configuration errors."""

    pass


def load_domain_manifest(domains_root: Path, domain: str) -> Dict[str, Any]:
    """Load and validate domain manifest for client operations."""
    domain_path = domains_root / domain
    manifest_path = domain_path / "domain.yaml"

    if not domain_path.exists():
        raise ClientConfigError(f"Domain directory does not exist: {domain_path}")

    if not manifest_path.exists():
        raise ClientConfigError(f"Domain manifest not found: {manifest_path}")

    try:
        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ClientConfigError(f"Invalid YAML in domain manifest {manifest_path}: {e}")

    if not isinstance(manifest, dict):
        raise ClientConfigError(f"Domain manifest must be a YAML object: {manifest_path}")

    # Basic validation
    required_fields = ["schemaVersion", "domain"]
    missing_fields = [field for field in required_fields if field not in manifest]
    if missing_fields:
        raise ClientConfigError(f"Domain manifest missing required fields: {missing_fields}")

    return manifest


def load_client_config(config_path: Path) -> Dict[str, Any]:
    """Load client configuration file."""
    if not config_path.exists():
        raise ClientConfigError(f"Client configuration not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ClientConfigError(f"Invalid YAML in client config {config_path}: {e}")

    if not isinstance(config, dict):
        raise ClientConfigError(f"Client config must be a YAML object: {config_path}")

    return config


def validate_client_config_against_manifest(
    client_config: Dict[str, Any], manifest: Dict[str, Any], config_path: Path, manifest_path: Path
) -> Dict[str, Any]:
    """Validate client configuration against domain manifest capabilities."""
    validation_result: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "capabilities_checked": [],
        "config_summary": {},
    }

    # Get manifest capabilities
    manifest_capabilities = manifest.get("capabilities", {})

    # Validate networking capabilities
    if "networking" in client_config:
        networking_config = client_config["networking"]
        networking_caps = manifest_capabilities.get("networking", {})

        validation_result["capabilities_checked"].append("networking")

        # Check internet access
        if networking_config.get("internet_access", False):
            if not networking_caps.get("internet_access", False):
                validation_result["errors"].append(
                    "Client requests internet access but domain manifest does not allow it"
                )
                validation_result["valid"] = False

    # Validate execution capabilities
    if "execution" in client_config:
        execution_config = client_config["execution"]
        execution_caps = manifest_capabilities.get("execution", {})

        validation_result["capabilities_checked"].append("execution")

        # Check required tools
        required_tools = execution_config.get("required_tools", [])
        available_tools = execution_caps.get("tools", [])

        missing_tools = [tool for tool in required_tools if tool not in available_tools]
        if missing_tools:
            validation_result["errors"].append(f"Client requires tools not available in domain: {missing_tools}")
            validation_result["valid"] = False

    # Build configuration summary
    validation_result["config_summary"] = {
        "domain": manifest.get("domain", {}).get("slug"),
        "schema_version": manifest.get("schemaVersion"),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "capabilities_validated": validation_result["capabilities_checked"],
    }

    return validation_result


@click.group()  # type: ignore[misc]
@click.version_option()  # type: ignore[misc]
def cli() -> None:
    """SABER Client CLI - Tools for SABER evaluation and log analysis."""
    pass


@cli.command("validate-config")  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to client configuration file (e.g., domains/cybench/client/saber.yaml)",
)
@click.option(  # type: ignore[misc]
    "--domain",
    type=str,
    help="Domain name (if not provided, will be inferred from config path)",
)
@click.option(  # type: ignore[misc]
    "--domains-root",
    type=click.Path(exists=True, path_type=Path),
    help="Root directory containing domain definitions (default: infer from config path)",
)
@click.option(  # type: ignore[misc]
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def validate_config_command(
    config: Path,
    domain: Optional[str],
    domains_root: Optional[Path],
    verbose: bool,
) -> None:
    """Validate client configuration against domain manifest.

    This command validates that the client configuration is compatible with
    the domain's capabilities as defined in the domain manifest.

    Examples:
      saber.client validate-config --config domains/cybench/client/saber.yaml
      saber.client validate-config --config ./client.yaml --domain cybench --domains-root ./domains
    """
    try:
        # Infer domain and domains-root from config path if not provided
        if not domain or not domains_root:
            # Try to infer from path like: domains/cybench/client/saber.yaml
            config_parts = config.parts

            if not domains_root:
                # Look for 'domains' in the path
                if "domains" in config_parts:
                    domains_idx = config_parts.index("domains")
                    domains_root = Path(*config_parts[: domains_idx + 1])
                else:
                    raise ClientConfigError("Cannot infer domains-root from config path. Please specify --domains-root")

            if not domain:
                # Look for domain name between 'domains' and 'client'
                if "domains" in config_parts and "client" in config_parts:
                    domains_idx = config_parts.index("domains")
                    client_idx = config_parts.index("client")
                    if client_idx > domains_idx + 1:
                        domain = config_parts[domains_idx + 1]
                    else:
                        raise ClientConfigError("Cannot infer domain from config path. Please specify --domain")
                else:
                    raise ClientConfigError("Cannot infer domain from config path. Please specify --domain")

        if verbose:
            click.echo("🔍 Validating configuration...")
            click.echo(f"   Config: {config}")
            click.echo(f"   Domain: {domain}")
            click.echo(f"   Domains Root: {domains_root}")
            click.echo()

        # Load domain manifest
        try:
            manifest = load_domain_manifest(domains_root, domain)
            if verbose:
                domain_info = manifest.get("domain", {})
                click.echo(f"✅ Loaded domain manifest: {domain_info.get('name', domain)}")
                click.echo(f"   Schema Version: {manifest.get('schemaVersion')}")
                click.echo()
        except ClientConfigError as e:
            click.echo(f"❌ Failed to load domain manifest: {e}", err=True)
            sys.exit(1)

        # Load client configuration
        try:
            client_config = load_client_config(config)
            if verbose:
                click.echo("✅ Loaded client configuration")
                click.echo(f"   Configuration keys: {list(client_config.keys())}")
                click.echo()
        except ClientConfigError as e:
            click.echo(f"❌ Failed to load client configuration: {e}", err=True)
            sys.exit(1)

        # Validate configuration against manifest
        manifest_path = domains_root / domain / "domain.yaml"
        validation_result = validate_client_config_against_manifest(client_config, manifest, config, manifest_path)

        # Display results
        if validation_result["valid"]:
            click.echo("✅ Configuration validation passed!")
        else:
            click.echo("❌ Configuration validation failed!")

        click.echo()
        click.echo("📋 Validation Summary:")
        summary = validation_result["config_summary"]
        click.echo(f"   Domain: {summary['domain']}")
        click.echo(f"   Schema Version: {summary['schema_version']}")
        click.echo(f"   Config Path: {summary['config_path']}")
        click.echo(f"   Manifest Path: {summary['manifest_path']}")
        click.echo(f"   Capabilities Checked: {', '.join(summary['capabilities_validated']) or 'none'}")

        if validation_result["errors"]:
            click.echo()
            click.echo("❌ Validation Errors:")
            for error in validation_result["errors"]:
                click.echo(f"   • {error}")

        if validation_result["warnings"]:
            click.echo()
            click.echo("⚠️  Validation Warnings:")
            for warning in validation_result["warnings"]:
                click.echo(f"   • {warning}")

        if verbose and not validation_result["errors"] and not validation_result["warnings"]:
            click.echo()
            click.echo("ℹ️  No validation issues found. Configuration is compatible with domain capabilities.")

        # Exit with appropriate code
        sys.exit(0 if validation_result["valid"] else 1)

    except ClientConfigError as e:
        click.echo(f"❌ Configuration Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.group()  # type: ignore[misc]
def inspect() -> None:
    """Inspect-AI integration commands."""
    pass


@inspect.command("view")  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--log-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing inspect-ai evaluation logs",
)
@click.option(  # type: ignore[misc]
    "--host",
    default="127.0.0.1",
    help="Host to bind the web server to (default: 127.0.0.1)",
)
@click.option(  # type: ignore[misc]
    "--port",
    type=int,
    default=7575,
    help="Port to bind the web server to (default: 7575)",
)
@click.option(  # type: ignore[misc]
    "--no-recursive",
    is_flag=True,
    help="Do not recursively scan subdirectories for logs",
)
@click.option(  # type: ignore[misc]
    "--no-browser",
    is_flag=True,
    help="Do not automatically open browser",
)
def view_command(
    log_dir: Path,
    host: str,
    port: int,
    no_recursive: bool,
    no_browser: bool,
) -> None:
    """Launch Inspect AI log viewer web interface.

    This command launches Inspect AI's built-in web UI for viewing evaluation logs.
    The web interface provides rich visualization of evaluation results, sample details,
    model conversations, and performance metrics.

    The viewer will be available at http://{host}:{port}
    """
    try:
        from inspect_ai._view.view import view
    except ImportError as e:
        click.echo(
            f"Error: Failed to import inspect_ai view components: {e}\n"
            "Please ensure inspect-ai is properly installed.",
            err=True,
        )
        sys.exit(1)

    # Validate log directory
    if not log_dir.exists():
        click.echo(f"Error: Log directory '{log_dir}' does not exist.", err=True)
        sys.exit(1)

    if not log_dir.is_dir():
        click.echo(f"Error: '{log_dir}' is not a directory.", err=True)
        sys.exit(1)

    # Check if directory contains any log files (quick scan)
    log_files = list(log_dir.rglob("*.eval")) + list(log_dir.rglob("*.json"))
    if not log_files:
        click.echo(
            f"Warning: No log files (*.eval or *.json) found in '{log_dir}'. "
            f"The viewer will start anyway, but may show an empty directory.",
            err=True,
        )

    # Initialize inspect_ai environment properly (following exact CLI pattern)
    try:
        # Import and set up exactly like the real inspect_ai CLI does
        from inspect_ai._cli.common import process_common_options

        # Set up the common options that inspect_ai CLI uses
        common_options = {
            "log_dir": str(log_dir.absolute()),
            "log_level": "info",
            "display": "full",
            "no_ansi": False,
            "traceback_locals": False,
            "env": [],
            "debug": False,
            "debug_port": 5678,
            "debug_errors": False,
        }

        # Process common options (this sets up logging and display properly)
        process_common_options(common_options)

    except Exception as e:
        click.echo(f"Warning: Failed to initialize inspect_ai environment: {e}", err=True)

    click.echo(click.style("🚀 Starting SABER Inspect AI Log Viewer", fg="green", bold=True))
    click.echo(f"📁 Log Directory: {log_dir.absolute()}")
    click.echo(f"🌐 Server: http://{host}:{port}")
    click.echo(f"🔄 Recursive: {not no_recursive}")

    # Open browser unless explicitly disabled
    if not no_browser:
        import threading
        import time
        import webbrowser

        def open_browser() -> None:
            """Open browser after a short delay to ensure server is ready."""
            time.sleep(2)  # Give server time to start
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=open_browser, daemon=True).start()
        click.echo("🔗 Browser will open automatically in 2 seconds...")

    click.echo("📝 Press Ctrl+C to stop the server")
    click.echo("=" * 60)

    # Clear problematic environment variables and set correct ones for frontend
    # This prevents URL construction errors in the frontend JavaScript
    os.environ.pop("__VIEW_SERVER_API_URL__", None)

    # Set environment variable to empty string to avoid URL construction issues
    # The frontend will default to relative paths which work correctly
    os.environ["VIEW_SERVER_API_URL"] = ""

    # Patch the frontend JavaScript to fix isApiCrossOrigin function
    try:
        # Patch the compiled JavaScript file to handle URL construction errors
        js_path = (
            Path(sys.executable).parent.parent
            / "lib"
            / "python3.11"
            / "site-packages"
            / "inspect_ai"
            / "_view"
            / "www"
            / "dist"
            / "assets"
            / "index.js"
        )
        if js_path.exists():
            # Read current JavaScript
            js_content = js_path.read_text()

            # Find and replace the isApiCrossOrigin function to handle the error properly
            original_function = """function isApiCrossOrigin() {
      try {
        console.log("API_BASE_URL:", API_BASE_URL);
        return Boolean(
          API_BASE_URL && new URL(API_BASE_URL).origin !== window.location.origin
        );
      } catch (e) {
        console.log("URL construction failed for API_BASE_URL:", API_BASE_URL, "Error:", e);
        return false;
      }
    }"""

            fixed_function = """function isApiCrossOrigin() {
      try {
        console.log("API_BASE_URL:", API_BASE_URL);
        // Handle relative URLs by returning false (same origin)
        if (!API_BASE_URL || API_BASE_URL.startsWith('/')) {
          return false;
        }
        return Boolean(
          API_BASE_URL && new URL(API_BASE_URL).origin !== window.location.origin
        );
      } catch (e) {
        console.log("URL construction failed for API_BASE_URL:", API_BASE_URL, "Error:", e);
        return false;
      }
    }"""

            if original_function in js_content:
                patched_js = js_content.replace(original_function, fixed_function)
                js_path.write_text(patched_js)
                click.echo("✅ Patched isApiCrossOrigin function to handle relative URLs")
            else:
                # Fallback: look for the function signature and patch it
                import re

                pattern = r"function isApiCrossOrigin\(\)\s*\{[^}]*new URL\(API_BASE_URL\)[^}]*\}"
                if re.search(pattern, js_content, re.DOTALL):
                    # Simple replacement: add check for relative URLs
                    simple_fix = js_content.replace(
                        "API_BASE_URL && new URL(API_BASE_URL)",
                        'API_BASE_URL && !API_BASE_URL.startsWith("/") && new URL(API_BASE_URL)',
                    )
                    js_path.write_text(simple_fix)
                    click.echo("✅ Patched API_BASE_URL check to handle relative paths")
    except Exception as e:
        click.echo(f"⚠️  Warning: Failed to patch frontend JavaScript: {e}", err=True)

    try:
        # Call view() with EXACT same parameters as the real CLI
        view(
            log_dir=str(log_dir.absolute()),
            recursive=not no_recursive,
            host=host,
            port=port,
            authorization=None,
            log_level="info",
        )
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped by user")
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e):
            click.echo(
                f"Error: Port {port} is already in use. "
                f"Try a different port with --port, or stop the existing service.",
                err=True,
            )
        else:
            click.echo(f"Error: Failed to start server: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: Unexpected failure starting viewer: {e}", err=True)
        raise  # Re-raise for debugging as per best practices


@inspect.command("eval")  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--log-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to inspect-ai .eval log file to analyze",
)
@click.option(  # type: ignore[misc]
    "--format",
    type=click.Choice(["json", "summary", "samples", "full"]),
    default="summary",
    help="Output format: json (raw), summary (header), samples (sample data), full (everything)",
)
@click.option(  # type: ignore[misc]
    "--pretty/--no-pretty",
    default=True,
    help="Pretty print JSON output",
)
@click.option(  # type: ignore[misc]
    "--sample-id",
    type=str,
    help="Show specific sample by ID (only with --format samples)",
)
@click.option(  # type: ignore[misc]
    "--max-samples",
    type=int,
    default=10,
    help="Maximum number of samples to show (only with --format samples/full)",
)
def eval_command(
    log_file: Path,
    format: str,
    pretty: bool,
    sample_id: Optional[str],
    max_samples: int,
) -> None:
    """Analyze and dump inspect-ai evaluation log files."""
    try:
        from inspect_ai.log import read_eval_log, read_eval_log_sample
    except ImportError:
        click.echo("Error: inspect_ai package not found. Please install inspect-ai.", err=True)
        sys.exit(1)

    try:
        if format == "json":
            # Read full log and dump as JSON
            log = read_eval_log(str(log_file))
            dump_as_json(log, pretty)

        elif format == "summary":
            # Read header only and show summary
            log = read_eval_log(str(log_file), header_only=True)
            dump_summary(log)

        elif format == "samples":
            if sample_id:
                # Read specific sample
                sample = read_eval_log_sample(str(log_file), sample_id)
                dump_as_json(sample, pretty)
            else:
                # Read full log and show samples
                log = read_eval_log(str(log_file))
                dump_samples(log, max_samples, pretty)

        elif format == "full":
            # Read full log and show everything
            log = read_eval_log(str(log_file))
            dump_full(log, max_samples, pretty)

    except Exception as e:
        click.echo(f"Error reading log file: {e}", err=True)
        sys.exit(1)


def dump_as_json(obj: Any, pretty: bool) -> None:
    """Dump object as JSON."""
    if hasattr(obj, "model_dump"):
        # Pydantic model
        data = obj.model_dump()
    elif hasattr(obj, "dict"):
        # Pydantic model (older versions)
        data = obj.dict()
    else:
        data = obj

    if pretty:
        json_str = json.dumps(data, indent=2, default=str)
    else:
        json_str = json.dumps(data, default=str)

    click.echo(json_str)


def dump_summary(log: Any) -> None:
    """Dump evaluation summary."""
    click.echo(click.style("🔍 SABER Evaluation Log Summary", fg="blue", bold=True))
    click.echo("=" * 50)

    # Basic info
    click.echo(f"📋 Task: {log.eval.task}")
    click.echo(f"🤖 Model: {log.eval.model}")
    click.echo(f"📅 Created: {log.eval.created}")
    click.echo(f"⚡ Status: {log.status}")

    if hasattr(log.eval, "task_version") and log.eval.task_version:
        click.echo(f"🔢 Task Version: {log.eval.task_version}")

    # Dataset info
    if log.eval.dataset:
        dataset_samples = len(log.eval.dataset.sample_ids) if log.eval.dataset.sample_ids else "Unknown"
        click.echo(f"📊 Dataset Samples: {dataset_samples}")
        if log.eval.dataset.name:
            click.echo(f"📁 Dataset Name: {log.eval.dataset.name}")

    # Results
    if log.results and log.results.scores:
        click.echo("\n📈 Scores:")
        try:
            if isinstance(log.results.scores, list):
                # Handle list of EvalScore objects
                for score_obj in log.results.scores:
                    if hasattr(score_obj, "metrics") and score_obj.metrics:
                        for metric_name, metric in score_obj.metrics.items():
                            if hasattr(metric, "value"):
                                click.echo(f"   • {score_obj.name}/{metric_name}: {metric.value}")
                            else:
                                click.echo(f"   • {score_obj.name}/{metric_name}: {metric}")
                    elif hasattr(score_obj, "name"):
                        click.echo(f"   • {score_obj.name}: {getattr(score_obj, 'value', 'N/A')}")
            elif isinstance(log.results.scores, dict):
                for scorer_name, metrics in log.results.scores.items():
                    if isinstance(metrics, dict):
                        for metric_name, score in metrics.items():
                            if isinstance(score, dict) and "value" in score:
                                value = score["value"]
                            else:
                                value = score
                            click.echo(f"   • {scorer_name}/{metric_name}: {value}")
                    else:
                        click.echo(f"   • {scorer_name}: {metrics}")
            else:
                click.echo(f"   • Scores: {str(log.results.scores)[:200]}...")
        except Exception as e:
            click.echo(f"   • Error parsing scores: {e}")

    # Stats
    if log.stats:
        click.echo("\n⏱️  Stats:")
        try:
            if hasattr(log.stats, "started_at") and hasattr(log.stats, "completed_at"):
                if log.stats.started_at and log.stats.completed_at:
                    # Handle different datetime formats
                    try:
                        from datetime import datetime

                        if isinstance(log.stats.started_at, str):
                            started = datetime.fromisoformat(log.stats.started_at.replace("Z", "+00:00"))
                        else:
                            started = log.stats.started_at

                        if isinstance(log.stats.completed_at, str):
                            completed = datetime.fromisoformat(log.stats.completed_at.replace("Z", "+00:00"))
                        else:
                            completed = log.stats.completed_at

                        duration = completed - started
                        click.echo(f"   • Duration: {duration}")
                    except Exception:
                        click.echo(f"   • Started: {log.stats.started_at}")
                        click.echo(f"   • Completed: {log.stats.completed_at}")

            if hasattr(log.stats, "model_usage") and log.stats.model_usage:
                for provider, usage in log.stats.model_usage.items():
                    if hasattr(usage, "total_tokens"):
                        click.echo(f"   • {provider} tokens: {usage.total_tokens}")
                    elif isinstance(usage, dict) and "total_tokens" in usage:
                        click.echo(f"   • {provider} tokens: {usage['total_tokens']}")
        except Exception as e:
            click.echo(f"   • Error parsing stats: {e}")

    # Error info
    if log.error:
        click.echo(f"\n❌ Error: {log.error.message}")
        if hasattr(log.error, "traceback") and log.error.traceback:
            click.echo(f"   Traceback: {log.error.traceback[:200]}...")


def dump_samples(log: Any, max_samples: int, pretty: bool) -> None:
    """Dump sample information."""
    if not log.samples:
        click.echo("No samples found in log file.")
        return

    if pretty:
        from rich.console import Console

        console = Console()
        total_samples = len(log.samples)
        showing_samples = min(max_samples, total_samples)
        console.print(f"\n[bold cyan]📝 Samples ({total_samples} total, showing {showing_samples})[/bold cyan]")
        console.print("=" * 50)

        for i, sample in enumerate(log.samples[:max_samples]):
            console.print(f"\n[bold yellow]🔸 Sample {i + 1} (ID: {sample.id})[/bold yellow]")

            # Input
            input_text = str(sample.input)
            if len(input_text) > 100:
                console.print(f"   [dim]Input:[/dim] {input_text[:100]}...")
            else:
                console.print(f"   [dim]Input:[/dim] {input_text}")

            # Target
            console.print(f"   [dim]Target:[/dim] {sample.target}")

            # Agent Conversation
            if hasattr(sample, "messages") and sample.messages:
                console.print(f"\n[bold cyan]🤖 Agent Conversation ({len(sample.messages)} messages)[/bold cyan]")

                for msg_idx, message in enumerate(sample.messages):
                    # Skip system messages for brevity
                    if hasattr(message, "role") and message.role == "system":
                        continue

                    role = getattr(message, "role", "unknown")
                    content = getattr(message, "content", "")

                    # Format role with color
                    if role == "user":
                        role_color = "[blue]👤 User[/blue]"
                    elif role == "assistant":
                        role_color = "[green]🤖 Assistant[/green]"
                    elif role == "tool":
                        role_color = "[yellow]🔧 Tool[/yellow]"
                    else:
                        role_color = f"[dim]{role}[/dim]"

                    console.print(f"\n{role_color}:")

                    # Show content (thinking)
                    if content:
                        # Handle content that might be a list or string
                        content_str = ""
                        if isinstance(content, list):
                            # If content is a list, extract text from items
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    content_str += item["text"]
                                else:
                                    content_str += str(item)
                        else:
                            content_str = str(content)

                        if content_str.strip():
                            # Extract "Thought:" sections for assistant messages
                            if role == "assistant" and "Thought:" in content_str:
                                lines = content_str.split("\n")
                                for line in lines:
                                    if line.strip().startswith("Thought:"):
                                        thought = line.replace("Thought:", "").strip()
                                        console.print(f"  [italic dim]💭 {thought}[/italic dim]")
                                    elif line.strip().startswith("Action:"):
                                        action = line.replace("Action:", "").strip()
                                        console.print(f"  [bold dim]⚡ {action}[/bold dim]")
                            else:
                                # Show truncated content for other messages
                                content_preview = content_str[:150] if len(content_str) > 150 else content_str
                                console.print(f"  {content_preview}{'...' if len(content_str) > 150 else ''}")

                    # Show tool calls
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            func_name = getattr(tool_call, "function", "unknown")
                            if hasattr(tool_call, "arguments"):
                                args = getattr(tool_call, "arguments", {})
                                if isinstance(args, dict) and "command" in args:
                                    console.print(f"  [cyan]🛠️  {func_name}[/cyan]: [code]{args['command']}[/code]")
                                else:
                                    console.print(f"  [cyan]🛠️  {func_name}[/cyan]: {str(args)[:100]}...")
                            else:
                                console.print(f"  [cyan]�️  {func_name}[/cyan]")

                    # Show tool results for tool role messages
                    if role == "tool" and content:
                        try:
                            # Handle content that might be a list or string
                            content_str = ""
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        content_str += item["text"]
                                    else:
                                        content_str += str(item)
                            else:
                                content_str = str(content)

                            # Try to parse tool output
                            if content_str.startswith("{") and "stdout" in content_str:
                                import json

                                tool_result = json.loads(content_str)
                                if "stdout" in tool_result:
                                    stdout = tool_result["stdout"]
                                    if stdout:
                                        stdout_preview = stdout[:100]
                                        stdout_suffix = "..." if len(stdout) > 100 else ""
                                        console.print(f"  [green]✅ Output:[/green] {stdout_preview}{stdout_suffix}")
                                if "stderr" in tool_result and tool_result["stderr"]:
                                    console.print(f"  [red]❌ Error:[/red] {tool_result['stderr'][:100]}")
                            else:
                                content_preview = content_str[:100]
                                content_suffix = "..." if len(content_str) > 100 else ""
                                console.print(f"  [dim]Result:[/dim] {content_preview}{content_suffix}")
                        except Exception:
                            fallback_content = content_str[:100] if "content_str" in locals() else str(content)[:100]
                            fallback_suffix = "..." if len(str(content)) > 100 else ""
                            console.print(f"  [dim]Result:[/dim] {fallback_content}{fallback_suffix}")

            # Scores
            if sample.scores:
                console.print("\n[bold magenta]📊 Scores[/bold magenta]")
                try:
                    if isinstance(sample.scores, list):
                        # Handle list of score objects
                        for score_obj in sample.scores:
                            if hasattr(score_obj, "metrics") and score_obj.metrics:
                                for metric_name, metric in score_obj.metrics.items():
                                    if hasattr(metric, "value"):
                                        console.print(f"   • {score_obj.name}/{metric_name}: {metric.value}")
                                    else:
                                        console.print(f"   • {score_obj.name}/{metric_name}: {metric}")
                    elif isinstance(sample.scores, dict):
                        for scorer_name, metrics in sample.scores.items():
                            if isinstance(metrics, dict):
                                for metric_name, score in metrics.items():
                                    if isinstance(score, dict) and "value" in score:
                                        value = score["value"]
                                    else:
                                        value = score
                                    console.print(f"   • {scorer_name}/{metric_name}: {value}")
                            else:
                                console.print(f"   • {scorer_name}: {metrics}")
                    else:
                        console.print(f"   • Scores: {str(sample.scores)[:100]}...")
                except Exception as e:
                    console.print(f"   [red]• Error parsing scores: {e}[/red]")

            # Error
            if sample.error:
                console.print(f"\n[red]❌ Error: {sample.error.message}[/red]")

    else:
        # Non-pretty output
        click.echo(
            click.style(
                f"�📝 Samples ({len(log.samples)} total, showing {min(max_samples, len(log.samples))})",
                fg="green",
                bold=True,
            )
        )
        click.echo("=" * 50)

        for i, sample in enumerate(log.samples[:max_samples]):
            click.echo(f"\n🔸 Sample {i + 1} (ID: {sample.id})")
            click.echo(
                f"   Input: {str(sample.input)[:100]}..."
                if len(str(sample.input)) > 100
                else f"   Input: {sample.input}"
            )
            click.echo(f"   Target: {sample.target}")

            # Show agent conversation in non-pretty format too
            if hasattr(sample, "messages") and sample.messages:
                click.echo(f"\n   Agent Conversation ({len(sample.messages)} messages):")
                for msg_idx, message in enumerate(sample.messages):
                    role = getattr(message, "role", "unknown")
                    content = getattr(message, "content", "")

                    if role == "system":
                        continue

                    click.echo(f"\n   {msg_idx+1}. {role.upper()}:")
                    if content:
                        click.echo(f"      {content[:200]}{'...' if len(content) > 200 else ''}")

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            func_name = getattr(tool_call, "function", "unknown")
                            args = getattr(tool_call, "arguments", {})
                            click.echo(f"      TOOL: {func_name} -> {args}")

            if sample.scores:
                click.echo("   Scores:")
                try:
                    if isinstance(sample.scores, list):
                        # Handle list of score objects
                        for score_obj in sample.scores:
                            if hasattr(score_obj, "metrics") and score_obj.metrics:
                                for metric_name, metric in score_obj.metrics.items():
                                    if hasattr(metric, "value"):
                                        click.echo(f"      • {score_obj.name}/{metric_name}: {metric.value}")
                                    else:
                                        click.echo(f"      • {score_obj.name}/{metric_name}: {metric}")
                    elif isinstance(sample.scores, dict):
                        for scorer_name, metrics in sample.scores.items():
                            if isinstance(metrics, dict):
                                for metric_name, score in metrics.items():
                                    if isinstance(score, dict) and "value" in score:
                                        value = score["value"]
                                    else:
                                        value = score
                                    click.echo(f"      • {scorer_name}/{metric_name}: {value}")
                            else:
                                click.echo(f"      • {scorer_name}: {metrics}")
                    else:
                        click.echo(f"      • Scores: {str(sample.scores)[:100]}...")
                except Exception as e:
                    click.echo(f"      • Error parsing scores: {e}")

            if sample.error:
                click.echo(f"   ❌ Error: {sample.error.message}")


def dump_full(log: Any, max_samples: int, pretty: bool) -> None:
    """Dump full log information."""
    # Show summary first
    dump_summary(log)

    # Then show samples
    if log.samples:
        click.echo("\n" + "=" * 50)
        dump_samples(log, max_samples, pretty)

    # Show additional details
    if log.plan:
        click.echo("\n🛠️  Plan:")
        click.echo(f"   Solvers: {len(log.plan.steps) if log.plan.steps else 0}")
        if log.plan.config:
            click.echo(f"   Config: {dict(log.plan.config)}")


def setup_client_logging(
    *, verbose: bool, enable_file: bool, log_dir_override: Optional[Path] = None, domain: Optional[str] = None
) -> LoggingConfig:
    """Setup SABER client logging with timestamped log files.

    Args:
        verbose: Enable verbose/debug logging
        enable_file: Enable file logging
        log_dir_override: Override default log directory
        domain: Domain name for organizing logs by domain (creates logs/{domain}/client-logs/)
    """

    # Create timestamped filename similar to server
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"saber_client_{timestamp}.log"

    # Get base configuration
    base_config = LoggingConfig.from_env()
    target_level = logging.DEBUG if verbose else base_config.level

    # Set up log directory
    if log_dir_override:
        log_directory = log_dir_override.expanduser().resolve()
    else:
        log_directory = base_config.log_dir

    # Create domain-specific directory structure if domain is provided
    if domain:
        # Organize by domain: logs/{domain}/client-logs/
        domain_logs_dir = log_directory / domain
        client_logs_dir = domain_logs_dir / "client-logs"
    else:
        # Fallback to existing structure: logs/client-logs/
        client_logs_dir = log_directory / "client-logs"

    client_logs_dir.mkdir(parents=True, exist_ok=True)

    # Create configuration with timestamped filename and custom directory
    config = replace(
        base_config,
        level=target_level,
        console=False,  # Disable console logging - file only
        enable_file=enable_file and base_config.enable_file,
        log_dir=client_logs_dir,
        file_name=timestamped_filename,
    )

    return init_logging(config, force=True)


@cli.command("run")  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--config",
    type=click.Path(path_type=Path),
    help="Path to SABER configuration YAML file (default: auto-detect saber.yaml in current directory)",
)
@click.option(  # type: ignore[misc]
    "--env-file",
    type=click.Path(path_type=Path),
    help="Path to .env file for environment variables (default: auto-detect .env in current directory)",
)
@click.option(  # type: ignore[misc]
    "--domain",
    type=str,
    help="Domain name for validation and logging organization (optional)",
)
@click.option(  # type: ignore[misc]
    "--domains-root",
    type=click.Path(exists=True, path_type=Path),
    help="Root directory containing domain definitions (enables domain-aware validation)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")  # type: ignore[misc]
@click.option("--no-log-file", is_flag=True, help="Disable file logging (console only)")  # type: ignore[misc]
@click.option("--no-ui", is_flag=True, help="Disable TUI and use rich console output instead")  # type: ignore[misc]
@click.option(
    "--validate-config", is_flag=True, help="Validate configuration against domain manifest before running"
)  # type: ignore[misc]
# Direct configuration options (alternative to --config file)
@click.option("--rest-url", type=str, help="SABER server REST API URL")  # type: ignore[misc]
@click.option("--mcp-url", type=str, help="SABER server MCP URL")  # type: ignore[misc]
@click.option("--model", type=str, help="Model specification")  # type: ignore[misc]
@click.option("--agent-id", type=str, help="Agent ID")  # type: ignore[misc]
@click.option("--task-ids", type=str, help="Comma-separated task IDs (or '*' for all)")  # type: ignore[misc]
def run_command(
    config: Optional[Path],
    env_file: Optional[Path],
    domain: Optional[str],
    domains_root: Optional[Path],
    verbose: bool,
    no_log_file: bool,
    no_ui: bool,
    validate_config: bool,
    rest_url: Optional[str],
    mcp_url: Optional[str],
    model: Optional[str],
    agent_id: Optional[str],
    task_ids: Optional[str],
) -> None:
    """[DEPRECATED] Run SABER evaluation with inspect-ai integration.

    ⚠️  This command has been DEPRECATED in favor of 'inspect eval'.

    The 'saber.client run' command (and 'saber-domain test') have been deprecated.
    Please use 'inspect eval' for better performance and compatibility.

    See 'saber-domain test --help' for migration guidance.
    """
    click.echo(click.style("\n⚠️  DEPRECATION WARNING", fg="yellow", bold=True))
    click.echo(click.style("=" * 60, fg="yellow"))
    click.echo()
    click.echo("The 'saber.client run' command has been deprecated.")
    click.echo("This command was part of the old 'saber-domain test' pathway.")
    click.echo()
    click.echo("Please use 'inspect eval' instead:")
    click.echo()

    if domain:
        click.echo(click.style(f"  uv run inspect eval domains/{domain} --model <your-model>", fg="green"))
    else:
        click.echo(click.style("  uv run inspect eval domains/<domain_name> --model <your-model>", fg="green"))

    click.echo()
    click.echo("Examples:")
    click.echo("  --model openai/azure/gpt-4")
    click.echo("  --model anthropic/claude-3-opus")
    click.echo()
    click.echo("For more information:")
    click.echo(click.style("  saber-domain test --help", fg="cyan"))
    click.echo("  README.md - Getting started with inspect eval")
    click.echo()
    click.echo(click.style("=" * 60, fg="yellow"))
    click.echo()

    sys.exit(1)


def main() -> None:
    """Main entry point for SABER client CLI."""
    cli()


if __name__ == "__main__":
    main()
