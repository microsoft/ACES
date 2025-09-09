#!/usr/bin/env python3
"""
SABER Client CLI

Command-line interface for SABER client operations including inspect-ai log analysis.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click

from . import __main__ as main_module


@click.group()  # type: ignore[misc]
@click.version_option()  # type: ignore[misc]
def cli() -> None:
    """SABER Client CLI - Tools for SABER evaluation and log analysis."""
    pass


@cli.group()  # type: ignore[misc]
def inspect() -> None:
    """Inspect-AI integration commands."""
    pass


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


@cli.command("run")  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to SABER configuration YAML file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")  # type: ignore[misc]
@click.option("--no-log-file", is_flag=True, help="Disable file logging (console only)")  # type: ignore[misc]
def run_command(config: Optional[Path], verbose: bool, no_log_file: bool) -> None:
    """Run SABER evaluation (wrapper around main CLI)."""
    # Build args for main module
    args = []
    if config:
        args.extend(["--config", str(config)])
    if verbose:
        args.append("--verbose")
    if no_log_file:
        args.append("--no-log-file")

    # Override sys.argv and call main
    original_argv = sys.argv[:]
    try:
        sys.argv = ["saber-client"] + args
        main_module.main()
    finally:
        sys.argv = original_argv


def main() -> None:
    """Main entry point for SABER client CLI."""
    cli()


if __name__ == "__main__":
    main()
