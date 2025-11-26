"""SABER Domain Task Factory for Inspect AI.

This module provides a task factory pattern that enables SABER domains to be
evaluated as standalone Inspect AI tasks. The factory starts the SABER server
on demand, queries it for task definitions, and constructs a fully-populated
Task with dataset loaded from the server.

Key features:
- Server as single source of truth (no duplicate YAML parsing)
- Process-wide BlockingPortal for async operations in sync context
- Concurrent domain evaluation blocking via _active_domains registry
- Preflight checks to detect already-running instances
- Health check retry logic with backoff
- Task filtering with exact and glob pattern matching
- Robust cleanup on all failure paths
"""

import asyncio
import fnmatch
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
import anyio
from inspect_ai import Task
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ..client.api.rest_client import SABERRestClient
from ..client.config_loader import RoleConfigLoader
from ..client.models import RoleBasedConfig
from ..logging_config import LogCategory, get_saber_logger
from ..models import BenchmarkInfo
from ..models.constants import MetadataKeys
from .agents import AgentNotFoundError, SABERAgentRegistry
from .saber_dataset import create_saber_dataset
from .saber_scorer import saber_scorer
from .server import DomainContext, DomainController

logger = get_saber_logger(LogCategory.AGENT, __name__)


def _all_roles_have_models(role_config: RoleBasedConfig) -> bool:
    """Check if all roles in configuration have models defined.

    Args:
        role_config: Role-based configuration object

    Returns:
        True if all roles have models (or defaults provide a model), False otherwise
    """
    # Must have at least one role defined
    if not role_config.roles:
        return False

    # If defaults has a model, all roles will inherit it
    if role_config.defaults and role_config.defaults.model:
        return True

    # Check if all explicit roles have models
    for role_name, role_conf in role_config.roles.items():
        if not role_conf.model:
            return False

    return True


# Process-wide registry tracking active domains
# Key: domain_slug
# Value: Dict with controller, context, ownership, rest_url, mcp_url, etc.
_active_domains: Dict[str, Dict[str, Any]] = {}
_active_domains_lock = threading.Lock()

# Process-wide BlockingPortal for running async operations from sync context
# Store both the context manager and the portal
_portal_cm: Any = None
_portal: Optional[anyio.abc.BlockingPortal] = None
_portal_lock = threading.Lock()


def _get_or_create_portal() -> anyio.abc.BlockingPortal:
    """Get or create the process-wide BlockingPortal.

    This portal allows the synchronous task callable to invoke async
    operations without disturbing Inspect AI's event loop.

    Returns:
        Shared BlockingPortal instance
    """
    global _portal, _portal_cm

    with _portal_lock:
        if _portal is None:
            # start_blocking_portal returns a context manager
            _portal_cm = anyio.from_thread.start_blocking_portal("asyncio")
            _portal = _portal_cm.__enter__()
            logger.debug("Created process-wide BlockingPortal for SABER task factory")
        return _portal


def create_domain_task(
    domain_slug: str,
    domains_root: Path,
    compose_template_path: Optional[Path] = None,
    default_agent: str = "react",
) -> Callable[..., Task]:
    """Create a task factory callable for a SABER domain.

    Returns a callable that Inspect AI can invoke with CLI parameters
    (-T flags). When invoked, the callable starts the SABER server,
    waits for health checks, queries for tasks, and returns a fully-
    populated Task object.

    Usage:
        # In domains/cybench/__init__.py:
        from pathlib import Path
        from saber.inspect_ai import create_domain_task

        cybench = create_domain_task(
            "cybench",
            Path(__file__).parent.parent,
            default_agent="react"
        )

    Args:
        domain_slug: SABER domain slug (e.g., "cybench")
        domains_root: Path to domains directory (workspace root)
        compose_template_path: Optional custom compose template
        default_agent: Default agent implementation for this domain (default: "react")

    Returns:
        Callable that accepts Inspect AI task parameters and returns Task
    """

    def task_callable(
        rest_port: int = 8000,
        mcp_port: int = 8001,
        task_filter: Optional[str] = None,
        agent: Optional[str] = None,
        log_level: str = "INFO",
        build: bool = False,
        rebuild: Optional[str] = None,
        rebuild_all: bool = False,
        stop_saber_after: bool = False,
        max_concurrent_episodes: int = 16,
        run_preflight: bool = False,
        enable_debug_logging: bool = False,
        roles: Optional[Union[str, dict]] = None,
        roles_file: Optional[str] = None,
        **kwargs: Any,
    ) -> Task:
        """Task callable invoked by Inspect AI with CLI parameters.

        Args:
            rest_port: REST API port (default: 8000)
            mcp_port: MCP API port (default: 8001)
            task_filter: Optional task filter (exact match, glob pattern, or comma-separated)
                Examples:
                - "xss_0_flag_capture" (exact match)
                - "xss_*" (glob pattern)
                - "xss_*,sql_*" (multiple patterns with OR logic)
                - "xss_0_flag_capture,sql_*,cmd_*" (mix of exact and glob)
            agent: Agent implementation to use (default: domain-specific default)
            log_level: Logging level for domain services (default: "INFO")
            build: Build missing images before starting (default: False)
            rebuild: Remove and rebuild images matching this prefix (e.g., 'server')
            rebuild_all: Remove and rebuild all images (default: False)
            stop_saber_after: Stop SABER domain after task completes (default: False).
                If False, server stays running for faster re-runs.
            max_concurrent_episodes: Max concurrent episodes (default: 8, 0 = unlimited)
            run_preflight: Run preflight check on all compose environments before evaluation
                (default: False). If any environments fail health checks, evaluation aborts.
            enable_debug_logging: Enable detailed episode lifecycle debug logging
                (default: False). When enabled, creates detailed logs in logs/saber_episode_debug_*.log
            roles: Role-based configuration for orchestrated tasks. Can be:
                - Dict (inline): {'red': {'agent': 'react', 'model': 'gpt-4'}, ...}
                - String as JSON: '{"red": {"agent": "react"}}'
                Note: For file-based configs, use roles_file parameter instead.
            roles_file: Path to YAML/JSON role configuration file (relative or absolute).
                Examples:
                - 'configs/saber_dual_roles.yaml' (relative to current directory)
                - '/absolute/path/to/roles.yaml'
                This is the preferred method for file-based configs (simpler than roles parameter).
            **kwargs: Additional parameters passed through (may include role overrides like red_model=...)

        Returns:
            Inspect AI Task with fully-populated dataset

        Raises:
            PrerequisiteError: If server startup, health checks, or task loading fails
        """
        # Validate mutually exclusive build options
        build_options_count = sum([build, rebuild_all, bool(rebuild)])
        if build_options_count > 1:
            raise PrerequisiteError(
                "Cannot specify multiple build options together. Use one of:\n"
                "  -T build=true: Build missing images only\n"
                "  -T rebuild_all=true: Remove and rebuild all images\n"
                "  -T rebuild=<prefix>: Remove and rebuild specific images"
            )

        # Convert to orchestrator format (same as CLI)
        build_param = "" if build else None
        rebuild_param = "" if rebuild_all else rebuild

        # Use CLI agent override or domain default
        agent_to_use = agent if agent is not None else default_agent

        # Process role-based configuration if provided
        role_config_obj = None
        roles_source = roles_file or roles  # Prefer roles_file if both provided

        if roles_source is not None:
            try:
                # Extract role-specific overrides from kwargs (e.g., red_model=gpt-4)
                role_overrides: Dict[str, Dict[str, Any]] = {}
                for key, value in list(kwargs.items()):
                    if "_" in key:
                        parts = key.split("_", 1)
                        if len(parts) == 2:
                            # Potential role override like "red_model"
                            role_name, field_name = parts
                            if role_name not in role_overrides:
                                role_overrides[role_name] = {}
                            role_overrides[role_name][field_name] = value
                            # Remove from kwargs so it's not passed downstream
                            del kwargs[key]

                # Load and merge role configuration
                base_config = RoleConfigLoader.load(roles_source)
                role_config_obj = RoleConfigLoader.merge_with_overrides(base_config, role_overrides)

                logger.info(
                    "Loaded role-based configuration",
                    extra={
                        "domain": domain_slug,
                        "roles": list(role_config_obj.roles.keys()) if role_config_obj.roles else [],
                        "has_defaults": role_config_obj.defaults is not None,
                        "override_count": len(role_overrides),
                    },
                )
            except Exception as e:
                raise PrerequisiteError(f"Failed to load role configuration: {e}") from e

        # Get the portal and run async startup
        portal = _get_or_create_portal()

        try:
            return portal.call(
                _start_and_load_tasks,
                domain_slug,
                domains_root,
                rest_port,
                mcp_port,
                task_filter,
                agent_to_use,
                log_level,
                build_param,
                rebuild_param,
                compose_template_path,
                stop_saber_after,
                max_concurrent_episodes,
                run_preflight,
                enable_debug_logging,
                role_config_obj,
            )
        except Exception as e:
            # Ensure we have a clean error message
            if isinstance(e, PrerequisiteError):
                raise
            raise PrerequisiteError(f"Failed to create SABER task for domain '{domain_slug}': {e}") from e

    return task_callable


def _create_saber_solver(agent_name: str, agent_factory: Callable, role_config: Optional[Any] = None) -> Solver:
    """Create solver with SABER agent that extracts prompts from metadata.

    This solver:
    1. Extracts the three prompts from sample metadata
    2. Uses saber_tools() to get MCP client from sandbox
    3. Creates agent with prompts and tools (using provided factory)
    4. Executes the agent

    Args:
        agent_name: Name of the agent implementation
        agent_factory: Callable that creates the agent with prompts
        role_config: Optional role-based configuration for model selection

    Returns:
        Solver that runs SABER agent with dynamic prompts
    """

    @solver  # type: ignore[misc]
    def saber_agent_solver() -> Solver:
        """SABER agent solver with dynamic prompt injection."""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            """Execute SABER agent with prompts from metadata."""

            # DEBUG: Log solver invocation
            from ..debug_logging import EpisodeDebugLogger

            debug_logger = EpisodeDebugLogger("solver")
            debug_logger.info(
                "🎯 SOLVER_INVOKED: SABER solver called by Inspect AI",
                agent_name=agent_name,
                state_metadata_keys=list(state.metadata.keys()) if state.metadata else [],
                state_messages_count=len(state.messages) if state.messages else 0,
            )

            # Extract prompts from sample metadata
            metadata = state.metadata
            instruction_prompt = metadata.get(MetadataKeys.INSTRUCTION_PROMPT)
            assistant_prompt = metadata.get(MetadataKeys.ASSISTANT_PROMPT)
            submit_prompt = metadata.get(MetadataKeys.SUBMIT_PROMPT)

            # Validate prompts are present
            if not instruction_prompt:
                raise ValueError(
                    "Missing 'instruction_prompt' in sample metadata. "
                    "Ensure the SABER server is providing all three prompts."
                )
            if not assistant_prompt:
                raise ValueError(
                    "Missing 'assistant_prompt' in sample metadata. "
                    "Ensure the SABER server is providing all three prompts."
                )
            if not submit_prompt:
                raise ValueError(
                    "Missing 'submit_prompt' in sample metadata. "
                    "Ensure the SABER server is providing all three prompts."
                )

            logger.info(
                f"SABER agent '{agent_name}' starting with prompts from metadata",
                extra={
                    "agent": agent_name,
                    "task_id": metadata.get(MetadataKeys.TASK_ID),
                    "instruction_length": len(instruction_prompt),
                    "assistant_length": len(assistant_prompt),
                    "submit_length": len(submit_prompt),
                },
            )

            # DEBUG: Log before calling agent factory
            debug_logger.info(
                "🏭 SOLVER: Calling agent_factory to create agent",
                agent_name=agent_name,
                agent_factory_type=type(agent_factory).__name__,
            )

            # Create agent using the factory
            # agent_factory() returns a function that accepts prompts
            # Call it with prompts to get the actual agent
            create_with_prompts = agent_factory()

            # DEBUG: Log factory result
            debug_logger.info(
                "🏗️ SOLVER: agent_factory returned, calling with prompts",
                agent_name=agent_name,
                create_with_prompts_type=type(create_with_prompts).__name__,
            )

            agent = create_with_prompts(
                instruction_prompt=instruction_prompt,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            )

            # DEBUG: Log agent creation complete
            debug_logger.info(
                "✅ SOLVER: Agent created, about to execute",
                agent_name=agent_name,
                agent_type=type(agent).__name__,
            )

            # Check for role-based model selection
            role = metadata.get(MetadataKeys.SUB_TASK_ROLE)

            # DEBUG: Log role detection
            debug_logger.info(
                "🔍 ROLE CHECK",
                role=role,
                has_role_config=role_config is not None,
                metadata_keys=list(metadata.keys()) if metadata else [],
            )

            if role and role_config:
                from inspect_ai.model import get_model
                from inspect_ai.model._model import active_model, active_model_context_var

                # Get role-specific model from configuration
                try:
                    role_agent_config = role_config.get_config_for_role(role)
                    role_model_name = role_agent_config.model

                    if role_model_name:
                        logger.info(
                            f"Using role-specific model for '{role}' role: {role_model_name}",
                            extra={
                                "role": role,
                                "model": role_model_name,
                                "task_id": metadata.get(MetadataKeys.TASK_ID),
                            },
                        )

                        # Get the model instance
                        role_model = get_model(role_model_name)

                        # Save current model to restore later
                        previous_model = active_model()

                        try:
                            # Set role-specific model as active using context var
                            active_model_context_var.set(role_model)

                            # Execute agent with role-specific model context
                            result = await agent(state)

                            # Push transcript to server after agent execution
                            await _push_transcript_if_enabled(result, metadata)
                        finally:
                            # Restore previous model
                            if previous_model:
                                active_model_context_var.set(previous_model)
                            else:
                                # If no previous model, clear the context
                                active_model_context_var.set(None)

                        logger.info(
                            f"SABER agent '{agent_name}' execution complete with role model",
                            extra={
                                "agent": agent_name,
                                "role": role,
                                "model": role_model_name,
                                "task_id": metadata.get(MetadataKeys.TASK_ID),
                            },
                        )
                        return result

                except Exception as e:
                    logger.warning(
                        "Failed to apply role-based model, using default",
                        extra={
                            "role": role,
                            "error": str(e),
                            "task_id": metadata.get(MetadataKeys.TASK_ID),
                        },
                    )

            # Execute the agent with default model
            logger.debug(
                f"Executing SABER agent '{agent_name}'",
                extra={"agent": agent_name, "task_id": metadata.get(MetadataKeys.TASK_ID)},
            )
            result = await agent(state)

            # Push transcript to server after agent execution
            await _push_transcript_if_enabled(result, metadata)

            logger.info(
                f"SABER agent '{agent_name}' execution complete",
                extra={
                    "agent": agent_name,
                    "task_id": metadata.get(MetadataKeys.TASK_ID),
                    "completion_length": len(result.output.completion) if result.output.completion else 0,
                },
            )

            return result

        return solve

    return saber_agent_solver()


async def _run_preflight_check(
    domain_slug: str,
    domains_root: Path,
    concurrency: int = 8,
    timeout: int = 180,
) -> None:
    """Run saber-domain preflight check before evaluation.

    Executes the preflight CLI command to validate all compose environments
    are healthy before starting the evaluation. This catches configuration
    issues early and prevents wasted evaluation time.

    Args:
        domain_slug: Domain to check
        domains_root: Path to domains directory
        concurrency: Number of environments to check in parallel (default: 8)
        timeout: Timeout per environment in seconds (default: 180)

    Raises:
        PrerequisiteError: If preflight check fails or command execution fails
    """
    import signal
    import sys

    logger.info(
        f"Running preflight check for domain '{domain_slug}'",
        extra={
            "domain": domain_slug,
            "concurrency": concurrency,
            "timeout": timeout,
        },
    )

    # Print to terminal for user feedback
    print(f"\n🚀 Running preflight check for domain '{domain_slug}'...")
    print(f"   Testing {concurrency} environments in parallel (timeout: {timeout}s per environment)")
    print("   Press Ctrl+C to cancel\n")

    # Find the saber-domain CLI
    # Try to use the same Python interpreter and environment
    cli_cmd = [sys.executable, "-m", "saber.domain.cli", "preflight", domain_slug]

    # Add options
    cli_cmd.extend(["-c", str(concurrency)])
    cli_cmd.extend(["--timeout", str(timeout)])
    cli_cmd.append("--verbose")  # Show detailed progress

    # Set working directory to domains_root parent for proper path resolution
    cwd = domains_root.parent if domains_root.parent.exists() else domains_root

    process = None
    try:
        # Run preflight command with direct stdout/stderr passthrough for real-time output
        process = await asyncio.create_subprocess_exec(
            *cli_cmd,
            stdout=None,  # Pass through to terminal
            stderr=None,  # Pass through to terminal
            cwd=cwd,
        )

        # Wait for completion with cancellation support
        try:
            exit_code = await process.wait()
        except asyncio.CancelledError:
            # User hit Ctrl+C - terminate the subprocess gracefully
            print("\n⚠️  Ctrl+C detected - cancelling preflight check...")
            print("   Sending termination signal to preflight process...")

            if process.returncode is None:  # Process still running
                try:
                    # Send SIGINT first (graceful)
                    process.send_signal(signal.SIGINT)
                    print("   Waiting for cleanup to complete (this may take a few seconds)...")

                    # Wait up to 10 seconds for graceful shutdown
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10.0)
                        print("   ✓ Preflight check cancelled and cleaned up")
                    except asyncio.TimeoutError:
                        # If graceful didn't work, force terminate
                        print("   Graceful shutdown timed out, forcing termination...")
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                            print("   ✓ Preflight check forcefully terminated")
                        except asyncio.TimeoutError:
                            # Last resort - kill
                            print("   Force termination timed out, killing process...")
                            process.kill()
                            await process.wait()
                            print("   ✓ Preflight check killed")
                except Exception as e:
                    logger.warning(f"Error during preflight cancellation: {e}")
                    print(f"   ⚠️  Warning: Error during cleanup: {e}")

            raise PrerequisiteError(
                "Preflight check cancelled by user (Ctrl+C). "
                "Some Docker containers may still be cleaning up in the background."
            )

        if exit_code != 0:
            raise PrerequisiteError(
                f"\nPreflight check failed for domain '{domain_slug}' (exit code: {exit_code}).\n\n"
                f"One or more compose environments failed health checks.\n"
                f"Review the preflight output above for details on which environments failed.\n\n"
                f"To fix:\n"
                f"  1. Check compose file configurations in domains/{domain_slug}/server/config/environments/\n"
                f"  2. Verify healthcheck endpoints match your service ports\n"
                f"  3. Run preflight manually for debugging: uv run saber-domain preflight {domain_slug} --verbose"
            )

        logger.info(
            f"Preflight check passed for domain '{domain_slug}'",
            extra={"domain": domain_slug},
        )
        print(f"\n✅ Preflight check passed for domain '{domain_slug}'\n")

    except FileNotFoundError as e:
        raise PrerequisiteError(
            f"Failed to run saber-domain preflight command.\n\n"
            f"Ensure SABER is properly installed:\n"
            f"  uv pip install -e external/saber\n\n"
            f"Error: {e}"
        ) from e
    except asyncio.CancelledError:
        # Re-raise CancelledError to properly propagate cancellation
        raise
    except Exception as e:
        if isinstance(e, PrerequisiteError):
            raise
        raise PrerequisiteError(
            f"Preflight check failed with unexpected error: {e}\n\n"
            f"Command: {' '.join(cli_cmd)}\n"
            f"Working directory: {cwd}"
        ) from e
    finally:
        # Ensure process is cleaned up even if something goes wrong
        if process is not None and process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass  # Best effort cleanup


async def _start_and_load_tasks(
    domain_slug: str,
    domains_root: Path,
    rest_port: int,
    mcp_port: int,
    task_filter: Optional[str],
    agent_name: str,
    log_level: str,
    build: Optional[str],
    rebuild: Optional[str],
    compose_template_path: Optional[Path],
    stop_saber_after: bool,
    max_concurrent_episodes: int,
    run_preflight: bool,
    enable_debug_logging: bool = False,
    role_config: Optional[Any] = None,
    **kwargs: Any,
) -> Task:
    """Start SABER domain and load tasks as dataset.

    This async helper:
    1. Validates parameters
    2. Checks for concurrent domain usage (process-local registry)
    3. Performs preflight check for already-running instances
    4. Runs preflight environment checks if requested (before domain startup)
    5. Starts DomainController with health checks
    6. Waits for server health with retry/backoff
    7. Queries REST API for benchmark info
    8. Applies task_filter (exact, glob, or comma-separated patterns with OR logic)
    9. Converts to dataset with pre-assigned IDs
    10. Returns Task with MemoryDataset and sandbox config

    Cleanup is guaranteed via try/finally on all failure paths.

    Args:
        domain_slug: Domain to start
        domains_root: Path to domains directory
        rest_port: REST API port
        mcp_port: MCP API port
        task_filter: Optional task filter pattern (supports comma-separated patterns)
        agent_name: Name of agent implementation to use
        log_level: Logging level
        build: Optional build filter
        rebuild: Optional rebuild filter
        compose_template_path: Optional compose template
        stop_saber_after: Stop domain after evaluation
        max_concurrent_episodes: Max concurrent episodes
        run_preflight: Run preflight check on all compose environments before starting
        **kwargs: Additional parameters

    Returns:
        Inspect AI Task with fully-populated dataset

    Raises:
        PrerequisiteError: On any failure (server start, health, task loading)
    """
    controller: Optional[DomainController] = None
    context: Optional[DomainContext] = None

    # Resolve agent implementation before starting domain
    agent_factory = _resolve_agent_implementation(domain_slug, domains_root, agent_name)

    try:
        # Check if domain already active in this process (reuse scenario)
        with _active_domains_lock:
            existing = _active_domains.get(domain_slug)
            if existing:
                # Domain already running in this process - reuse it
                logger.info(
                    f"Reusing existing SABER domain '{domain_slug}'",
                    extra={
                        "domain": domain_slug,
                        "rest_port": existing["rest_port"],
                        "mcp_port": existing["mcp_port"],
                    },
                )
                context = existing["context"]
                # Skip server startup, go directly to task loading
                controller = existing["controller"]
                # Don't update registry - already there

        # Perform preflight check - detect already-running SABER instance (other process or ports)
        controller = DomainController(domains_root=Path(domains_root))
        running_domain = await controller.check_running_domain(rest_port, mcp_port)

        if running_domain and not existing:
            # Server running but not in our registry
            if running_domain == domain_slug:
                # Same domain, different process or manual start - reuse it!
                logger.info(
                    f"Found existing SABER domain '{domain_slug}' running externally, reusing it",
                    extra={
                        "domain": domain_slug,
                        "rest_port": rest_port,
                        "mcp_port": mcp_port,
                    },
                )
                # Create a dummy controller (we won't stop it since we didn't start it)
                controller = DomainController(domains_root=Path(domains_root))
                context = DomainContext(
                    domain=domain_slug,
                    rest_url=f"http://localhost:{rest_port}",
                    mcp_url=f"http://localhost:{mcp_port}",
                    rest_port=rest_port,
                    mcp_port=mcp_port,
                    project_slug=domain_slug,
                    domains_root=Path(domains_root),
                )
                # Register it with ownership=False (we didn't start it)
                with _active_domains_lock:
                    _active_domains[domain_slug] = {
                        "owner": f"domain_task_{domain_slug}",
                        "domain_slug": domain_slug,
                        "controller": controller,
                        "context": context,
                        "ownership": False,  # Don't stop it on cleanup
                        "rest_port": rest_port,
                        "mcp_port": mcp_port,
                        "rest_url": context.rest_url,
                        "mcp_url": context.mcp_url,
                    }
            else:
                # Different domain running on these ports - fail!
                raise PrerequisiteError(
                    f"SABER domain '{running_domain}' is already running on ports "
                    f"{rest_port}/{mcp_port}.\n"
                    f"Task requires domain '{domain_slug}'.\n\n"
                    f"Options:\n"
                    f"  1. Stop the running domain: uv run saber-domain --domains-root "
                    f"{domains_root} stop {running_domain}\n"
                    f"  2. Use different ports: inspect eval domains/{domain_slug} "
                    f"-T rest_port=9000 -T mcp_port=9001"
                )

        if not existing and not running_domain:
            # No domain running - start it fresh

            # Run preflight check if requested (before starting domain)
            if run_preflight:
                await _run_preflight_check(
                    domain_slug=domain_slug,
                    domains_root=domains_root,
                    concurrency=max_concurrent_episodes,
                    timeout=180,
                )

            # Create controller (CLI-based, no compose_template_path needed)
            controller = DomainController(
                domains_root=Path(domains_root),
            )

            logger.info(
                f"Starting SABER domain '{domain_slug}' for task construction",
                extra={
                    "domain": domain_slug,
                    "rest_port": rest_port,
                    "mcp_port": mcp_port,
                },
            )

            # Start domain with health checks
            context = await controller.start(
                domain=domain_slug,
                rest_port=rest_port,
                mcp_port=mcp_port,
                log_level=log_level,
                build=build,
                rebuild=rebuild,
            )

            # Register in active domains (for preflight and sandbox coordination)
            with _active_domains_lock:
                _active_domains[domain_slug] = {
                    "owner": f"domain_task_{domain_slug}",
                    "domain_slug": domain_slug,
                    "controller": controller,
                    "context": context,
                    "ownership": True,  # We started it, we can stop it
                    "rest_port": rest_port,
                    "mcp_port": mcp_port,
                    "rest_url": context.rest_url,
                    "mcp_url": context.mcp_url,
                }

        logger.info(f"SABER domain '{domain_slug}' started, waiting for health checks")

        # Wait for server health with retry/backoff
        assert context is not None  # Help mypy understand context is not None
        await _wait_for_server_health(context.rest_url, max_retries=30, backoff=2.0)

        logger.info(f"SABER domain '{domain_slug}' is healthy, querying tasks")

        # Query REST API for benchmark info
        rest_client = SABERRestClient(context.rest_url, request_timeout=60.0)
        benchmark_info: BenchmarkInfo = await rest_client.get_benchmark_info()

        logger.info(
            f"Retrieved {benchmark_info.total_tasks} tasks from SABER server",
            extra={
                "domain": domain_slug,
                "total_tasks": benchmark_info.total_tasks,
                "total_episodes": benchmark_info.total_episodes,
            },
        )

        # Apply task filter if specified
        tasks_to_load = benchmark_info.tasks
        if task_filter:
            tasks_to_load = _apply_task_filter(
                benchmark_info.tasks,
                task_filter,
                domain_slug,
            )
            logger.info(
                f"Applied task_filter '{task_filter}': {len(tasks_to_load)} tasks matched",
                extra={
                    "domain": domain_slug,
                    "task_filter": task_filter,
                    "matched_count": len(tasks_to_load),
                },
            )

        # Convert to SABERDataset with orchestration-aware slicing
        dataset = await create_saber_dataset(tasks_to_load)  # type: ignore[arg-type]

        logger.info(
            f"Created dataset with {len(dataset)} samples",
            extra={
                "domain": domain_slug,
                "sample_count": len(dataset),
                "task_count": len(tasks_to_load),
            },
        )

        # Check for orchestrated tasks and log info (SABERDataset handles --limit automatically)
        orchestrations: Dict[str, List[str]] = {}
        for sample in dataset:
            orch_id = sample.metadata.get(MetadataKeys.ORCHESTRATION_ID) if sample.metadata else None
            if orch_id:
                if orch_id not in orchestrations:
                    orchestrations[orch_id] = []
                orchestrations[orch_id].append(sample.id)

        if orchestrations:
            logger.info(
                f"Dataset contains {len(orchestrations)} orchestrated task(s). "
                f"SABERDataset will automatically preserve orchestration boundaries when --limit is applied.",
                extra={
                    "domain": domain_slug,
                    "orchestration_count": len(orchestrations),
                    "total_samples": len(dataset),
                    "orchestrations": {k: len(v) for k, v in orchestrations.items()},
                },
            )
        # Create solver with the resolved agent and role_config for model selection
        saber_solver = _create_saber_solver(agent_name, agent_factory, role_config)

        # Construct Task with SABERDataset, sandbox config, solver, and scorer
        # The SABERDataset automatically handles orchestration-aware slicing
        # The sandbox config points to SABERSandboxEnvironment
        # The scorer performs client-side evaluation after agent execution

        # Check if role_config provides all models (allows bypassing --model requirement)
        task_model: Optional[str] = None
        if role_config and _all_roles_have_models(role_config):
            # All roles have models - use first role's model as Task default
            #
            # NOTE: This Task is used ONLY when running via `inspect eval` command.
            # Role-based model assignment is fully supported only when using SABER's
            # native runner (EvaluationOrchestrator), which creates separate Task
            # instances per role via AgentManager.create_agent_task().
            #
            # When using `inspect eval`, there is only one Task and one model context,
            # so we use the first role's model as a reasonable default.
            first_role = list(role_config.roles.keys())[0]
            first_role_config = role_config.get_config_for_role(first_role)
            task_model = first_role_config.model
            logger.info(
                "All roles have models defined - using first role's model as Task default",
                extra={
                    "domain": domain_slug,
                    "roles": list(role_config.roles.keys()) if role_config.roles else [],
                    "default_model": task_model,
                    "default_from_role": first_role,
                },
            )

        task = Task(
            dataset=dataset,
            model=task_model,  # None if all roles have models, otherwise uses Inspect AI's default
            sandbox=(
                "saber",
                {
                    "domain_slug": domain_slug,
                    "domains_root": domains_root,
                    "rest_port": rest_port,
                    "mcp_port": mcp_port,
                    "compose_template_path": compose_template_path,
                    "cleanup": stop_saber_after,  # Pass cleanup flag to sandbox
                    "max_concurrent_episodes": max_concurrent_episodes,  # Limit concurrent episode execution
                    "enable_debug_logging": enable_debug_logging,  # Enable detailed lifecycle debug logging
                },
            ),
            solver=saber_solver,
            scorer=saber_scorer(),  # Client-side evaluation
            # SABER defaults for evaluation behavior
            # These can be overridden via inspect eval CLI flags
            epochs=1,  # Run each sample once by default (use --epochs N to override)
        )

        logger.info(
            f"SABER task construction complete for domain '{domain_slug}'",
            extra={
                "domain": domain_slug,
                "sample_count": len(dataset),
            },
        )

        return task

    except Exception as e:
        # Cleanup on failure
        logger.error(
            f"Failed to start and load tasks for domain '{domain_slug}': {e}",
            extra={"domain": domain_slug},
            exc_info=True,
        )

        # Check if we need to cleanup (only if we started the domain)
        should_cleanup = False
        with _active_domains_lock:
            if domain_slug in _active_domains:
                should_cleanup = _active_domains[domain_slug].get("ownership", False)
                del _active_domains[domain_slug]
                logger.debug(f"Removed '{domain_slug}' from active domains after failure")

        # Stop controller only if we started it
        if should_cleanup and controller is not None:
            try:
                logger.info(f"Stopping domain '{domain_slug}' after failure (we started it)")
                await controller.stop(domain_slug)
            except Exception as stop_error:
                logger.warning(
                    f"Failed to stop domain '{domain_slug}' during cleanup: {stop_error}", extra={"domain": domain_slug}
                )

        # Re-raise as PrerequisiteError for Inspect AI
        if isinstance(e, PrerequisiteError):
            raise
        raise PrerequisiteError(f"Failed to construct task for SABER domain '{domain_slug}'.\n\n{e}") from e


async def _wait_for_server_health(
    rest_url: str,
    max_retries: int = 30,
    backoff: float = 2.0,
) -> None:
    """Wait for SABER server to become healthy with retry/backoff.

    Polls the health endpoint until successful or max retries exceeded.

    Args:
        rest_url: Base URL for REST API
        max_retries: Maximum number of retry attempts (default: 30)
        backoff: Backoff multiplier between retries (default: 2.0s)

    Raises:
        PrerequisiteError: If server doesn't become healthy within retries
    """
    health_url = f"{rest_url}/api/v1/health"

    for attempt in range(1, max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(
                            f"SABER server health check passed (attempt {attempt})",
                            extra={
                                "rest_url": rest_url,
                                "attempt": attempt,
                                "domain": data.get("domain", "unknown"),
                            },
                        )
                        return
                    else:
                        logger.warning(
                            f"Health check returned {response.status} (attempt {attempt}/{max_retries})",
                            extra={"rest_url": rest_url, "attempt": attempt},
                        )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(
                f"Health check failed (attempt {attempt}/{max_retries}): {e}",
                extra={"rest_url": rest_url, "attempt": attempt},
            )

        if attempt < max_retries:
            await asyncio.sleep(backoff)

    # Max retries exceeded
    raise PrerequisiteError(
        f"SABER server health check failed after {max_retries} attempts.\n\n"
        f"REST URL: {rest_url}\n\n"
        "The server may have failed to start or is not responding.\n"
        "Check server logs for details:\n"
        "  docker logs <container_name>\n"
        "  docker compose -p saber-<domain> logs server"
    )


def _apply_task_filter(
    tasks: List[Any],
    task_filter: Union[str, List[str]],
    domain_slug: str,
) -> List[Any]:
    """Apply task filter with exact and glob pattern matching using polymorphic dispatch.

    Supports both legacy TaskInfo and new polymorphic BenchmarkTask types.
    Uses the polymorphic matches_filter() method on each task for type-safe filtering.

    Supports:
    - Exact match: task_filter="labyrinth_linguist_task_hard"
    - Glob pattern: task_filter="labyrinth_*"
    - Glob pattern: task_filter="*_hard"
    - Multiple filters (OR logic): task_filter="xss_*,sql_*"
    - Multiple filters with exact: task_filter="xss_0_flag_capture,sql_*,cmd_injection_task"

    For OrchestratedTask: Matching ANY sub-task includes the entire orchestration.
    For SingleEpisodeTask: Matches against the single task_id.
    For TaskInfo: Legacy exact/glob matching against task_id.

    Multiple filters are separated by commas and matched with OR logic.
    Each filter can be an exact match or a glob pattern.

    Args:
        tasks: List of TaskInfo or BenchmarkTask objects from server
        task_filter: Filter pattern (exact, glob, or comma-separated patterns)
                    Can be a string or a list (Inspect AI may parse comma-separated values as lists)
        domain_slug: Domain slug for error messages

    Returns:
        Filtered list of tasks (deduplicated)

    Raises:
        PrerequisiteError: If no tasks match any of the filters
    """
    # Handle both string and list inputs (Inspect AI may parse "a,b" as ["a", "b"])
    if isinstance(task_filter, list):
        filter_patterns = [str(p).strip() for p in task_filter]
    else:
        # Split on comma to support multiple filters
        filter_patterns = [pattern.strip() for pattern in task_filter.split(",")]

    # Collect all matching tasks across all patterns
    # Use dict to deduplicate - key depends on task type
    matched_tasks: Dict[str, Any] = {}

    for pattern in filter_patterns:
        if not pattern:  # Skip empty patterns
            continue

        # Use polymorphic matches_filter() if available (BenchmarkTask types)
        # Otherwise fall back to legacy exact/glob matching (TaskInfo)
        for task in tasks:
            if hasattr(task, "matches_filter"):
                # New polymorphic task types (SingleEpisodeTask, OrchestratedTask)
                if task.matches_filter(pattern):
                    # Use benchmark_task_id as key for deduplication
                    matched_tasks[task.benchmark_task_id] = task
            else:
                # Legacy TaskInfo - use exact/glob matching
                task_id = task.task_id
                if task_id == pattern or fnmatch.fnmatch(task_id, pattern):
                    matched_tasks[task_id] = task

    # Convert back to list and maintain consistent ordering
    if matched_tasks:
        # Sort by task ID for deterministic ordering
        def get_sort_key(t: Any) -> str:
            if hasattr(t, "benchmark_task_id"):
                return str(t.benchmark_task_id)
            else:
                return str(t.task_id)

        return sorted(matched_tasks.values(), key=get_sort_key)

    # No matches - provide helpful error
    available_ids = []
    for task in tasks:
        if hasattr(task, "benchmark_task_id"):
            available_ids.append(task.benchmark_task_id)
        else:
            available_ids.append(task.task_id)

    # Format task_filter for error message
    filter_display = task_filter if isinstance(task_filter, str) else ",".join(task_filter)
    raise PrerequisiteError(
        f"No tasks matched filter '{filter_display}' in domain '{domain_slug}'.\n\n"
        f"Available tasks ({len(available_ids)}):\n"
        + "\n".join(f"  - {task_id}" for task_id in sorted(available_ids)[:20])
        + (f"\n  ... and {len(available_ids) - 20} more" if len(available_ids) > 20 else "")
    )


# Module-level function to access active domains (for sandbox coordination)
def get_active_domain(domain_slug: str) -> Optional[Dict[str, Any]]:
    """Get active domain registry entry if exists.

    Used by SABERSandboxEnvironment to detect if the factory already
    started the server, enabling ownership transfer.

    Args:
        domain_slug: Domain to lookup

    Returns:
        Registry entry dict or None if not active
    """
    with _active_domains_lock:
        return _active_domains.get(domain_slug)


def _resolve_agent_implementation(
    domain_slug: str,
    domains_root: Path,
    agent_name: str,
) -> Callable:
    """Resolve agent implementation with three-tier discovery.

    Discovery order:
    1. Domain-local agents (domains/{domain}/client/{agent}.py)
    2. Core SABER agents (saber.inspect_ai.agents.{agent})
    3. Error if not found

    Args:
        domain_slug: Domain name
        domains_root: Path to domains directory
        agent_name: Name of agent to resolve

    Returns:
        Agent factory callable

    Raises:
        AgentNotFoundError: If agent not found in any registry
    """
    logger.debug(
        f"Resolving agent '{agent_name}' for domain '{domain_slug}'", extra={"agent": agent_name, "domain": domain_slug}
    )

    # 1. Try domain-local agent first
    domain_agent = _load_domain_agent(domain_slug, domains_root, agent_name)
    if domain_agent is not None:
        logger.info(
            f"Using domain-local agent '{agent_name}' from {domain_slug}/client/",
            extra={"agent": agent_name, "domain": domain_slug, "source": "domain-local"},
        )
        return domain_agent

    # 2. Try core SABER agent registry
    core_agent = SABERAgentRegistry.get(agent_name)
    if core_agent is not None:
        logger.info(
            f"Using core SABER agent '{agent_name}'",
            extra={"agent": agent_name, "domain": domain_slug, "source": "core-registry"},
        )
        return core_agent

    # 3. Not found - provide helpful error
    core_agents = SABERAgentRegistry.list_agents()
    domain_client_dir = domains_root / domain_slug / "client"

    error_msg = (
        f"Agent '{agent_name}' not found for domain '{domain_slug}'.\n\n"
        f"Agent discovery order:\n"
        f"  1. Domain-local: {domain_client_dir}/{agent_name}.py\n"
        f"  2. Core SABER agents: {', '.join(core_agents) if core_agents else '(none)'}\n\n"
        f"To fix:\n"
        f"  - Use a core agent: -T agent={core_agents[0] if core_agents else 'react'}\n"
        f"  - Create domain agent: {domain_client_dir}/{agent_name}.py with create_agent() function\n"
        f"  - Check spelling: '{agent_name}'"
    )

    logger.error(
        f"Agent '{agent_name}' not found",
        extra={
            "agent": agent_name,
            "domain": domain_slug,
            "core_agents": core_agents,
            "domain_client_dir": str(domain_client_dir),
        },
    )

    raise AgentNotFoundError(error_msg)


def _load_domain_agent(
    domain_slug: str,
    domains_root: Path,
    agent_name: str,
) -> Optional[Callable]:
    """Load agent from domain's client folder.

    Searches: domains/{domain_slug}/client/{agent_name}.py
    Expects: A module with create_agent() function

    Args:
        domain_slug: Domain name
        domains_root: Path to domains directory
        agent_name: Agent name (filename without .py)

    Returns:
        Agent factory callable, or None if not found
    """
    agent_file = domains_root / domain_slug / "client" / f"{agent_name}.py"

    if not agent_file.exists():
        logger.debug(
            f"Domain-local agent file not found: {agent_file}", extra={"agent": agent_name, "domain": domain_slug}
        )
        return None

    try:
        # Dynamically import the module
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(f"domain_{domain_slug}_agent_{agent_name}", agent_file)
        if spec is None or spec.loader is None:
            logger.warning(
                f"Failed to load agent module spec: {agent_file}", extra={"agent": agent_name, "domain": domain_slug}
            )
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Look for create_agent function
        if hasattr(module, "create_agent"):
            logger.debug(
                f"Loaded domain-local agent from {agent_file}", extra={"agent": agent_name, "domain": domain_slug}
            )
            return module.create_agent  # type: ignore[no-any-return]
        else:
            logger.warning(
                f"Domain agent file {agent_file} missing create_agent() function",
                extra={"agent": agent_name, "domain": domain_slug, "file": str(agent_file)},
            )
            return None

    except Exception as e:
        logger.warning(
            f"Failed to load domain agent from {agent_file}: {e}",
            extra={"agent": agent_name, "domain": domain_slug, "error": str(e)},
        )
        return None


def remove_active_domain(domain_slug: str) -> None:
    """Remove domain from active registry.

    Called by SABERSandboxEnvironment.task_cleanup() to clean up
    after evaluation completes.

    Args:
        domain_slug: Domain to remove
    """
    with _active_domains_lock:
        if domain_slug in _active_domains:
            del _active_domains[domain_slug]
            logger.debug(f"Removed '{domain_slug}' from active domains registry")


# ============================================================================
# Transcript Synchronization Functions
# ============================================================================


async def _push_transcript_if_enabled(
    state: TaskState,
    metadata: Dict[str, Any],
) -> None:
    """Helper to push transcript if episode context is available.

    This function extracts episode context from metadata and active domain
    registry, then delegates to _push_transcript() if all required data is present.

    Args:
        state: TaskState after agent execution
        metadata: Sample metadata containing episode/session IDs and domain slug

    Returns:
        None (gracefully handles missing context and errors)
    """
    try:
        # Extract episode context from metadata
        session_id = metadata.get(MetadataKeys.SESSION_ID)
        episode_id = metadata.get(MetadataKeys.EPISODE_ID)
        domain_slug = metadata.get(MetadataKeys.SABER_DOMAIN_SLUG)

        # Skip if episode context is missing
        if not session_id or not episode_id:
            return

        # Get REST URL from active domain registry
        if domain_slug:
            domain_context = get_active_domain(domain_slug)
            if domain_context:
                rest_url = domain_context.get("rest_url")
                if rest_url:
                    # Push transcript using extracted context
                    await _push_transcript(
                        state=state,
                        session_id=session_id,
                        episode_id=episode_id,
                        rest_url=rest_url,
                    )
    except Exception as e:
        # Log error but don't raise - transcript sync failures should not crash agent execution
        logger.warning(
            "Failed to push transcript, continuing with agent execution",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "episode_id": metadata.get(MetadataKeys.EPISODE_ID),
                "session_id": metadata.get(MetadataKeys.SESSION_ID),
            },
        )


def _serialize_message(msg: ChatMessage) -> Dict[str, Any]:
    """Convert ChatMessage to JSON-safe dict.

    Args:
        msg: Inspect AI ChatMessage object (ChatMessageSystem, ChatMessageUser,
             ChatMessageAssistant, or ChatMessageTool)

    Returns:
        Dictionary with keys: role, content, tool_calls (optional),
        tool_call_id (optional), name (optional), reasoning (optional)

    Raises:
        ValueError: If message type is unknown or unsupported
    """
    result: Dict[str, Any] = {}

    # Extract role
    if isinstance(msg, ChatMessageSystem):
        result["role"] = "system"
    elif isinstance(msg, ChatMessageUser):
        result["role"] = "user"
    elif isinstance(msg, ChatMessageAssistant):
        result["role"] = "assistant"
    elif isinstance(msg, ChatMessageTool):
        result["role"] = "tool"
    else:
        raise ValueError(f"Unknown message type: {type(msg)}")

    # Extract content - handle both string and list of Content objects
    if isinstance(msg.content, str):
        result["content"] = msg.content
    elif isinstance(msg.content, list):
        # Extract text content parts and concatenate
        text_parts = []
        reasoning_text = None

        for content_item in msg.content:
            if isinstance(content_item, ContentText):
                text_parts.append(content_item.text)
            elif isinstance(content_item, ContentReasoning):
                reasoning_text = content_item.reasoning

        result["content"] = "".join(text_parts)

        # Add reasoning if present (for assistant messages)
        if reasoning_text:
            result["reasoning"] = reasoning_text
    else:
        result["content"] = ""

    # Handle assistant-specific fields
    if isinstance(msg, ChatMessageAssistant):
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": tc.function,
                    "arguments": tc.arguments,
                }
                for tc in msg.tool_calls
            ]

    # Handle tool message-specific fields
    if isinstance(msg, ChatMessageTool):
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if msg.function:
            result["name"] = msg.function

    return result


async def _push_transcript(
    state: TaskState,
    session_id: str,
    episode_id: str,
    rest_url: str,
) -> None:
    """Push current transcript to SABER server.

    Args:
        state: Current TaskState with messages
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")

    Returns:
        None (logs errors but doesn't raise exceptions)

    Side Effects:
        - Logs success/failure
        - Makes HTTP POST request to server
        - Retries on transient failures (max 3 attempts)
    """
    # Constants
    MAX_PAYLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB (matches server limit)
    REQUEST_TIMEOUT_SECONDS = 5.0
    MAX_RETRIES = 3
    RETRY_DELAYS_SECONDS = [0.5, 1.0, 2.0]
    FEATURE_FLAG_ENV_VAR = "SABER_ENABLE_TRANSCRIPT_SYNC"
    FEATURE_FLAG_DEFAULT = "true"

    # Check feature flag (evaluate at runtime, not module load time)
    if os.getenv(FEATURE_FLAG_ENV_VAR, FEATURE_FLAG_DEFAULT).lower() != "true":
        return

    # Validate rest_url format (defense against SSRF)
    if not rest_url.startswith(("http://", "https://")):
        logger.warning(
            "Invalid rest_url format - must start with http:// or https://",
            extra={
                "episode_id": episode_id,
                "rest_url": rest_url,
            },
        )
        return

    # Serialize all messages
    try:
        messages = [_serialize_message(msg) for msg in state.messages]
    except Exception as e:
        logger.warning(
            "Failed to serialize transcript messages",
            extra={
                "episode_id": episode_id,
                "error": str(e),
            },
        )
        return

    # Prepare metadata
    metadata = {
        "step_number": len(state.messages),
        "timestamp": datetime.utcnow().isoformat(),
        "source": "inspect_ai",
    }

    # Prepare payload
    payload = {
        "messages": messages,
        "metadata": metadata,
    }

    # Check payload size before sending (client-side validation)
    import json

    try:
        payload_json = json.dumps(payload)
        payload_size = len(payload_json.encode("utf-8"))

        if payload_size > MAX_PAYLOAD_SIZE_BYTES:
            logger.warning(
                "Transcript payload too large, skipping push",
                extra={
                    "episode_id": episode_id,
                    "payload_size_bytes": payload_size,
                    "max_size_bytes": MAX_PAYLOAD_SIZE_BYTES,
                    "message_count": len(messages),
                },
            )
            return
    except Exception as e:
        logger.warning(
            "Failed to validate payload size",
            extra={
                "episode_id": episode_id,
                "error": str(e),
            },
        )
        return

    # Construct URL
    url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/transcript"

    # Retry configuration
    max_retries = MAX_RETRIES
    retry_delays = RETRY_DELAYS_SECONDS

    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        logger.info(
                            "Transcript pushed successfully",
                            extra={
                                "episode_id": episode_id,
                                "message_count": len(messages),
                                "response_status": resp.status,
                            },
                        )
                        return
                    elif resp.status == 422:
                        # Validation error - don't retry
                        error_text = await resp.json()
                        logger.warning(
                            "Failed to push transcript - validation error",
                            extra={
                                "episode_id": episode_id,
                                "status_code": resp.status,
                                "error": error_text,
                            },
                        )
                        return
                    else:
                        # Server error - retry
                        error_text = await resp.text()
                        logger.warning(
                            f"Failed to push transcript (attempt {attempt + 1}/{max_retries})",
                            extra={
                                "episode_id": episode_id,
                                "status_code": resp.status,
                                "error": error_text,
                            },
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delays[attempt])

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout pushing transcript (attempt {attempt + 1}/{max_retries})",
                extra={
                    "episode_id": episode_id,
                    "error": "Request timeout",
                },
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])

        except aiohttp.ClientError as e:
            logger.warning(
                f"Network error pushing transcript (attempt {attempt + 1}/{max_retries})",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])

        except Exception as e:
            logger.warning(
                f"Unexpected error pushing transcript (attempt {attempt + 1}/{max_retries})",
                extra={
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])

    # If we get here, all retries failed
    logger.error(
        "Failed to push transcript after all retry attempts",
        extra={
            "episode_id": episode_id,
            "max_retries": max_retries,
        },
    )
