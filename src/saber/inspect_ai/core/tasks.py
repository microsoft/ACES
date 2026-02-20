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

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
from inspect_ai import Task
from inspect_ai._util.error import PrerequisiteError

from ...client.api.rest_client import SABERRestClient
from ...logging_config import LogCategory, LoggingConfig, get_saber_logger, init_logging
from ...models import BenchmarkInfo
from ...models.constants import MetadataKeys, ScoreAggregationStrategy
from ..agents.agent_resolver import resolve_agent_implementation
from ..agents.role_config_processor import all_roles_have_models, process_role_configuration
from ..agents.solver_factory import create_saber_solver
from ..constants import SandboxTimeouts
from ..server.domain_manager import get_active_domain, register_domain, unregister_domain_on_failure
from ..server.health_check import wait_for_server_health
from ..server.preflight import run_preflight_check
from ..server.server import DomainContext, DomainController
from .saber_dataset import create_saber_dataset
from .saber_scorer import saber_scorer
from .task_filter import apply_task_filter

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Track whether logging has been initialized for this process
_logging_initialized = False
_logging_init_lock = threading.Lock()


def _initialize_inspect_logging(domain_slug: str, domains_root: Path, log_level: str = "INFO") -> None:
    """Initialize SABER logging for inspect eval runs.

    Sets up logging with:
    - Console output DISABLED (to avoid breaking Inspect AI's rich display)
    - File logging to logs/saber_inspect_{domain}_{timestamp}.log
    - SaberLogger formatters active for structured output in log files

    This is called once per process to ensure SABER logging is properly configured
    when running via `uv run inspect eval domains/...`.

    Args:
        domain_slug: Domain being evaluated (used for log directory organization)
        domains_root: Path to domains directory
        log_level: Logging level (default: "INFO")
    """
    global _logging_initialized

    with _logging_init_lock:
        if _logging_initialized:
            return

        from datetime import datetime

        # Resolve log level
        level = getattr(logging, log_level.upper(), logging.INFO)

        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_filename = f"saber_inspect_{domain_slug}_{timestamp}.log"

        # Determine log directory: logs/ at workspace root
        workspace_root = domains_root.parent if domains_root.name == "domains" else domains_root
        log_dir = workspace_root / "logs"

        # Check if we can create the log directory (may fail in tests with fake paths)
        enable_file = True
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # Can't create log directory (e.g., test environment with fake paths)
            # Disable file logging entirely - we don't want console output either
            # as it breaks Inspect AI's rich display
            enable_file = False
            log_dir = Path(".")  # Placeholder, won't be used

        # Initialize logging with file output only (no console)
        # Console output breaks Inspect AI's rich terminal display
        config = LoggingConfig(
            level=level,
            console=False,  # Disable console to avoid breaking Inspect AI's rich display
            structured=False,  # Human-readable text format in log files
            enable_file=enable_file,
            log_dir=log_dir,
            file_name=log_filename,
        )

        init_logging(config, force=True)
        _logging_initialized = True

        # Log initialization to file (won't appear on console)
        if enable_file:
            logger.info(
                "SABER logging initialized for inspect eval",
                extra={"domain": domain_slug, "log_file": str(log_dir / log_filename)},
            )


# Process-wide BlockingPortal for running async operations from sync context
# Store both the context manager and the portal
_portal_cm: Any = None
_portal: anyio.abc.BlockingPortal | None = None
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
        default_agent: Default agent implementation for this domain (default: "react")

    Returns:
        Callable that accepts Inspect AI task parameters and returns Task
    """

    def task_callable(
        rest_port: int = 8000,
        mcp_port: int = 8001,
        task_filter: str | None = None,
        agent: str | None = None,
        log_level: str = "INFO",
        build: bool = False,
        rebuild: str | None = None,
        rebuild_all: bool = False,
        stop_saber_after: bool = False,
        run_preflight: bool = False,
        enable_debug_logging: bool = False,
        roles: str | dict | None = None,
        roles_file: str | None = None,
        skills_dir: str | None = None,
        agent_persona: str | None = None,
        score_aggregation: str | None = None,
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
            skills_dir: Path(s) to directory containing Copilot skill files (optional).
                Skills are markdown files with YAML frontmatter that provide
                contextual knowledge/instructions to guide agent behavior.
                Multiple directories can be specified as comma-separated paths.
                Only applicable for 'copilot' agent type.
            agent_persona: Path to a custom agent YAML file that defines the agent persona.
                The file should contain agent_name, system_prompt, and optionally tools.
                Only applicable for 'copilot' agent type.
            score_aggregation: Score aggregation strategy override. Valid values:
                'average' (default), 'weighted_sum', 'max'.
                Overrides any YAML-configured scoring_config.aggregation.
            **kwargs: Additional parameters passed through (may include role overrides like red_model=...)

        Returns:
            Inspect AI Task with fully-populated dataset

        Raises:
            PrerequisiteError: If server startup, health checks, or task loading fails
        """
        # Initialize SABER logging for inspect runs
        # This ensures the SaberLogger formatters are active and extra= fields are visible
        # Log files go to logs/{domain}/client-logs/ with console output enabled
        _initialize_inspect_logging(domain_slug, domains_root, log_level)

        # Validate score_aggregation if provided
        if score_aggregation is not None:
            try:
                ScoreAggregationStrategy(score_aggregation)
            except ValueError as exc:
                valid = [s.value for s in ScoreAggregationStrategy]
                raise PrerequisiteError(
                    f"Invalid score_aggregation '{score_aggregation}'. Valid values: {valid}"
                ) from exc

        # Validate mutually exclusive build options
        # Note: rebuild is str (filter prefix) or None; rebuild_all is bool for rebuild-all
        has_rebuild = bool(isinstance(rebuild, str) and rebuild)
        build_options_count = sum([build, rebuild_all, has_rebuild])
        if build_options_count > 1:
            raise PrerequisiteError(
                "Cannot specify multiple build options together. Use one of:\n"
                "  -T build=true: Build missing images only\n"
                "  -T rebuild_all=true: Remove and rebuild all images\n"
                "  -T rebuild=<prefix>: Remove and rebuild specific images"
            )

        # Convert to orchestrator format (same as CLI)
        build_param = "" if build else None

        # Handle rebuild: string (filter prefix) or None; rebuild_all for rebuild-everything
        if rebuild_all:
            rebuild_param = ""  # Empty string = rebuild all
        elif isinstance(rebuild, str) and rebuild:
            rebuild_param = rebuild  # Use as filter prefix
        else:
            rebuild_param = None

        # Use CLI agent override or domain default
        agent_to_use = agent if agent is not None else default_agent

        # Process role-based configuration if provided
        role_config_obj = process_role_configuration(roles_file, roles, kwargs, domain_slug)

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
                stop_saber_after,
                run_preflight,
                enable_debug_logging,
                role_config_obj,
                skills_dir,
                agent_persona,
                score_aggregation=score_aggregation,
            )
        except Exception as e:
            # Ensure we have a clean error message
            if isinstance(e, PrerequisiteError):
                raise
            raise PrerequisiteError(f"Failed to create SABER task for domain '{domain_slug}': {e}") from e

    return task_callable


async def _start_and_load_tasks(
    domain_slug: str,
    domains_root: Path,
    rest_port: int,
    mcp_port: int,
    task_filter: str | None,
    agent_name: str,
    log_level: str,
    build: str | None,
    rebuild: str | None,
    stop_saber_after: bool,
    run_preflight: bool,
    enable_debug_logging: bool = False,
    role_config: Any | None = None,
    skills_dir: str | None = None,
    agent_persona: str | None = None,
    score_aggregation: str | None = None,
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
        stop_saber_after: Stop domain after evaluation
        run_preflight: Run preflight check on all compose environments before starting
        enable_debug_logging: Enable detailed episode lifecycle debug logging
        role_config: Optional role-based configuration for orchestrated tasks
        skills_dir: Optional path(s) to directory containing Copilot skill files.
            Multiple directories can be specified as comma-separated paths.
        **kwargs: Additional parameters

    Returns:
        Inspect AI Task with fully-populated dataset

    Raises:
        PrerequisiteError: On any failure (server start, health, task loading)
    """
    controller: DomainController | None = None
    context: DomainContext | None = None

    try:
        # Resolve agent implementation before starting domain
        agent_factory = resolve_agent_implementation(domain_slug, domains_root, agent_name)
        # Check if domain already active in this process (reuse scenario)
        existing = get_active_domain(domain_slug)
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

        # Check if rebuild requested - need to determine if we should stop existing domain
        rebuild_requested = build is not None or rebuild is not None

        # Perform preflight check - detect already-running SABER instance (other process or ports)
        controller = DomainController(domains_root=Path(domains_root))
        running_domain = await controller.check_running_domain(rest_port, mcp_port)

        if running_domain and not existing:
            # Server running but not in our registry
            if running_domain == domain_slug:
                # Same domain, different process or manual start
                if rebuild_requested:
                    # Rebuild requested - must stop existing domain first
                    logger.info(
                        f"Rebuild requested but domain '{domain_slug}' already running. "
                        "Stopping existing domain to rebuild...",
                        extra={
                            "domain": domain_slug,
                            "rest_port": rest_port,
                            "mcp_port": mcp_port,
                        },
                    )
                    print(
                        f"[SABER] Rebuild requested - stopping existing domain '{domain_slug}' first...",
                        flush=True,
                    )
                    # Stop the running domain
                    await controller.stop(domain_slug)
                    # Wait for ports to be released (containers may take a moment to fully stop)
                    import asyncio
                    import socket

                    def is_port_free(port: int) -> bool:
                        """Check if a port is free."""
                        try:
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                s.bind(("localhost", port))
                                return True
                        except OSError:
                            return False

                    print(f"[SABER] Waiting for ports {rest_port} and {mcp_port} to be released...", flush=True)
                    for i in range(30):  # Up to 15 seconds
                        await asyncio.sleep(0.5)
                        if is_port_free(rest_port) and is_port_free(mcp_port):
                            print(f"[SABER] Ports released after {(i + 1) * 0.5:.1f}s", flush=True)
                            break
                    else:
                        print("[SABER] Warning: Ports may still be in use after 15s", flush=True)

                    running_domain = None  # Clear so we start fresh below
                else:
                    # No rebuild - reuse existing domain
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
                    register_domain(
                        domain_slug=domain_slug,
                        controller=controller,
                        context=context,
                        ownership=False,  # Don't stop it on cleanup
                        rest_port=rest_port,
                        mcp_port=mcp_port,
                    )
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
            # Use default concurrency of 8 for preflight environment validation
            if run_preflight:
                await run_preflight_check(
                    domain_slug=domain_slug,
                    domains_root=domains_root,
                    concurrency=8,  # Default for preflight validation
                    timeout=180,
                )

            # Create controller for host-based server deployment
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
            register_domain(
                domain_slug=domain_slug,
                controller=controller,
                context=context,
                ownership=True,  # We started it, we can stop it
                rest_port=rest_port,
                mcp_port=mcp_port,
            )

        logger.info(f"SABER domain '{domain_slug}' started, waiting for health checks")
        print(f"[SABER] Domain '{domain_slug}' started, beginning REST API health check...", flush=True)

        # Wait for server health - polls indefinitely every 15s until healthy
        assert context is not None  # Help mypy understand context is not None
        await wait_for_server_health(
            context.rest_url,
            domain=domain_slug,
            domains_root=Path(domains_root),
        )

        print(f"[SABER] Domain '{domain_slug}' health check passed!", flush=True)
        logger.info(f"SABER domain '{domain_slug}' is healthy, querying tasks")

        # Query REST API for benchmark info
        rest_client = SABERRestClient(context.rest_url, request_timeout=SandboxTimeouts.REST_API_SECONDS)
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
            tasks_to_load = apply_task_filter(
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
        orchestrations: dict[str, list[str]] = {}
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
        saber_solver = create_saber_solver(agent_name, agent_factory, role_config, skills_dir, agent_persona)

        # Construct Task with SABERDataset, sandbox config, solver, and scorer
        # The SABERDataset automatically handles orchestration-aware slicing
        # The sandbox config points to SABERSandboxEnvironment
        # The scorer performs client-side evaluation after agent execution

        # Check if role_config provides all models (allows bypassing --model requirement)
        task_model: str | None = None
        if role_config and all_roles_have_models(role_config):
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
        # NOTE: We intentionally do NOT set task_model = f"agent/{agent_name}" here anymore.
        # This was overriding the --model flag from the command line.
        # Instead, we let Inspect AI use whatever model the user specified via --model.
        # The agent solver will use active_model() to get the actual model for BYOK config.

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
                    "cleanup": stop_saber_after,  # Pass cleanup flag to sandbox
                    "enable_debug_logging": enable_debug_logging,  # Enable detailed lifecycle debug logging
                },
            ),
            solver=saber_solver,
            scorer=saber_scorer(score_aggregation_override=score_aggregation),  # Client-side evaluation
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
        should_cleanup = unregister_domain_on_failure(domain_slug)

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
