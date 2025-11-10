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

Component 1 of the SABER Domain Task factory pattern.
"""

import asyncio
import fnmatch
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import anyio
from inspect_ai import Task
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ..client.api.rest_client import SABERRestClient
from ..client.inspect_ai.saber_dataset import create_saber_dataset
from ..client.inspect_ai.saber_scorer import saber_scorer
from ..logging_config import LogCategory, get_saber_logger
from ..models import BenchmarkInfo
from .agents import AgentNotFoundError, SABERAgentRegistry
from .server import DomainContext, DomainController

logger = get_saber_logger(LogCategory.AGENT, __name__)

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
        max_concurrent_episodes: int = 8,
        **kwargs: Any,
    ) -> Task:
        """Task callable invoked by Inspect AI with CLI parameters.

        Args:
            rest_port: REST API port (default: 8000)
            mcp_port: MCP API port (default: 8001)
            task_filter: Optional task filter (exact match or glob pattern)
            agent: Agent implementation to use (default: domain-specific default)
            log_level: Logging level for domain services (default: "INFO")
            build: Build missing images before starting (default: False)
            rebuild: Remove and rebuild images matching this prefix (e.g., 'server')
            rebuild_all: Remove and rebuild all images (default: False)
            stop_saber_after: Stop SABER domain after task completes (default: False).
                If False, server stays running for faster re-runs.
            max_concurrent_episodes: Max concurrent episodes (default: 8, 0 = unlimited)
            **kwargs: Additional parameters passed through

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
                **kwargs,
            )
        except Exception as e:
            # Ensure we have a clean error message
            if isinstance(e, PrerequisiteError):
                raise
            raise PrerequisiteError(f"Failed to create SABER task for domain '{domain_slug}': {e}") from e

    return task_callable


def _create_saber_solver(agent_name: str, agent_factory: Callable) -> Solver:
    """Create solver with SABER agent that extracts prompts from metadata.

    This solver:
    1. Extracts the three prompts from sample metadata
    2. Uses saber_tools() to get MCP client from sandbox
    3. Creates agent with prompts and tools (using provided factory)
    4. Executes the agent

    Args:
        agent_name: Name of the agent implementation
        agent_factory: Callable that creates the agent with prompts

    Returns:
        Solver that runs SABER agent with dynamic prompts
    """

    @solver  # type: ignore[misc]
    def saber_agent_solver() -> Solver:
        """SABER agent solver with dynamic prompt injection."""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            """Execute SABER agent with prompts from metadata."""

            # Extract prompts from sample metadata
            metadata = state.metadata
            instruction_prompt = metadata.get("instruction_prompt")
            assistant_prompt = metadata.get("assistant_prompt")
            submit_prompt = metadata.get("submit_prompt")

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
                    "task_id": metadata.get("task_id"),
                    "instruction_length": len(instruction_prompt),
                    "assistant_length": len(assistant_prompt),
                    "submit_length": len(submit_prompt),
                },
            )

            # Create agent using the factory
            # agent_factory() returns a function that accepts prompts
            # Call it with prompts to get the actual agent
            create_with_prompts = agent_factory()
            agent = create_with_prompts(
                instruction_prompt=instruction_prompt,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            )

            # Execute the agent
            logger.debug(
                f"Executing SABER agent '{agent_name}'", extra={"agent": agent_name, "task_id": metadata.get("task_id")}
            )
            result = await agent(state)

            logger.info(
                f"SABER agent '{agent_name}' execution complete",
                extra={
                    "agent": agent_name,
                    "task_id": metadata.get("task_id"),
                    "completion_length": len(result.output.completion) if result.output.completion else 0,
                },
            )

            return result

        return solve

    return saber_agent_solver()


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
    **kwargs: Any,
) -> Task:
    """Start SABER domain and load tasks as dataset.

    This async helper:
    1. Validates parameters
    2. Checks for concurrent domain usage (process-local registry)
    3. Performs preflight check for already-running instances
    4. Starts DomainController with health checks
    5. Waits for server health with retry/backoff
    6. Queries REST API for benchmark info
    7. Applies task_filter (exact + glob)
    8. Converts to dataset with pre-assigned IDs
    9. Returns Task with MemoryDataset and sandbox config

    Cleanup is guaranteed via try/finally on all failure paths.

    Args:
        domain_slug: Domain to start
        domains_root: Path to domains directory
        rest_port: REST API port
        mcp_port: MCP API port
        task_filter: Optional task filter pattern
        agent_name: Name of agent implementation to use
        log_level: Logging level
        build: Optional build filter
        rebuild: Optional rebuild filter
        compose_template_path: Optional compose template
        stop_saber_after: Stop domain after evaluation
        max_concurrent_episodes: Max concurrent episodes
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

        # Convert to dataset with pre-assigned IDs
        samples: List[Sample] = await create_saber_dataset(tasks_to_load)

        logger.info(
            f"Created dataset with {len(samples)} samples",
            extra={
                "domain": domain_slug,
                "sample_count": len(samples),
                "task_count": len(tasks_to_load),
            },
        )

        # Create solver with the resolved agent
        saber_solver = _create_saber_solver(agent_name, agent_factory)

        # Construct Task with MemoryDataset, sandbox config, solver, and scorer
        # The sandbox config points to SABERSandboxEnvironment
        # The scorer performs client-side evaluation after agent execution
        task = Task(
            dataset=MemoryDataset(samples),
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
                },
            ),
            solver=saber_solver,
            scorer=saber_scorer(),  # Client-side evaluation
            # SABER defaults for evaluation behavior
            # These can be overridden via inspect eval CLI flags
            fail_on_error=False,  # Continue evaluation even if samples fail (use --fail-on-error to override)
            epochs=1,  # Run each sample once by default (use --epochs N to override)
        )

        logger.info(
            f"SABER task construction complete for domain '{domain_slug}'",
            extra={
                "domain": domain_slug,
                "sample_count": len(samples),
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
    task_filter: str,
    domain_slug: str,
) -> List[Any]:
    """Apply task filter with exact and glob pattern matching.

    Supports:
    - Exact match: task_filter="labyrinth_linguist_task_hard"
    - Glob pattern: task_filter="labyrinth_*"
    - Glob pattern: task_filter="*_hard"

    Args:
        tasks: List of TaskInfo objects from server
        task_filter: Filter pattern (exact or glob)
        domain_slug: Domain slug for error messages

    Returns:
        Filtered list of TaskInfo objects

    Raises:
        PrerequisiteError: If no tasks match the filter
    """
    # Try exact match first
    exact_matches = [t for t in tasks if t.task_id == task_filter]
    if exact_matches:
        return exact_matches

    # Try glob pattern
    glob_matches = [t for t in tasks if fnmatch.fnmatch(t.task_id, task_filter)]
    if glob_matches:
        return glob_matches

    # No matches - provide helpful error
    available_ids = [t.task_id for t in tasks]
    raise PrerequisiteError(
        f"No tasks matched filter '{task_filter}' in domain '{domain_slug}'.\n\n"
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
