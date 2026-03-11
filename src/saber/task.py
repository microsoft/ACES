"""High-level domain task factory.

Wires ConfigLoader + PromptRenderer + converter + ScorerFactory +
ToolRegistry + AgentRegistry + SaberSandboxEnvironment into a single
``inspect_ai.Task`` that domain ``@task`` functions call.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inspect_ai import Epochs, Task
from inspect_ai.approval import ApprovalPolicy
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Scorer
from inspect_ai.tool import Tool

from saber.agents.resolver import resolve_agent
from saber.agents.solver_factory import create_saber_solver
from saber.config.converter import tasks_to_samples
from saber.config.loader import ConfigLoader
from saber.environments import resolve_sandbox_spec
from saber.environments.images import parse_rebuild_param
from saber.environments.preflight import validate_domain
from saber.logging import configure_logging, get_logger
from saber.prompts.renderer import PromptRenderer
from saber.scoring.factory import ScorerFactory
from saber.scoring.registry import ScoringStrategyRegistry
from saber.scoring.strategies import SaberScoringStrategy
from saber.tools import ToolRegistry
from saber.tools.security import ToolSecurityConfig, build_tool_approval

logger = get_logger(__name__)


def _cli_bool(value: str | bool | None) -> bool:
    """Coerce a CLI ``-T`` flag to bool.

    ``inspect eval -T flag=true`` may deliver either ``True`` (bool,
    via ``yaml.safe_load``) or ``"true"`` (str) depending on the
    inspect_ai version.  This helper normalises both forms.
    """
    return value is True or value == "true"


def create_task(
    domain_root: Path | None = None,
    task_filter: str | None = None,
    dataset: str | None = None,
    agent: str = "react",
    sandbox_compose: str = "compose/sandbox.compose.yml",
    permanent_compose: str | None = None,
    permanent_project: str = "saber-permanent",
    extra_tools: dict[str, Callable[..., Tool]] | None = None,
    extra_strategies: dict[str, SaberScoringStrategy] | None = None,
    approval: list[ApprovalPolicy] | None = None,
    rebuild: str | None = None,
    run_preflight: str | bool = False,
    keep_permanent: str | bool = False,
    **kwargs: object,
) -> Task:
    """Create an inspect_ai Task from a SABER domain directory.

    Wires together all saber components:
    - ConfigLoader for YAML inheritance cascade
    - PromptRenderer for Jinja2 prompt templates
    - tasks_to_samples for Sample conversion
    - ScorerFactory for scoring
    - ToolRegistry for tool resolution
    - AgentRegistry for agent lookup
    - SaberSandboxEnvironment for permanent services

    Args:
        domain_root: Path to the domain directory (e.g., domains/excytin).
            If ``None``, auto-detected from the caller's file location.
        task_filter: Optional glob/comma-separated task name filter.
        dataset: Optional dataset name for filtering tasks. When set, only
            tasks whose ``dataset`` field matches are loaded. Falls back to
            ``default_dataset`` from ``global.yaml`` when ``None``.
            Pass ``"all"`` to bypass dataset filtering entirely and
            return every task regardless of its ``dataset`` value.

            .. note::

               Setup hooks receive the raw *dataset* value (including
               ``"all"``) but **not** the resolved ``default_dataset``
               fallback.  Hooks should prepare data for all tasks and
               let filtering happen inside :meth:`ConfigLoader.load_tasks`.
        agent: Agent name (default: "react"). Registered agents: react, copilot,
            claude_code.
        sandbox_compose: Relative path to sandbox compose file within domain.
        permanent_compose: Relative path to permanent services compose, or ``None``.
            Each domain passes this explicitly (default: ``None``).
        permanent_project: Docker Compose project name for permanent services.
        extra_tools: Optional mapping of tool name to factory callable for
            domain-specific tools (e.g., ``{"kql_query": kql_query}``).
            Merged with tools auto-discovered from ``<domain>/tools/``.
        extra_strategies: Optional mapping of strategy name to
            :class:`~saber.scoring.strategies.SaberScoringStrategy` instance.
            These are registered alongside the built-in strategies so that
            domain task YAMLs can reference them by name.  Merged with
            strategies auto-discovered from ``<domain>/scoring/``.
        approval: Optional pre-built list of approval policies.
            - ``list[ApprovalPolicy]`` to pass pre-built policies directly
            - ``None`` for auto-wiring from tool security configs or no approval
        rebuild: Optional rebuild parameter string. Accepted values:
            - ``None`` → no rebuild
            - ``"true"`` → rebuild all images
            - ``"false"`` → no rebuild
            - ``"name1,name2"`` → rebuild specific images
        run_preflight: Validate compose files before creating the task.
            Accepts ``True``, ``"true"`` (from CLI ``-T``), or ``False``.
            Errors cause a ``RuntimeError``; warnings are logged.
        keep_permanent: Keep permanent Docker Compose services alive after
            evaluation so the next run can reuse them.
            Accepts ``True``, ``"true"`` (from CLI ``-T``), or ``False``.

    Returns:
        Fully configured inspect_ai Task.

    Raises:
        AgentNotFoundError: If the agent name is not registered.
        FileNotFoundError: If domain_root or required config files don't exist.
        RuntimeError: If preflight validation finds errors.
        ValueError: If no tasks are found.
    """
    # 0a. Auto-detect domain_root from caller if not provided.
    #     Must happen BEFORE the kwargs-unwrap below so the recursive
    #     call gets the real domain path instead of saber's own source dir.
    if domain_root is None:
        caller_frame = inspect.stack()[1]
        domain_root = Path(caller_frame.filename).resolve().parent

    # -1. Unwrap double-nested kwargs from inspect eval-retry.
    #     When eval-retry reconstructs a task whose @task function uses **kwargs,
    #     inspect_ai stores the original kwargs dict under the key "kwargs",
    #     producing create_task(kwargs={...}) instead of create_task(**original).
    #     Detect this and recursively call with the flattened arguments.
    if "kwargs" in kwargs and isinstance(kwargs["kwargs"], dict):
        nested: dict[str, Any] = dict(kwargs.pop("kwargs"))  # type: ignore[call-overload]
        # Merge remaining extra kwargs (shouldn't normally exist)
        nested.update({k: v for k, v in kwargs.items() if k != "kwargs"})
        # Re-invoke with the unwrapped arguments (domain_root already resolved)
        return create_task(
            domain_root=domain_root,
            task_filter=nested.pop("task_filter", task_filter),
            dataset=nested.pop("dataset", dataset),
            agent=nested.pop("agent", agent),
            sandbox_compose=nested.pop("sandbox_compose", sandbox_compose),
            permanent_compose=nested.pop("permanent_compose", permanent_compose),
            permanent_project=nested.pop("permanent_project", permanent_project),
            extra_tools=nested.pop("extra_tools", extra_tools),
            extra_strategies=nested.pop("extra_strategies", extra_strategies),
            approval=nested.pop("approval", approval),
            rebuild=nested.pop("rebuild", rebuild),
            run_preflight=nested.pop("run_preflight", run_preflight),
            keep_permanent=nested.pop("keep_permanent", keep_permanent),
            **nested,
        )

    # 0. Normalise CLI flags
    run_preflight_bool = _cli_bool(run_preflight)
    keep_permanent_bool = _cli_bool(keep_permanent)

    # 0b. Configure saber logging
    configure_logging()

    # 0c. Run domain setup hooks (data downloads, task generation, etc.)
    from saber.hooks import run_setup_hooks
    from saber.setup_discovery import _discover_setup_hooks

    # Forward `dataset` into kwargs so setup hooks can scope downloads
    # to only the data needed for the selected dataset.
    setup_kwargs: dict[str, object] = dict(kwargs)
    if dataset is not None:
        setup_kwargs["dataset"] = dataset

    hooks = _discover_setup_hooks(domain_root, setup_kwargs)
    if hooks:
        hooks_result = run_setup_hooks(hooks, domain_root)
        if not hooks_result.all_succeeded:
            failed = "; ".join(f"{r.name}: {r.message}" for r in hooks_result.failed_hooks)
            raise RuntimeError(f"Setup hook(s) failed: {failed}")

    # 1. Load task configs from YAML
    config_root = _find_config_root(domain_root)
    loader = ConfigLoader(config_root)
    global_config = loader.load_global_config()
    tasks = loader.load_tasks(task_filter=task_filter, dataset=dataset)

    # 1b. Resolve permanent compose from global.yaml (explicit kwargs win)
    perm_env = global_config.permanent_environment
    effective_permanent_compose = (
        permanent_compose if permanent_compose is not None else (perm_env.compose if perm_env else None)
    )
    effective_permanent_project = (
        permanent_project
        if permanent_project != "saber-permanent"
        else (perm_env.name if perm_env else "saber-permanent")
    )

    # 0b. Compose preflight validation
    if run_preflight_bool:
        preflight_result = validate_domain(
            domain_root=domain_root,
            sandbox_compose=sandbox_compose,
            permanent_compose=effective_permanent_compose,
        )
        if not preflight_result.passed:
            error_summary = "; ".join(e.message for e in preflight_result.errors)
            raise RuntimeError(
                f"Compose preflight failed with {len(preflight_result.errors)} error(s): {error_summary}"
            )

    if not tasks:
        msg = f"No tasks found in {config_root}"
        if dataset:
            msg += f" for dataset '{dataset}'"
        if task_filter:
            msg += f" matching filter '{task_filter}'"
        raise ValueError(msg)

    # 2. Render prompts and convert to Sample objects
    prompts_dir = _find_prompts_dir(domain_root)
    renderer = PromptRenderer(prompts_dir)
    samples = tasks_to_samples(tasks, domain_root, renderer)

    # 3. Build scorers from task configs
    all_strategies = _discover_strategies(domain_root)
    if extra_strategies:
        all_strategies.update(extra_strategies)

    registry: ScoringStrategyRegistry | None = None
    if all_strategies:
        registry = ScoringStrategyRegistry()
        for name, strategy in all_strategies.items():
            registry.register(name, strategy)
    scorer_factory = ScorerFactory(domain_root=domain_root, registry=registry)

    # saber_overall goes FIRST so it becomes scores[0] (the headline).
    # It computes the aggregate independently and caches LLM results
    # in state.metadata so subsequent per-task scorers skip repeat calls.
    scorers: list[Scorer] = [scorer_factory.create_overall_scorer(tasks)]
    for task_cfg in tasks:
        scorers.extend(scorer_factory.create_scorers(task_cfg))

    # 4. Register domain tools
    all_extra_tools = _discover_tools(domain_root)
    if extra_tools:
        all_extra_tools.update(extra_tools)

    tool_registry = ToolRegistry()
    for tool_name, tool_factory in all_extra_tools.items():
        tool_registry.register(tool_name, tool_factory)

    # 5. Resolve agent
    agent_factory = resolve_agent(agent)
    solver = create_saber_solver(
        agent_name=agent,
        agent_factory=agent_factory,
        tool_registry=tool_registry,
        **kwargs,
    )

    # 6. Parse rebuild param
    rebuild_mode = parse_rebuild_param(rebuild)

    # 7. Resolve sandbox
    sandbox_spec = resolve_sandbox_spec(
        domain_root=domain_root,
        sandbox_compose=sandbox_compose,
        permanent_compose=effective_permanent_compose,
        permanent_project=effective_permanent_project,
        rebuild=rebuild_mode,
        keep_permanent=keep_permanent_bool,
    )

    # 8. Build approval policies from ALL tasks' security configs
    effective_approval: list[ApprovalPolicy] | None = None
    if approval is not None:
        # Explicit override — use as-is (empty list = no approval)
        effective_approval = approval if approval else None
    else:
        # Union of security configs across all tasks
        all_security_configs: dict[str, ToolSecurityConfig] = {}
        for task_cfg in tasks:
            for tool_name, tool_config in task_cfg.tools.items():
                if tool_config.security is not None:
                    # Later task configs override earlier ones for same tool
                    all_security_configs[tool_name] = tool_config.security
        if all_security_configs:
            effective_approval = build_tool_approval(all_security_configs)

    # Default per-sample time limit (seconds).  Can be overridden from
    # the CLI with ``--time-limit``.  The value propagates automatically
    # to bridge-based agents (copilot, claude_code) via sample_limits().
    _DEFAULT_TIME_LIMIT = 3600

    # inspect_ai retries transient errors (429, 5xx) using tenacity.
    # When max_retries is None the retry loop uses stop_never, giving
    # unlimited attempts with exponential backoff (capped at 30 min)
    # so samples survive prolonged rate-limit bursts.

    return Task(
        dataset=samples,
        solver=solver,
        scorer=scorers,
        sandbox=sandbox_spec,
        approval=effective_approval,
        config=GenerateConfig(max_retries=None),
        # Use max across all tasks as the ceiling; per-sample tightening
        # happens in the solver via state.tool_call_limit from metadata.
        tool_call_limit=max(t.max_steps for t in tasks),
        time_limit=_DEFAULT_TIME_LIMIT,
        # Empty reducer list suppresses the spurious "(mean)" display
        # suffix caused by an inspect_ai variable-shadowing bug in
        # resolve_reducer().  With reducer=[], metrics are computed
        # directly from per-sample scores without reduction.
        epochs=Epochs(1, reducer=[]),
    )


def _find_config_root(domain_root: Path) -> Path:
    """Find the config root directory within a domain.

    Expects flat layout: ``domain_root/tasks/``.

    Args:
        domain_root: Domain root directory.

    Returns:
        Path to directory containing tasks/ subdirectory.

    Raises:
        FileNotFoundError: If no tasks directory is found.
    """
    if (domain_root / "tasks").is_dir():
        return domain_root
    raise FileNotFoundError(f"No tasks/ directory found in {domain_root}")


def _find_prompts_dir(domain_root: Path) -> Path:
    """Find the prompts directory within a domain.

    Expects flat layout: ``domain_root/prompts/``.

    Args:
        domain_root: Domain root directory.

    Returns:
        Path to prompts directory.

    Raises:
        FileNotFoundError: If no prompts directory is found.
    """
    prompts = domain_root / "prompts"
    if prompts.is_dir():
        return prompts
    raise FileNotFoundError(f"No prompts/ directory found in {domain_root}")


def _discover_strategies(domain_root: Path) -> dict[str, SaberScoringStrategy]:
    """Auto-discover domain scoring strategies.

    Looks for ``<domain_root>/scoring/__init__.py`` with a
    ``get_strategies(domain_root: Path)`` function and calls it.

    Args:
        domain_root: Domain root directory.

    Returns:
        Mapping of strategy name to strategy instance, or empty dict.
    """
    scoring_init = domain_root / "scoring" / "__init__.py"
    if not scoring_init.is_file():
        return {}

    module = _import_domain_module(domain_root, "scoring")
    factory = getattr(module, "get_strategies", None)
    if factory is None:
        return {}

    result: dict[str, Any] = factory(domain_root)
    logger.debug("Auto-discovered %d scoring strategies from %s", len(result), scoring_init)
    return result


def _discover_tools(domain_root: Path) -> dict[str, Callable[..., Tool]]:
    """Auto-discover domain tools.

    Looks for ``<domain_root>/tools/__init__.py`` with a
    ``get_tools(domain_root: Path)`` function and calls it.

    Args:
        domain_root: Domain root directory.

    Returns:
        Mapping of tool name to tool factory, or empty dict.
    """
    tools_init = domain_root / "tools" / "__init__.py"
    if not tools_init.is_file():
        return {}

    module = _import_domain_module(domain_root, "tools")
    factory = getattr(module, "get_tools", None)
    if factory is None:
        return {}

    result: dict[str, Callable[..., Any]] = factory(domain_root)
    logger.debug("Auto-discovered %d domain tools from %s", len(result), tools_init)
    return result


def _import_domain_module(domain_root: Path, subpackage: str) -> object:
    """Import a domain subpackage by path.

    If the domain's parent directory is not on ``sys.path``, it is
    temporarily added so that relative imports within the package work.

    Args:
        domain_root: Domain root directory (e.g., ``domains/cti_realm``).
        subpackage: Subpackage name (``"scoring"`` or ``"tools"``).

    Returns:
        The imported module object.
    """
    domain_name = domain_root.name
    module_name = f"{domain_name}.{subpackage}"

    # Ensure parent dir is on sys.path for the import
    parent = str(domain_root.parent)
    added = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        added = True

    try:
        return importlib.import_module(module_name)
    finally:
        if added:
            sys.path.remove(parent)
