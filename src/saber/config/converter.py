"""Convert TaskConfig objects to inspect_ai Sample objects.

This module bridges SABER's task configuration layer with the
inspect_ai evaluation framework by transforming each ``TaskConfig``
into a ``Sample`` that can be consumed by an inspect_ai ``Task``.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from saber.config.models import (
    DomainCriteria,
    LLMJudgeCriteria,
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
    TaskConfig,
    ToolCallCriteria,
)
from saber.prompts.renderer import PromptRenderer

_PROMPT_FIELD_MAP: dict[str, str] = {
    "instruction": "instruction_prompt",
    "assistant": "assistant_prompt",
}
# Mapping from render key (used in ``PromptRenderer.render_all_prompts``) to metadata key.


def tasks_to_samples(
    tasks: list[TaskConfig],
    domain_root: Path,
    prompt_renderer: PromptRenderer,
) -> list[Sample]:
    """Convert a list of TaskConfig objects to inspect_ai Samples.

    For each task:
    1. Render the instruction prompt using PromptRenderer.
    2. Build a Sample with system message (instruction) + user message (description).
    3. Extract a target answer from the submission eval config.
    4. Populate metadata with all task config for downstream scorers.
    5. Resolve sandbox environment to a Docker Compose file path.
    6. Resolve initial_files to domain data/ directory.

    Args:
        tasks: Task configurations to convert.
        domain_root: Root directory of the domain.
        prompt_renderer: Renderer for Jinja2 prompt templates.

    Returns:
        List of inspect_ai Sample objects.
    """
    return [_task_to_sample(task, domain_root, prompt_renderer) for task in tasks]


def _task_to_sample(
    task: TaskConfig,
    domain_root: Path,
    prompt_renderer: PromptRenderer,
) -> Sample:
    """Convert a single TaskConfig to an inspect_ai Sample."""
    rendered_prompts = prompt_renderer.render_all_prompts(task.prompts, task)
    rendered_instruction = rendered_prompts.get("instruction", "")

    return Sample(
        id=task.task_id,
        input=[
            ChatMessageSystem(content=rendered_instruction),
            ChatMessageUser(content=task.description),
        ],
        target=_extract_target(task),
        metadata=_build_metadata(task, rendered_prompts),
        sandbox=_resolve_sandbox(task, domain_root),
        files=_resolve_files(task, domain_root),
        setup=task.setup,
    )


def _extract_target(task: TaskConfig) -> str:
    """Extract a target answer from the task's scoring configuration.

    Finds the first scorer with ``target == ScorerTarget.SUBMISSION``
    and extracts the expected answer.

    Args:
        task: The task configuration.

    Returns:
        The target answer string.
    """
    return _extract_target_from_scorer(task.scorers)


def _extract_target_from_scorer(scorers: tuple[ScorerConfig, ...]) -> str:
    """Extract a target from the first submission scorer."""
    for scorer in scorers:
        if scorer.target == ScorerTarget.SUBMISSION:
            return _answer_from_criteria(scorer.criteria)
    return ""


def _answer_from_criteria(
    criteria: StaticCriteria | LLMJudgeCriteria | ToolCallCriteria | DomainCriteria,
) -> str:
    """Extract an answer string from a criteria object."""
    if isinstance(criteria, StaticCriteria):
        return criteria.expected_answers[0] if criteria.expected_answers else ""
    if isinstance(criteria, LLMJudgeCriteria):
        return ""
    return ""


def _resolve_sandbox(task: TaskConfig, domain_root: Path) -> tuple[str, str] | None:
    """Resolve a sandbox name to a Docker Compose file path.

    Maps ``sandbox_environment: "labyrinth_linguist_sandbox"`` to
    ``("docker", "<domain_root>/compose/labyrinth_linguist_sandbox.compose.yaml")``.

    Tries ``.compose.yaml`` first (required by Inspect AI's
    ``is_compose_yaml`` pattern), then falls back to ``.compose.yml``
    for backwards compatibility with existing domains.

    Args:
        task: The task configuration.
        domain_root: Root directory of the domain.

    Returns:
        A ``("docker", path)`` tuple, or ``None`` if the task has no sandbox.
    """
    if task.sandbox is None:
        return None

    # Prefer .compose.yaml (Inspect AI's is_compose_yaml regex requires it)
    compose_path = domain_root / "compose" / f"{task.sandbox}.compose.yaml"
    if not compose_path.exists():
        compose_path = domain_root / "compose" / f"{task.sandbox}.compose.yml"
    return ("docker", str(compose_path))


def _resolve_files(task: TaskConfig, domain_root: Path) -> dict[str, str]:
    """Resolve initial_files paths relative to the domain root directory.

    Transforms ``{"dest_path": "source_file"}`` into
    ``{"dest_path": "<domain_root>/source_file"}``.

    Args:
        task: The task configuration.
        domain_root: Root directory of the domain.

    Returns:
        A mapping from destination paths to resolved source paths.
    """
    if not task.initial_files:
        return {}

    return {dest: str(domain_root / source) for dest, source in task.initial_files.items()}


def _build_metadata(task: TaskConfig, rendered_prompts: dict[str, str]) -> dict[str, object]:
    """Build the metadata dictionary carried through to inspect_ai scorers.

    Includes both prompt *paths* (for reference) and *rendered* prompt strings
    (for solver_factory consumption via ``state.metadata``).

    Args:
        task: The task configuration.
        rendered_prompts: Mapping of prompt name to rendered string
            (from ``PromptRenderer.render_all_prompts``).

    Returns:
        A metadata dictionary with all task configuration fields.
    """
    ctx_dump = task.initial_context.model_dump()

    common: dict[str, object] = {
        "task_id": task.task_id,
        "title": task.title,
        "description": task.description,
        "initial_context": ctx_dump,
        "prompts": task.prompts.model_dump(),
        "tools": {k: v.model_dump() for k, v in task.tools.items()},
        "max_steps": task.max_steps,
        "tool_call_limit": task.max_steps,
        "aggregation": task.aggregation.value,
        **{meta_key: rendered_prompts.get(render_key, "") for render_key, meta_key in _PROMPT_FIELD_MAP.items()},
    }

    common["scorers"] = [s.model_dump(by_alias=True) for s in task.scorers]
    common["scoring_aggregation"] = (
        task.scoring_aggregation.model_dump() if task.scoring_aggregation is not None else None
    )

    # Promote all string-valued initial_context keys to top-level metadata.
    # This allows compose env var interpolation via SAMPLE_METADATA_* keys
    # (resolve_config_environment only reads top-level metadata keys).
    # Non-string values (dicts, ints, bools) are skipped — only strings
    # are useful as Docker Compose environment variable values.
    for key, value in ctx_dump.items():
        if isinstance(value, str) and key not in common:
            common[key] = value

    return common
