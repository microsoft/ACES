# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Scorer factory — creates inspect_ai ``@scorer`` instances from YAML task config.

Each :class:`TaskConfig` produces N unit scorers + 1 aggregate scorer.
All unit scorers share the same creation path via
``_create_unit_scorer_from_config``.

Batch optimization for LLM-judge checkpoint scorers lives exclusively
in ``compute_task_aggregate`` (called by ``saber_overall``).  Individual
unit scorers are cache-or-run: they check the unified cache first
(populated by ``saber_overall``) and fall back to running their
strategy directly when no cache exists.

When a domain loads many YAML tasks into one inspect task, registering
every per-task scorer directly would cause inspect_ai to open scorer spans
for all of them on every sample. ``create_runtime_scorers()`` avoids that
by registering one runtime scorer per logical scorer name
(``submission``, ``checkpoint_1``, etc.) and dispatching to the current
sample's task config at call time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState

from saber.agents.bridge_events import SABER_BRIDGE_TOOL_STEPS_KEY
from saber.config.models import (
    ScorerConfig,
    ScorerTarget,
    SubmissionFallback,
    SubmissionSourceType,
    TaskConfig,
)
from saber.scoring.aggregation import (
    aggregate_scorer_results,
    build_scorer_summary,
)
from saber.scoring.context import ScoringContext, ToolStep
from saber.scoring.registry import ScoringStrategyRegistry
from saber.scoring.templates import TemplateRenderer
from saber.scoring.trajectory import extract_tool_steps


class StrategyName(StrEnum):
    """Built-in scoring strategy names.

    Used for factory-internal comparisons only.
    ``ScorerConfig.strategy`` remains ``str`` for YAML flexibility.
    """

    STATIC = "static"
    LLM_JUDGE = "llm_judge"
    NONE = "none"
    STATIC_JACCARD = "static_jaccard"
    TOOL_CALL = "tool_call"
    TOOL_CALL_COUNT = "tool_call_count"


@dataclass(frozen=True)
class CacheKeys:
    """Namespaced metadata cache keys for the scoring pipeline."""

    _UNIFIED_PREFIX: ClassVar[str] = "_saber_all_scores_"

    @staticmethod
    def unified_scores(task_id: str) -> str:
        """Key for the unified per-scorer result cache in state.metadata."""
        return f"{CacheKeys._UNIFIED_PREFIX}{task_id}"


def _is_batch_eligible(sc: ScorerConfig) -> bool:
    """Return True if *sc* should be scored via the batched LLM judge pipeline.

    Only LLM judge checkpoint (non-submission) scorers qualify.
    """
    return bool(sc.strategy == StrategyName.LLM_JUDGE and sc.target != ScorerTarget.SUBMISSION)


logger = logging.getLogger("saber.scoring.factory")


def _bridge_tool_steps_from_metadata(
    metadata: dict[str, object],
    sample_store: object | None = None,
) -> tuple[ToolStep, ...] | None:
    """Return generic bridged tool steps from state metadata or sample store."""
    raw_steps = metadata.get(SABER_BRIDGE_TOOL_STEPS_KEY)
    if not isinstance(raw_steps, list) and sample_store is not None:
        raw_steps = sample_store.get(SABER_BRIDGE_TOOL_STEPS_KEY, None)
    if not isinstance(raw_steps, list):
        try:
            from inspect_ai.util import store

            stored_steps = store().get(SABER_BRIDGE_TOOL_STEPS_KEY, None)
        except Exception:
            stored_steps = None
        raw_steps = stored_steps
    if not isinstance(raw_steps, list):
        return None

    steps: list[ToolStep] = []
    for idx, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            continue
        raw_input = raw.get("tool_input")
        tool_input = raw_input if isinstance(raw_input, dict) else {}
        try:
            step_number = int(raw.get("step_number", len(steps) + 1))
        except (TypeError, ValueError):
            step_number = len(steps) + 1
        steps.append(
            ToolStep(
                step_number=step_number or idx + 1,
                tool_name=str(raw.get("tool_name") or ""),
                tool_input=tool_input,
                output=str(raw.get("output") or ""),
                is_error=bool(raw.get("is_error", False)),
                error_type=(
                    str(raw["error_type"])
                    if raw.get("error_type") is not None
                    else None
                ),
                assistant_message=(
                    str(raw["assistant_message"])
                    if raw.get("assistant_message") is not None
                    else None
                ),
                reasoning=(
                    str(raw["reasoning"])
                    if raw.get("reasoning") is not None
                    else None
                ),
            )
        )
    return tuple(steps) if steps else None


def _task_from_state(
    task_map: dict[str, TaskConfig],
    state: TaskState,
) -> TaskConfig | None:
    """Return the task config for the current sample state, if known."""
    sample_task_id = state.metadata.get("task_id", "")
    if not isinstance(sample_task_id, str):
        return None
    return task_map.get(sample_task_id)


def _scorer_config_by_name(task: TaskConfig, scorer_name: str) -> ScorerConfig | None:
    """Return the task-local scorer config matching *scorer_name*, if any."""
    for task_scorer in task.scorers:
        if task_scorer.scorer_name == scorer_name:
            return task_scorer
    return None


class SubmissionSourceError(Exception):
    """Raised when a scorer's ``submission_source`` cannot be resolved."""


async def _resolve_submission(state: TaskState, scorer: ScorerConfig) -> str:
    """Resolve the submission text for a scorer.

    Default behavior (when ``scorer.submission_source`` is None or
    ``type=completion``): returns ``state.output.completion`` — the
    agent's final assistant message text.

    When ``scorer.submission_source.type == file``: reads the configured
    sandbox file via inspect_ai's sandbox API and returns its contents.
    If the file is missing and ``fallback == completion``, falls back to
    the chat message; otherwise raises :class:`SubmissionSourceError`.

    See :class:`saber.config.models.SubmissionSource` for the contract.
    """
    src = scorer.submission_source
    if src is None or src.type == SubmissionSourceType.COMPLETION:
        return state.output.completion if state.output else ""

    if src.type == SubmissionSourceType.FILE:
        from inspect_ai.util import sandbox as sandbox_env

        try:
            content: str = await sandbox_env().read_file(src.path)
            if src.encoding != "utf-8":
                # read_file returns str (UTF-8 decoded); re-decode if
                # a different encoding was requested.
                content = content.encode("utf-8").decode(src.encoding)
            return content
        except FileNotFoundError:
            if src.fallback == SubmissionFallback.COMPLETION:
                return state.output.completion if state.output else ""
            raise SubmissionSourceError(
                f"submission_source file not found: {src.path}. Set fallback: completion to use chat message instead."
            ) from None
        except (PermissionError, OSError, UnicodeDecodeError) as exc:
            if src.fallback == SubmissionFallback.COMPLETION:
                return state.output.completion if state.output else ""
            raise SubmissionSourceError(f"submission_source file read failed: {src.path}: {exc}") from exc

    raise SubmissionSourceError(  # pragma: no cover — guarded by enum
        f"Unknown submission_source.type: {src.type!r}"
    )


async def _make_scoring_context(
    *,
    state: TaskState,
    target: Target,
    task_id: str,
    domain_slug: str,
    scorer: ScorerConfig,
    tool_steps: tuple[ToolStep, ...] | None = None,
) -> ScoringContext:
    """Build a ScoringContext from a TaskState.

    DRY helper used by both ``ScorerFactory`` methods and the closures
    created for unit / batch scorers.

    Args:
        state: Current TaskState for this sample.
        target: The target answer.
        task_id: Unique task identifier.
        domain_slug: Domain name (directory basename).
        scorer: Scorer configuration (may specify ``submission_source``).
        tool_steps: Pre-extracted tool steps; extracted from
            ``state.messages`` when *None*.

    Returns:
        Fully-populated :class:`ScoringContext`.
    """
    return ScoringContext(
        submission=await _resolve_submission(state, scorer),
        tool_steps=(
            tool_steps
            if tool_steps is not None
            else _bridge_tool_steps_from_metadata(state.metadata or {}, state.store)
            or tuple(extract_tool_steps(state.messages))
        ),
        messages=tuple(state.messages),
        target=target.text,
        task_id=task_id,
        domain=domain_slug,
        metadata=state.metadata or {},
        scorer=scorer,
    )


class ScorerFactory:
    """Creates inspect_ai ``@scorer`` instances from YAML task config.

    Each :class:`TaskConfig` produces its own namespaced scorers.
    Scorer names use dot-separated ``task_id.component`` format.
    """

    def __init__(
        self,
        domain_root: Path,
        registry: ScoringStrategyRegistry | None = None,
    ) -> None:
        self._domain_root = domain_root
        self._domain_slug = domain_root.name
        self._registry = registry or ScoringStrategyRegistry()
        self._renderer = TemplateRenderer(domain_root / "prompts")

    def create_scorers(self, task: TaskConfig) -> list[Scorer]:
        """Build all scorers for *task* from its YAML config.

        Returns an ordered list — aggregate scorer is always last so it
        can read ``state.scores`` populated by the preceding scorers.

        Args:
            task: The task configuration to build scorers for.

        Returns:
            Ordered list of :class:`Scorer` instances.
        """
        return self._create_scorers_from_config(task)

    def create_runtime_scorers(self, tasks: list[TaskConfig]) -> list[Scorer]:
        """Build the inspect runtime scorer list for a multi-task domain.

        ``inspect_ai`` applies every scorer in ``Task.scorer`` to every sample
        and emits scorer span events before the scorer has a chance to skip.
        Registering one scorer per YAML task therefore scales transcript
        traffic with the total number of tasks rather than the current sample.

        To keep runtime scoring bounded, this method registers:

        - ``saber_overall`` once
        - one unit scorer per distinct logical scorer name across tasks
        - one aggregate scorer that dispatches by ``state.metadata["task_id"]``

        Args:
            tasks: All loaded task configs for the current inspect task.

        Returns:
            Ordered list of runtime scorers for ``Task.scorer``.
        """
        task_map = {task.task_id: task for task in tasks}
        logical_names: list[str] = []
        seen_names: set[str] = set()

        for task in tasks:
            self._validate_strategies(task)
            self._validate_aggregation_refs(task)
            for task_scorer in task.scorers:
                if task_scorer.scorer_name not in seen_names:
                    seen_names.add(task_scorer.scorer_name)
                    logical_names.append(task_scorer.scorer_name)

        runtime_scorers: list[Scorer] = [self.create_overall_scorer(tasks)]
        runtime_scorers.extend(self._create_runtime_unit_scorer(name, task_map) for name in logical_names)
        runtime_scorers.append(self._create_runtime_aggregate_scorer(task_map))
        return runtime_scorers

    async def compute_task_aggregate(
        self,
        task: TaskConfig,
        state: TaskState,
        target: Target,
    ) -> Score:
        """Compute a task's aggregate score by running all strategies inline.

        Unlike the aggregate ``@scorer`` (which reads ``state.scores``),
        this method independently runs every scoring strategy and then
        aggregates the results.  It is designed to be called from the
        ``saber_overall`` scorer **before** individual per-task scorers
        have populated ``state.scores``.

        Results are written to a unified cache in ``state.metadata`` so
        that individual per-task unit scorers hit the cache and avoid
        duplicate strategy calls.

        Args:
            task: Task configuration with scorers and aggregation config.
            state: Current TaskState for this sample.
            target: The target answer.

        Returns:
            Aggregate :class:`Score` with normalized value.
        """
        from saber.scoring.batch import score_checkpoints_llm_batch

        self._validate_strategies(task)

        task_id = task.task_id
        tool_steps = (
            _bridge_tool_steps_from_metadata(state.metadata or {}, state.store)
            or tuple(extract_tool_steps(state.messages))
        )

        # --- Run all strategies and collect per-scorer results -----------
        individual_scores: dict[str, Score] = {}
        llm_batch_configs: list[ScorerConfig] = []

        for sc in task.scorers:
            if _is_batch_eligible(sc):
                # Checkpoint LLM judge scorers → batched
                llm_batch_configs.append(sc)
            else:
                # Submission LLM judges, static, tool_call, etc. → individual
                strategy = self._registry.get(sc.strategy)
                ctx = await _make_scoring_context(
                    state=state,
                    target=target,
                    task_id=task_id,
                    domain_slug=self._domain_slug,
                    scorer=sc,
                    tool_steps=tool_steps,
                )
                individual_scores[sc.scorer_name] = await strategy.score(ctx, renderer=self._renderer)

        # Batch LLM judge checkpoint scorers
        if llm_batch_configs:
            contexts = [
                await _make_scoring_context(
                    state=state,
                    target=target,
                    task_id=task_id,
                    domain_slug=self._domain_slug,
                    scorer=cfg,
                    tool_steps=tool_steps,
                )
                for cfg in llm_batch_configs
            ]
            batch_results = await score_checkpoints_llm_batch(
                contexts=contexts,
                renderer=self._renderer,
            )

            for cfg in llm_batch_configs:
                if cfg.scorer_name in batch_results:
                    individual_scores[cfg.scorer_name] = batch_results[cfg.scorer_name]
                else:
                    logger.warning(
                        "Scorer %r not found in batch results for task %r — defaulting to 0.0",
                        cfg.scorer_name,
                        task_id,
                    )
                    individual_scores[cfg.scorer_name] = Score(value=0.0)

        # --- Cache ALL results for individual scorers to read ----------
        state.metadata[CacheKeys.unified_scores(task_id)] = individual_scores

        # --- Aggregate ---------------------------------------------------
        raw_scores = {name: score.as_float() for name, score in individual_scores.items()}
        agg_config = task.scoring_aggregation
        normalized, scorer_details = aggregate_scorer_results(
            scorer_configs=task.scorers,
            raw_scores=raw_scores,
            agg_config=agg_config,
        )

        return Score(
            value=normalized,
            answer=state.output.completion if state.output else "",
            explanation=build_scorer_summary(scorer_details),
            metadata={
                "task_id": task_id,
                "scorer_details": scorer_details,
                "saber_score": normalized,
                "aggregation": (agg_config.strategy.value if agg_config else "average"),
            },
        )

    def _validate_strategies(self, task: TaskConfig) -> None:
        """Fail fast if any scorer references an unregistered strategy.

        Raises:
            ValueError: If any scorer's strategy is not in the registry.
        """
        unknown = {s.strategy for s in task.scorers if s.strategy not in self._registry.available()}
        if unknown:
            raise ValueError(
                f"Task {task.task_id!r} references unregistered scoring strategies: "
                f"{sorted(unknown)}. Available: {self._registry.available()}. "
                f"Register custom strategies via extra_strategies parameter."
            )

    def create_overall_scorer(self, tasks: list[TaskConfig]) -> Scorer:
        """Create the ``saber_overall`` scorer — the headline metric.

        This scorer is placed **first** in the scorer list so that
        inspect_ai uses it as ``results.scores[0]`` (the headline).
        It computes the per-task aggregate independently by running all
        scoring strategies inline, then returns the aggregate value.

        Because it runs **before** per-task scorers, it caches LLM-judge
        results in ``state.metadata`` so subsequent batch scorers skip
        duplicate LLM calls.

        Args:
            tasks: All task configurations in this evaluation.

        Returns:
            A :class:`Scorer` named ``saber_overall`` using ``mean()``
            metric so the headline averages per-sample aggregate values.
        """
        task_map = {task.task_id: task for task in tasks}
        factory = self

        @scorer(metrics=[mean(), stderr()], name="saber_overall")  # type: ignore[misc]
        def overall() -> Scorer:
            async def do_score(state: TaskState, target: Target) -> Score:
                sample_task_id = state.metadata.get("task_id", "")
                task_cfg = task_map.get(sample_task_id)
                if task_cfg is None:
                    return Score(
                        value=0.0,
                        explanation=(f"No task config for {sample_task_id!r}"),
                    )
                return await factory.compute_task_aggregate(
                    task_cfg,
                    state,
                    target,
                )

            return do_score

        return overall()

    def _validate_aggregation_refs(self, task: TaskConfig) -> None:
        """Fail fast if aggregation references scorer names not in task.scorers.

        Raises:
            ValueError: If ``scoring_aggregation.scores`` contains a name
                not present in ``task.scorers``.
        """
        if task.scoring_aggregation is None:
            return
        known = {sc.scorer_name for sc in task.scorers}
        referenced: set[str] = set()
        for entry in task.scoring_aggregation.scores:
            if isinstance(entry, str):
                referenced.add(entry)
            else:
                referenced.update(entry)
        unknown = referenced - known
        if unknown:
            raise ValueError(
                f"Task {task.task_id!r} aggregation references unknown scorer names: "
                f"{sorted(unknown)}. Available scorers: {sorted(known)}."
            )

    def _create_scorers_from_config(self, task: TaskConfig) -> list[Scorer]:
        """Build scorers from task.scorers config list.

        Each :class:`ScorerConfig` produces one unit scorer via
        ``_create_unit_scorer_from_config``.  An aggregate scorer is
        appended last so it can read ``state.scores`` populated by
        the preceding scorers.

        Returns ordered list with aggregate scorer last.
        """
        self._validate_strategies(task)
        self._validate_aggregation_refs(task)

        scorers: list[Scorer] = []

        for sc in task.scorers:
            scorers.append(
                self._create_unit_scorer_from_config(
                    name=f"{task.task_id}.{sc.scorer_name}",
                    scorer_config=sc,
                    task=task,
                )
            )

        scorers.append(self._create_aggregate_scorer_from_config(task))
        return scorers

    def _create_unit_scorer_from_config(
        self,
        name: str,
        scorer_config: ScorerConfig,
        task: TaskConfig,
    ) -> Scorer:
        """Create a single atomic scorer from a ScorerConfig."""
        strategy = self._registry.get(scorer_config.strategy)
        task_id = task.task_id
        renderer = self._renderer
        domain_slug = self._domain_slug

        @scorer(metrics=[accuracy(), stderr()], name=name)  # type: ignore[misc]
        def unit_scorer() -> Scorer:
            async def do_score(state: TaskState, target: Target) -> Score | None:
                if state.metadata.get("task_id") != task_id:
                    return None

                # Check unified cache (populated by saber_overall)
                unified = state.metadata.get(CacheKeys.unified_scores(task_id))
                if isinstance(unified, dict) and scorer_config.scorer_name in unified:
                    return unified[scorer_config.scorer_name]

                ctx = await _make_scoring_context(
                    state=state,
                    target=target,
                    task_id=task_id,
                    domain_slug=domain_slug,
                    scorer=scorer_config,
                )
                return await strategy.score(ctx, renderer=renderer)

            return do_score

        return unit_scorer()

    def _create_runtime_unit_scorer(
        self,
        scorer_name: str,
        task_map: dict[str, TaskConfig],
    ) -> Scorer:
        """Create a logical unit scorer that dispatches by sample task id."""
        renderer = self._renderer
        domain_slug = self._domain_slug

        @scorer(metrics=[accuracy(), stderr()], name=scorer_name)  # type: ignore[misc]
        def runtime_unit_scorer() -> Scorer:
            async def do_score(state: TaskState, target: Target) -> Score | None:
                task = _task_from_state(task_map, state)
                if task is None:
                    return None

                scorer_config = _scorer_config_by_name(task, scorer_name)
                if scorer_config is None:
                    return None

                task_id = task.task_id

                # Check unified cache (populated by saber_overall)
                unified = state.metadata.get(CacheKeys.unified_scores(task_id))
                if isinstance(unified, dict) and scorer_name in unified:
                    return unified[scorer_name]

                strategy = self._registry.get(scorer_config.strategy)
                ctx = await _make_scoring_context(
                    state=state,
                    target=target,
                    task_id=task_id,
                    domain_slug=domain_slug,
                    scorer=scorer_config,
                )
                return await strategy.score(ctx, renderer=renderer)

            return do_score

        return runtime_unit_scorer()

    def _create_aggregate_scorer_from_config(self, task: TaskConfig) -> Scorer:
        """Create aggregate scorer using task.scorers and task.scoring_aggregation."""
        task_id = task.task_id
        agg_name = f"{task_id}.aggregate"
        task_scorers = task.scorers
        agg_config = task.scoring_aggregation

        @scorer(metrics=[accuracy(), stderr()], name=agg_name)  # type: ignore[misc]
        def aggregate() -> Scorer:
            async def do_score(state: TaskState, target: Target) -> Score | None:
                if state.metadata.get("task_id") != task_id:
                    return None
                scores = state.scores or {}

                raw_scores = {
                    sc.scorer_name: scores.get(f"{task_id}.{sc.scorer_name}", Score(value=0.0)).as_float()
                    for sc in task_scorers
                }
                normalized, scorer_details = aggregate_scorer_results(
                    scorer_configs=task_scorers,
                    raw_scores=raw_scores,
                    agg_config=agg_config,
                )

                answer = state.output.completion if state.output else ""
                return Score(
                    value=normalized,
                    answer=answer,
                    explanation=build_scorer_summary(scorer_details),
                    metadata={
                        "task_id": task_id,
                        "scorer_details": scorer_details,
                        "saber_score": normalized,
                        "aggregation": (agg_config.strategy.value if agg_config else "average"),
                    },
                )

            return do_score

        return aggregate()

    def _create_runtime_aggregate_scorer(
        self,
        task_map: dict[str, TaskConfig],
    ) -> Scorer:
        """Create a logical aggregate scorer that dispatches by sample task id."""

        @scorer(metrics=[accuracy(), stderr()], name="aggregate")  # type: ignore[misc]
        def runtime_aggregate() -> Scorer:
            async def do_score(state: TaskState, target: Target) -> Score | None:
                task = _task_from_state(task_map, state)
                if task is None:
                    return None

                scores = state.scores or {}
                raw_scores = {
                    sc.scorer_name: scores.get(sc.scorer_name, Score(value=0.0)).as_float() for sc in task.scorers
                }
                normalized, scorer_details = aggregate_scorer_results(
                    scorer_configs=task.scorers,
                    raw_scores=raw_scores,
                    agg_config=task.scoring_aggregation,
                )

                answer = state.output.completion if state.output else ""
                return Score(
                    value=normalized,
                    answer=answer,
                    explanation=build_scorer_summary(scorer_details),
                    metadata={
                        "task_id": task.task_id,
                        "scorer_details": scorer_details,
                        "saber_score": normalized,
                        "aggregation": (
                            task.scoring_aggregation.strategy.value if task.scoring_aggregation else "average"
                        ),
                    },
                )

            return do_score

        return runtime_aggregate()
