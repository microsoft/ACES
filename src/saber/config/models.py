"""Pydantic v2 data models for SABER configuration.

All models use ``ConfigDict(frozen=True)`` for immutability.
"""

from __future__ import annotations

import types
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saber.tools.security import ToolSecurityConfig

# ── Template Helpers ────────────────────────────────────────────────


def to_template_vars(model: BaseModel) -> dict[str, object]:
    """Convert a Pydantic model to Jinja2 template variables.

    Returns a flat dict where keys are field names and values are
    Python objects (not serialized). Extra fields from models with
    ``extra='allow'`` are included. Nested Pydantic models are preserved
    as objects for Jinja2 attribute access.
    """
    result: dict[str, object] = {}
    for name in model.model_fields:
        result[name] = getattr(model, name)
    if model.model_extra:
        result.update(model.model_extra)
    return result


# ── Domain-Specific Initial Context ─────────────────────────────────


class InitialContext(BaseModel):
    """Domain-specific initial context from YAML.

    Accepts arbitrary keys via ``extra='allow'``. Each domain defines
    its own fields (database_connection, vulnerability_info, etc.).
    Jinja2 templates access fields via attribute notation.

    Supports dict-like ``__contains__`` and ``__getitem__`` for
    attribute access via string keys.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    def __contains__(self, key: object) -> bool:
        """Support ``"key" in initial_context``."""
        if not isinstance(key, str):
            return False
        return key in (self.model_extra or {})

    def __getitem__(self, key: str) -> object:
        """Support ``initial_context["key"]``."""
        extras = self.model_extra or {}
        if key in extras:
            return extras[key]
        raise KeyError(key)


# ── Enums ───────────────────────────────────────────────────────────


class ScoreAggregation(StrEnum):
    """How scores from multiple attempts/subtasks are aggregated."""

    MAX = "max"
    AVERAGE = "average"


class ScorerTarget(StrEnum):
    """What data a scorer evaluates."""

    SUBMISSION = "submission"
    TRAJECTORY = "trajectory"


class LLMJudgeResponseFormat(StrEnum):
    """Expected response format from an LLM judge."""

    BINARY = "binary"
    CONTINUOUS = "continuous"
    STEP_EVALUATIONS = "step_evaluations"


# ── Scoring Criteria ────────────────────────────────────────────────


class StaticCriteria(BaseModel):
    """Criteria that compare against expected static answers."""

    model_config = ConfigDict(frozen=True)

    expected_answers: list[str]


class LLMJudgeCriteria(BaseModel):
    """Criteria evaluated by an LLM judge."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    model: str
    system_template: str = Field(alias="judge_system_template")
    user_template: str = Field(alias="judge_user_template")
    steps_per_message: int | None = None
    response_format: LLMJudgeResponseFormat = LLMJudgeResponseFormat.BINARY


class ToolCallCriteria(BaseModel):
    """Criteria based on tool-call execution during an episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str = ""
    min_executions: int = 1
    expected_tools: list[str] = Field(default_factory=list)
    param_name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_expected_tools(cls, data: dict[str, object]) -> dict[str, object]:
        """Coerce ``expected_tools`` from a bare string to a list.

        YAML authors commonly write ``expected_tools: bash`` instead of
        ``expected_tools: ["bash"]``.  This validator also splits
        comma-separated strings like ``"bash,python"``.
        """
        if not isinstance(data, dict):
            return data  # type: ignore[unreachable]
        tools = data.get("expected_tools")
        if isinstance(tools, str):
            data = {**data, "expected_tools": [t.strip() for t in tools.split(",") if t.strip()]}
        return data

    @model_validator(mode="after")
    def _require_tool_specification(self) -> ToolCallCriteria:
        """Ensure at least one of tool_name or expected_tools is set."""
        if not self.tool_name and not self.expected_tools:
            raise ValueError("At least one of 'tool_name' or 'expected_tools' must be non-empty")
        return self


class DomainCriteria(BaseModel):
    """Domain-specific criteria — accepts arbitrary extra fields."""

    model_config = ConfigDict(frozen=True, extra="allow")


# ── New Scoring Models ──────────────────────────────────────────────


class AggregationConfig(BaseModel):
    """Structured aggregation configuration."""

    model_config = ConfigDict(frozen=True)

    strategy: ScoreAggregation
    scores: list[str | list[str]]

    @model_validator(mode="before")
    @classmethod
    def _parse_strategy_key(cls, data: dict[str, object]) -> dict[str, object]:
        """Parse strategy-as-key format: {"max": {"scores": [...]}} → {"strategy": "max", "scores": [...]}."""
        if not isinstance(data, dict):
            return data  # type: ignore[unreachable]
        if "strategy" in data:
            return data
        for key in ("max", "average"):
            if key in data:
                inner = data[key]
                if isinstance(inner, dict):
                    return {"strategy": key, "scores": inner.get("scores", [])}
        return data


_COMMON_FIELDS: frozenset[str] = frozenset(
    {
        "target",
        "max_score",
        "weight",
        "title",
        "description",
        "hints",
        "submission_source",
    }
)


class SubmissionSourceType(StrEnum):
    """Where ScoringContext.submission is sourced from."""

    COMPLETION = "completion"
    FILE = "file"


class SubmissionFallback(StrEnum):
    """Behavior when a file-type submission_source cannot be resolved."""

    COMPLETION = "completion"
    ERROR = "error"


class SubmissionSource(BaseModel):
    """Per-scorer configuration for where to source ``submission`` from.

    When unset on a :class:`ScorerConfig`, behavior is unchanged: submission
    is populated from ``state.output.completion`` (agent's final chat msg).

    When ``type=file``, SABER reads the file from the sample's sandbox via
    ``inspect_ai.util.sandbox().read_file(path)`` and uses its content as the
    submission. Use this for agents whose primary deliverable is a structured
    artifact written to disk rather than emitted inline.

    Example YAML::

        gate_report_complete:
          target: submission
          submission_source:
            type: file
            path: "/workspace/shared/posture_analysis_report.md"
            fallback: completion
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    type: SubmissionSourceType = SubmissionSourceType.COMPLETION
    path: str | None = None
    fallback: SubmissionFallback = SubmissionFallback.ERROR
    encoding: str = "utf-8"

    @model_validator(mode="after")
    def _require_path_for_file_type(self) -> SubmissionSource:
        if self.type == SubmissionSourceType.FILE and not self.path:
            raise ValueError(
                "submission_source.path is required when type=file"
            )
        return self


class ScorerConfig(BaseModel):
    """A single scorer within a task's scoring block."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    scorer_name: str  # e.g. "submission", "checkpoint_1"
    strategy: str  # e.g. "static", "llm_judge"

    target: ScorerTarget = ScorerTarget.TRAJECTORY
    max_score: Annotated[float, Field(gt=0, le=100)] = 1.0
    weight: Annotated[float, Field(ge=0, le=1)] = 1.0

    # Optional: override where ScoringContext.submission is sourced from.
    # When None (default), submission = state.output.completion.
    # See SubmissionSource docstring for file-resident artifact scoring.
    submission_source: SubmissionSource | None = None

    title: str = ""
    description: str = ""
    hints: list[str] = Field(default_factory=list)

    criteria: StaticCriteria | LLMJudgeCriteria | ToolCallCriteria | DomainCriteria = Field(
        default_factory=DomainCriteria
    )


_STRATEGY_TO_CRITERIA: types.MappingProxyType[str, type[BaseModel]] = types.MappingProxyType(
    {
        "static": StaticCriteria,
        "llm_judge": LLMJudgeCriteria,
        "tool_call": ToolCallCriteria,
        "tool_call_count": ToolCallCriteria,
        "static_jaccard": StaticCriteria,
    }
)


# ── Prompt Paths ────────────────────────────────────────────────────


class PromptPaths(BaseModel):
    """Paths to prompt template files."""

    model_config = ConfigDict(frozen=True)

    instruction: str
    assistant: str = "assistants/inspect_assistant.j2"


# ── Tool Config ─────────────────────────────────────────────────────


class ToolConfig(BaseModel):
    """Per-tool configuration (e.g. timeout, security)."""

    model_config = ConfigDict(frozen=True)

    timeout: int = 180
    security: ToolSecurityConfig | None = None


# ── Task Config ─────────────────────────────────────────────────────


class TaskConfig(BaseModel):
    """Complete configuration for a single benchmark task."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    task_id: Annotated[str, Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")]
    dataset: str | None = None
    title: str

    @field_validator("dataset")
    @classmethod
    def _reject_reserved_dataset(cls, v: str | None) -> str | None:
        """Reject the reserved sentinel value ``"all"``.

        ``"all"`` is used by :data:`saber.config.loader._DATASET_ALL` to
        bypass dataset filtering and must not appear as a task-level value.
        """
        if v == "all":
            raise ValueError(
                '"all" is a reserved dataset sentinel '
                "(see saber.config.loader._DATASET_ALL) and cannot be "
                "used as a task dataset value"
            )
        return v

    description: str
    prompts: PromptPaths
    sandbox: str | None = Field(default=None, alias="sandbox_environment")
    initial_context: InitialContext = Field(default_factory=InitialContext)
    initial_files: dict[str, str] = Field(default_factory=dict)
    tools: dict[str, ToolConfig] = Field(default_factory=dict)
    max_steps: int = 25
    timeout: int | None = None
    scorers: tuple[ScorerConfig, ...] = ()
    scoring_aggregation: AggregationConfig | None = None
    attempts: int = 1
    aggregation: ScoreAggregation = ScoreAggregation.MAX
    role: str | None = None
    is_template: bool = False
    dependency_template: str | None = None
    setup: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_scoring_block(cls, data: dict[str, object]) -> dict[str, object]:
        """Parse new-format ``scoring:`` block into ``scorers`` list.

        If the data contains a ``scoring`` key whose value is a strategy-grouped
        dict (new format), this validator converts it to a list of
        ``ScorerConfig`` dicts and stores them in ``data["scorers"]``.
        The raw ``scoring`` key is removed.

        If ``scoring`` is absent the data passes through unchanged.
        """
        if not isinstance(data, dict):
            return data  # type: ignore[unreachable]

        raw_scoring = data.get("scoring")
        if not isinstance(raw_scoring, dict):
            # Even without scoring block, clean up scoring_defaults
            if "scoring_defaults" in data:
                data = {**data}
                del data["scoring_defaults"]
            return data

        # Detect strategy-grouped dict: keys are strategy names,
        # values are dicts of scorer_name → config.
        # Heuristic: if any value is itself a dict-of-dicts, it's the grouped format.
        is_new_format = any(
            isinstance(v, dict) and any(isinstance(vv, dict) for vv in v.values()) for v in raw_scoring.values()
        )
        if not is_new_format:
            if "scoring_defaults" in data:
                data = {**data}
                del data["scoring_defaults"]
            return data

        # Extract scoring defaults (consumed here, not passed to model fields)
        scoring_defaults: dict[str, object] = {}
        raw_defaults = data.get("scoring_defaults")
        if isinstance(raw_defaults, dict):
            scoring_defaults = raw_defaults

        scorers: list[dict[str, object]] = []
        for strategy, scorer_group in raw_scoring.items():
            if not isinstance(scorer_group, dict):
                continue

            # Extract inline defaults: non-dict values at the strategy level
            inline_defaults: dict[str, object] = {k: v for k, v in scorer_group.items() if not isinstance(v, dict)}

            # Get strategy-level defaults from scoring_defaults (higher priority than inline)
            strategy_defaults = scoring_defaults.get(strategy, {})
            if not isinstance(strategy_defaults, dict):
                strategy_defaults = {}

            for scorer_name, cfg in scorer_group.items():
                if not isinstance(cfg, dict):
                    continue  # skip inline defaults (scalars)

                # Apply defaults: inline -> scoring_defaults scalars -> target-specific -> per-scorer
                target = cfg.get("target", "trajectory")
                scalar_defaults = {k: v for k, v in strategy_defaults.items() if not isinstance(v, dict)}
                target_specific = strategy_defaults.get(str(target), {})
                if not isinstance(target_specific, dict):
                    target_specific = {}

                merged = {**inline_defaults, **scalar_defaults, **target_specific, **cfg}

                scorer_dict: dict[str, object] = {
                    "scorer_name": scorer_name,
                    "strategy": strategy,
                }
                criteria_fields: dict[str, object] = {}
                for k, v in merged.items():
                    if k in _COMMON_FIELDS:
                        scorer_dict[k] = v
                    else:
                        criteria_fields[k] = v

                # Backward compat: move golden_answer → description
                if "golden_answer" in criteria_fields:
                    ga = criteria_fields.pop("golden_answer")
                    if ga and "description" not in scorer_dict:
                        scorer_dict["description"] = str(ga)

                criteria_cls = _STRATEGY_TO_CRITERIA.get(strategy)
                if criteria_cls is not None:
                    scorer_dict["criteria"] = criteria_cls(**criteria_fields)
                elif criteria_fields:
                    scorer_dict["criteria"] = DomainCriteria(**criteria_fields)

                scorers.append(scorer_dict)

        data = {**data}
        data["scorers"] = scorers
        del data["scoring"]
        data.pop("scoring_defaults", None)

        return data


# ── Global Defaults ─────────────────────────────────────────────────


class PermanentEnvironment(BaseModel):
    """Permanent Docker Compose environment for shared services.

    Defines the compose file and project name for long-lived services
    (databases, caches) that persist across evaluation samples.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "saber-permanent"
    compose: str


class GlobalDefaults(BaseModel):
    """Domain-wide default settings applied to all tasks unless overridden."""

    model_config = ConfigDict(frozen=True)

    default_dataset: str | None = None
    permanent_environment: PermanentEnvironment | None = None
    prompts: PromptPaths | None = None
    max_steps: int = 25
    aggregation: ScoreAggregation = ScoreAggregation.MAX
    scoring_defaults: dict[str, dict[str, object]] | None = None
    setup: str | None = None


# ── Domain Config ───────────────────────────────────────────────────


class ImageConfig(BaseModel):
    """Docker image build specification."""

    model_config = ConfigDict(frozen=True)

    tag: str
    dockerfile: str
    context: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    build_args: dict[str, str] = Field(default_factory=dict)


class DomainConfig(BaseModel):
    """Top-level configuration for a SABER benchmark domain."""

    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    description: str
    version: str = "1.0.0"
    images: dict[str, ImageConfig] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    maintainer: str | None = None
    documentation: str | None = None
    repository: str | None = None
