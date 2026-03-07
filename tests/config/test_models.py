"""Tests for saber.config.models — Pydantic v2 data models for SABER configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saber.config.models import (
    AggregationConfig,
    DomainConfig,
    DomainCriteria,
    GlobalDefaults,
    ImageConfig,
    InitialContext,
    LLMJudgeCriteria,
    LLMJudgeResponseFormat,
    PermanentEnvironment,
    PromptPaths,
    ScoreAggregation,
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
    TaskConfig,
    ToolCallCriteria,
    ToolConfig,
    to_template_vars,
)

# ── InitialContext ──────────────────────────────────────────────────


class TestInitialContext:
    """InitialContext: frozen model with extra='allow' for domain-specific fields."""

    def test_empty_construction(self) -> None:
        ic = InitialContext()
        assert ic.model_extra == {} or ic.model_extra is None

    def test_arbitrary_keys(self) -> None:
        ic = InitialContext(database="postgres://db:5432", api_key="secret")
        assert ic.database == "postgres://db:5432"  # type: ignore[attr-defined]
        assert ic.api_key == "secret"  # type: ignore[attr-defined]

    def test_nested_dict_access(self) -> None:
        ic = InitialContext(database_connection={"hostname": "db.local", "port": 5432})
        conn = ic.database_connection  # type: ignore[attr-defined]
        assert isinstance(conn, dict)
        assert conn["hostname"] == "db.local"

    def test_frozen(self) -> None:
        ic = InitialContext(foo="bar")
        with pytest.raises(Exception):  # noqa: B017
            ic.foo = "baz"  # type: ignore[attr-defined]

    def test_equality(self) -> None:
        ic1 = InitialContext(key="val")
        ic2 = InitialContext(key="val")
        assert ic1 == ic2


class TestToTemplateVars:
    """to_template_vars converts Pydantic models to Jinja2 template dicts."""

    def test_extra_fields_included(self) -> None:
        ic = InitialContext(database="pg", timeout=30)
        result = to_template_vars(ic)
        assert result["database"] == "pg"
        assert result["timeout"] == 30

    def test_nested_models_preserved(self) -> None:
        """Nested Pydantic models remain as objects (not serialized to dicts)."""
        sc = ScorerConfig(scorer_name="sub", strategy="static")
        ic = InitialContext(scorer=sc)
        result = to_template_vars(ic)
        assert isinstance(result["scorer"], ScorerConfig)


class TestTaskConfigInitialContext:
    """TaskConfig.initial_context is now InitialContext, auto-coerced from dicts."""

    def test_dict_coerced_to_initial_context(self) -> None:
        tc = TaskConfig(
            task_id="test_task",
            title="T",
            description="D",
            prompts=PromptPaths(instruction="i.j2"),
            initial_context={"incident": "breach"},
        )
        assert isinstance(tc.initial_context, InitialContext)
        assert tc.initial_context.incident == "breach"  # type: ignore[attr-defined]

    def test_initial_context_model_accepted(self) -> None:
        ic = InitialContext(incident="breach")
        tc = TaskConfig(
            task_id="test_task",
            title="T",
            description="D",
            prompts=PromptPaths(instruction="i.j2"),
            initial_context=ic,
        )
        assert tc.initial_context is ic or tc.initial_context == ic

    def test_default_is_empty_initial_context(self) -> None:
        tc = TaskConfig(
            task_id="test_task",
            title="T",
            description="D",
            prompts=PromptPaths(instruction="i.j2"),
        )
        assert isinstance(tc.initial_context, InitialContext)


# ── Enum tests ──────────────────────────────────────────────────────


class TestScoreAggregation:
    def test_members(self) -> None:
        assert ScoreAggregation.MAX == "max"
        assert ScoreAggregation.AVERAGE == "average"

    def test_is_str(self) -> None:
        assert isinstance(ScoreAggregation.MAX, str)

    def test_member_count(self) -> None:
        assert len(ScoreAggregation) == 2


class TestLLMJudgeResponseFormat:
    def test_members(self) -> None:
        assert LLMJudgeResponseFormat.BINARY == "binary"
        assert LLMJudgeResponseFormat.CONTINUOUS == "continuous"
        assert LLMJudgeResponseFormat.STEP_EVALUATIONS == "step_evaluations"

    def test_is_str(self) -> None:
        assert isinstance(LLMJudgeResponseFormat.BINARY, str)

    def test_member_count(self) -> None:
        assert len(LLMJudgeResponseFormat) == 3


# ── StaticCriteria ──────────────────────────────────────────────────


class TestStaticCriteria:
    def test_happy_path(self) -> None:
        sc = StaticCriteria(expected_answers=["yes", "no"])
        assert sc.expected_answers == ["yes", "no"]

    def test_frozen(self) -> None:
        sc = StaticCriteria(expected_answers=["a"])
        with pytest.raises(ValidationError):
            sc.expected_answers = ["b"]  # type: ignore[misc]

    def test_empty_list(self) -> None:
        sc = StaticCriteria(expected_answers=[])
        assert sc.expected_answers == []


# ── LLMJudgeCriteria ───────────────────────────────────────────────


class TestLLMJudgeCriteria:
    def test_happy_path_with_aliases(self) -> None:
        c = LLMJudgeCriteria(
            model="gpt-4",
            judge_system_template="sys",
            judge_user_template="usr",
        )
        assert c.model == "gpt-4"
        assert c.system_template == "sys"
        assert c.user_template == "usr"

    def test_field_name_access(self) -> None:
        c = LLMJudgeCriteria(
            model="m",
            judge_system_template="s",
            judge_user_template="u",
        )
        assert c.system_template == "s"
        assert c.user_template == "u"

    def test_populate_by_name(self) -> None:
        """Can also construct using the Python field name directly."""
        c = LLMJudgeCriteria(
            model="m",
            system_template="s",
            user_template="u",
        )
        assert c.system_template == "s"

    def test_defaults(self) -> None:
        c = LLMJudgeCriteria(
            model="m",
            judge_system_template="s",
            judge_user_template="u",
        )
        assert c.steps_per_message is None
        assert c.response_format == LLMJudgeResponseFormat.BINARY

    def test_frozen(self) -> None:
        c = LLMJudgeCriteria(model="m", judge_system_template="s", judge_user_template="u")
        with pytest.raises(ValidationError):
            c.model = "other"  # type: ignore[misc]


# ── ToolCallCriteria ───────────────────────────────────────────────


class TestToolCallCriteria:
    def test_happy_path(self) -> None:
        tc = ToolCallCriteria(tool_name="exec")
        assert tc.tool_name == "exec"
        assert tc.min_executions == 1
        assert tc.expected_tools == []
        assert tc.param_name is None

    def test_custom_values(self) -> None:
        tc = ToolCallCriteria(
            tool_name="scan",
            min_executions=3,
            expected_tools=["nmap", "nessus"],
            param_name="target",
        )
        assert tc.min_executions == 3
        assert tc.expected_tools == ["nmap", "nessus"]
        assert tc.param_name == "target"

    def test_tool_name_defaults_to_empty(self) -> None:
        """tool_name defaults to '' so YAML without it still parses."""
        tc = ToolCallCriteria(expected_tools=["bash"])
        assert tc.tool_name == ""
        assert tc.expected_tools == ["bash"]

    def test_expected_tools_string_coerced_to_list(self) -> None:
        """A bare string 'bash' in YAML should be coerced to ['bash']."""
        tc = ToolCallCriteria(expected_tools="bash")  # type: ignore[arg-type]
        assert tc.expected_tools == ["bash"]

    def test_expected_tools_csv_string_coerced(self) -> None:
        """Comma-separated string should be split into list."""
        tc = ToolCallCriteria(expected_tools="bash,python")  # type: ignore[arg-type]
        assert tc.expected_tools == ["bash", "python"]


# ── DomainCriteria ─────────────────────────────────────────────────


class TestDomainCriteria:
    def test_empty(self) -> None:
        dc = DomainCriteria()
        assert dc.model_fields_set == set()

    def test_extra_fields_accepted(self) -> None:
        dc = DomainCriteria(custom_key="value", another=42)  # type: ignore[call-arg]
        assert dc.custom_key == "value"  # type: ignore[attr-defined]
        assert dc.another == 42  # type: ignore[attr-defined]

    def test_frozen(self) -> None:
        dc = DomainCriteria()
        with pytest.raises(ValidationError):
            dc.foo = "bar"  # type: ignore[misc]


# ── PromptPaths ────────────────────────────────────────────────────


class TestPromptPaths:
    def test_minimal(self) -> None:
        pp = PromptPaths(instruction="instructions/task.j2")
        assert pp.instruction == "instructions/task.j2"
        assert pp.assistant == "assistants/inspect_assistant.j2"

    def test_frozen(self) -> None:
        pp = PromptPaths(instruction="x")
        with pytest.raises(ValidationError):
            pp.instruction = "y"  # type: ignore[misc]


# ── ToolConfig ─────────────────────────────────────────────────────


class TestToolConfig:
    def test_defaults(self) -> None:
        tc = ToolConfig()
        assert tc.timeout == 180
        assert tc.security is None

    def test_custom(self) -> None:
        tc = ToolConfig(timeout=60)
        assert tc.timeout == 60

    def test_security_with_model(self) -> None:
        from saber.tools.security import ToolSecurityConfig

        sec = ToolSecurityConfig()
        tc = ToolConfig(security=sec)
        assert tc.security is not None
        assert tc.security.max_content_length == 4096

    def test_security_dict_coercion(self) -> None:
        """A dict value for security is auto-coerced to ToolSecurityConfig."""
        tc = ToolConfig(security={"allow_semicolons": True})  # type: ignore[arg-type]
        assert tc.security is not None
        assert tc.security.allow_semicolons is True

    def test_backward_compat_timeout_only(self) -> None:
        """Existing ToolConfig(timeout=180) usage still works."""
        tc = ToolConfig(timeout=180)
        assert tc.timeout == 180
        assert tc.security is None


# ── TaskConfig ─────────────────────────────────────────────────────


class TestTaskConfig:
    @staticmethod
    def _minimal_data() -> dict[str, object]:
        return {
            "task_id": "test_task",
            "title": "Test Task",
            "description": "A test",
            "prompts": {"instruction": "instructions/test.j2"},
        }

    def test_happy_path(self) -> None:
        tc = TaskConfig(**self._minimal_data())
        assert tc.task_id == "test_task"
        assert tc.title == "Test Task"
        assert tc.max_steps == 25
        assert tc.timeout is None
        assert tc.attempts == 1
        assert tc.aggregation == ScoreAggregation.MAX
        assert tc.role is None
        assert tc.is_template is False
        assert tc.dependency_template is None

    def test_sandbox_alias(self) -> None:
        data = {**self._minimal_data(), "sandbox_environment": "compose.yaml"}
        tc = TaskConfig(**data)
        assert tc.sandbox == "compose.yaml"

    def test_task_id_min_length(self) -> None:
        data = {**self._minimal_data(), "task_id": ""}
        with pytest.raises(ValidationError, match="task_id"):
            TaskConfig(**data)

    def test_task_id_pattern_rejects_spaces(self) -> None:
        data = {**self._minimal_data(), "task_id": "bad id!"}
        with pytest.raises(ValidationError, match="task_id"):
            TaskConfig(**data)

    def test_task_id_pattern_rejects_special_chars(self) -> None:
        data = {**self._minimal_data(), "task_id": "bad@task"}
        with pytest.raises(ValidationError, match="task_id"):
            TaskConfig(**data)

    def test_task_id_allows_hyphens_underscores(self) -> None:
        data = {**self._minimal_data(), "task_id": "good-task_123"}
        tc = TaskConfig(**data)
        assert tc.task_id == "good-task_123"

    def test_initial_context_accepts_nested_data(self) -> None:
        data = {
            **self._minimal_data(),
            "initial_context": {"key": "value", "nested": {"a": 1}},
        }
        tc = TaskConfig(**data)
        assert isinstance(tc.initial_context, InitialContext)
        assert tc.initial_context.key == "value"  # type: ignore[attr-defined]
        assert tc.initial_context.nested == {"a": 1}  # type: ignore[attr-defined]

    def test_tools_dict(self) -> None:
        data = {
            **self._minimal_data(),
            "tools": {"exec": {"timeout": 60}},
        }
        tc = TaskConfig(**data)
        assert tc.tools["exec"].timeout == 60

    def test_frozen(self) -> None:
        tc = TaskConfig(**self._minimal_data())
        with pytest.raises(ValidationError):
            tc.title = "changed"  # type: ignore[misc]


# ── GlobalDefaults ─────────────────────────────────────────────────


class TestGlobalDefaults:
    def test_defaults(self) -> None:
        gd = GlobalDefaults()
        assert gd.permanent_environment is None
        assert gd.prompts is None
        assert gd.max_steps == 25
        assert gd.aggregation == ScoreAggregation.MAX

    def test_with_permanent_environment(self) -> None:
        perm = PermanentEnvironment(name="my-proj", compose="compose/db.yml")
        gd = GlobalDefaults(permanent_environment=perm)
        assert gd.permanent_environment is not None
        assert gd.permanent_environment.name == "my-proj"
        assert gd.permanent_environment.compose == "compose/db.yml"

    def test_permanent_environment_default_name(self) -> None:
        perm = PermanentEnvironment(compose="compose/db.yml")
        assert perm.name == "saber-permanent"

    def test_with_prompts(self) -> None:
        gd = GlobalDefaults(
            prompts=PromptPaths(instruction="inst.j2"),
            max_steps=50,
            aggregation=ScoreAggregation.AVERAGE,
        )
        assert gd.prompts is not None
        assert gd.prompts.instruction == "inst.j2"
        assert gd.max_steps == 50
        assert gd.aggregation == ScoreAggregation.AVERAGE

    def test_with_scoring_defaults(self) -> None:
        gd = GlobalDefaults(
            scoring_defaults={
                "llm_judge": {
                    "model": "openai/gpt-4",
                    "trajectory": {
                        "judge_system_template": "sys.md",
                        "judge_user_template": "user.md",
                    },
                }
            }
        )
        assert gd.scoring_defaults is not None
        assert "llm_judge" in gd.scoring_defaults


# ── ImageConfig ────────────────────────────────────────────────────


class TestImageConfig:
    def test_minimal(self) -> None:
        ic = ImageConfig(tag="v1", dockerfile="Dockerfile")
        assert ic.tag == "v1"
        assert ic.dockerfile == "Dockerfile"
        assert ic.context is None
        assert ic.labels == {}
        assert ic.build_args == {}

    def test_full(self) -> None:
        ic = ImageConfig(
            tag="v2",
            dockerfile="Dockerfile.dev",
            context=".",
            labels={"app": "saber"},
            build_args={"BASE": "python:3.12"},
        )
        assert ic.context == "."
        assert ic.labels["app"] == "saber"
        assert ic.build_args["BASE"] == "python:3.12"

    def test_frozen(self) -> None:
        ic = ImageConfig(tag="v1", dockerfile="Dockerfile")
        with pytest.raises(ValidationError):
            ic.tag = "v2"  # type: ignore[misc]


# ── DomainConfig ───────────────────────────────────────────────────


class TestDomainConfig:
    def test_minimal(self) -> None:
        dc = DomainConfig(slug="excytin", name="ExCyTIn", description="Demo domain")
        assert dc.slug == "excytin"
        assert dc.name == "ExCyTIn"
        assert dc.version == "1.0.0"
        assert dc.images == {}
        assert dc.tags == []
        assert dc.maintainer is None
        assert dc.documentation is None
        assert dc.repository is None

    def test_full(self) -> None:
        dc = DomainConfig(
            slug="cybench",
            name="CyBench",
            description="Cyber benchmark",
            version="2.0.0",
            images={
                "base": ImageConfig(tag="latest", dockerfile="Dockerfile"),
            },
            tags=["security", "benchmark"],
            maintainer="team@saber.dev",
            documentation="https://docs.saber.dev",
            repository="https://github.com/saber/saber",
        )
        assert dc.version == "2.0.0"
        assert "base" in dc.images
        assert dc.tags == ["security", "benchmark"]

    def test_frozen(self) -> None:
        dc = DomainConfig(slug="x", name="X", description="d")
        with pytest.raises(ValidationError):
            dc.slug = "y"  # type: ignore[misc]


# ── ScorerTarget ───────────────────────────────────────────────────


class TestScorerTarget:
    def test_enum_values(self) -> None:
        assert ScorerTarget.SUBMISSION == "submission"
        assert ScorerTarget.TRAJECTORY == "trajectory"

    def test_string_coercion(self) -> None:
        assert ScorerTarget("submission") == ScorerTarget.SUBMISSION


# ── ScorerConfig ───────────────────────────────────────────────────


class TestScorerConfig:
    def test_construction(self) -> None:
        cfg = ScorerConfig(
            scorer_name="sub",
            strategy="static",
            criteria=StaticCriteria(expected_answers=["a"]),
        )
        assert cfg.scorer_name == "sub"
        assert cfg.strategy == "static"

    def test_default_target(self) -> None:
        cfg = ScorerConfig(scorer_name="x", strategy="s")
        assert cfg.target == ScorerTarget.TRAJECTORY

    def test_frozen(self) -> None:
        cfg = ScorerConfig(scorer_name="x", strategy="s")
        with pytest.raises(ValidationError):
            cfg.scorer_name = "y"  # type: ignore[misc]


# ── AggregationConfig ─────────────────────────────────────────────


class TestAggregationConfig:
    def test_canonical_form(self) -> None:
        cfg = AggregationConfig(strategy="max", scores=["a", ["b", "c"]])
        assert cfg.strategy == ScoreAggregation.MAX
        assert cfg.scores == ["a", ["b", "c"]]

    def test_strategy_as_key_format(self) -> None:
        cfg = AggregationConfig(**{"max": {"scores": ["a", "b"]}})
        assert cfg.strategy == ScoreAggregation.MAX

    def test_frozen(self) -> None:
        cfg = AggregationConfig(strategy="max", scores=["a"])
        with pytest.raises(ValidationError):
            cfg.strategy = "average"  # type: ignore[misc]


# ── TaskConfig scoring block ──────────────────────────────────────


class TestTaskConfigScoringBlock:
    """Tests for the new scoring: block parsing."""

    def _make_task(self, **overrides: object) -> TaskConfig:
        """Helper to build minimal TaskConfig data with scoring block."""
        base: dict[str, object] = {
            "task_id": "test_task",
            "title": "Test",
            "description": "Test task",
            "prompts": {"instruction": "test.md"},
        }
        base.update(overrides)
        return TaskConfig(**base)

    def test_static_scorer(self) -> None:
        task = self._make_task(
            scoring={
                "static": {
                    "submission": {
                        "target": "submission",
                        "expected_answers": ["42"],
                        "max_score": 1.0,
                    }
                }
            }
        )
        assert len(task.scorers) == 1
        s = task.scorers[0]
        assert s.scorer_name == "submission"
        assert s.strategy == "static"
        assert s.target == ScorerTarget.SUBMISSION
        assert isinstance(s.criteria, StaticCriteria)
        assert s.criteria.expected_answers == ["42"]

    def test_llm_judge_scorer(self) -> None:
        task = self._make_task(
            scoring={
                "llm_judge": {
                    "checkpoint_1": {
                        "model": "openai/gpt-4",
                        "judge_system_template": "sys.md",
                        "judge_user_template": "user.md",
                        "max_score": 1.0,
                        "title": "CP1",
                        "description": "test checkpoint",
                    }
                }
            }
        )
        assert len(task.scorers) == 1
        s = task.scorers[0]
        assert s.strategy == "llm_judge"
        assert isinstance(s.criteria, LLMJudgeCriteria)
        assert s.criteria.model == "openai/gpt-4"

    def test_tool_call_scorer(self) -> None:
        task = self._make_task(scoring={"tool_call": {"cp": {"expected_tools": ["bash"], "max_score": 1.0}}})
        assert isinstance(task.scorers[0].criteria, ToolCallCriteria)

    def test_mixed_strategies(self) -> None:
        task = self._make_task(
            scoring={
                "static": {"sub": {"target": "submission", "expected_answers": ["a"]}},
                "llm_judge": {
                    "cp1": {
                        "model": "m",
                        "judge_system_template": "s",
                        "judge_user_template": "u",
                    }
                },
            }
        )
        assert len(task.scorers) == 2

    def test_unknown_strategy_uses_domain_criteria(self) -> None:
        task = self._make_task(
            scoring={"trajectory_analysis": {"sub": {"detection_objective": "...", "max_score": 10.0}}}
        )
        assert isinstance(task.scorers[0].criteria, DomainCriteria)
        assert task.scorers[0].criteria.model_extra["detection_objective"] == "..."

    def test_scoring_aggregation_parsed(self) -> None:
        task = self._make_task(
            scoring={"static": {"sub": {"target": "submission", "expected_answers": ["a"]}}},
            scoring_aggregation={"max": {"scores": ["sub"]}},
        )
        assert task.scoring_aggregation is not None
        assert task.scoring_aggregation.strategy == ScoreAggregation.MAX


class TestTaskConfigScoringDefaults:
    """Tests for scoring_defaults merging in _parse_scoring_block."""

    def _make_task(self, **overrides: object) -> TaskConfig:
        base: dict[str, object] = {
            "task_id": "test_task",
            "title": "Test",
            "description": "Test task",
            "prompts": {"instruction": "test.md"},
        }
        base.update(overrides)
        return TaskConfig(**base)

    def test_defaults_applied_to_llm_judge(self) -> None:
        """scoring_defaults provide model and templates when not in scorer config."""
        task = self._make_task(
            scoring_defaults={
                "llm_judge": {
                    "model": "openai/gpt-4",
                    "submission": {
                        "judge_system_template": "sub_sys.md",
                        "judge_user_template": "sub_user.md",
                    },
                    "trajectory": {
                        "judge_system_template": "cp_sys.md",
                        "judge_user_template": "cp_user.md",
                        "steps_per_message": 50,
                    },
                }
            },
            scoring={
                "llm_judge": {
                    "submission": {
                        "target": "submission",
                        "description": "42",
                        "max_score": 1.0,
                    },
                    "checkpoint_1": {
                        "target": "trajectory",
                        "max_score": 0.5,
                        "title": "CP1",
                        "description": "Find the flag",
                    },
                }
            },
        )
        assert len(task.scorers) == 2

        # Find submission scorer
        sub = next(s for s in task.scorers if s.scorer_name == "submission")
        assert isinstance(sub.criteria, LLMJudgeCriteria)
        assert sub.criteria.model == "openai/gpt-4"
        assert sub.criteria.system_template == "sub_sys.md"
        assert sub.criteria.user_template == "sub_user.md"
        assert sub.description == "42"

        # Find checkpoint scorer
        cp = next(s for s in task.scorers if s.scorer_name == "checkpoint_1")
        assert isinstance(cp.criteria, LLMJudgeCriteria)
        assert cp.criteria.model == "openai/gpt-4"
        assert cp.criteria.system_template == "cp_sys.md"
        assert cp.criteria.user_template == "cp_user.md"
        assert cp.criteria.steps_per_message == 50

    def test_scorer_overrides_defaults(self) -> None:
        """Per-scorer values override scoring_defaults."""
        task = self._make_task(
            scoring_defaults={
                "llm_judge": {
                    "model": "openai/gpt-4",
                    "trajectory": {
                        "judge_system_template": "default_sys.md",
                        "judge_user_template": "default_user.md",
                        "steps_per_message": 50,
                    },
                }
            },
            scoring={
                "llm_judge": {
                    "checkpoint_1": {
                        "target": "trajectory",
                        "model": "openai/gpt-3.5",
                        "judge_system_template": "custom_sys.md",
                        "steps_per_message": 100,
                        "max_score": 0.5,
                        "description": "test",
                    }
                }
            },
        )
        cp = task.scorers[0]
        assert isinstance(cp.criteria, LLMJudgeCriteria)
        assert cp.criteria.model == "openai/gpt-3.5"  # overridden
        assert cp.criteria.system_template == "custom_sys.md"  # overridden
        assert cp.criteria.user_template == "default_user.md"  # from default
        assert cp.criteria.steps_per_message == 100  # overridden

    def test_no_defaults_works(self) -> None:
        """When scoring_defaults absent, behavior unchanged."""
        task = self._make_task(
            scoring={
                "llm_judge": {
                    "cp": {
                        "model": "m",
                        "judge_system_template": "s",
                        "judge_user_template": "u",
                        "max_score": 1.0,
                    }
                }
            }
        )
        assert len(task.scorers) == 1
        assert isinstance(task.scorers[0].criteria, LLMJudgeCriteria)

    def test_defaults_dont_leak_to_model_fields(self) -> None:
        """scoring_defaults is consumed and not stored on TaskConfig."""
        task = self._make_task(
            scoring_defaults={"llm_judge": {"model": "m"}},
            scoring={
                "llm_judge": {
                    "cp": {
                        "judge_system_template": "s",
                        "judge_user_template": "u",
                        "max_score": 1.0,
                    }
                }
            },
        )
        assert not hasattr(task, "scoring_defaults")

    def test_defaults_for_non_llm_strategy_ignored(self) -> None:
        """scoring_defaults for unrelated strategies is harmless."""
        task = self._make_task(
            scoring_defaults={"other_strategy": {"key": "val"}},
            scoring={"static": {"sub": {"target": "submission", "expected_answers": ["a"]}}},
        )
        assert len(task.scorers) == 1
        assert isinstance(task.scorers[0].criteria, StaticCriteria)

    def test_defaults_with_default_trajectory_target(self) -> None:
        """When target is not specified in scorer, it defaults to 'trajectory'."""
        task = self._make_task(
            scoring_defaults={
                "llm_judge": {
                    "model": "m",
                    "trajectory": {
                        "judge_system_template": "cp_sys.md",
                        "judge_user_template": "cp_user.md",
                    },
                }
            },
            scoring={
                "llm_judge": {
                    "checkpoint_1": {
                        # No target specified -> defaults to trajectory
                        "max_score": 0.5,
                        "description": "test",
                    }
                }
            },
        )
        cp = task.scorers[0]
        assert isinstance(cp.criteria, LLMJudgeCriteria)
        assert cp.criteria.system_template == "cp_sys.md"


class TestTaskConfigInlineStrategyDefaults:
    """Tests for inline strategy-level defaults in the scoring block."""

    def _make_task(self, **overrides: object) -> TaskConfig:
        base: dict[str, object] = {
            "task_id": "test_task",
            "title": "Test",
            "description": "Test task",
            "prompts": {"instruction": "test.md"},
        }
        base.update(overrides)
        return TaskConfig(**base)

    def test_inline_defaults_applied(self) -> None:
        """Non-dict values in strategy block are inherited by all children."""
        task = self._make_task(
            scoring={
                "llm_judge": {
                    "model": "openai/gpt-4",
                    "steps_per_message": 50,
                    "judge_system_template": "cp_sys.md",
                    "judge_user_template": "cp_user.md",
                    "checkpoint_1": {
                        "target": "trajectory",
                        "max_score": 0.5,
                        "title": "CP1",
                        "description": "Find the flag",
                    },
                    "checkpoint_2": {
                        "target": "trajectory",
                        "max_score": 0.5,
                        "title": "CP2",
                        "description": "Extract data",
                    },
                }
            },
        )
        assert len(task.scorers) == 2

        cp1 = next(s for s in task.scorers if s.scorer_name == "checkpoint_1")
        assert isinstance(cp1.criteria, LLMJudgeCriteria)
        assert cp1.criteria.model == "openai/gpt-4"
        assert cp1.criteria.steps_per_message == 50
        assert cp1.criteria.system_template == "cp_sys.md"
        assert cp1.criteria.user_template == "cp_user.md"

        cp2 = next(s for s in task.scorers if s.scorer_name == "checkpoint_2")
        assert isinstance(cp2.criteria, LLMJudgeCriteria)
        assert cp2.criteria.model == "openai/gpt-4"
        assert cp2.criteria.steps_per_message == 50

    def test_inline_defaults_overridden_by_scorer(self) -> None:
        """Per-scorer values override inline defaults."""
        task = self._make_task(
            scoring={
                "llm_judge": {
                    "model": "openai/gpt-4",
                    "judge_system_template": "default_sys.md",
                    "judge_user_template": "default_user.md",
                    "steps_per_message": 50,
                    "submission": {
                        "target": "submission",
                        "judge_system_template": "sub_sys.md",
                        "judge_user_template": "sub_user.md",
                        "description": "42",
                        "max_score": 1.0,
                    },
                    "checkpoint_1": {
                        "target": "trajectory",
                        "max_score": 0.5,
                        "title": "CP1",
                        "description": "test",
                    },
                }
            },
        )
        sub = next(s for s in task.scorers if s.scorer_name == "submission")
        assert isinstance(sub.criteria, LLMJudgeCriteria)
        assert sub.criteria.model == "openai/gpt-4"  # from inline default
        assert sub.criteria.system_template == "sub_sys.md"  # overridden
        assert sub.criteria.user_template == "sub_user.md"  # overridden
        assert sub.description == "42"

        cp = next(s for s in task.scorers if s.scorer_name == "checkpoint_1")
        assert isinstance(cp.criteria, LLMJudgeCriteria)
        assert cp.criteria.system_template == "default_sys.md"  # from inline default
        assert cp.criteria.steps_per_message == 50  # from inline default

    def test_scoring_defaults_override_inline(self) -> None:
        """scoring_defaults have higher priority than inline defaults."""
        task = self._make_task(
            scoring_defaults={
                "llm_judge": {
                    "model": "openai/gpt-4-turbo",  # higher priority
                }
            },
            scoring={
                "llm_judge": {
                    "model": "openai/gpt-4",  # inline default (lower priority)
                    "judge_system_template": "sys.md",
                    "judge_user_template": "user.md",
                    "checkpoint_1": {
                        "target": "trajectory",
                        "max_score": 1.0,
                        "description": "test",
                    },
                }
            },
        )
        cp = task.scorers[0]
        assert isinstance(cp.criteria, LLMJudgeCriteria)
        assert cp.criteria.model == "openai/gpt-4-turbo"  # from scoring_defaults

    def test_inline_defaults_with_mixed_strategies(self) -> None:
        """Inline defaults only apply within their own strategy block."""
        task = self._make_task(
            scoring={
                "llm_judge": {
                    "model": "openai/gpt-4",
                    "judge_system_template": "sys.md",
                    "judge_user_template": "user.md",
                    "cp1": {
                        "target": "trajectory",
                        "max_score": 1.0,
                        "description": "test",
                    },
                },
                "static": {
                    "sub": {
                        "target": "submission",
                        "expected_answers": ["42"],
                    }
                },
            }
        )
        assert len(task.scorers) == 2
        llm = next(s for s in task.scorers if s.strategy == "llm_judge")
        static = next(s for s in task.scorers if s.strategy == "static")
        assert isinstance(llm.criteria, LLMJudgeCriteria)
        assert llm.criteria.model == "openai/gpt-4"
        assert isinstance(static.criteria, StaticCriteria)

    def test_no_inline_defaults_works(self) -> None:
        """When no inline defaults present, behavior is unchanged."""
        task = self._make_task(
            scoring={
                "llm_judge": {
                    "cp": {
                        "model": "m",
                        "judge_system_template": "s",
                        "judge_user_template": "u",
                        "max_score": 1.0,
                    }
                }
            }
        )
        assert len(task.scorers) == 1
        assert isinstance(task.scorers[0].criteria, LLMJudgeCriteria)
