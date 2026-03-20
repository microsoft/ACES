"""Tests for ScorerFactory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai._util.registry import registry_unqualified_name
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target

from saber.config.models import (
    AggregationConfig,
    LLMJudgeCriteria,
    PromptPaths,
    ScoreAggregation,
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
    TaskConfig,
)
from saber.scoring.factory import CacheKeys, ScorerFactory, StrategyName, _is_batch_eligible
from saber.scoring.registry import ScoringStrategyRegistry

# ── Helpers ─────────────────────────────────────────────────────────


def _make_prompts() -> PromptPaths:
    return PromptPaths(instruction="instructions/test.md")


def _make_scorer(
    name: str = "checkpoint_1",
    strategy: str = "static",
    target: ScorerTarget = ScorerTarget.TRAJECTORY,
    max_score: float = 1.0,
    weight: float = 1.0,
    expected_answers: list[str] | None = None,
) -> ScorerConfig:
    return ScorerConfig(
        scorer_name=name,
        strategy=strategy,
        target=target,
        max_score=max_score,
        weight=weight,
        criteria=StaticCriteria(expected_answers=expected_answers or ["answer"]),
    )


def _make_task(
    *,
    task_id: str = "test_task",
    scorers: tuple[ScorerConfig, ...] = (),
    scoring_aggregation: AggregationConfig | None = None,
) -> TaskConfig:
    return TaskConfig(
        task_id=task_id,
        title="Test Task",
        description="A test task",
        prompts=_make_prompts(),
        scorers=scorers,
        scoring_aggregation=scoring_aggregation,
    )


def _make_mock_state(
    completion: str = "my answer",
    scores: dict[str, Score] | None = None,
    task_id: str = "test_task",
) -> MagicMock:
    state = MagicMock()
    state.output = MagicMock()
    state.output.completion = completion
    state.messages = [ChatMessageUser(content="test prompt")]
    state.metadata = {"task_id": task_id}
    state.scores = scores or {}
    return state


def _make_factory() -> ScorerFactory:
    return ScorerFactory(domain_root=Path("/fake/domain"))


# ── TestStrategyName ─────────────────────────────────────────────────


class TestStrategyName:
    """Verify StrategyName enum values and identity."""

    def test_enum_values(self) -> None:
        assert StrategyName.STATIC.value == "static"
        assert StrategyName.LLM_JUDGE.value == "llm_judge"
        assert StrategyName.NONE.value == "none"
        assert StrategyName.STATIC_JACCARD.value == "static_jaccard"
        assert StrategyName.TOOL_CALL.value == "tool_call"
        assert StrategyName.TOOL_CALL_COUNT.value == "tool_call_count"

    def test_string_lookup(self) -> None:
        assert StrategyName("llm_judge") is StrategyName.LLM_JUDGE
        assert StrategyName("static") is StrategyName.STATIC

    def test_str_comparison(self) -> None:
        """StrEnum members compare equal to their string values."""
        assert StrategyName.LLM_JUDGE == "llm_judge"
        assert StrategyName.STATIC == "static"


# ── TestCacheKeys ───────────────────────────────────────────────────


class TestCacheKeys:
    """Verify CacheKeys generates correct cache key strings."""

    def test_unified_scores_key(self) -> None:
        assert CacheKeys.unified_scores("task_1") == "_saber_all_scores_task_1"

    def test_unified_scores_empty_task_id(self) -> None:
        assert CacheKeys.unified_scores("") == "_saber_all_scores_"


# ── TestIsBatchEligible ─────────────────────────────────────────────


class TestIsBatchEligible:
    """Verify _is_batch_eligible predicate."""

    def test_llm_judge_trajectory_is_eligible(self) -> None:
        sc = _make_scorer(name="cp1", strategy="llm_judge", target=ScorerTarget.TRAJECTORY)
        # Need LLMJudgeCriteria for llm_judge
        sc = ScorerConfig(
            scorer_name="cp1",
            strategy="llm_judge",
            target=ScorerTarget.TRAJECTORY,
            criteria=LLMJudgeCriteria(
                model="mockmodel/test",
                judge_system_template="s.txt",
                judge_user_template="u.txt",
            ),
        )
        assert _is_batch_eligible(sc) is True

    def test_llm_judge_submission_not_eligible(self) -> None:
        sc = ScorerConfig(
            scorer_name="sub",
            strategy="llm_judge",
            target=ScorerTarget.SUBMISSION,
            criteria=LLMJudgeCriteria(
                model="mockmodel/test",
                judge_system_template="s.txt",
                judge_user_template="u.txt",
            ),
        )
        assert _is_batch_eligible(sc) is False

    def test_static_trajectory_not_eligible(self) -> None:
        sc = _make_scorer(name="cp1", strategy="static", target=ScorerTarget.TRAJECTORY)
        assert _is_batch_eligible(sc) is False

    def test_none_strategy_not_eligible(self) -> None:
        sc = _make_scorer(name="cp1", strategy="none", target=ScorerTarget.TRAJECTORY)
        assert _is_batch_eligible(sc) is False


# ── TestCreateScorersCount ──────────────────────────────────────────


class TestCreateScorersCount:
    """Verify scorer counts: N unit scorers + 1 aggregate."""

    def test_zero_scorers_returns_one_aggregate(self) -> None:
        factory = _make_factory()
        task = _make_task(scorers=())
        result = factory.create_scorers(task)
        assert len(result) == 1

    def test_one_scorer_returns_two(self) -> None:
        factory = _make_factory()
        task = _make_task(scorers=(_make_scorer(),))
        result = factory.create_scorers(task)
        assert len(result) == 2

    def test_three_scorers_returns_four(self) -> None:
        factory = _make_factory()
        scorers = (
            _make_scorer(name="cp1"),
            _make_scorer(name="cp2"),
            _make_scorer(name="cp3"),
        )
        task = _make_task(scorers=scorers)
        result = factory.create_scorers(task)
        assert len(result) == 4


# ── TestUnitScorer ──────────────────────────────────────────────────


class TestUnitScorer:
    """Test per-checkpoint unit scorer behavior."""

    @pytest.mark.asyncio
    async def test_static_match_scores_max(self) -> None:
        factory = _make_factory()
        sc = _make_scorer(
            name="sub",
            target=ScorerTarget.SUBMISSION,
            expected_answers=["my answer"],
        )
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        unit = scorers[0]  # first is the unit scorer

        state = _make_mock_state(completion="my answer")
        target = Target("expected")
        result = await unit(state, target)

        assert result is not None
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_static_no_match_scores_zero(self) -> None:
        factory = _make_factory()
        sc = _make_scorer(
            name="sub",
            target=ScorerTarget.SUBMISSION,
            expected_answers=["correct answer"],
        )
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        unit = scorers[0]

        state = _make_mock_state(completion="wrong answer")
        target = Target("expected")
        result = await unit(state, target)

        assert result is not None
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_none_strategy_scores_zero(self) -> None:
        factory = _make_factory()
        sc = _make_scorer(name="noop", strategy="none")
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        unit = scorers[0]

        state = _make_mock_state()
        target = Target("expected")
        result = await unit(state, target)

        assert result is not None
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_wrong_task_id_returns_none(self) -> None:
        factory = _make_factory()
        sc = _make_scorer(name="sub", target=ScorerTarget.SUBMISSION)
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        unit = scorers[0]

        state = _make_mock_state(task_id="other_task")
        target = Target("expected")
        result = await unit(state, target)

        assert result is None


# ── TestAggregateScorer ─────────────────────────────────────────────


class TestAggregateScorer:
    """Test the aggregate scorer that reads state.scores."""

    @pytest.mark.asyncio
    async def test_aggregate_with_scores(self) -> None:
        factory = _make_factory()
        sc = _make_scorer(name="cp1", max_score=1.0, weight=1.0)
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        aggregate = scorers[-1]  # aggregate is last

        state = _make_mock_state(
            scores={"test_task.cp1": Score(value=1.0)},
        )
        target = Target("expected")
        result = await aggregate(state, target)

        assert result is not None
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_aggregate_wrong_task_id_returns_none(self) -> None:
        factory = _make_factory()
        sc = _make_scorer(name="cp1")
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        aggregate = scorers[-1]

        state = _make_mock_state(task_id="other_task")
        target = Target("expected")
        result = await aggregate(state, target)

        assert result is None

    @pytest.mark.asyncio
    async def test_aggregate_empty_scores_returns_zero(self) -> None:
        factory = _make_factory()
        sc = _make_scorer(name="cp1")
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        aggregate = scorers[-1]

        state = _make_mock_state(scores={})
        target = Target("expected")
        result = await aggregate(state, target)

        assert result is not None
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_aggregate_with_max_aggregation(self) -> None:
        factory = _make_factory()
        scorers_cfg = (
            _make_scorer(name="cp1", max_score=1.0, weight=1.0),
            _make_scorer(name="cp2", max_score=1.0, weight=1.0),
        )
        agg = AggregationConfig(
            strategy=ScoreAggregation.MAX,
            scores=["cp1", "cp2"],
        )
        task = _make_task(scorers=scorers_cfg, scoring_aggregation=agg)
        all_scorers = factory.create_scorers(task)
        aggregate = all_scorers[-1]

        # cp1=1.0, cp2=0.0 → max aggregation → 1.0
        state = _make_mock_state(
            scores={
                "test_task.cp1": Score(value=1.0),
                "test_task.cp2": Score(value=0.0),
            },
        )
        target = Target("expected")
        result = await aggregate(state, target)

        assert result is not None
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_aggregate_average_normalized(self) -> None:
        """Default flat aggregation: average normalized by max possible."""
        factory = _make_factory()
        scorers_cfg = (
            _make_scorer(name="cp1", max_score=2.0, weight=1.0),
            _make_scorer(name="cp2", max_score=2.0, weight=1.0),
        )
        task = _make_task(scorers=scorers_cfg)
        all_scorers = factory.create_scorers(task)
        aggregate = all_scorers[-1]

        # cp1=2.0, cp2=0.0 → (2*1 + 0*1) / (2*1 + 2*1) = 0.5
        state = _make_mock_state(
            scores={
                "test_task.cp1": Score(value=2.0),
                "test_task.cp2": Score(value=0.0),
            },
        )
        target = Target("expected")
        result = await aggregate(state, target)

        assert result is not None
        assert result.value == pytest.approx(0.5)


# ── TestFactoryValidation ──────────────────────────────────────────


class TestFactoryValidation:
    """Test validation in ScorerFactory."""

    def test_unknown_strategy_raises(self) -> None:
        factory = _make_factory()
        sc = ScorerConfig(
            scorer_name="bad",
            strategy="nonexistent_strategy",
        )
        task = _make_task(scorers=(sc,))
        with pytest.raises(ValueError, match="unregistered scoring strategies"):
            factory.create_scorers(task)

    def test_aggregation_unknown_scorer_name_raises(self) -> None:
        factory = _make_factory()
        sc = _make_scorer(name="cp1")
        agg = AggregationConfig(
            strategy=ScoreAggregation.MAX,
            scores=["cp1", "unknown_scorer"],
        )
        task = _make_task(scorers=(sc,), scoring_aggregation=agg)
        with pytest.raises(ValueError, match="unknown scorer names"):
            factory.create_scorers(task)


# ── TestFactoryCustomRegistry ──────────────────────────────────────


class TestFactoryCustomRegistry:
    """Test factory with custom strategy registry."""

    def test_custom_strategy_used(self) -> None:
        registry = ScoringStrategyRegistry()
        factory = ScorerFactory(domain_root=Path("/fake/domain"), registry=registry)
        sc = _make_scorer(name="noop", strategy="none")
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        # 1 unit + 1 aggregate
        assert len(scorers) == 2

    def test_unknown_strategy_in_custom_registry_raises(self) -> None:
        registry = ScoringStrategyRegistry()
        factory = ScorerFactory(domain_root=Path("/fake/domain"), registry=registry)
        sc = ScorerConfig(
            scorer_name="bad",
            strategy="totally_custom",
        )
        task = _make_task(scorers=(sc,))
        with pytest.raises(ValueError, match="unregistered scoring strategies"):
            factory.create_scorers(task)


# ── TestOverallScorer ───────────────────────────────────────────────


class TestOverallScorer:
    """Test the saber_overall scorer creation."""

    def test_create_overall_scorer_returns_scorer(self) -> None:
        factory = _make_factory()
        task = _make_task(scorers=(_make_scorer(),))
        overall = factory.create_overall_scorer([task])
        assert callable(overall)

    @pytest.mark.asyncio
    async def test_overall_scorer_unknown_task_id_returns_zero(self) -> None:
        factory = _make_factory()
        task = _make_task(scorers=(_make_scorer(),))
        overall = factory.create_overall_scorer([task])

        state = _make_mock_state(task_id="nonexistent_task")
        target = Target("expected")
        result = await overall(state, target)

        assert result is not None
        assert result.value == 0.0


# ── TestExceptionPropagation ────────────────────────────────────────


def _make_llm_scorer(name: str = "llm_cp1") -> ScorerConfig:
    """Create an llm_judge ScorerConfig for testing batch paths."""
    return ScorerConfig(
        scorer_name=name,
        strategy="llm_judge",
        target=ScorerTarget.TRAJECTORY,
        max_score=1.0,
        weight=1.0,
        criteria=LLMJudgeCriteria(
            model="mockmodel/test",
            judge_system_template="system.txt",
            judge_user_template="user.txt",
        ),
    )


def _make_llm_submission_scorer(name: str = "submission") -> ScorerConfig:
    """Create an llm_judge submission ScorerConfig for testing."""
    return ScorerConfig(
        scorer_name=name,
        strategy="llm_judge",
        target=ScorerTarget.SUBMISSION,
        max_score=1.0,
        weight=1.0,
        criteria=LLMJudgeCriteria(
            model="mockmodel/test",
            judge_system_template="submission_system.txt",
            judge_user_template="submission_user.txt",
        ),
    )


class TestSubmissionScorerRouting:
    """Verify LLM judge submission scorers are individually scored, not batched."""

    def test_llm_submission_not_in_batch(self) -> None:
        """Mixed submission + checkpoint: correct scorer count."""
        factory = _make_factory()
        sub = _make_llm_submission_scorer()
        cp = _make_llm_scorer("cp1")
        task = _make_task(scorers=(sub, cp))

        scorers = factory.create_scorers(task)
        # 2 unit scorers + 1 aggregate = 3
        assert len(scorers) == 3

    def test_llm_submission_only_no_batch(self) -> None:
        """When only submission llm_judge scorers exist, correct count returned."""
        factory = _make_factory()
        sub = _make_llm_submission_scorer()
        task = _make_task(scorers=(sub,))

        scorers = factory.create_scorers(task)
        # 1 unit scorer + 1 aggregate = 2
        assert len(scorers) == 2


# ── TestExceptionPropagation ────────────────────────────────────────


class TestExceptionPropagation:
    """Verify that batch scoring exceptions propagate instead of being swallowed."""

    @pytest.mark.asyncio
    async def test_compute_task_aggregate_propagates_batch_error(self) -> None:
        """score_checkpoints_llm_batch error must propagate from compute_task_aggregate."""
        factory = _make_factory()
        llm_sc = _make_llm_scorer()
        task = _make_task(scorers=(llm_sc,))

        state = _make_mock_state(task_id="test_task")
        target = Target("expected")

        with patch(
            "saber.scoring.batch.score_checkpoints_llm_batch",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM provider exploded"),
        ):
            with pytest.raises(RuntimeError, match="LLM provider exploded"):
                await factory.compute_task_aggregate(task, state, target)

    @pytest.mark.asyncio
    async def test_batch_unit_scorer_propagates_batch_error(self) -> None:
        """LLM judge strategy error must propagate from unit scorer."""
        factory = _make_factory()
        llm_sc = _make_llm_scorer()
        task = _make_task(scorers=(llm_sc,))

        mock_strategy = AsyncMock(side_effect=RuntimeError("strategy kaboom"))

        with patch.object(
            factory._registry.get("llm_judge"),
            "score",
            mock_strategy,
        ):
            scorers = factory.create_scorers(task)
            unit_scorer = scorers[0]

            state = _make_mock_state(task_id="test_task")
            target = Target("expected")

            with pytest.raises(RuntimeError, match="strategy kaboom"):
                await unit_scorer(state, target)


# ── TestUnifiedScoreCache ───────────────────────────────────────────


class TestUnifiedScoreCache:
    """Verify saber_overall populates a unified cache that individual scorers read."""

    @pytest.mark.asyncio
    async def test_compute_task_aggregate_populates_unified_cache(self) -> None:
        """compute_task_aggregate must write all scores to state.metadata."""
        factory = _make_factory()
        sc = _make_scorer(name="sub", target=ScorerTarget.SUBMISSION, expected_answers=["correct"])
        task = _make_task(scorers=(sc,))

        state = _make_mock_state(task_id="test_task", completion="correct")
        target = Target("expected")
        await factory.compute_task_aggregate(task, state, target)

        cache_key = "_saber_all_scores_test_task"
        assert cache_key in state.metadata
        assert "sub" in state.metadata[cache_key]
        assert state.metadata[cache_key]["sub"].value == 1.0
        # Batch cache key must NOT exist
        assert "_saber_batch_cfg_test_task" not in state.metadata

    @pytest.mark.asyncio
    async def test_unit_scorer_reads_unified_cache(self) -> None:
        """Unit scorers must return cached results without re-running the strategy."""
        factory = _make_factory()
        sc = _make_scorer(name="sub", target=ScorerTarget.SUBMISSION)
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        unit = scorers[0]

        # Pre-populate the unified cache as saber_overall would
        cached_score = Score(value=1.0, answer="cached answer", explanation="from cache")
        state = _make_mock_state(task_id="test_task", completion="wrong answer")
        state.metadata["_saber_all_scores_test_task"] = {"sub": cached_score}

        target = Target("expected")
        result = await unit(state, target)

        assert result is not None
        assert result.value == 1.0
        assert result.explanation == "from cache"


class TestRuntimeScorers:
    """Verify runtime scorer dispatch avoids cross-task scorer explosions."""

    def test_create_runtime_scorers_deduplicates_logical_names(self) -> None:
        factory = _make_factory()
        task_one = _make_task(
            task_id="task_one",
            scorers=(
                _make_scorer(
                    name="submission",
                    target=ScorerTarget.SUBMISSION,
                    expected_answers=["alpha"],
                ),
                _make_scorer(name="checkpoint_1"),
            ),
        )
        task_two = _make_task(
            task_id="task_two",
            scorers=(
                _make_scorer(
                    name="submission",
                    target=ScorerTarget.SUBMISSION,
                    expected_answers=["beta"],
                ),
                _make_scorer(name="checkpoint_2"),
            ),
        )

        runtime_scorers = factory.create_runtime_scorers([task_one, task_two])

        assert [registry_unqualified_name(scorer) for scorer in runtime_scorers] == [
            "saber_overall",
            "submission",
            "checkpoint_1",
            "checkpoint_2",
            "aggregate",
        ]

    @pytest.mark.asyncio
    async def test_runtime_unit_scorer_dispatches_by_current_task_id(self) -> None:
        factory = _make_factory()
        task_one = _make_task(
            task_id="task_one",
            scorers=(
                _make_scorer(
                    name="submission",
                    target=ScorerTarget.SUBMISSION,
                    expected_answers=["alpha"],
                ),
            ),
        )
        task_two = _make_task(
            task_id="task_two",
            scorers=(
                _make_scorer(
                    name="submission",
                    target=ScorerTarget.SUBMISSION,
                    expected_answers=["beta"],
                ),
            ),
        )

        runtime_scorers = factory.create_runtime_scorers([task_one, task_two])
        submission = next(
            scorer
            for scorer in runtime_scorers
            if registry_unqualified_name(scorer) == "submission"
        )

        result = await submission(
            _make_mock_state(task_id="task_two", completion="beta"),
            Target("expected"),
        )

        assert result is not None
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_runtime_aggregate_reads_generic_score_names(self) -> None:
        factory = _make_factory()
        task = _make_task(
            task_id="task_one",
            scorers=(
                _make_scorer(name="checkpoint_1", max_score=1.0),
                _make_scorer(name="checkpoint_2", max_score=1.0),
            ),
            scoring_aggregation=AggregationConfig(
                strategy=ScoreAggregation.AVERAGE,
                scores=["checkpoint_1", "checkpoint_2"],
            ),
        )

        runtime_scorers = factory.create_runtime_scorers([task])
        aggregate = next(
            scorer
            for scorer in runtime_scorers
            if registry_unqualified_name(scorer) == "aggregate"
        )

        result = await aggregate(
            _make_mock_state(
                task_id="task_one",
                scores={
                    "checkpoint_1": Score(value=1.0),
                    "checkpoint_2": Score(value=0.0),
                },
            ),
            Target("expected"),
        )

        assert result is not None
        assert result.value == 0.5
        assert result.explanation is not None
        assert "checkpoint_1" in result.explanation
        assert "checkpoint_2" in result.explanation

    @pytest.mark.asyncio
    async def test_llm_checkpoint_scorer_reads_unified_cache(self) -> None:
        """LLM checkpoint scorers must return cached results from unified cache."""
        factory = _make_factory()
        llm_sc = _make_llm_scorer("cp1")
        task = _make_task(scorers=(llm_sc,))

        mock_strategy = AsyncMock(side_effect=RuntimeError("should not be called"))

        with patch.object(
            factory._registry.get("llm_judge"),
            "score",
            mock_strategy,
        ):
            scorers = factory.create_scorers(task)
            unit_scorer = scorers[0]

            # Pre-populate the unified cache
            cached_score = Score(value=1.0, answer="cached", explanation="from unified cache")
            state = _make_mock_state(task_id="test_task")
            state.metadata["_saber_all_scores_test_task"] = {"cp1": cached_score}

            target = Target("expected")
            result = await unit_scorer(state, target)

            assert result is not None
            assert result.value == 1.0
            assert result.explanation == "from unified cache"
            mock_strategy.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_judge_checkpoint_fallback_to_strategy(self) -> None:
        """LLM judge checkpoint scorer falls back to strategy without cache."""
        factory = _make_factory()
        llm_sc = _make_llm_scorer("cp1")
        task = _make_task(scorers=(llm_sc,))

        mock_strategy = AsyncMock(return_value=Score(value=1.0, explanation="from strategy"))

        with patch.object(
            factory._registry.get("llm_judge"),
            "score",
            mock_strategy,
        ):
            scorers = factory.create_scorers(task)
            unit_scorer = scorers[0]

            # No unified cache — scorer should call strategy directly
            state = _make_mock_state(task_id="test_task")
            target = Target("expected")
            result = await unit_scorer(state, target)

            assert result is not None
            assert result.value == 1.0
            assert result.explanation == "from strategy"
            mock_strategy.assert_called_once()

    @pytest.mark.asyncio
    async def test_unit_scorer_falls_through_without_cache(self) -> None:
        """Without unified cache, unit scorers must still run strategy directly."""
        factory = _make_factory()
        sc = _make_scorer(name="sub", target=ScorerTarget.SUBMISSION, expected_answers=["correct"])
        task = _make_task(scorers=(sc,))
        scorers = factory.create_scorers(task)
        unit = scorers[0]

        state = _make_mock_state(task_id="test_task", completion="correct")
        target = Target("expected")
        result = await unit(state, target)

        assert result is not None
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_compute_task_aggregate_batch_happy_path(self) -> None:
        """compute_task_aggregate batches LLM judge checkpoint scorers and caches results."""
        factory = _make_factory()
        cp1 = _make_llm_scorer("cp1")
        cp2 = _make_llm_scorer("cp2")
        task = _make_task(scorers=(cp1, cp2))

        state = _make_mock_state(task_id="test_task")
        target = Target("expected")

        mock_batch_results = {
            "cp1": Score(value=1.0, explanation="cp1 passed"),
            "cp2": Score(value=0.5, explanation="cp2 partial"),
        }

        with patch(
            "saber.scoring.batch.score_checkpoints_llm_batch",
            new_callable=AsyncMock,
            return_value=mock_batch_results,
        ):
            result = await factory.compute_task_aggregate(task, state, target)

        # Aggregate should reflect both scores
        assert result is not None
        assert result.value > 0.0

        # Both scores should be in the unified cache
        from saber.scoring.factory import CacheKeys

        cache = state.metadata[CacheKeys.unified_scores("test_task")]
        assert "cp1" in cache
        assert cache["cp1"].value == 1.0
        assert "cp2" in cache
        assert cache["cp2"].value == 0.5

    @pytest.mark.asyncio
    async def test_overall_scorer_happy_path(self) -> None:
        """saber_overall scorer returns aggregate score for a known task."""
        factory = _make_factory()
        sc = _make_scorer(name="sub", target=ScorerTarget.SUBMISSION, expected_answers=["correct"])
        task = _make_task(scorers=(sc,))
        overall = factory.create_overall_scorer([task])

        state = _make_mock_state(task_id="test_task", completion="correct")
        target = Target("expected")
        result = await overall(state, target)

        assert result is not None
        assert result.value == 1.0
