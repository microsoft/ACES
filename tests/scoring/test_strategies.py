"""Tests for SaberScoringStrategy protocol and built-in strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.model import ChatMessageUser

from saber.config.models import (
    LLMJudgeCriteria,
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
    ToolCallCriteria,
)
from saber.scoring.context import (
    CheckpointObjective,
    EpisodeContext,
    JudgeTemplateContext,
    ScoringContext,
    ToolStep,
)
from saber.scoring.strategies import (
    LLMJudgeStrategy,
    NoneStrategy,
    SaberScoringStrategy,
    StaticJaccardStrategy,
    StaticStrategy,
    ToolCallCountStrategy,
    ToolCallStrategy,
    _build_judge_context,
    _normalize_tool_name,
)

# ── Helper ──────────────────────────────────────────────────────────


def _make_ctx(
    *,
    submission: str = "",
    tool_steps: tuple[ToolStep, ...] = (),
    target: str = "",
    strategy: str = "static",
    criteria: StaticCriteria | None = None,
    max_score: float = 1.0,
    weight: float = 1.0,
    scorer_name: str | None = None,
    scorer_target: ScorerTarget | None = None,
) -> ScoringContext:
    resolved_criteria = criteria or StaticCriteria(expected_answers=["answer"])

    resolved_target = scorer_target if scorer_target is not None else ScorerTarget.SUBMISSION
    resolved_name = scorer_name or ("submission" if resolved_target == ScorerTarget.SUBMISSION else "checkpoint")

    scorer_config = ScorerConfig(
        scorer_name=resolved_name,
        strategy=strategy,
        target=resolved_target,
        criteria=resolved_criteria,
        max_score=max_score,
        weight=weight,
    )

    return ScoringContext(
        submission=submission,
        tool_steps=tool_steps,
        messages=(ChatMessageUser(content="test"),),
        target=target,
        task_id="test_task",
        domain="test",
        metadata={},
        scorer=scorer_config,
    )


def _make_tool_step(
    *,
    step_number: int = 1,
    tool_name: str = "run",
    output: str = "",
) -> ToolStep:
    return ToolStep(
        step_number=step_number,
        tool_name=tool_name,
        tool_input={},
        output=output,
    )


# ── NoneStrategy ────────────────────────────────────────────────────


class TestNoneStrategy:
    """NoneStrategy always returns Score(value=0.0)."""

    async def test_returns_zero(self) -> None:
        ctx = _make_ctx(submission="anything")
        score = await NoneStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_answer_is_submission(self) -> None:
        ctx = _make_ctx(submission="my answer")
        score = await NoneStrategy().score(ctx, renderer=object())
        assert score.answer == "my answer"

    async def test_explanation_mentions_no_evaluation(self) -> None:
        ctx = _make_ctx(submission="x")
        score = await NoneStrategy().score(ctx, renderer=object())
        assert score.explanation is not None
        assert "No evaluation configured" in score.explanation

    async def test_returns_zero_regardless_of_submission(self) -> None:
        for sub in ["", "correct answer", "answer", "  "]:
            ctx = _make_ctx(submission=sub)
            score = await NoneStrategy().score(ctx, renderer=object())
            assert score.value == 0.0


# ── StaticStrategy — submission scoring ─────────────────────────────


class TestStaticStrategySubmission:
    """StaticStrategy with submission target checks submission text."""

    async def test_exact_match(self) -> None:
        ctx = _make_ctx(
            submission="answer",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_substring_match(self) -> None:
        ctx = _make_ctx(
            submission="the answer is here",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_case_insensitive(self) -> None:
        ctx = _make_ctx(
            submission="ANSWER",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_no_match(self) -> None:
        ctx = _make_ctx(
            submission="something else",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_multiple_expected_first_match_wins(self) -> None:
        ctx = _make_ctx(
            submission="beta result",
            criteria=StaticCriteria(expected_answers=["alpha", "beta"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 1.0
        assert score.explanation is not None
        assert "beta" in score.explanation

    async def test_whitespace_stripping(self) -> None:
        ctx = _make_ctx(
            submission="  answer  ",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_no_match_explanation_lists_expected(self) -> None:
        ctx = _make_ctx(
            submission="nope",
            criteria=StaticCriteria(expected_answers=["x", "y"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0
        assert score.explanation is not None
        assert "x" in score.explanation
        assert "y" in score.explanation

    async def test_answer_field_is_submission(self) -> None:
        ctx = _make_ctx(submission="my sub")
        score = await StaticStrategy().score(ctx, renderer=object())
        # Regardless of match, answer should be the submission
        assert score.answer == "my sub"

    async def test_empty_submission_no_match(self) -> None:
        ctx = _make_ctx(
            submission="",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0


# ── StaticStrategy — subtask scoring ────────────────────────────────


class TestStaticStrategySubtask:
    """StaticStrategy with ctx.subtask set scans tool outputs."""

    async def test_match_in_tool_output(self) -> None:
        ctx = _make_ctx(
            submission="",
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_steps=(_make_tool_step(output="the answer is here"),),
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_match_across_multiple_outputs(self) -> None:
        """Match spans the join boundary (outputs joined with space)."""
        ctx = _make_ctx(
            submission="",
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_steps=(
                _make_tool_step(step_number=1, output="found the"),
                _make_tool_step(step_number=2, output="answer here"),
            ),
            criteria=StaticCriteria(expected_answers=["the answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_no_match_in_tool_outputs(self) -> None:
        ctx = _make_ctx(
            submission="",
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_steps=(_make_tool_step(output="nothing useful"),),
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_empty_tool_steps(self) -> None:
        ctx = _make_ctx(
            submission="",
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_steps=(),
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0


# ── StaticStrategy — max_score scaling ──────────────────────────────


class TestStaticStrategyMaxScore:
    """StaticStrategy returns configured max_score on match."""

    async def test_max_score_half(self) -> None:
        ctx = _make_ctx(
            submission="answer",
            max_score=0.5,
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.5

    async def test_max_score_two(self) -> None:
        ctx = _make_ctx(
            submission="answer",
            max_score=2.0,
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 2.0

    async def test_no_match_always_zero(self) -> None:
        ctx = _make_ctx(
            submission="wrong",
            max_score=5.0,
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0


# ── Protocol compliance ─────────────────────────────────────────────


class TestProtocolCompliance:
    """StaticStrategy and NoneStrategy satisfy SaberScoringStrategy."""

    def test_static_strategy_satisfies_protocol(self) -> None:
        strategy: SaberScoringStrategy = StaticStrategy()
        assert hasattr(strategy, "score")

    def test_none_strategy_satisfies_protocol(self) -> None:
        strategy: SaberScoringStrategy = NoneStrategy()
        assert hasattr(strategy, "score")

    async def test_static_callable_through_protocol(self) -> None:
        strategy: SaberScoringStrategy = StaticStrategy()
        ctx = _make_ctx(submission="answer")
        score = await strategy.score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_none_callable_through_protocol(self) -> None:
        strategy: SaberScoringStrategy = NoneStrategy()
        ctx = _make_ctx(submission="answer")
        score = await strategy.score(ctx, renderer=object())
        assert score.value == 0.0


# ── StaticStrategy — empty/blank answer handling ────────────────────


class TestStaticStrategyEmptyAnswers:
    """Empty or whitespace-only expected answers must NOT match everything."""

    async def test_empty_expected_answer_does_not_match(self) -> None:
        ctx = _make_ctx(
            submission="anything at all",
            criteria=StaticCriteria(expected_answers=[""]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_whitespace_only_expected_answer_does_not_match(self) -> None:
        ctx = _make_ctx(
            submission="anything at all",
            criteria=StaticCriteria(expected_answers=["   "]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_mixed_empty_and_valid_answers(self) -> None:
        """Empty answers are skipped; valid answers still match."""
        ctx = _make_ctx(
            submission="found the answer",
            criteria=StaticCriteria(expected_answers=["", "answer", "   "]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 1.0
        assert score.explanation is not None
        assert "answer" in score.explanation

    async def test_all_empty_answers_no_match(self) -> None:
        ctx = _make_ctx(
            submission="any text",
            criteria=StaticCriteria(expected_answers=["", "  ", "\t"]),
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0


# ── StaticStrategy — criteria type mismatch ─────────────────────────


class TestStaticStrategyCriteriaMismatch:
    """StaticStrategy gracefully handles non-StaticCriteria."""

    async def test_llm_judge_criteria_returns_zero(self) -> None:
        llm_criteria = LLMJudgeCriteria(
            model="gpt-4",
            judge_system_template="sys.j2",
            judge_user_template="usr.j2",
        )
        scorer_config = ScorerConfig(
            scorer_name="submission",
            strategy="llm_judge",
            target=ScorerTarget.SUBMISSION,
            criteria=llm_criteria,
            max_score=1.0,
            weight=1.0,
        )
        ctx = ScoringContext(
            submission="answer",
            tool_steps=(),
            messages=(ChatMessageUser(content="test"),),
            target="",
            task_id="test_task",
            domain="test",
            metadata={},
            scorer=scorer_config,
        )
        score = await StaticStrategy().score(ctx, renderer=object())
        assert score.value == 0.0
        assert score.explanation is not None
        assert "StaticCriteria" in score.explanation
        assert "LLMJudgeCriteria" in score.explanation


# ── Helpers for ToolCallCriteria-based tests ────────────────────────


def _make_tool_call_ctx(
    *,
    submission: str = "",
    tool_steps: tuple[ToolStep, ...] = (),
    target: str = "",
    tool_name: str = "run",
    min_executions: int = 1,
    expected_tools: list[str] | None = None,
    max_score: float = 1.0,
    scorer_target: ScorerTarget | None = None,
) -> ScoringContext:
    tool_criteria = ToolCallCriteria(
        tool_name=tool_name,
        min_executions=min_executions,
        expected_tools=expected_tools or [],
    )

    resolved_target = scorer_target if scorer_target is not None else ScorerTarget.SUBMISSION
    resolved_name = "submission" if resolved_target == ScorerTarget.SUBMISSION else "checkpoint"

    scorer_config = ScorerConfig(
        scorer_name=resolved_name,
        strategy="tool_call",
        target=resolved_target,
        criteria=tool_criteria,
        max_score=max_score,
        weight=1.0,
    )

    return ScoringContext(
        submission=submission,
        tool_steps=tool_steps,
        messages=(ChatMessageUser(content="test"),),
        target=target,
        task_id="test_task",
        domain="test",
        metadata={},
        scorer=scorer_config,
    )


def _make_llm_judge_ctx(
    *,
    submission: str = "my answer",
    tool_steps: tuple[ToolStep, ...] = (),
    target: str = "expected",
    scorer_target: ScorerTarget | None = None,
    scorer_name: str | None = None,
    title: str = "",
    description: str = "",
    metadata: dict[str, object] | None = None,
) -> ScoringContext:
    llm_criteria = LLMJudgeCriteria(
        model="gpt-4",
        judge_system_template="sys.j2",
        judge_user_template="usr.j2",
    )

    resolved_target = scorer_target if scorer_target is not None else ScorerTarget.SUBMISSION
    resolved_name = scorer_name or ("submission" if resolved_target == ScorerTarget.SUBMISSION else "checkpoint")

    scorer_config = ScorerConfig(
        scorer_name=resolved_name,
        strategy="llm_judge",
        target=resolved_target,
        criteria=llm_criteria,
        max_score=1.0,
        weight=1.0,
        title=title,
        description=description or "gold",
    )

    return ScoringContext(
        submission=submission,
        tool_steps=tool_steps,
        messages=(ChatMessageUser(content="test"),),
        target=target,
        task_id="task_42",
        domain="test_domain",
        metadata=metadata or {"description": "A test task"},
        scorer=scorer_config,
    )


# ── ToolCallStrategy ───────────────────────────────────────────────


class TestToolCallStrategy:
    """ToolCallStrategy binary pass/fail for expected tool calls."""

    async def test_returns_zero_for_submission(self) -> None:
        ctx = _make_tool_call_ctx()
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 0.0
        assert "N/A" in (score.explanation or "")

    async def test_finds_expected_tool(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            tool_steps=(_make_tool_step(tool_name="run", output="ok"),),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_no_match(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            tool_steps=(_make_tool_step(tool_name="read_file", output="ok"),),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_multiple_expected_one_matches(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            expected_tools=["run", "exec"],
            tool_steps=(_make_tool_step(tool_name="exec", output="ok"),),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_empty_tool_steps(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            tool_steps=(),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_wrong_criteria_type(self) -> None:
        ctx = _make_ctx(
            submission="x",
            scorer_target=ScorerTarget.TRAJECTORY,
            criteria=StaticCriteria(expected_answers=["a"]),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 0.0
        assert "ToolCallCriteria" in (score.explanation or "")

    async def test_max_score_scaling(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            tool_steps=(_make_tool_step(tool_name="run", output="ok"),),
            max_score=5.0,
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 5.0


# ── ToolCallCountStrategy ──────────────────────────────────────────


class TestToolCallCountStrategy:
    """ToolCallCountStrategy binary pass if tool executed >= threshold."""

    async def test_returns_zero_for_submission(self) -> None:
        ctx = _make_tool_call_ctx()
        score = await ToolCallCountStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_count_meets_threshold(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            min_executions=2,
            tool_steps=(
                _make_tool_step(step_number=1, tool_name="run", output="a"),
                _make_tool_step(step_number=2, tool_name="run", output="b"),
                _make_tool_step(step_number=3, tool_name="run", output="c"),
            ),
        )
        score = await ToolCallCountStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_count_below_threshold(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            min_executions=3,
            tool_steps=(_make_tool_step(step_number=1, tool_name="run", output="a"),),
        )
        score = await ToolCallCountStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_exactly_at_threshold(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            min_executions=2,
            tool_steps=(
                _make_tool_step(step_number=1, tool_name="run", output="a"),
                _make_tool_step(step_number=2, tool_name="run", output="b"),
            ),
        )
        score = await ToolCallCountStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_wrong_criteria_type(self) -> None:
        ctx = _make_ctx(
            submission="x",
            scorer_target=ScorerTarget.TRAJECTORY,
            criteria=StaticCriteria(expected_answers=["a"]),
        )
        score = await ToolCallCountStrategy().score(ctx, renderer=object())
        assert score.value == 0.0
        assert "ToolCallCountStrategy" in (score.explanation or "")


# ── StaticJaccardStrategy ──────────────────────────────────────────


class TestStaticJaccardStrategy:
    """StaticJaccardStrategy — Jaccard similarity scoring."""

    async def test_perfect_overlap(self) -> None:
        ctx = _make_ctx(
            submission="answer",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticJaccardStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_no_overlap(self) -> None:
        ctx = _make_ctx(
            submission="completely different",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticJaccardStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_partial_overlap(self) -> None:
        # expected = {"alpha", "beta"}, actual = {"alpha", "gamma"}
        # intersection = {"alpha"}, union = {"alpha", "beta", "gamma"}
        # jaccard = 1/3
        ctx = _make_ctx(
            submission="alpha gamma",
            criteria=StaticCriteria(expected_answers=["alpha", "beta"]),
            max_score=3.0,
        )
        score = await StaticJaccardStrategy().score(ctx, renderer=object())
        assert score.value == pytest.approx(1.0, abs=0.01)

    async def test_empty_expected_set(self) -> None:
        ctx = _make_ctx(
            submission="anything",
            criteria=StaticCriteria(expected_answers=[""]),
        )
        score = await StaticJaccardStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_empty_actual_text(self) -> None:
        ctx = _make_ctx(
            submission="",
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticJaccardStrategy().score(ctx, renderer=object())
        assert score.value == 0.0

    async def test_subtask_mode_scans_tool_outputs(self) -> None:
        ctx = _make_ctx(
            submission="",
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_steps=(_make_tool_step(step_number=1, output="answer found"),),
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        score = await StaticJaccardStrategy().score(ctx, renderer=object())
        # expected={"answer"}, actual={"answer", "found"} → 1/2 = 0.5
        assert score.value == pytest.approx(0.5, abs=0.01)

    async def test_wrong_criteria_type(self) -> None:
        llm_criteria = LLMJudgeCriteria(
            model="gpt-4",
            judge_system_template="sys.j2",
            judge_user_template="usr.j2",
        )
        scorer_config = ScorerConfig(
            scorer_name="submission",
            strategy="llm_judge",
            target=ScorerTarget.SUBMISSION,
            criteria=llm_criteria,
            max_score=1.0,
            weight=1.0,
        )
        ctx = ScoringContext(
            submission="x",
            tool_steps=(),
            messages=(ChatMessageUser(content="test"),),
            target="",
            task_id="test_task",
            domain="test",
            metadata={},
            scorer=scorer_config,
        )
        score = await StaticJaccardStrategy().score(ctx, renderer=object())
        assert score.value == 0.0
        assert "StaticCriteria" in (score.explanation or "")


# ── LLMJudgeStrategy — error paths ─────────────────────────────────


class TestLLMJudgeStrategyErrors:
    """LLMJudgeStrategy returns 0.0 for wrong criteria/renderer types."""

    async def test_wrong_criteria_type(self) -> None:
        ctx = _make_ctx(
            submission="x",
            criteria=StaticCriteria(expected_answers=["a"]),
        )
        score = await LLMJudgeStrategy().score(ctx, renderer=object())
        assert score.value == 0.0
        assert "LLMJudgeCriteria" in (score.explanation or "")

    async def test_wrong_renderer_type(self) -> None:
        ctx = _make_llm_judge_ctx()
        score = await LLMJudgeStrategy().score(ctx, renderer=object())
        assert score.value == 0.0
        assert "TemplateRenderer" in (score.explanation or "")


# ── LLMJudgeStrategy — happy path ──────────────────────────────────


class TestLLMJudgeStrategyHappyPath:
    """LLMJudgeStrategy end-to-end with mocked LLM calls."""

    @pytest.mark.asyncio
    @patch("inspect_ai.model.get_model")
    async def test_binary_correct_returns_max_score(self, mock_get_model: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.completion = "The answer is CORRECT."
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value=mock_result)
        mock_get_model.return_value = mock_model

        from saber.scoring.templates import TemplateRenderer

        renderer = TemplateRenderer(templates={"sys.j2": "system", "usr.j2": "user"})
        ctx = _make_llm_judge_ctx(submission="my answer")

        score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == ctx.scorer.max_score
        assert score.answer == "my answer"
        assert score.metadata is not None
        assert score.metadata["judge_model"] == "gpt-4"
        assert score.metadata["judge_response"] == "The answer is CORRECT."

    @pytest.mark.asyncio
    @patch("inspect_ai.model.get_model")
    async def test_continuous_format_scales_score(self, mock_get_model: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.completion = '{"score": 0.75}'
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value=mock_result)
        mock_get_model.return_value = mock_model

        from saber.config.models import LLMJudgeResponseFormat
        from saber.scoring.templates import TemplateRenderer

        renderer = TemplateRenderer(templates={"sys.j2": "system", "usr.j2": "user"})

        llm_criteria = LLMJudgeCriteria(
            model="gpt-4",
            judge_system_template="sys.j2",
            judge_user_template="usr.j2",
            response_format=LLMJudgeResponseFormat.CONTINUOUS,
        )
        scorer_config = ScorerConfig(
            scorer_name="submission",
            strategy="llm_judge",
            target=ScorerTarget.SUBMISSION,
            criteria=llm_criteria,
            max_score=2.0,
            weight=1.0,
        )
        ctx = ScoringContext(
            submission="my answer",
            tool_steps=(),
            messages=(ChatMessageUser(content="test"),),
            target="expected",
            task_id="task_42",
            domain="test_domain",
            metadata={"description": "A test task"},
            scorer=scorer_config,
        )

        score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == pytest.approx(0.75 * 2.0)
        assert score.metadata is not None
        assert score.metadata["judge_model"] == "gpt-4"

    @pytest.mark.asyncio
    @patch("inspect_ai.model.get_model")
    async def test_checkpoint_step_format_returns_max_score(self, mock_get_model: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.completion = "[3: cp1]"
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value=mock_result)
        mock_get_model.return_value = mock_model

        from saber.scoring.templates import TemplateRenderer

        renderer = TemplateRenderer(templates={"sys.j2": "system", "usr.j2": "user"})

        ctx = _make_llm_judge_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            scorer_name="cp1",
            description="Find the flag",
        )

        score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == ctx.scorer.max_score
        assert score.metadata is not None
        assert score.metadata["judge_model"] == "gpt-4"
        assert score.metadata["judge_response"] == "[3: cp1]"


# ── _build_judge_context ────────────────────────────────────────────


class TestBuildJudgeContext:
    """_build_judge_context builds correct typed JudgeTemplateContext."""

    def test_returns_judge_template_context(self) -> None:
        ctx = _make_llm_judge_ctx(submission="my answer", target="expected")
        criteria = ctx.scorer.criteria
        assert isinstance(criteria, LLMJudgeCriteria)
        result = _build_judge_context(ctx, criteria)
        assert isinstance(result, JudgeTemplateContext)

    def test_submission_mode_fields(self) -> None:
        ctx = _make_llm_judge_ctx(submission="my answer", target="expected")
        criteria = ctx.scorer.criteria
        assert isinstance(criteria, LLMJudgeCriteria)

        result = _build_judge_context(ctx, criteria)

        assert result.submission == "my answer"
        assert result.description == "gold"
        assert result.task_id == "task_42"
        assert result.domain == "test_domain"
        assert result.question == "A test task"
        assert isinstance(result.episode, EpisodeContext)
        assert result.episode.step_count == 0

    def test_submission_mode_no_subtask_fields(self) -> None:
        ctx = _make_llm_judge_ctx()
        criteria = ctx.scorer.criteria
        assert isinstance(criteria, LLMJudgeCriteria)

        result = _build_judge_context(ctx, criteria)
        assert result.objective == "gold"
        assert result.checkpoint_id == "submission"
        assert len(result.checkpoints) == 1

    def test_subtask_mode_includes_subtask_fields(self) -> None:
        ctx = _make_llm_judge_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            scorer_name="cp1",
            description="Find the flag",
        )
        criteria = ctx.scorer.criteria
        assert isinstance(criteria, LLMJudgeCriteria)

        result = _build_judge_context(ctx, criteria)
        assert result.objective == "Find the flag"
        assert result.checkpoint_id == "cp1"
        assert len(result.checkpoints) == 1
        assert isinstance(result.checkpoints[0], CheckpointObjective)
        assert result.checkpoints[0].id == "cp1"
        assert result.checkpoints[0].objective == "Find the flag"

    def test_subtask_mode_includes_title_and_description(self) -> None:
        ctx = _make_llm_judge_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            scorer_name="cp1",
            title="Find C2",
            description="Locate the C2 server",
        )
        criteria = ctx.scorer.criteria
        assert isinstance(criteria, LLMJudgeCriteria)

        result = _build_judge_context(ctx, criteria)
        assert result.checkpoints[0].title == "Find C2"
        assert result.checkpoints[0].description == "Locate the C2 server"

    def test_tool_steps_in_episode(self) -> None:
        steps = (
            _make_tool_step(step_number=1, tool_name="exec", output="result1"),
            _make_tool_step(step_number=2, tool_name="read", output="result2"),
        )
        ctx = _make_llm_judge_ctx(tool_steps=steps)
        criteria = ctx.scorer.criteria
        assert isinstance(criteria, LLMJudgeCriteria)

        result = _build_judge_context(ctx, criteria)
        assert isinstance(result.episode, EpisodeContext)
        assert len(result.episode.steps) == 2
        assert result.episode.steps[0].tool_name == "exec"
        assert result.episode.steps[1].output == "result2"
        assert result.episode.step_count == 2

    def test_golden_answer_fallback_to_target(self) -> None:
        """description comes from ScorerConfig.description."""
        llm_criteria = LLMJudgeCriteria(
            model="gpt-4",
            judge_system_template="sys.j2",
            judge_user_template="usr.j2",
        )
        scorer_config = ScorerConfig(
            scorer_name="submission",
            strategy="llm_judge",
            target=ScorerTarget.SUBMISSION,
            criteria=llm_criteria,
            max_score=1.0,
            weight=1.0,
            description="from_scorer_config",
        )
        ctx = ScoringContext(
            submission="x",
            tool_steps=(),
            messages=(ChatMessageUser(content="test"),),
            target="fallback_target",
            task_id="t",
            domain="d",
            metadata={},
            scorer=scorer_config,
        )
        criteria = ctx.scorer.criteria
        assert isinstance(criteria, LLMJudgeCriteria)
        result = _build_judge_context(ctx, criteria)
        assert result.description == "from_scorer_config"


# ── StaticStrategy — new ScorerConfig path ──────────────────────────


class TestStaticStrategyNew:
    """Tests for StaticStrategy with new ScorerConfig context."""

    @pytest.fixture()
    def _make_ctx(self):
        def factory(
            target: str,
            answers: list[str],
            submission: str = "hello",
            tool_output: str = "world",
        ):
            from saber.config.models import ScorerConfig, ScorerTarget, StaticCriteria
            from saber.scoring.context import ScoringContext, ToolStep

            scorer = ScorerConfig(
                scorer_name="test",
                strategy="static",
                target=ScorerTarget(target),
                criteria=StaticCriteria(expected_answers=answers),
                max_score=1.0,
            )
            steps = (
                (ToolStep(step_number=1, tool_name="bash", tool_input={}, output=tool_output),) if tool_output else ()
            )
            return ScoringContext(
                submission=submission,
                tool_steps=steps,
                messages=(),
                target_text="",
                task_id="t",
                domain="d",
                metadata={},
                scorer=scorer,
            )

        return factory

    @pytest.mark.asyncio
    async def test_submission_target_checks_submission(self, _make_ctx):
        ctx = _make_ctx("submission", ["hello"], submission="hello world")
        result = await StaticStrategy().score(ctx, None)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_trajectory_target_checks_tool_output(self, _make_ctx):
        ctx = _make_ctx("trajectory", ["world"], submission="no match", tool_output="world")
        result = await StaticStrategy().score(ctx, None)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_submission_target_no_match(self, _make_ctx):
        ctx = _make_ctx("submission", ["missing"], submission="hello")
        result = await StaticStrategy().score(ctx, None)
        assert result.value == 0.0


# ── ToolCallStrategy — new ScorerConfig path ───────────────────────


class TestToolCallStrategyNew:
    @pytest.mark.asyncio
    async def test_trajectory_target_with_match(self):
        from saber.config.models import ScorerConfig, ScorerTarget, ToolCallCriteria
        from saber.scoring.context import ScoringContext, ToolStep

        scorer = ScorerConfig(
            scorer_name="cp",
            strategy="tool_call",
            target=ScorerTarget.TRAJECTORY,
            criteria=ToolCallCriteria(expected_tools=["bash"]),
        )
        ctx = ScoringContext(
            submission="x",
            tool_steps=(ToolStep(step_number=1, tool_name="bash", tool_input={}, output="ok"),),
            messages=(),
            target_text="",
            task_id="t",
            domain="d",
            metadata={},
            scorer=scorer,
        )
        result = await ToolCallStrategy().score(ctx, None)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_submission_target_returns_zero(self):
        from saber.config.models import ScorerConfig, ScorerTarget, ToolCallCriteria
        from saber.scoring.context import ScoringContext

        scorer = ScorerConfig(
            scorer_name="cp",
            strategy="tool_call",
            target=ScorerTarget.SUBMISSION,
            criteria=ToolCallCriteria(expected_tools=["bash"]),
        )
        ctx = ScoringContext(
            submission="x",
            tool_steps=(),
            messages=(),
            target_text="",
            task_id="t",
            domain="d",
            metadata={},
            scorer=scorer,
        )
        result = await ToolCallStrategy().score(ctx, None)
        assert result.value == 0.0


# ── ToolCallCountStrategy — new ScorerConfig path ──────────────────


class TestToolCallCountStrategyNew:
    @pytest.mark.asyncio
    async def test_trajectory_with_enough_calls(self):
        from saber.config.models import ScorerConfig, ScorerTarget, ToolCallCriteria
        from saber.scoring.context import ScoringContext, ToolStep

        scorer = ScorerConfig(
            scorer_name="cp",
            strategy="tool_call_count",
            target=ScorerTarget.TRAJECTORY,
            criteria=ToolCallCriteria(tool_name="bash", min_executions=2),
        )
        steps = tuple(ToolStep(step_number=i, tool_name="bash", tool_input={}, output="ok") for i in range(3))
        ctx = ScoringContext(
            submission="x",
            tool_steps=steps,
            messages=(),
            target_text="",
            task_id="t",
            domain="d",
            metadata={},
            scorer=scorer,
        )
        result = await ToolCallCountStrategy().score(ctx, None)
        assert result.value == 1.0


# ── TestScorerRequired ─────────────────────────────────────────────


class TestScorerRequired:
    """ScoringContext requires scorer field (Pydantic enforces at construction)."""

    def test_construction_without_scorer_raises(self) -> None:
        with pytest.raises(Exception, match="scorer"):
            ScoringContext(
                submission="test",
                tool_steps=(),
                messages=(ChatMessageUser(content="test"),),
                target="",
                task_id="t",
                domain="d",
                metadata={},
            )


# ── _normalize_tool_name ────────────────────────────────────────────


class TestNormalizeToolName:
    """Unit tests for _normalize_tool_name helper."""

    def test_lowercase(self) -> None:
        assert _normalize_tool_name("Bash") == "bash"

    def test_already_lowercase(self) -> None:
        assert _normalize_tool_name("bash") == "bash"

    def test_all_caps(self) -> None:
        assert _normalize_tool_name("READ") == "read"

    def test_mixed_case(self) -> None:
        assert _normalize_tool_name("Edit") == "edit"

    def test_mcp_prefix_single_segment(self) -> None:
        assert _normalize_tool_name("mcp__saber_tools__submit_patch") == "submit_patch"

    def test_mcp_prefix_preserves_tool_name(self) -> None:
        assert _normalize_tool_name("mcp__my_server__run_command") == "run_command"

    def test_mcp_prefix_with_uppercase(self) -> None:
        assert _normalize_tool_name("MCP__Saber_Tools__Submit_Patch") == "submit_patch"

    def test_no_prefix(self) -> None:
        assert _normalize_tool_name("submit_patch") == "submit_patch"

    def test_empty_string(self) -> None:
        assert _normalize_tool_name("") == ""

    def test_single_double_underscore(self) -> None:
        # Only MCP-style triple-segment pattern should be stripped
        assert _normalize_tool_name("foo__bar") == "foo__bar"

    def test_trailing_double_underscore_returns_original(self) -> None:
        """mcp__server__ → empty after prefix strip; fall back to original."""
        assert _normalize_tool_name("mcp__server__") == "mcp__server__"

    def test_four_plus_segments(self) -> None:
        """mcp__a__b__c → joins everything after the server segment."""
        assert _normalize_tool_name("mcp__a__b__c") == "b__c"


# ── ToolCallStrategy with normalization ─────────────────────────────


class TestToolCallStrategyNormalization:
    """ToolCallStrategy matches tools after normalizing names."""

    async def test_capitalized_tool_matches_lowercase_expected(self) -> None:
        """Claude Code sends 'Bash' but expected_tools has 'bash'."""
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="bash",
            expected_tools=["bash"],
            tool_steps=(_make_tool_step(tool_name="Bash", output="ok"),),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_mcp_prefixed_tool_matches_base_name(self) -> None:
        """MCP-bridged 'mcp__saber_tools__submit_patch' matches 'submit_patch'."""
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="submit_patch",
            expected_tools=["submit_patch"],
            tool_steps=(_make_tool_step(tool_name="mcp__saber_tools__submit_patch", output="ok"),),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_expected_tool_also_normalized(self) -> None:
        """Expected tool names are also normalized for fair comparison."""
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="Bash",
            expected_tools=["Bash", "Submit_Patch"],
            tool_steps=(_make_tool_step(tool_name="bash", output="ok"),),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_no_match_after_normalization(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="run",
            expected_tools=["run"],
            tool_steps=(_make_tool_step(tool_name="Bash", output="ok"),),
        )
        score = await ToolCallStrategy().score(ctx, renderer=object())
        assert score.value == 0.0


# ── ToolCallCountStrategy with normalization ────────────────────────


class TestToolCallCountStrategyNormalization:
    """ToolCallCountStrategy normalizes tool names before counting."""

    async def test_capitalized_tool_counted(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="bash",
            min_executions=2,
            tool_steps=(
                _make_tool_step(step_number=1, tool_name="Bash", output="a"),
                _make_tool_step(step_number=2, tool_name="bash", output="b"),
            ),
        )
        score = await ToolCallCountStrategy().score(ctx, renderer=object())
        assert score.value == 1.0

    async def test_mcp_prefixed_tool_counted(self) -> None:
        ctx = _make_tool_call_ctx(
            scorer_target=ScorerTarget.TRAJECTORY,
            tool_name="submit_patch",
            min_executions=1,
            tool_steps=(
                _make_tool_step(tool_name="mcp__saber_tools__submit_patch", output="ok"),
            ),
        )
        score = await ToolCallCountStrategy().score(ctx, renderer=object())
        assert score.value == 1.0
