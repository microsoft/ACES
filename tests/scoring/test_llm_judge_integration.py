# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Integration tests for LLMJudgeStrategy with mocked model.

Exercises the full flow: context building → template rendering →
model call → response parsing → Score.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.config.models import (
    LLMJudgeCriteria,
    LLMJudgeResponseFormat,
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
)
from saber.scoring.context import ScoringContext, ToolStep
from saber.scoring.registry import ScoringStrategyRegistry
from saber.scoring.strategies import (
    LLMJudgeStrategy,
    NoneStrategy,
    StaticJaccardStrategy,
    StaticStrategy,
    ToolCallCountStrategy,
    ToolCallStrategy,
)
from saber.scoring.templates import TemplateRenderer

# ── Helpers ─────────────────────────────────────────────────────────


def _mock_model_response(completion: str) -> MagicMock:
    """Create a mock ModelOutput with .completion property."""
    result = MagicMock()
    result.completion = completion
    return result


def _mock_model(completion: str) -> MagicMock:
    """Create a mock model whose .generate() returns *completion*."""
    model = MagicMock()
    model.generate = AsyncMock(return_value=_mock_model_response(completion))
    return model


def _make_llm_judge_ctx(
    *,
    submission: str = "test answer",
    target: str = "expected",
    task_id: str = "task_1",
    response_format: LLMJudgeResponseFormat = LLMJudgeResponseFormat.BINARY,
    system_template: str = "system.j2",
    user_template: str = "user.j2",
    model: str = "openai/gpt-4",
    description: str = "",
    subtask: dict[str, str] | None = None,
    tool_steps: tuple[ToolStep, ...] = (),
    max_score: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> ScoringContext:
    """Build a ScoringContext with LLMJudgeCriteria."""
    criteria = LLMJudgeCriteria(
        model=model,
        judge_system_template=system_template,
        judge_user_template=user_template,
        response_format=response_format,
    )

    if subtask is None:
        scorer_target = ScorerTarget.SUBMISSION
        scorer_name = "submission"
    else:
        scorer_target = ScorerTarget.TRAJECTORY
        scorer_name = subtask["id"]

    scorer_config = ScorerConfig(
        scorer_name=scorer_name,
        strategy="llm_judge",
        target=scorer_target,
        criteria=criteria,
        max_score=max_score,
        weight=1.0,
        title=subtask["title"] if subtask else "",
        description=description or (subtask["description"] if subtask else ""),
    )

    return ScoringContext(
        submission=submission,
        tool_steps=tool_steps,
        messages=(),
        target=target,
        task_id=task_id,
        domain="test_domain",
        metadata=metadata or {"description": "Test task"},
        scorer=scorer_config,
    )


def _make_subtask(
    subtask_id: str = "checkpoint_1",
    objective: str = "Find the flag",
) -> dict[str, str]:
    """Return subtask info as a dict (SubtaskConfig was removed)."""
    return {
        "id": subtask_id,
        "title": "Sub",
        "objective": objective,
        "description": objective,
    }


def _make_tool_step(
    *,
    step_number: int = 1,
    tool_name: str = "run_command",
    output: str = "ok",
    tool_input: dict[str, object] | None = None,
) -> ToolStep:
    return ToolStep(
        step_number=step_number,
        tool_name=tool_name,
        tool_input=tool_input or {},
        output=output,
    )


def _default_renderer() -> TemplateRenderer:
    """Return a TemplateRenderer with simple dict templates."""
    return TemplateRenderer(
        templates={
            "system.j2": "You are a judge. Evaluate the answer.",
            "user.j2": "Answer: {{ submission }}. Expected: {{ description }}",
        }
    )


# ── TestLLMJudgeSubmissionScoring ───────────────────────────────────


class TestLLMJudgeSubmissionScoring:
    """Submission mode: scorer targets SUBMISSION."""

    @pytest.mark.asyncio
    async def test_binary_correct(self) -> None:
        """Model returns 'CORRECT' → Score(value=max_score)."""
        mock_model = _mock_model("CORRECT")
        renderer = _default_renderer()
        ctx = _make_llm_judge_ctx(submission="42", max_score=5.0)

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == 5.0
        assert score.answer == "42"

        # Verify mock was called with system + user messages
        mock_model.generate.assert_awaited_once()
        messages = mock_model.generate.call_args[0][0]
        assert len(messages) == 2
        assert messages[0].content == "You are a judge. Evaluate the answer."
        assert "42" in messages[1].content

    @pytest.mark.asyncio
    async def test_binary_incorrect(self) -> None:
        """Model returns 'INCORRECT' → Score(value=0.0)."""
        mock_model = _mock_model("INCORRECT")
        renderer = _default_renderer()
        ctx = _make_llm_judge_ctx(submission="wrong answer")

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == 0.0
        assert score.answer == "wrong answer"

    @pytest.mark.asyncio
    async def test_continuous_score(self) -> None:
        """Model returns JSON {"score": 0.75} with max_score=10.0 → 7.5."""
        mock_model = _mock_model('{"score": 0.75}')
        renderer = _default_renderer()
        ctx = _make_llm_judge_ctx(
            submission="partial answer",
            response_format=LLMJudgeResponseFormat.CONTINUOUS,
            max_score=10.0,
        )

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == pytest.approx(7.5)

    @pytest.mark.asyncio
    async def test_template_renders_with_context(self) -> None:
        """Verify the rendered prompts contain actual task values."""
        mock_model = _mock_model("CORRECT")
        renderer = TemplateRenderer(
            templates={
                "system.j2": "Evaluate {{ submission }} against {{ description }}",
                "user.j2": "Task: {{ task_id }}, Domain: {{ domain }}",
            }
        )
        ctx = _make_llm_judge_ctx(
            submission="my_submission",
            description="the_description",
            task_id="task_42",
        )

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        messages = mock_model.generate.call_args[0][0]
        # System prompt should contain rendered context
        assert "my_submission" in messages[0].content
        assert "the_description" in messages[0].content
        # User prompt should contain task info
        assert "task_42" in messages[1].content
        assert "test_domain" in messages[1].content
        assert score.value == 1.0  # max_score default

    @pytest.mark.asyncio
    async def test_score_metadata_includes_judge_info(self) -> None:
        """Score.metadata has 'judge_model' and 'judge_response'."""
        mock_model = _mock_model("CORRECT")
        renderer = _default_renderer()
        ctx = _make_llm_judge_ctx(model="openai/gpt-4o")

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.metadata is not None
        assert score.metadata["judge_model"] == "openai/gpt-4o"
        assert score.metadata["judge_response"] == "CORRECT"

    @pytest.mark.asyncio
    async def test_description_from_scorer_config(self) -> None:
        """description comes from ScorerConfig.description."""
        mock_model = _mock_model("CORRECT")
        renderer = TemplateRenderer(
            templates={
                "system.j2": "Judge.",
                "user.j2": "Description: {{ description }}",
            }
        )
        ctx = _make_llm_judge_ctx(target="expected_target", description="my_description")

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            await LLMJudgeStrategy().score(ctx, renderer)

        messages = mock_model.generate.call_args[0][0]
        assert "my_description" in messages[1].content


# ── TestLLMJudgeSubtaskScoring ──────────────────────────────────────


class TestLLMJudgeSubtaskScoring:
    """Subtask mode: scorer targets TRAJECTORY."""

    @pytest.mark.asyncio
    async def test_subtask_context_includes_checkpoint_info(self) -> None:
        """Model receives prompt with objective and checkpoint_id."""
        mock_model = _mock_model("CORRECT")
        renderer = TemplateRenderer(
            templates={
                "system.j2": "Objective: {{ objective }}",
                "user.j2": "Checkpoint: {{ checkpoint_id }}",
            }
        )
        subtask = _make_subtask(subtask_id="chk_42", objective="Exfiltrate credentials")
        ctx = _make_llm_judge_ctx(subtask=subtask)

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        messages = mock_model.generate.call_args[0][0]
        assert "Exfiltrate credentials" in messages[0].content
        assert "chk_42" in messages[1].content
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_subtask_with_trajectory(self) -> None:
        """Tool steps appear in rendered prompt."""
        mock_model = _mock_model("CORRECT")
        renderer = TemplateRenderer(
            templates={
                "system.j2": "Judge steps.",
                "user.j2": ("Steps: {% for s in episode.steps %}{{ s.step_number }}{% endfor %}"),
            }
        )
        steps = (
            _make_tool_step(step_number=1, output="first"),
            _make_tool_step(step_number=2, output="second"),
            _make_tool_step(step_number=3, output="third"),
        )
        subtask = _make_subtask()
        ctx = _make_llm_judge_ctx(subtask=subtask, tool_steps=steps)

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        messages = mock_model.generate.call_args[0][0]
        assert "123" in messages[1].content
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_subtask_step_evaluations_format(self) -> None:
        """STEP_EVALUATIONS format computes proportional score."""
        response = (
            '{"step_evaluations": [{"step_number": 1, "completed": true}, {"step_number": 2, "completed": false}]}'
        )
        mock_model = _mock_model(response)
        renderer = _default_renderer()
        subtask = _make_subtask()
        ctx = _make_llm_judge_ctx(
            subtask=subtask,
            response_format=LLMJudgeResponseFormat.STEP_EVALUATIONS,
            max_score=10.0,
            tool_steps=(
                _make_tool_step(step_number=1),
                _make_tool_step(step_number=2),
            ),
        )

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        # 1 of 2 steps completed → 0.5 * 10.0 = 5.0
        assert score.value == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_subtask_tool_step_details_in_context(self) -> None:
        """Tool step action/response details are available in templates."""
        mock_model = _mock_model("CORRECT")
        renderer = TemplateRenderer(
            templates={
                "system.j2": "Judge.",
                "user.j2": ("{% for s in episode.steps %}Tool: {{ s.tool_name }}, Output: {{ s.output }}{% endfor %}"),
            }
        )
        steps = (
            _make_tool_step(
                step_number=1,
                tool_name="nmap_scan",
                output="port 22 open",
            ),
        )
        ctx = _make_llm_judge_ctx(subtask=_make_subtask(), tool_steps=steps)

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            await LLMJudgeStrategy().score(ctx, renderer)

        messages = mock_model.generate.call_args[0][0]
        assert "nmap_scan" in messages[1].content
        assert "port 22 open" in messages[1].content


# ── TestLLMJudgeErrorHandling ───────────────────────────────────────


class TestLLMJudgeErrorHandling:
    """Error cases return Score(0.0) with explanatory message."""

    @pytest.mark.asyncio
    async def test_wrong_criteria_type(self) -> None:
        """StaticCriteria instead of LLMJudgeCriteria → Score 0.0."""
        static_criteria = StaticCriteria(expected_answers=["x"])
        scorer_config = ScorerConfig(
            scorer_name="submission",
            strategy="llm_judge",
            target=ScorerTarget.SUBMISSION,
            criteria=static_criteria,
            max_score=1.0,
        )
        ctx = ScoringContext(
            submission="answer",
            tool_steps=(),
            messages=(),
            target="",
            task_id="task_1",
            domain="test_domain",
            metadata={},
            scorer=scorer_config,
        )
        renderer = _default_renderer()

        score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == 0.0
        assert score.explanation is not None
        assert "LLMJudgeCriteria" in score.explanation
        assert "StaticCriteria" in score.explanation

    @pytest.mark.asyncio
    async def test_wrong_renderer_type(self) -> None:
        """Pass object() instead of TemplateRenderer → Score 0.0."""
        ctx = _make_llm_judge_ctx()

        score = await LLMJudgeStrategy().score(ctx, object())

        assert score.value == 0.0
        assert score.explanation is not None
        assert "TemplateRenderer" in score.explanation

    @pytest.mark.asyncio
    async def test_binary_ambiguous_response(self) -> None:
        """Model returns text without CORRECT/INCORRECT → Score 0.0."""
        mock_model = _mock_model("I'm not sure about this one.")
        renderer = _default_renderer()
        ctx = _make_llm_judge_ctx()

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_continuous_invalid_json(self) -> None:
        """Model returns non-JSON text with CONTINUOUS format → Score 0.0."""
        mock_model = _mock_model("not valid json")
        renderer = _default_renderer()
        ctx = _make_llm_judge_ctx(
            response_format=LLMJudgeResponseFormat.CONTINUOUS,
        )

        with patch("inspect_ai.model.get_model", return_value=mock_model):
            score = await LLMJudgeStrategy().score(ctx, renderer)

        assert score.value == 0.0


# ── TestRegistryLookupIntegration ───────────────────────────────────


class TestRegistryLookupIntegration:
    """Verify the registry resolves strategy names to correct types."""

    def test_registry_resolves_llm_judge(self) -> None:
        """Registry returns LLMJudgeStrategy for 'llm_judge'."""
        registry = ScoringStrategyRegistry()
        strategy = registry.get("llm_judge")
        assert isinstance(strategy, LLMJudgeStrategy)

    def test_registry_resolves_all_strategies(self) -> None:
        """Each registered name returns the expected strategy type."""
        registry = ScoringStrategyRegistry()
        expected: dict[str, type[object]] = {
            "static": StaticStrategy,
            "none": NoneStrategy,
            "llm_judge": LLMJudgeStrategy,
            "static_jaccard": StaticJaccardStrategy,
            "tool_call": ToolCallStrategy,
            "tool_call_count": ToolCallCountStrategy,
        }
        for name, cls in expected.items():
            assert isinstance(registry.get(name), cls), (
                f"Registry returned {type(registry.get(name)).__name__} for '{name}', expected {cls.__name__}"
            )

    def test_registry_available_lists_all(self) -> None:
        """Registry.available() includes all built-in strategies."""
        registry = ScoringStrategyRegistry()
        available = registry.available()
        for name in ("static", "none", "llm_judge", "tool_call", "tool_call_count"):
            assert name in available
