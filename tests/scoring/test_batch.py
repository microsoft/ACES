"""Tests for batched LLM judge scoring (score_checkpoints_llm_batch)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.model import ChatMessageUser

from saber.config.models import (
    LLMJudgeCriteria,
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
)
from saber.scoring.batch import score_checkpoints_llm_batch
from saber.scoring.context import ScoringContext, ToolStep
from saber.scoring.templates import TemplateRenderer

# ── Helpers ─────────────────────────────────────────────────────────


def _mock_model_response(completion: str) -> MagicMock:
    """Create a mock model generate result."""
    result = MagicMock()
    result.completion = completion
    return result


def _make_renderer() -> TemplateRenderer:
    return TemplateRenderer(
        templates={
            "system.j2": "You are a judge. Checkpoints: {{ checkpoints }}",
            "user.j2": "Steps: {{ episode.steps }} Chunk {{ chunk_index }}/{{ total_chunks }}",
        }
    )


def _make_criteria(
    *,
    steps_per_message: int | None = None,
) -> LLMJudgeCriteria:
    return LLMJudgeCriteria(
        model="mockmodel/judge",
        judge_system_template="system.j2",
        judge_user_template="user.j2",
        steps_per_message=steps_per_message,
    )


def _make_subtask(
    subtask_id: str,
    objective: str = "Complete the task",
    max_score: float = 1.0,
    title: str = "",
    description: str = "",
) -> dict[str, object]:
    """Return subtask info as a dict (SubtaskConfig was removed)."""
    return {
        "id": subtask_id,
        "title": title or f"Subtask {subtask_id}",
        "description": description or objective,
        "objective": objective,
        "max_score": max_score,
    }


def _make_tool_step(
    step_number: int,
    output: str = "ok",
    tool_name: str = "run_command",
) -> ToolStep:
    return ToolStep(
        step_number=step_number,
        tool_name=tool_name,
        tool_input={},
        output=output,
    )


def _make_context(
    *,
    subtask: dict[str, object],
    criteria: LLMJudgeCriteria | None = None,
    tool_steps: tuple[ToolStep, ...] = (),
    max_score: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> ScoringContext:
    crit = criteria or _make_criteria()
    scorer_config = ScorerConfig(
        scorer_name=str(subtask["id"]),
        strategy="llm_judge",
        target=ScorerTarget.TRAJECTORY,
        criteria=crit,
        max_score=max_score,
        weight=1.0,
        title=str(subtask["title"]),
        description=str(subtask["description"]),
    )
    return ScoringContext(
        submission="",
        tool_steps=tool_steps,
        messages=(ChatMessageUser(content="test"),),
        target="",
        task_id="task_1",
        domain="test_domain",
        metadata=metadata or {"description": "Test task"},
        scorer=scorer_config,
    )


# ── Test Cases ──────────────────────────────────────────────────────


class TestScoreSubtasksLLMBatchEmpty:
    """Empty contexts → empty dict."""

    @pytest.mark.asyncio
    async def test_empty_contexts_returns_empty_dict(self) -> None:
        renderer = _make_renderer()
        result = await score_checkpoints_llm_batch(contexts=[], renderer=renderer)
        assert result == {}


class TestScoreSubtasksLLMBatchSingleCompleted:
    """Single subtask, single chunk, checkpoint completed."""

    @pytest.mark.asyncio
    async def test_single_subtask_completed(self) -> None:
        subtask = _make_subtask("cp_1")
        steps = (_make_tool_step(1, "found it"), _make_tool_step(2, "done"))
        ctx = _make_context(subtask=subtask, tool_steps=steps, max_score=5.0)

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=_mock_model_response("[1: cp_1]"))

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx], renderer=_make_renderer())

        assert "cp_1" in result
        assert result["cp_1"].value == 5.0


class TestScoreSubtasksLLMBatchSingleNotCompleted:
    """Single subtask, response is [NO_COMPLETIONS] → 0.0."""

    @pytest.mark.asyncio
    async def test_single_subtask_not_completed(self) -> None:
        subtask = _make_subtask("cp_1")
        steps = (_make_tool_step(1, "nope"),)
        ctx = _make_context(subtask=subtask, tool_steps=steps)

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=_mock_model_response("[NO_COMPLETIONS]"))

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx], renderer=_make_renderer())

        assert result["cp_1"].value == 0.0


class TestScoreSubtasksLLMBatchMultipleCompleted:
    """Multiple subtasks, both completed in one chunk."""

    @pytest.mark.asyncio
    async def test_multiple_subtasks_both_completed(self) -> None:
        sub_a = _make_subtask("cp_a", max_score=2.0)
        sub_b = _make_subtask("cp_b", max_score=3.0)
        steps = (
            _make_tool_step(1, "step one"),
            _make_tool_step(2, "step two"),
            _make_tool_step(3, "step three"),
        )
        ctx_a = _make_context(subtask=sub_a, tool_steps=steps, max_score=2.0)
        ctx_b = _make_context(subtask=sub_b, tool_steps=steps, max_score=3.0)

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=_mock_model_response("[1: cp_a] [3: cp_b]"))

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx_a, ctx_b], renderer=_make_renderer())

        assert result["cp_a"].value == 2.0
        assert result["cp_b"].value == 3.0


class TestScoreSubtasksLLMBatchPartialCompletion:
    """Multiple subtasks, only one completed."""

    @pytest.mark.asyncio
    async def test_partial_completion(self) -> None:
        sub_a = _make_subtask("cp_a")
        sub_b = _make_subtask("cp_b")
        steps = (_make_tool_step(1, "data"),)
        ctx_a = _make_context(subtask=sub_a, tool_steps=steps)
        ctx_b = _make_context(subtask=sub_b, tool_steps=steps)

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=_mock_model_response("[1: cp_a]"))

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx_a, ctx_b], renderer=_make_renderer())

        assert result["cp_a"].value == 1.0
        assert result["cp_b"].value == 0.0


class TestScoreSubtasksLLMBatchChunked:
    """Chunked trajectory: steps_per_message=2, 4 steps → 2 chunks."""

    @pytest.mark.asyncio
    async def test_chunked_trajectory_checkpoint_in_second_chunk(self) -> None:
        criteria = _make_criteria(steps_per_message=2)
        subtask = _make_subtask("cp_x", max_score=10.0)
        steps = tuple(_make_tool_step(i, f"step {i}") for i in range(1, 5))
        ctx = _make_context(subtask=subtask, criteria=criteria, tool_steps=steps, max_score=10.0)

        mock_model = AsyncMock()
        # First chunk: not found; second chunk: found
        mock_model.generate = AsyncMock(
            side_effect=[
                _mock_model_response("[NO_COMPLETIONS]"),
                _mock_model_response("[3: cp_x]"),
            ]
        )

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx], renderer=_make_renderer())

        assert result["cp_x"].value == 10.0
        assert mock_model.generate.call_count == 2
        # Check metadata indicates chunk 1 (0-indexed)
        assert result["cp_x"].metadata is not None
        assert result["cp_x"].metadata.get("chunk_index") == 1


class TestScoreSubtasksLLMBatchErrorNoSubtask:
    """Context with wrong criteria type → ValueError."""

    @pytest.mark.asyncio
    async def test_context_with_static_criteria_raises(self) -> None:
        """scorer is required; passing non-LLMJudgeCriteria raises ValueError."""
        subtask = _make_subtask("cp_bad")
        static_criteria = StaticCriteria(expected_answers=["x"])
        scorer_config = ScorerConfig(
            scorer_name=str(subtask["id"]),
            strategy="static",
            target=ScorerTarget.TRAJECTORY,
            criteria=static_criteria,
            max_score=1.0,
        )
        ctx = ScoringContext(
            submission="",
            tool_steps=(),
            messages=(ChatMessageUser(content="test"),),
            target="",
            task_id="task_1",
            domain="test_domain",
            metadata={},
            scorer=scorer_config,
        )

        with pytest.raises(ValueError, match="LLMJudgeCriteria"):
            await score_checkpoints_llm_batch(contexts=[ctx], renderer=_make_renderer())


class TestScoreSubtasksLLMBatchErrorWrongCriteria:
    """Wrong criteria type → ValueError."""

    @pytest.mark.asyncio
    async def test_wrong_criteria_type_raises(self) -> None:
        subtask = _make_subtask("cp_1")
        static_criteria = StaticCriteria(expected_answers=["x"])
        scorer_config = ScorerConfig(
            scorer_name=str(subtask["id"]),
            strategy="static",
            target=ScorerTarget.TRAJECTORY,
            criteria=static_criteria,
            max_score=1.0,
        )
        ctx = ScoringContext(
            submission="",
            tool_steps=(),
            messages=(ChatMessageUser(content="test"),),
            target="",
            task_id="task_1",
            domain="test_domain",
            metadata={},
            scorer=scorer_config,
        )

        with pytest.raises(ValueError, match="LLMJudgeCriteria"):
            await score_checkpoints_llm_batch(contexts=[ctx], renderer=_make_renderer())


class TestScoreSubtasksLLMBatchEarlyStop:
    """All checkpoints completed in first chunk → second chunk skipped."""

    @pytest.mark.asyncio
    async def test_all_completed_early_skips_remaining_chunks(self) -> None:
        criteria = _make_criteria(steps_per_message=2)
        sub_a = _make_subtask("cp_a")
        sub_b = _make_subtask("cp_b")
        steps = tuple(_make_tool_step(i, f"step {i}") for i in range(1, 5))
        ctx_a = _make_context(subtask=sub_a, criteria=criteria, tool_steps=steps)
        ctx_b = _make_context(subtask=sub_b, criteria=criteria, tool_steps=steps)

        mock_model = AsyncMock()
        # First chunk completes both → second chunk should be skipped
        mock_model.generate = AsyncMock(return_value=_mock_model_response("[1: cp_a] [2: cp_b]"))

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx_a, ctx_b], renderer=_make_renderer())

        assert result["cp_a"].value == 1.0
        assert result["cp_b"].value == 1.0
        # Only one LLM call made (second chunk skipped)
        assert mock_model.generate.call_count == 1


class TestScoreSubtasksLLMBatchCriteriaGrouping:
    """Contexts with different criteria are grouped and processed independently."""

    @pytest.mark.asyncio
    async def test_different_model_groups_independently(self) -> None:
        sub_a = _make_subtask("cp_a")
        sub_b = _make_subtask("cp_b")
        criteria_a = _make_criteria()
        criteria_b = LLMJudgeCriteria(
            model="other/model",
            judge_system_template="system.j2",
            judge_user_template="user.j2",
        )
        ctx_a = _make_context(subtask=sub_a, criteria=criteria_a)
        ctx_b = _make_context(subtask=sub_b, criteria=criteria_b)

        mock_model_a = AsyncMock()
        mock_model_a.generate = AsyncMock(return_value=_mock_model_response("[1: cp_a]"))
        mock_model_b = AsyncMock()
        mock_model_b.generate = AsyncMock(return_value=_mock_model_response("[1: cp_b]"))

        def get_model_side_effect(model_name: str) -> AsyncMock:
            if model_name == "mockmodel/judge":
                return mock_model_a
            if model_name == "other/model":
                return mock_model_b
            raise ValueError(f"Unexpected model: {model_name}")

        with patch("saber.scoring.batch.get_model", side_effect=get_model_side_effect):
            result = await score_checkpoints_llm_batch(contexts=[ctx_a, ctx_b], renderer=_make_renderer())

        assert result["cp_a"].value == 1.0
        assert result["cp_b"].value == 1.0
        assert mock_model_a.generate.call_count == 1
        assert mock_model_b.generate.call_count == 1


class TestScoreSubtasksLLMBatchMixedTemplates:
    """Mixed templates (submission vs checkpoint) are grouped correctly."""

    @pytest.mark.asyncio
    async def test_mixed_templates_grouped_correctly(self) -> None:
        submission_criteria = LLMJudgeCriteria(
            model="mockmodel/judge",
            judge_system_template="submission_system.j2",
            judge_user_template="submission_user.j2",
        )
        checkpoint_criteria = LLMJudgeCriteria(
            model="mockmodel/judge",
            judge_system_template="checkpoint_system.j2",
            judge_user_template="checkpoint_user.j2",
        )

        sub_submit = _make_subtask("submission")
        sub_cp1 = _make_subtask("checkpoint_1")
        sub_cp2 = _make_subtask("checkpoint_2")

        ctx_submit = _make_context(subtask=sub_submit, criteria=submission_criteria)
        ctx_cp1 = _make_context(subtask=sub_cp1, criteria=checkpoint_criteria)
        ctx_cp2 = _make_context(subtask=sub_cp2, criteria=checkpoint_criteria)

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(
            side_effect=[
                _mock_model_response("[1: submission]"),
                _mock_model_response("[1: checkpoint_1] [1: checkpoint_2]"),
            ]
        )

        renderer = TemplateRenderer(
            templates={
                "submission_system.j2": "submission system {{ checkpoints }}",
                "submission_user.j2": "submission user {{ episode.steps }}",
                "checkpoint_system.j2": "checkpoint system {{ checkpoints }}",
                "checkpoint_user.j2": "checkpoint user {{ episode.steps }}",
            }
        )

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx_submit, ctx_cp1, ctx_cp2], renderer=renderer)

        assert result["submission"].value == 1.0
        assert result["checkpoint_1"].value == 1.0
        assert result["checkpoint_2"].value == 1.0
        assert mock_model.generate.call_count == 2


class TestScoreSubtasksLLMBatchMixedCriteriaGroups:
    """Multiple criteria groups processed independently with different models."""

    @pytest.mark.asyncio
    async def test_mixed_criteria_groups_independently(self) -> None:
        criteria_a = LLMJudgeCriteria(
            model="model_a/judge",
            judge_system_template="template_a_system.j2",
            judge_user_template="template_a_user.j2",
        )
        criteria_b = LLMJudgeCriteria(
            model="model_b/judge",
            judge_system_template="template_b_system.j2",
            judge_user_template="template_b_user.j2",
        )

        sub_a1 = _make_subtask("cp_a1")
        sub_a2 = _make_subtask("cp_a2")
        sub_b1 = _make_subtask("cp_b1")
        sub_b2 = _make_subtask("cp_b2")

        ctx_a1 = _make_context(subtask=sub_a1, criteria=criteria_a)
        ctx_a2 = _make_context(subtask=sub_a2, criteria=criteria_a)
        ctx_b1 = _make_context(subtask=sub_b1, criteria=criteria_b)
        ctx_b2 = _make_context(subtask=sub_b2, criteria=criteria_b)

        mock_model_a = AsyncMock()
        mock_model_a.generate = AsyncMock(return_value=_mock_model_response("[1: cp_a1] [1: cp_a2]"))
        mock_model_b = AsyncMock()
        mock_model_b.generate = AsyncMock(return_value=_mock_model_response("[1: cp_b1] [1: cp_b2]"))

        def get_model_side_effect(model_name: str) -> AsyncMock:
            if model_name == "model_a/judge":
                return mock_model_a
            if model_name == "model_b/judge":
                return mock_model_b
            raise ValueError(f"Unexpected model: {model_name}")

        renderer = TemplateRenderer(
            templates={
                "template_a_system.j2": "a system {{ checkpoints }}",
                "template_a_user.j2": "a user {{ episode.steps }}",
                "template_b_system.j2": "b system {{ checkpoints }}",
                "template_b_user.j2": "b user {{ episode.steps }}",
            }
        )

        with patch("saber.scoring.batch.get_model", side_effect=get_model_side_effect):
            result = await score_checkpoints_llm_batch(contexts=[ctx_a1, ctx_a2, ctx_b1, ctx_b2], renderer=renderer)

        assert result["cp_a1"].value == 1.0
        assert result["cp_a2"].value == 1.0
        assert result["cp_b1"].value == 1.0
        assert result["cp_b2"].value == 1.0
        assert mock_model_a.generate.call_count == 1
        assert mock_model_b.generate.call_count == 1
        # Verify correct model was used via metadata
        assert result["cp_a1"].metadata is not None
        assert result["cp_a1"].metadata["judge_model"] == "model_a/judge"
        assert result["cp_b1"].metadata is not None
        assert result["cp_b1"].metadata["judge_model"] == "model_b/judge"


class TestScoreSubtasksLLMBatchTrajectoryMismatch:
    """Contexts with different tool_steps → ValueError."""

    @pytest.mark.asyncio
    async def test_different_steps_raises(self) -> None:
        sub_a = _make_subtask("cp_a")
        sub_b = _make_subtask("cp_b")
        criteria = _make_criteria()
        ctx_a = _make_context(
            subtask=sub_a,
            criteria=criteria,
            tool_steps=(_make_tool_step(1, "out_1"),),
        )
        ctx_b = _make_context(
            subtask=sub_b,
            criteria=criteria,
            tool_steps=(_make_tool_step(1, "different"),),
        )

        with pytest.raises(ValueError, match="different tool_steps"):
            await score_checkpoints_llm_batch(contexts=[ctx_a, ctx_b], renderer=_make_renderer())


class TestScoreSubtasksLLMBatchEmptySteps:
    """Contexts with empty tool_steps → still calls LLM with empty chunk."""

    @pytest.mark.asyncio
    async def test_empty_tool_steps(self) -> None:
        sub = _make_subtask("cp_1")
        criteria = _make_criteria()
        ctx = _make_context(subtask=sub, criteria=criteria, tool_steps=())

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=_mock_model_response("[1: cp_1]"))

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx], renderer=_make_renderer())

        assert result["cp_1"].value == 1.0
        assert mock_model.generate.call_count == 1


class TestScoreSubtasksLLMBatchCheckpointFields:
    """Checkpoints include title and description from ScorerConfig."""

    @pytest.mark.asyncio
    async def test_checkpoint_has_title_and_description(self) -> None:
        from saber.scoring.context import CheckpointObjective

        subtask = _make_subtask(
            "cp_1",
            title="Identify C2 Server",
            description="Find the command-and-control server IP",
            objective="Locate the C2 IP address",
        )
        steps = (_make_tool_step(1, "found it"),)
        ctx = _make_context(subtask=subtask, tool_steps=steps, max_score=1.0)

        captured_checkpoints: list[tuple[CheckpointObjective, ...]] = []

        real_renderer = _make_renderer()
        original_render = real_renderer.render

        def _capture_render(template_name: str, context: dict[str, object]) -> str:
            if "checkpoints" in context:
                captured_checkpoints.append(context["checkpoints"])  # type: ignore[arg-type]
            return original_render(template_name, context)

        real_renderer.render = _capture_render  # type: ignore[method-assign]

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=_mock_model_response("[1: cp_1]"))

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            await score_checkpoints_llm_batch(contexts=[ctx], renderer=real_renderer)

        assert len(captured_checkpoints) > 0
        cp = captured_checkpoints[0][0]
        assert cp.id == "cp_1"
        assert cp.title == "Identify C2 Server"
        assert cp.description == "Find the command-and-control server IP"
        assert cp.objective == "Find the command-and-control server IP"


class TestScoreSubtasksLLMBatchOnlyLLMJudgeSubtasks:
    """Only LLM judge subtasks should appear in checkpoints, not mixed strategies.

    This verifies that when batch scoring is invoked with contexts,
    only the provided contexts (which are pre-filtered to llm_judge) appear.
    Static strategy subtasks are excluded before reaching batch scoring.
    """

    @pytest.mark.asyncio
    async def test_batch_only_receives_llm_judge_contexts(self) -> None:
        """score_checkpoints_llm_batch rejects non-LLMJudgeCriteria contexts."""
        sub_static = _make_subtask("cp_static")
        static_criteria = StaticCriteria(expected_answers=["flag"])
        scorer_static = ScorerConfig(
            scorer_name=str(sub_static["id"]),
            strategy="static",
            target=ScorerTarget.TRAJECTORY,
            criteria=static_criteria,
            max_score=1.0,
        )
        ctx_static = ScoringContext(
            submission="",
            tool_steps=(_make_tool_step(1, "ok"),),
            messages=(ChatMessageUser(content="test"),),
            target="",
            task_id="task_1",
            domain="test_domain",
            metadata={"description": "Test task"},
            scorer=scorer_static,
        )

        sub_judge = _make_subtask("cp_judge")
        ctx_judge = _make_context(
            subtask=sub_judge,
            tool_steps=(_make_tool_step(1, "ok"),),
        )

        # Mixing strategies raises ValueError (contexts must all be LLMJudgeCriteria)
        with pytest.raises(ValueError, match="LLMJudgeCriteria"):
            await score_checkpoints_llm_batch(contexts=[ctx_judge, ctx_static], renderer=_make_renderer())


class TestScoreSubtasksLLMBatchSubmissionFields:
    """Submission and description are available in judge templates."""

    @pytest.mark.asyncio
    async def test_submission_and_description_in_template(self) -> None:
        """Verify submission and description are rendered in judge templates."""
        criteria = LLMJudgeCriteria(
            model="mockmodel/judge",
            judge_system_template="sub_system.j2",
            judge_user_template="sub_user.j2",
        )
        _make_subtask("submission", objective="Judge submission")
        scorer_config = ScorerConfig(
            scorer_name="submission",
            strategy="llm_judge",
            target=ScorerTarget.SUBMISSION,
            criteria=criteria,
            max_score=1.0,
            description="mimilove.exe",
        )
        ctx = ScoringContext(
            submission="The answer is mimilove.exe",
            tool_steps=(_make_tool_step(1, "found it"),),
            messages=(ChatMessageUser(content="test"),),
            target="",
            task_id="task_1",
            domain="test_domain",
            metadata={"description": "Test task"},
            scorer=scorer_config,
        )

        captured_contexts: list[dict[str, object]] = []

        renderer = TemplateRenderer(
            templates={
                "sub_system.j2": "System prompt",
                "sub_user.j2": "Submission: {{ submission }} Description: {{ description }}",
            }
        )
        original_render = renderer.render

        def _capture_render(template_name: str, context: dict[str, object]) -> str:
            captured_contexts.append(context)
            return original_render(template_name, context)

        renderer.render = _capture_render  # type: ignore[method-assign]

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=_mock_model_response("[1: submission]"))

        with patch("saber.scoring.batch.get_model", return_value=mock_model):
            result = await score_checkpoints_llm_batch(contexts=[ctx], renderer=renderer)

        assert result["submission"].value == 1.0
        # Verify submission and description were in the template context
        assert len(captured_contexts) >= 1
        user_ctx = captured_contexts[-1]  # user template is rendered second
        assert user_ctx["submission"] == "The answer is mimilove.exe"
        assert user_ctx["description"] == "mimilove.exe"
