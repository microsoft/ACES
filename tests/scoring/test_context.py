# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for ToolStep and ScoringContext Pydantic models."""

from __future__ import annotations

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
)
from inspect_ai.tool import ToolCall

from saber.config.models import (
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
    to_template_vars,
)
from saber.scoring.context import (
    CheckpointObjective,
    EpisodeContext,
    JudgeTemplateContext,
    ScoringContext,
    TaskMetadata,
    ToolStep,
)

# ── ToolStep ────────────────────────────────────────────────────────


class TestToolStepConstruction:
    """Happy-path construction and field verification."""

    def test_minimal_construction(self) -> None:
        step = ToolStep(
            step_number=1,
            tool_name="bash",
            tool_input={"cmd": "ls"},
            output="file1.txt",
        )
        assert step.step_number == 1
        assert step.tool_name == "bash"
        assert step.tool_input == {"cmd": "ls"}
        assert step.output == "file1.txt"

    def test_all_fields(self) -> None:
        step = ToolStep(
            step_number=3,
            tool_name="python",
            tool_input={"code": "print(1)"},
            output="1",
            is_error=True,
            error_type="runtime",
            assistant_message="Let me run that.",
            reasoning="I should execute the code.",
        )
        assert step.step_number == 3
        assert step.tool_name == "python"
        assert step.tool_input == {"code": "print(1)"}
        assert step.output == "1"
        assert step.is_error is True
        assert step.error_type == "runtime"
        assert step.assistant_message == "Let me run that."
        assert step.reasoning == "I should execute the code."


class TestToolStepDefaults:
    """Default values for optional fields."""

    def test_defaults(self) -> None:
        step = ToolStep(
            step_number=1,
            tool_name="bash",
            tool_input={},
            output="ok",
        )
        assert step.is_error is False
        assert step.error_type is None
        assert step.assistant_message is None
        assert step.reasoning is None


class TestToolStepInputTypes:
    """tool_input accepts non-string values (dict[str, object])."""

    def test_non_string_values(self) -> None:
        step = ToolStep(
            step_number=1,
            tool_name="api_call",
            tool_input={"count": 42, "verbose": True, "tags": ["a", "b"]},
            output="ok",
        )
        assert step.tool_input["count"] == 42
        assert step.tool_input["verbose"] is True
        assert step.tool_input["tags"] == ["a", "b"]

    def test_mixed_types(self) -> None:
        step = ToolStep(
            step_number=1,
            tool_name="complex_tool",
            tool_input={"cmd": "ls", "timeout": 30, "nested": {"key": "val"}},
            output="result",
        )
        assert step.tool_input["cmd"] == "ls"
        assert step.tool_input["timeout"] == 30
        assert step.tool_input["nested"] == {"key": "val"}


class TestToolStepFrozen:
    """Immutability enforcement."""

    def test_cannot_mutate(self) -> None:
        step = ToolStep(
            step_number=1,
            tool_name="bash",
            tool_input={},
            output="ok",
        )
        with pytest.raises(Exception):  # noqa: B017
            step.output = "changed"  # type: ignore[misc]


class TestEpisodeContext:
    """EpisodeContext wraps tool steps with helper methods for templates."""

    def test_construction_with_steps(self) -> None:
        steps = (
            ToolStep(step_number=1, tool_name="bash", tool_input={"cmd": "ls"}, output="ok"),
            ToolStep(step_number=2, tool_name="python", tool_input={"code": "1+1"}, output="2"),
        )
        ctx = EpisodeContext(steps=steps)
        assert len(ctx.steps) == 2
        assert isinstance(ctx.steps[0], ToolStep)
        assert isinstance(ctx.steps[1], ToolStep)

    def test_step_count_property(self) -> None:
        steps = (
            ToolStep(step_number=1, tool_name="bash", tool_input={}, output="ok"),
            ToolStep(step_number=2, tool_name="python", tool_input={}, output="2"),
            ToolStep(step_number=3, tool_name="read", tool_input={}, output="x"),
        )
        ctx = EpisodeContext(steps=steps)
        assert ctx.step_count == 3

    def test_get_step_count_method(self) -> None:
        """Callable form for Jinja2 templates: {{ episode.get_step_count() }}."""
        steps = (
            ToolStep(step_number=1, tool_name="bash", tool_input={}, output="ok"),
            ToolStep(step_number=2, tool_name="python", tool_input={}, output="2"),
        )
        ctx = EpisodeContext(steps=steps)
        assert ctx.get_step_count() == 2
        assert ctx.get_step_count() == ctx.step_count

    def test_empty_steps_returns_zero_count(self) -> None:
        ctx = EpisodeContext(steps=())
        assert ctx.step_count == 0
        assert ctx.get_step_count() == 0

    def test_frozen_immutable(self) -> None:
        ctx = EpisodeContext(steps=())
        with pytest.raises(Exception):  # noqa: B017
            ctx.steps = ()  # type: ignore[misc]

    def test_steps_are_toolstep_instances(self) -> None:
        steps = (ToolStep(step_number=1, tool_name="bash", tool_input={"cmd": "ls"}, output="ok"),)
        ctx = EpisodeContext(steps=steps)
        assert all(isinstance(s, ToolStep) for s in ctx.steps)


# ── CheckpointObjective ─────────────────────────────────────────────


class TestCheckpointObjective:
    """CheckpointObjective: frozen model for subtask checkpoint in judge templates."""

    def test_construction(self) -> None:
        cp = CheckpointObjective(id="cp1", objective="Find the flag")
        assert cp.id == "cp1"
        assert cp.objective == "Find the flag"

    def test_frozen(self) -> None:
        cp = CheckpointObjective(id="cp1", objective="test")
        with pytest.raises(Exception):  # noqa: B017
            cp.id = "cp2"  # type: ignore[misc]

    def test_equality(self) -> None:
        cp1 = CheckpointObjective(id="cp1", objective="obj")
        cp2 = CheckpointObjective(id="cp1", objective="obj")
        assert cp1 == cp2

    def test_all_four_fields(self) -> None:
        """CheckpointObjective supports id, objective, title, and description."""
        cp = CheckpointObjective(
            id="cp1",
            objective="Find the flag",
            title="Flag Finder",
            description="Locate the hidden flag in /tmp",
        )
        assert cp.id == "cp1"
        assert cp.objective == "Find the flag"
        assert cp.title == "Flag Finder"
        assert cp.description == "Locate the hidden flag in /tmp"

    def test_title_description_default_to_empty(self) -> None:
        """title and description default to empty string for backward compat."""
        cp = CheckpointObjective(id="cp1", objective="obj")
        assert cp.title == ""
        assert cp.description == ""

    def test_equality_with_all_fields(self) -> None:
        cp1 = CheckpointObjective(id="cp1", objective="obj", title="T", description="D")
        cp2 = CheckpointObjective(id="cp1", objective="obj", title="T", description="D")
        assert cp1 == cp2

    def test_inequality_different_title(self) -> None:
        cp1 = CheckpointObjective(id="cp1", objective="obj", title="A")
        cp2 = CheckpointObjective(id="cp1", objective="obj", title="B")
        assert cp1 != cp2

    def test_hints_default_empty(self) -> None:
        """hints defaults to empty tuple."""
        cp = CheckpointObjective(id="cp1", objective="obj")
        assert cp.hints == ()

    def test_hints_stored_as_tuple(self) -> None:
        """hints field stores hint strings."""
        cp = CheckpointObjective(
            id="cp1",
            objective="Find the flag",
            hints=("Look in /tmp", "Check ENV vars"),
        )
        assert cp.hints == ("Look in /tmp", "Check ENV vars")
        assert len(cp.hints) == 2

    def test_equality_with_hints(self) -> None:
        cp1 = CheckpointObjective(id="cp1", objective="obj", hints=("h1",))
        cp2 = CheckpointObjective(id="cp1", objective="obj", hints=("h1",))
        assert cp1 == cp2

    def test_inequality_different_hints(self) -> None:
        cp1 = CheckpointObjective(id="cp1", objective="obj", hints=("h1",))
        cp2 = CheckpointObjective(id="cp1", objective="obj", hints=("h2",))
        assert cp1 != cp2


# ── TaskMetadata ────────────────────────────────────────────────────


class TestTaskMetadata:
    """TaskMetadata: frozen model with extra='allow' for judge template {{ task.field }}."""

    def test_default_values(self) -> None:
        tm = TaskMetadata()
        assert tm.task_id == ""
        assert tm.title == ""
        assert tm.description == ""

    def test_explicit_values(self) -> None:
        tm = TaskMetadata(task_id="t1", title="Task 1", description="Do thing")
        assert tm.task_id == "t1"
        assert tm.title == "Task 1"
        assert tm.description == "Do thing"

    def test_extra_fields_accepted(self) -> None:
        tm = TaskMetadata(task_id="t1", custom_field="extra_value")
        assert tm.custom_field == "extra_value"  # type: ignore[attr-defined]

    def test_frozen(self) -> None:
        tm = TaskMetadata(task_id="t1")
        with pytest.raises(Exception):  # noqa: B017
            tm.task_id = "t2"  # type: ignore[misc]

    def test_constructed_from_metadata_dict(self) -> None:
        """TaskMetadata(**ctx.metadata) pattern works."""
        metadata: dict[str, object] = {
            "task_id": "t1",
            "title": "T",
            "description": "D",
            "initial_context": {"key": "val"},
        }
        tm = TaskMetadata(**metadata)
        assert tm.task_id == "t1"
        assert tm.initial_context == {"key": "val"}  # type: ignore[attr-defined]


# ── JudgeTemplateContext ────────────────────────────────────────────


class TestJudgeTemplateContext:
    """JudgeTemplateContext: typed context for LLM judge template rendering."""

    def _make_episode(self) -> EpisodeContext:
        return EpisodeContext(steps=(ToolStep(step_number=1, tool_name="bash", tool_input={}, output="ok"),))

    def _make_task_metadata(self) -> TaskMetadata:
        return TaskMetadata(task_id="t1", title="T", description="D")

    def test_minimal_construction(self) -> None:
        ctx = JudgeTemplateContext(
            task_id="t1",
            domain="test",
            question="What happened?",
            episode=self._make_episode(),
            task=self._make_task_metadata(),
        )
        assert ctx.task_id == "t1"
        assert ctx.domain == "test"
        assert ctx.question == "What happened?"
        assert ctx.submission is None
        assert ctx.description == ""
        assert ctx.objective is None
        assert ctx.checkpoint_id is None
        assert ctx.checkpoints == ()
        assert ctx.chunk_index is None
        assert ctx.total_chunks is None

    def test_full_construction(self) -> None:
        cp = CheckpointObjective(id="cp1", objective="Find flag")
        ctx = JudgeTemplateContext(
            task_id="t1",
            domain="test",
            question="Q",
            episode=self._make_episode(),
            task=self._make_task_metadata(),
            submission="my answer",
            description="correct answer",
            objective="Find flag",
            checkpoint_id="cp1",
            checkpoints=(cp,),
            chunk_index=0,
            total_chunks=3,
        )
        assert ctx.submission == "my answer"
        assert ctx.description == "correct answer"
        assert ctx.objective == "Find flag"
        assert ctx.checkpoint_id == "cp1"
        assert len(ctx.checkpoints) == 1
        assert ctx.checkpoints[0].id == "cp1"
        assert ctx.chunk_index == 0
        assert ctx.total_chunks == 3

    def test_frozen(self) -> None:
        ctx = JudgeTemplateContext(
            task_id="t1",
            domain="d",
            question="q",
            episode=self._make_episode(),
            task=self._make_task_metadata(),
        )
        with pytest.raises(Exception):  # noqa: B017
            ctx.task_id = "t2"  # type: ignore[misc]

    def test_to_template_vars_includes_all_fields(self) -> None:
        """to_template_vars preserves nested models for Jinja2."""
        ctx = JudgeTemplateContext(
            task_id="t1",
            domain="test",
            question="Q",
            episode=self._make_episode(),
            task=self._make_task_metadata(),
            submission="answer",
        )
        tvars = to_template_vars(ctx)
        assert tvars["task_id"] == "t1"
        assert tvars["submission"] == "answer"
        assert isinstance(tvars["episode"], EpisodeContext)
        assert isinstance(tvars["task"], TaskMetadata)


# ── ScoringContext ──────────────────────────────────────────────────


def _make_messages() -> tuple[
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    ChatMessageTool,
]:
    """Build a minimal set of inspect_ai messages for testing."""
    sys_msg = ChatMessageSystem(content="You are a helper.")
    user_msg = ChatMessageUser(content="Run ls")
    assistant_msg = ChatMessageAssistant(
        content=[ContentText(text="Sure, let me run that.")],
        tool_calls=[
            ToolCall(id="call_1", function="bash", arguments={"cmd": "ls"}, type="function"),
        ],
    )
    tool_msg = ChatMessageTool(content="file1.txt", tool_call_id="call_1")
    return sys_msg, user_msg, assistant_msg, tool_msg


class TestScoringContextConstruction:
    """Full construction with all fields."""

    def test_full_construction(self) -> None:
        msgs = _make_messages()
        tool_step = ToolStep(
            step_number=1,
            tool_name="bash",
            tool_input={"cmd": "ls"},
            output="file1.txt",
        )
        scorer = ScorerConfig(
            scorer_name="sub",
            strategy="static",
            target=ScorerTarget.SUBMISSION,
            criteria=StaticCriteria(expected_answers=["file1.txt"]),
        )

        ctx = ScoringContext(
            submission="file1.txt",
            tool_steps=(tool_step,),
            messages=tuple(msgs),
            target="file1.txt",
            task_id="task_1",
            domain="test_domain",
            metadata={"key": "value"},
            scorer=scorer,
        )

        assert ctx.submission == "file1.txt"
        assert len(ctx.tool_steps) == 1
        assert ctx.tool_steps[0].tool_name == "bash"
        assert len(ctx.messages) == 4
        assert ctx.target == "file1.txt"
        assert ctx.task_id == "task_1"
        assert ctx.domain == "test_domain"
        assert ctx.metadata == {"key": "value"}
        assert ctx.scorer.scorer_name == "sub"


class TestScoringContextSubtaskNone:
    """Submission scoring mode — no subtask."""

    def test_scorer_is_required(self) -> None:
        msgs = _make_messages()
        scorer = ScorerConfig(
            scorer_name="submission",
            strategy="static",
            target=ScorerTarget.SUBMISSION,
            criteria=StaticCriteria(expected_answers=["answer"]),
        )
        ctx = ScoringContext(
            submission="answer",
            tool_steps=(),
            messages=tuple(msgs),
            target="answer",
            task_id="task_2",
            domain="demo",
            metadata={},
            scorer=scorer,
        )
        assert ctx.scorer.scorer_name == "submission"


class TestScoringContextFrozen:
    """Immutability enforcement."""

    def test_cannot_mutate(self) -> None:
        msgs = _make_messages()
        scorer = ScorerConfig(
            scorer_name="sub",
            strategy="static",
            target=ScorerTarget.SUBMISSION,
            criteria=StaticCriteria(expected_answers=["x"]),
        )
        ctx = ScoringContext(
            submission="x",
            tool_steps=(),
            messages=tuple(msgs),
            target="x",
            task_id="t",
            domain="d",
            metadata={},
            scorer=scorer,
        )
        with pytest.raises(Exception):  # noqa: B017
            ctx.submission = "y"  # type: ignore[misc]


class TestScoringContextMessages:
    """Verify messages stored as tuple of ChatMessage."""

    def test_messages_as_tuple(self) -> None:
        msgs = _make_messages()
        scorer = ScorerConfig(
            scorer_name="sub",
            strategy="static",
            target=ScorerTarget.SUBMISSION,
            criteria=StaticCriteria(expected_answers=["x"]),
        )
        ctx = ScoringContext(
            submission="x",
            tool_steps=(),
            messages=tuple(msgs),
            target="x",
            task_id="t",
            domain="d",
            metadata={},
            scorer=scorer,
        )
        assert isinstance(ctx.messages, tuple)
        assert len(ctx.messages) == 4


# ── ScoringContext — new scorer field ───────────────────────────────


class TestScoringContextNew:
    """Tests for updated ScoringContext with scorer field."""

    def test_construction_with_scorer(self) -> None:
        """ScoringContext accepts new scorer field."""
        from saber.config.models import ScorerConfig, ScorerTarget, StaticCriteria

        scorer = ScorerConfig(
            scorer_name="sub",
            strategy="static",
            target=ScorerTarget.SUBMISSION,
            criteria=StaticCriteria(expected_answers=["42"]),
        )
        ctx = ScoringContext(
            submission="42",
            tool_steps=(),
            messages=(),
            target_text="42",
            task_id="t",
            domain="d",
            metadata={},
            scorer=scorer,
        )
        assert ctx.scorer is not None
        assert ctx.scorer.scorer_name == "sub"

    def test_target_text_field(self) -> None:
        """Field is now target_text, not target."""
        from saber.config.models import ScorerConfig, ScorerTarget, StaticCriteria

        scorer = ScorerConfig(
            scorer_name="sub",
            strategy="static",
            target=ScorerTarget.SUBMISSION,
            criteria=StaticCriteria(expected_answers=["golden"]),
        )
        ctx = ScoringContext(
            submission="x",
            tool_steps=(),
            messages=(),
            target_text="golden",
            task_id="t",
            domain="d",
            metadata={},
            scorer=scorer,
        )
        assert ctx.target_text == "golden"
