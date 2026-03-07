"""Tests for saber.config.converter — TaskConfig to inspect_ai Sample conversion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from saber.config.converter import (
    _PROMPT_FIELD_MAP,
    _build_metadata,
    _extract_target,
    _resolve_files,
    _resolve_sandbox,
    tasks_to_samples,
)
from saber.config.models import (
    AggregationConfig,
    DomainCriteria,
    LLMJudgeCriteria,
    PromptPaths,
    ScoreAggregation,
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
    TaskConfig,
    ToolCallCriteria,
    ToolConfig,
)
from saber.prompts.renderer import PromptRenderer

# ── Fixtures ────────────────────────────────────────────────────────


def _minimal_prompts() -> PromptPaths:
    return PromptPaths(
        instruction="instructions/demo.md",
        assistant="assistants/inspect_assistant.md",
    )


def _minimal_task(
    *,
    task_id: str = "task_1",
    title: str = "Test Task",
    description: str = "A test task",
    sandbox: str | None = None,
    initial_files: dict[str, str] | None = None,
    tools: dict[str, ToolConfig] | None = None,
    scorers: tuple[ScorerConfig, ...] = (),
    scoring_aggregation: AggregationConfig | None = None,
    max_steps: int = 25,
    aggregation: ScoreAggregation = ScoreAggregation.MAX,
) -> TaskConfig:
    return TaskConfig(
        task_id=task_id,
        title=title,
        description=description,
        prompts=_minimal_prompts(),
        sandbox_environment=sandbox,
        initial_files=initial_files or {},
        tools=tools or {},
        scorers=scorers,
        scoring_aggregation=scoring_aggregation,
        max_steps=max_steps,
        aggregation=aggregation,
    )


def _mock_renderer(rendered_instruction: str = "mocked instruction") -> PromptRenderer:
    renderer = create_autospec(PromptRenderer, instance=True)
    renderer.render.return_value = rendered_instruction
    # build_task_context returns a realistic dict so render() receives it
    renderer.build_task_context.side_effect = lambda task: {
        "task_id": task.task_id,
        "title": task.title,
        "description": task.description,
        "initial_context": task.initial_context,
        "tools": list(task.tools.keys()),
        "max_steps": task.max_steps,
    }

    def _mock_render_all(prompts: object, task: object) -> dict[str, str]:
        result: dict[str, str] = {}
        for key in ("instruction", "assistant"):
            path = getattr(prompts, key, None)
            if path:
                result[key] = rendered_instruction
        return result

    renderer.render_all_prompts.side_effect = _mock_render_all
    return renderer


def _mock_renderer_per_path() -> PromptRenderer:
    """Mock renderer that returns different values per template path."""
    renderer = create_autospec(PromptRenderer, instance=True)
    renderer.build_task_context.side_effect = lambda task: {
        "task_id": task.task_id,
        "title": task.title,
        "description": task.description,
        "initial_context": task.initial_context,
        "tools": list(task.tools.keys()),
        "max_steps": task.max_steps,
    }
    renderer.render.side_effect = lambda path, _ctx: f"rendered:{path}"

    def _mock_render_all(prompts: object, task: object) -> dict[str, str]:
        result: dict[str, str] = {}
        for key in ("instruction", "assistant"):
            path = getattr(prompts, key, None)
            if path:
                result[key] = f"rendered:{path}"
        return result

    renderer.render_all_prompts.side_effect = _mock_render_all
    return renderer


# ── _extract_target tests ───────────────────────────────────────────


class TestExtractTarget:
    def test_static_criteria_returns_first_expected_answer(self) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="submission",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                    criteria=StaticCriteria(expected_answers=["answer1", "answer2"]),
                ),
            ),
        )
        assert _extract_target(task) == "answer1"

    def test_static_criteria_single_answer(self) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="submission",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                    criteria=StaticCriteria(expected_answers=["only_answer"]),
                ),
            ),
        )
        assert _extract_target(task) == "only_answer"

    def test_llm_judge_returns_empty(self) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="submission",
                    strategy="llm_judge",
                    target=ScorerTarget.SUBMISSION,
                    criteria=LLMJudgeCriteria(
                        model="gpt-4",
                        judge_system_template="sys.md",
                        judge_user_template="usr.md",
                    ),
                ),
            ),
        )
        assert _extract_target(task) == ""

    def test_no_scorers_returns_empty(self) -> None:
        task = _minimal_task()
        assert _extract_target(task) == ""

    def test_domain_criteria_returns_empty(self) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="domain_check",
                    strategy="domain",
                    target=ScorerTarget.SUBMISSION,
                    criteria=DomainCriteria(),
                ),
            ),
        )
        assert _extract_target(task) == ""

    def test_static_criteria_empty_answers_returns_empty(self) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="submission",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                    criteria=StaticCriteria(expected_answers=[]),
                ),
            ),
        )
        assert _extract_target(task) == ""

    def test_tool_call_criteria_returns_empty(self) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="tool_check",
                    strategy="tool_call",
                    target=ScorerTarget.SUBMISSION,
                    criteria=ToolCallCriteria(tool_name="run_command"),
                ),
            ),
        )
        assert _extract_target(task) == ""

    def test_trajectory_scorer_not_used_for_target(self) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="checkpoint",
                    strategy="static",
                    target=ScorerTarget.TRAJECTORY,
                    criteria=StaticCriteria(expected_answers=["nope"]),
                ),
            ),
        )
        assert _extract_target(task) == ""

    def test_first_submission_scorer_wins(self) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="trajectory_check",
                    strategy="static",
                    target=ScorerTarget.TRAJECTORY,
                    criteria=StaticCriteria(expected_answers=["skip_me"]),
                ),
                ScorerConfig(
                    scorer_name="submission",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                    criteria=StaticCriteria(expected_answers=["first"]),
                ),
                ScorerConfig(
                    scorer_name="submission_2",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                    criteria=StaticCriteria(expected_answers=["second"]),
                ),
            ),
        )
        assert _extract_target(task) == "first"


# ── _resolve_sandbox tests ──────────────────────────────────────────


class TestResolveSandbox:
    def test_task_with_sandbox(self, tmp_path: Path) -> None:
        task = _minimal_task(sandbox="excytin_sandbox")
        result = _resolve_sandbox(task, tmp_path)
        expected_path = str(tmp_path / "compose" / "excytin_sandbox.compose.yml")
        assert result == ("docker", expected_path)

    def test_task_without_sandbox(self, tmp_path: Path) -> None:
        task = _minimal_task(sandbox=None)
        result = _resolve_sandbox(task, tmp_path)
        assert result is None

    def test_sandbox_path_format(self, tmp_path: Path) -> None:
        task = _minimal_task(sandbox="my-env")
        result = _resolve_sandbox(task, tmp_path)
        assert result is not None
        _, path = result
        assert path.endswith("my-env.compose.yml")
        assert "compose/" in path


# ── _resolve_files tests ────────────────────────────────────────────


class TestResolveFiles:
    def test_empty_initial_files(self, tmp_path: Path) -> None:
        task = _minimal_task(initial_files={})
        result = _resolve_files(task, tmp_path)
        assert result == {}

    def test_single_file(self, tmp_path: Path) -> None:
        task = _minimal_task(initial_files={"/home/user/data.csv": "data.csv"})
        result = _resolve_files(task, tmp_path)
        expected = {"/home/user/data.csv": str(tmp_path / "data.csv")}
        assert result == expected

    def test_multiple_files(self, tmp_path: Path) -> None:
        task = _minimal_task(
            initial_files={
                "/opt/file1.txt": "file1.txt",
                "/opt/file2.json": "subdir/file2.json",
            }
        )
        result = _resolve_files(task, tmp_path)
        assert result == {
            "/opt/file1.txt": str(tmp_path / "file1.txt"),
            "/opt/file2.json": str(tmp_path / "subdir" / "file2.json"),
        }


# ── tasks_to_samples integration tests ─────────────────────────────


class TestTasksToSamples:
    def test_single_task_produces_single_sample(self, tmp_path: Path) -> None:
        task = _minimal_task()
        renderer = _mock_renderer("rendered instruction")
        samples = tasks_to_samples([task], tmp_path, renderer)
        assert len(samples) == 1
        assert isinstance(samples[0], Sample)

    def test_multiple_tasks_produce_multiple_samples(self, tmp_path: Path) -> None:
        tasks = [
            _minimal_task(task_id="task_a"),
            _minimal_task(task_id="task_b"),
            _minimal_task(task_id="task_c"),
        ]
        renderer = _mock_renderer()
        samples = tasks_to_samples(tasks, tmp_path, renderer)
        assert len(samples) == 3

    def test_sample_id_matches_task_id(self, tmp_path: Path) -> None:
        task = _minimal_task(task_id="unique_id")
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.id == "unique_id"

    def test_sample_input_has_system_and_user_messages(self, tmp_path: Path) -> None:
        task = _minimal_task(description="user desc")
        renderer = _mock_renderer("system instruction")
        sample = tasks_to_samples([task], tmp_path, renderer)[0]

        assert isinstance(sample.input, list)
        assert len(sample.input) == 2
        assert isinstance(sample.input[0], ChatMessageSystem)
        assert sample.input[0].content == "system instruction"
        assert isinstance(sample.input[1], ChatMessageUser)
        assert sample.input[1].content == "user desc"

    def test_sample_target_extracted_correctly(self, tmp_path: Path) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="submission",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                    criteria=StaticCriteria(expected_answers=["correct"]),
                ),
            ),
        )
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.target == "correct"

    def test_sample_metadata_has_all_expected_keys(self, tmp_path: Path) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="submission",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                    criteria=StaticCriteria(expected_answers=["ans"]),
                ),
            ),
            tools={"bash": ToolConfig(timeout=60)},
        )
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]

        expected_keys = {
            "task_id",
            "title",
            "description",
            "initial_context",
            "prompts",
            "tools",
            "max_steps",
            "tool_call_limit",
            "aggregation",
            # Rendered prompt strings for solver_factory
            "instruction_prompt",
            "assistant_prompt",
            # Scorer configuration
            "scorers",
            "scoring_aggregation",
        }
        assert sample.metadata is not None
        assert set(sample.metadata.keys()) == expected_keys

    def test_sample_metadata_values(self, tmp_path: Path) -> None:
        task = _minimal_task(
            task_id="meta_task",
            title="Meta Title",
            description="Meta Desc",
            max_steps=10,
            aggregation=ScoreAggregation.AVERAGE,
        )
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]

        assert sample.metadata is not None
        assert sample.metadata["task_id"] == "meta_task"
        assert sample.metadata["title"] == "Meta Title"
        assert sample.metadata["description"] == "Meta Desc"
        assert sample.metadata["max_steps"] == 10
        assert sample.metadata["tool_call_limit"] == 10
        assert sample.metadata["aggregation"] == "average"

    def test_sample_sandbox_resolved(self, tmp_path: Path) -> None:
        task = _minimal_task(sandbox="my_sandbox")
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.sandbox is not None
        assert sample.sandbox.type == "docker"
        assert "my_sandbox.compose.yml" in str(sample.sandbox.config)

    def test_sample_sandbox_none_when_no_sandbox(self, tmp_path: Path) -> None:
        task = _minimal_task(sandbox=None)
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.sandbox is None

    def test_sample_files_resolved(self, tmp_path: Path) -> None:
        task = _minimal_task(initial_files={"/dest/file.txt": "src.txt"})
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.files is not None
        assert "/dest/file.txt" in sample.files
        assert sample.files["/dest/file.txt"] == str(tmp_path / "src.txt")

    def test_prompt_renderer_called_with_correct_args(self, tmp_path: Path) -> None:
        task = _minimal_task(
            task_id="render_test",
            title="Render Title",
            description="Render Desc",
            max_steps=15,
        )
        renderer = _mock_renderer("called instruction")
        tasks_to_samples([task], tmp_path, renderer)

        # render_all_prompts is called once per task
        renderer.render_all_prompts.assert_called_once()
        call_args = renderer.render_all_prompts.call_args[0]
        prompts_arg = call_args[0]
        task_arg = call_args[1]

        assert prompts_arg.instruction == "instructions/demo.md"
        assert task_arg.task_id == "render_test"
        assert task_arg.title == "Render Title"
        assert task_arg.description == "Render Desc"
        assert task_arg.max_steps == 15

    def test_empty_task_list_returns_empty(self, tmp_path: Path) -> None:
        renderer = _mock_renderer()
        samples = tasks_to_samples([], tmp_path, renderer)
        assert samples == []


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_task_with_no_scorers(self, tmp_path: Path) -> None:
        task = _minimal_task()
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        assert sample.metadata["scorers"] == []

    def test_task_with_no_tools(self, tmp_path: Path) -> None:
        task = _minimal_task(tools={})
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        assert sample.metadata["tools"] == {}

    def test_task_with_no_scorers_empty_target(self, tmp_path: Path) -> None:
        task = _minimal_task()
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        assert sample.target == ""

    def test_scorers_serialized_in_metadata(self, tmp_path: Path) -> None:
        task = _minimal_task(
            scorers=(
                ScorerConfig(
                    scorer_name="sub_a",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                    criteria=StaticCriteria(expected_answers=["a"]),
                ),
                ScorerConfig(
                    scorer_name="sub_b",
                    strategy="llm_judge",
                    target=ScorerTarget.TRAJECTORY,
                    criteria=LLMJudgeCriteria(
                        model="gpt-4",
                        judge_system_template="sys.md",
                        judge_user_template="usr.md",
                    ),
                ),
            ),
        )
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        scorers = sample.metadata["scorers"]
        assert len(scorers) == 2
        assert scorers[0]["scorer_name"] == "sub_a"
        assert scorers[1]["scorer_name"] == "sub_b"

    def test_tools_serialized_in_metadata(self, tmp_path: Path) -> None:
        task = _minimal_task(
            tools={
                "bash": ToolConfig(timeout=60),
                "python": ToolConfig(timeout=120),
            },
        )
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        tools = sample.metadata["tools"]
        assert "bash" in tools
        assert tools["bash"]["timeout"] == 60
        assert "python" in tools
        assert tools["python"]["timeout"] == 120

    def test_scorer_config_serialized(self, tmp_path: Path) -> None:
        scorer = ScorerConfig(
            scorer_name="submission",
            strategy="static",
            target=ScorerTarget.SUBMISSION,
            max_score=10.0,
            weight=0.5,
            criteria=StaticCriteria(expected_answers=["ans"]),
        )
        task = _minimal_task(scorers=(scorer,))
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        scorers = sample.metadata["scorers"]
        assert len(scorers) == 1
        assert scorers[0]["strategy"] == "static"
        assert scorers[0]["max_score"] == 10.0
        assert scorers[0]["weight"] == 0.5

    def test_prompts_serialized_in_metadata(self, tmp_path: Path) -> None:
        task = _minimal_task()
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        prompts = sample.metadata["prompts"]
        assert prompts["instruction"] == "instructions/demo.md"

    def test_initial_context_in_metadata(self, tmp_path: Path) -> None:
        task = TaskConfig(
            task_id="ctx_task",
            title="Context Task",
            description="desc",
            prompts=_minimal_prompts(),
            initial_context={"key": "value", "num": 42},
        )
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        assert sample.metadata["initial_context"] == {"key": "value", "num": 42}

    def test_scoring_aggregation_none(self, tmp_path: Path) -> None:
        task = _minimal_task()
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        assert sample.metadata["scoring_aggregation"] is None

    def test_scoring_aggregation_present(self, tmp_path: Path) -> None:
        task = _minimal_task(
            scoring_aggregation=AggregationConfig(
                strategy=ScoreAggregation.MAX,
                scores=["submission", "checkpoint_1"],
            ),
        )
        renderer = _mock_renderer()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]
        assert sample.metadata is not None
        agg = sample.metadata["scoring_aggregation"]
        assert agg is not None
        assert agg["strategy"] == "max"
        assert agg["scores"] == ["submission", "checkpoint_1"]


# ── Rendered prompts in metadata ─────────────────────────────────────


class TestRenderAllPrompts:
    """Tests for PromptRenderer.render_all_prompts via mock."""

    def test_renders_all_prompts(self) -> None:
        task = _minimal_task()
        renderer = _mock_renderer_per_path()
        result = renderer.render_all_prompts(task.prompts, task)

        assert "instruction" in result
        assert "assistant" in result

    def test_rendered_values_come_from_renderer(self) -> None:
        task = _minimal_task()
        renderer = _mock_renderer_per_path()
        result = renderer.render_all_prompts(task.prompts, task)

        assert result["instruction"] == "rendered:instructions/demo.md"
        assert result["assistant"] == "rendered:assistants/inspect_assistant.md"

    def test_custom_prompt_paths(self) -> None:
        prompts = PromptPaths(
            instruction="custom/inst.md",
            assistant="custom/asst.md",
        )
        task = TaskConfig(
            task_id="custom_prompts",
            title="Custom",
            description="desc",
            prompts=prompts,
        )
        renderer = _mock_renderer_per_path()
        result = renderer.render_all_prompts(task.prompts, task)

        assert result["instruction"] == "rendered:custom/inst.md"
        assert result["assistant"] == "rendered:custom/asst.md"


class TestBuildMetadataWithRenderedPrompts:
    """Tests for _build_metadata including rendered prompt strings."""

    def test_metadata_contains_rendered_prompts(self) -> None:
        task = _minimal_task()
        rendered = {
            "instruction": "You are a security analyst...",
            "assistant": "Assistant prompt text",
        }
        metadata = _build_metadata(task, rendered)

        assert metadata["instruction_prompt"] == "You are a security analyst..."
        assert metadata["assistant_prompt"] == "Assistant prompt text"

    def test_metadata_empty_rendered_prompts_default_to_empty_string(self) -> None:
        task = _minimal_task()
        rendered: dict[str, str] = {}
        metadata = _build_metadata(task, rendered)

        assert metadata["instruction_prompt"] == ""
        assert metadata["assistant_prompt"] == ""

    def test_metadata_still_has_prompt_paths(self) -> None:
        task = _minimal_task()
        rendered = {"instruction": "rendered"}
        metadata = _build_metadata(task, rendered)

        # Still has the path-based prompts field
        assert "prompts" in metadata
        assert metadata["prompts"]["instruction"] == "instructions/demo.md"

    def test_metadata_preserves_existing_fields(self) -> None:
        task = _minimal_task(task_id="preserved", max_steps=42)
        rendered = {"instruction": "x"}
        metadata = _build_metadata(task, rendered)

        assert metadata["task_id"] == "preserved"
        assert metadata["max_steps"] == 42
        assert metadata["tool_call_limit"] == 42


class TestRenderedPromptsEndToEnd:
    """Integration tests: rendered prompts flow through tasks_to_samples."""

    def test_all_rendered_prompts_in_sample_metadata(self, tmp_path: Path) -> None:
        task = _minimal_task()
        renderer = _mock_renderer_per_path()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]

        assert sample.metadata is not None
        assert sample.metadata["instruction_prompt"] == "rendered:instructions/demo.md"
        assert sample.metadata["assistant_prompt"] == "rendered:assistants/inspect_assistant.md"

    def test_solver_factory_keys_present(self, tmp_path: Path) -> None:
        """Verify the metadata keys match what solver_factory.py reads."""
        task = _minimal_task()
        renderer = _mock_renderer_per_path()
        sample = tasks_to_samples([task], tmp_path, renderer)[0]

        assert sample.metadata is not None
        # These are the exact keys solver_factory.py reads
        for key in (
            "instruction_prompt",
            "assistant_prompt",
        ):
            assert key in sample.metadata
            assert isinstance(sample.metadata[key], str)


class TestPromptFieldMap:
    """Tests for _PROMPT_FIELD_MAP consistency."""

    def test_map_has_all_prompt_fields(self) -> None:
        """The mapping covers all prompt types."""
        assert set(_PROMPT_FIELD_MAP.keys()) == {
            "instruction",
            "assistant",
        }

    def test_map_values_end_with_prompt(self) -> None:
        """All metadata keys end with _prompt for consistency."""
        for key, value in _PROMPT_FIELD_MAP.items():
            assert value.endswith("_prompt"), f"{key} -> {value} doesn't end with _prompt"

    def test_render_and_metadata_use_same_keys(self) -> None:
        """Keys from render_all_prompts match _PROMPT_FIELD_MAP keys."""
        task = _minimal_task()
        renderer = _mock_renderer_per_path()
        rendered = renderer.render_all_prompts(task.prompts, task)

        for raw_key in rendered:
            assert raw_key in _PROMPT_FIELD_MAP, f"Key {raw_key!r} not in field map"
