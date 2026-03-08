"""Validate excytin domain configuration loads correctly with saber pipeline.

These tests verify that the real excytin YAML files, prompts, and task
configs work end-to-end through ConfigLoader → PromptRenderer → converter
→ ScorerFactory. No Docker or network access required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from saber.config.converter import tasks_to_samples
from saber.config.loader import ConfigLoader
from saber.config.models import TaskConfig
from saber.prompts.renderer import PromptRenderer
from saber.scoring.factory import ScorerFactory
from saber.task import _find_config_root, _find_prompts_dir

_EXCYTIN_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "domains" / "excytin"
_HAS_EXCYTIN = (_EXCYTIN_ROOT / "tasks" / "global.yaml").exists()

pytestmark = pytest.mark.skipif(not _HAS_EXCYTIN, reason="excytin domain not available")


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def config_root() -> Path:
    """Resolved config root for excytin domain."""
    return _find_config_root(_EXCYTIN_ROOT)


@pytest.fixture(scope="module")
def loader(config_root: Path) -> ConfigLoader:
    """ConfigLoader pointed at excytin's config directory."""
    return ConfigLoader(config_root)


@pytest.fixture(scope="module")
def all_tasks(loader: ConfigLoader) -> list[TaskConfig]:
    """All excytin tasks loaded via ConfigLoader."""
    return loader.load_tasks()


@pytest.fixture(scope="module")
def incident_5_task(loader: ConfigLoader) -> TaskConfig:
    """The specific incident_5_latest_test_set_task_1 task."""
    tasks = loader.load_tasks(task_filter="incident_5_latest_test_set_task_1")
    assert len(tasks) == 1
    return tasks[0]


@pytest.fixture(scope="module")
def prompts_dir() -> Path:
    """Resolved prompts directory for excytin domain."""
    return _find_prompts_dir(_EXCYTIN_ROOT)


@pytest.fixture(scope="module")
def renderer(prompts_dir: Path) -> PromptRenderer:
    """PromptRenderer pointed at excytin's prompts directory."""
    return PromptRenderer(prompts_dir)


@pytest.fixture(scope="module")
def incident_5_samples(loader: ConfigLoader, renderer: PromptRenderer) -> list:
    """Samples generated from incident_5_latest_test_set_task_1."""
    tasks = loader.load_tasks(task_filter="incident_5_latest_test_set_task_1")
    return tasks_to_samples(tasks, _EXCYTIN_ROOT, renderer)


# ── Test 1: ConfigLoader loads all tasks ────────────────────────────


class TestConfigLoaderLoadsExcytinTasks:
    """Verify ConfigLoader discovers and validates all excytin tasks."""

    def test_loads_nonzero_tasks(self, all_tasks: list[TaskConfig]) -> None:
        assert len(all_tasks) > 0

    def test_all_tasks_have_valid_ids(self, all_tasks: list[TaskConfig]) -> None:
        for task_cfg in all_tasks:
            assert task_cfg.task_id, f"Task has empty task_id: {task_cfg}"
            assert isinstance(task_cfg.task_id, str)

    def test_task_count_above_minimum(self, all_tasks: list[TaskConfig]) -> None:
        """Excytin should have at least 2000 tasks across all incident sets."""
        assert len(all_tasks) >= 2000


# ── Test 2: Specific task loads with correct fields ─────────────────


class TestIncident5TaskFields:
    """Verify incident_5_latest_test_set_task_1 loads with correct config."""

    def test_task_id(self, incident_5_task: TaskConfig) -> None:
        assert incident_5_task.task_id == "incident_5_latest_test_set_task_1"

    def test_has_five_scorers(self, incident_5_task: TaskConfig) -> None:
        """1 submission + 4 checkpoints = 5 scorers."""
        assert len(incident_5_task.scorers) == 5

    def test_scorer_names(self, incident_5_task: TaskConfig) -> None:
        names = [s.scorer_name for s in incident_5_task.scorers]
        assert "submission" in names
        assert "checkpoint_1" in names
        assert "checkpoint_2" in names
        assert "checkpoint_3" in names
        assert "checkpoint_4" in names

    def test_submission_scorer_uses_llm_judge(self, incident_5_task: TaskConfig) -> None:
        submission = [s for s in incident_5_task.scorers if s.scorer_name == "submission"][0]
        assert submission.strategy == "llm_judge"
        assert submission.target.value == "submission"

    def test_max_steps_from_global(self, incident_5_task: TaskConfig) -> None:
        assert incident_5_task.max_steps == 25

    def test_prompt_paths_from_global(self, incident_5_task: TaskConfig) -> None:
        assert incident_5_task.prompts.instruction == "instructions/excytin_demo.md"
        assert incident_5_task.prompts.assistant == "assistants/inspect_assistant.md"

    def test_initial_context_has_database_connection(self, incident_5_task: TaskConfig) -> None:
        assert "database_connection" in incident_5_task.initial_context
        db_conn = incident_5_task.initial_context["database_connection"]
        assert isinstance(db_conn, dict)
        assert db_conn["hostname"] == "incident-5-db"  # type: ignore[index]

    def test_tools_have_timeout_configs(self, incident_5_task: TaskConfig) -> None:
        """Verify global.yaml tools round-trip through ConfigLoader."""
        assert "bash" in incident_5_task.tools
        bash_cfg = incident_5_task.tools["bash"]
        assert bash_cfg.timeout == 180
        assert bash_cfg.security is None

        assert "python" in incident_5_task.tools
        python_cfg = incident_5_task.tools["python"]
        assert python_cfg.timeout == 180
        assert python_cfg.security is None


# ── Test 3: PromptRenderer renders excytin prompts ──────────────────


class TestPromptRendererExcytin:
    """Verify PromptRenderer successfully renders excytin templates."""

    def test_renders_instruction_prompt(self, incident_5_task: TaskConfig, renderer: PromptRenderer) -> None:
        context = renderer.build_task_context(incident_5_task)
        rendered = renderer.render(incident_5_task.prompts.instruction, context)
        assert isinstance(rendered, str)
        assert len(rendered) > 100  # Real prompts are substantial

    def test_instruction_contains_expected_content(self, incident_5_task: TaskConfig, renderer: PromptRenderer) -> None:
        context = renderer.build_task_context(incident_5_task)
        rendered = renderer.render(incident_5_task.prompts.instruction, context)
        # The excytin instruction prompt should mention the agent role
        lower = rendered.lower()
        assert "security" in lower or "agent" in lower or "autonomous" in lower

    def test_renders_all_prompts(self, incident_5_task: TaskConfig, renderer: PromptRenderer) -> None:
        result = renderer.render_all_prompts(incident_5_task.prompts, incident_5_task)
        assert len(result["instruction"]) > 0
        assert len(result["assistant"]) > 0


# ── Test 4: tasks_to_samples produces valid samples ─────────────────


class TestTasksToSamples:
    """Verify tasks_to_samples produces well-formed inspect_ai Samples."""

    def test_produces_correct_number_of_samples(self, incident_5_samples: list) -> None:
        assert len(incident_5_samples) == 1

    def test_sample_id_matches_task_id(self, incident_5_samples: list) -> None:
        assert incident_5_samples[0].id == "incident_5_latest_test_set_task_1"

    def test_sample_input_has_system_and_user_messages(self, incident_5_samples: list) -> None:
        sample = incident_5_samples[0]
        assert isinstance(sample.input, list)
        assert len(sample.input) == 2
        assert isinstance(sample.input[0], ChatMessageSystem)
        assert isinstance(sample.input[1], ChatMessageUser)

    def test_sample_metadata_has_required_keys(self, incident_5_samples: list) -> None:
        metadata = incident_5_samples[0].metadata
        assert metadata is not None
        required_keys = {"task_id", "title", "description", "max_steps", "aggregation", "scorers"}
        assert required_keys.issubset(set(metadata.keys()))

    def test_sample_metadata_aggregation_value(self, incident_5_samples: list) -> None:
        metadata = incident_5_samples[0].metadata
        assert metadata is not None
        assert metadata["aggregation"] == "max"

    def test_sample_target_is_golden_answer(self, incident_5_samples: list) -> None:
        # The submission scorer uses llm_judge strategy, so _extract_target
        # returns "" (no static expected answer for LLM-judged submissions).
        assert incident_5_samples[0].target == ""


# ── Test 5: ScorerFactory creates scorers ───────────────────────────


class TestScorerFactoryExcytin:
    """Verify ScorerFactory creates the right number and type of scorers."""

    def test_creates_correct_number_of_scorers(self, incident_5_task: TaskConfig) -> None:
        factory = ScorerFactory(domain_root=_EXCYTIN_ROOT)
        scorers = factory.create_scorers(incident_5_task)
        # 1 submission + 4 checkpoints + 1 aggregate = 6
        assert len(scorers) == 6

    def test_all_scorers_are_scorer_instances(self, incident_5_task: TaskConfig) -> None:
        from inspect_ai.scorer import Scorer

        factory = ScorerFactory(domain_root=_EXCYTIN_ROOT)
        scorers = factory.create_scorers(incident_5_task)
        for s in scorers:
            assert isinstance(s, Scorer)


# ── Test 6: _find_config_root resolves excytin legacy layout ────────


class TestFindConfigRoot:
    """Verify _find_config_root resolves the excytin flat tasks/ layout."""

    def test_resolves_to_tasks_dir(self) -> None:
        result = _find_config_root(_EXCYTIN_ROOT)
        expected = _EXCYTIN_ROOT
        assert result == expected

    def test_has_tasks_subdirectory(self) -> None:
        result = _find_config_root(_EXCYTIN_ROOT)
        assert (result / "tasks").is_dir()


# ── Test 7: _find_prompts_dir resolves excytin legacy layout ────────


class TestFindPromptsDir:
    """Verify _find_prompts_dir resolves the excytin flat prompts/ layout."""

    def test_resolves_to_prompts_dir(self) -> None:
        result = _find_prompts_dir(_EXCYTIN_ROOT)
        expected = _EXCYTIN_ROOT / "prompts"
        assert result == expected

    def test_has_instructions_subdirectory(self) -> None:
        result = _find_prompts_dir(_EXCYTIN_ROOT)
        assert (result / "instructions").is_dir()
