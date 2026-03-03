"""Tests for migrate_scoring_yaml script."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

# Import the module under test — skip entire module if not yet implemented
try:
    from scripts.migrate_scoring_yaml import (
        migrate_file,
        migrate_global_yaml,
        migrate_task,
    )
except ModuleNotFoundError:
    pytest.skip(
        "scripts.migrate_scoring_yaml not yet implemented",
        allow_module_level=True,
    )

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(**overrides: object) -> dict[str, object]:
    """Build a minimal old-format task dict, applying *overrides*."""
    base: dict[str, object] = {
        "task_id": "test_task",
        "title": "Test Task",
        "description": "A test task.",
    }
    base.update(overrides)
    return base


# =========================================================================
# TestMigrateTask
# =========================================================================


class TestMigrateTask:
    """Unit tests for migrate_task()."""

    # ----- E1: excytin (all llm_judge) -----
    def test_excytin_llm_judge_only(self) -> None:
        """Pattern E1: submission + subtask checkpoints, all llm_judge."""
        task = _make_task(
            submission_evaluation_config={
                "strategy": "llm_judge",
                "criteria": {
                    "model": "openai/azure/gpt-4.1",
                    "judge_system_template": "judge/submission_judge_system.md",
                    "judge_user_template": "judge/submission_judge_user.md",
                    "golden_answer": "mimilove.exe",
                },
                "scoring": {"max_score": 1.0},
            },
            subtasks=[
                {
                    "subtask_id": "checkpoint_1",
                    "title": "Checkpoint 1",
                    "description": "Some checkpoint description.",
                    "objective": "Identify key details related to ...",
                    "step_evaluation_config": {
                        "strategy": "llm_judge",
                        "criteria": {
                            "model": "openai/azure/gpt-4.1",
                            "judge_system_template": "judge/checkpoint_judge_system.md",
                            "judge_user_template": "judge/checkpoint_judge_user.md",
                            "steps_per_message": 50,
                        },
                        "scoring": {"max_score": 0.25, "weight": 1.0},
                    },
                },
            ],
        )

        result = migrate_task(task)

        # Old keys removed
        assert "submission_evaluation_config" not in result
        assert "step_evaluation_config" not in result
        assert "subtasks" not in result

        # New scoring block
        scoring = result["scoring"]
        assert "llm_judge" in scoring

        # Submission scorer
        sub = scoring["llm_judge"]["submission"]
        assert sub["target"] == "submission"
        assert sub["model"] == "openai/azure/gpt-4.1"
        assert sub["golden_answer"] == "mimilove.exe"
        assert sub["max_score"] == 1.0

        # Checkpoint scorer
        cp = scoring["llm_judge"]["checkpoint_1"]
        assert cp["target"] == "trajectory"
        assert cp["steps_per_message"] == 50
        assert cp["max_score"] == 0.25
        assert cp["weight"] == 1.0
        assert cp["title"] == "Checkpoint 1"
        assert cp["description"] == "Some checkpoint description."
        # objective should be dropped by default
        assert "objective" not in cp

    # ----- D1: mixed strategies (already migrated) -----
    def test_mixed_strategies(self) -> None:
        """Pattern D1: tasks with mixed scoring strategies."""
        task = _make_task(
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["flag{test}"]},
                "scoring": {"max_score": 1.0},
            },
            step_evaluation_config={
                "strategy": "llm_judge",
                "criteria": {
                    "model": "openai/azure/gpt-4.1",
                    "judge_system_template": "judge/system.md",
                    "judge_user_template": "judge/user.md",
                    "steps_per_message": 10,
                },
            },
            subtasks=[
                {
                    "subtask_id": "checkpoint_1",
                    "title": "CP1",
                    "description": "desc1",
                    "scoring": {"max_score": 1.0, "weight": 0.5},
                },
            ],
        )

        result = migrate_task(task)

        scoring = result["scoring"]
        # Static submission
        assert scoring["static"]["submission"]["target"] == "submission"
        assert scoring["static"]["submission"]["expected_answers"] == ["flag{test}"]

        # Task-level step_eval is NOT a scorer – only provides inherited defaults
        assert "trajectory" not in scoring.get("llm_judge", {})

        # Subtask inherits task-level step_eval strategy
        assert scoring["llm_judge"]["checkpoint_1"]["target"] == "trajectory"
        assert scoring["llm_judge"]["checkpoint_1"]["weight"] == 0.5

    # ----- D2: metadata-only subtasks -----
    def test_metadata_only_subtasks_dropped(self) -> None:
        """Pattern D2: subtasks without step_evaluation_config are dropped."""
        task = _make_task(
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["42"]},
                "scoring": {"max_score": 1.0},
            },
            subtasks=[
                {
                    "subtask_id": "meta_only",
                    "title": "Metadata",
                    "description": "No eval config",
                },
            ],
        )

        result = migrate_task(task)

        scoring = result["scoring"]
        assert "meta_only" not in scoring.get("static", {})
        # No llm_judge group should exist either
        assert "llm_judge" not in scoring

    # ----- CB: cybench – inherited step_eval -----
    def test_inherited_step_eval(self) -> None:
        """Pattern CB: subtasks inherit task-level step_evaluation_config."""
        task = _make_task(
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["HTB{flag}"]},
                "scoring": {"max_score": 1.0},
            },
            step_evaluation_config={
                "strategy": "llm_judge",
                "criteria": {
                    "model": "openai/azure/gpt-4.1",
                    "judge_system_template": "judge/cybench_evaluation_system.j2",
                    "judge_user_template": "judge/cybench_evaluation_user.j2",
                    "steps_per_message": 10,
                },
            },
            subtasks=[
                {
                    "subtask_id": "checkpoint_1",
                    "title": "Contextualize the problem",
                    "description": "Identify the programming language.",
                    "objective": "What language? ANSWER: Java",
                    "hints": ["Check the codebase"],
                    "scoring": {"max_score": 1.0, "weight": 0.17},
                },
            ],
        )

        result = migrate_task(task, merge_objective=True)

        scoring = result["scoring"]
        # Task-level step_eval is NOT a scorer
        assert "trajectory" not in scoring["llm_judge"]
        cp = scoring["llm_judge"]["checkpoint_1"]
        assert cp["target"] == "trajectory"
        assert cp["model"] == "openai/azure/gpt-4.1"
        assert cp["max_score"] == 1.0
        assert cp["weight"] == 0.17
        assert cp["hints"] == ["Check the codebase"]
        # objective merged into description
        assert cp["description"] == "Identify the programming language. What language? ANSWER: Java"

    # ----- Objective handling -----
    def test_objective_merged_when_flag_set(self) -> None:
        """With merge_objective=True, objective appended to description."""
        task = _make_task(
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["x"]},
                "scoring": {"max_score": 1.0},
            },
            subtasks=[
                {
                    "subtask_id": "cp1",
                    "title": "T",
                    "description": "Base desc.",
                    "objective": "Extra info.",
                    "step_evaluation_config": {
                        "strategy": "llm_judge",
                        "criteria": {"model": "m"},
                        "scoring": {"max_score": 0.5, "weight": 1.0},
                    },
                },
            ],
        )

        result = migrate_task(task, merge_objective=True)

        cp = result["scoring"]["llm_judge"]["cp1"]
        assert cp["description"] == "Base desc. Extra info."

    def test_objective_dropped_by_default(self) -> None:
        """Without merge_objective, objective is dropped."""
        task = _make_task(
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["x"]},
                "scoring": {"max_score": 1.0},
            },
            subtasks=[
                {
                    "subtask_id": "cp1",
                    "title": "T",
                    "description": "Base desc.",
                    "objective": "Extra info.",
                    "step_evaluation_config": {
                        "strategy": "llm_judge",
                        "criteria": {"model": "m"},
                        "scoring": {"max_score": 0.5, "weight": 1.0},
                    },
                },
            ],
        )

        result = migrate_task(task)

        cp = result["scoring"]["llm_judge"]["cp1"]
        assert cp["description"] == "Base desc."
        assert "objective" not in cp

    # ----- CTI: custom domain strategies -----
    def test_custom_domain_strategies(self) -> None:
        """Pattern CTI: multiple custom strategies across submission + subtasks."""
        task = _make_task(
            submission_evaluation_config={
                "strategy": "trajectory_analysis",
                "criteria": {
                    "detection_objective": "Detect APT",
                    "expected_techniques": ["T1059"],
                    "regex_patterns": {"filename": "^bash$"},
                },
                "scoring": {"max_score": 10.0},
            },
            subtasks=[
                {
                    "subtask_id": "c0_cti_analysis",
                    "title": "C0: CTI Report Usage",
                    "description": "Evidence of threat intelligence analysis",
                    "objective": "Agent must demonstrate use of CTI reports",
                    "step_evaluation_config": {
                        "strategy": "cti_tool_llm",
                        "criteria": {
                            "model": "openai/azure/gpt-5.1",
                            "steps_per_message": 20,
                        },
                        "scoring": {"max_score": 1.25, "weight": 1.0},
                    },
                },
                {
                    "subtask_id": "c1_threat_context",
                    "title": "C1: Threat Context",
                    "description": "MITRE techniques mentioned",
                    "objective": "Agent must reference MITRE techniques",
                    "step_evaluation_config": {
                        "strategy": "trajectory_jaccard",
                        "criteria": {
                            "expected_items": ["T1059"],
                            "regex_pattern": r"T\d{4}",
                        },
                        "scoring": {"max_score": 0.75, "weight": 1.0},
                    },
                },
            ],
        )

        result = migrate_task(task)

        scoring = result["scoring"]
        # trajectory_analysis strategy
        sub = scoring["trajectory_analysis"]["submission"]
        assert sub["target"] == "submission"
        assert sub["detection_objective"] == "Detect APT"
        assert sub["max_score"] == 10.0

        # cti_tool_llm strategy
        c0 = scoring["cti_tool_llm"]["c0_cti_analysis"]
        assert c0["target"] == "trajectory"
        assert c0["model"] == "openai/azure/gpt-5.1"

        # trajectory_jaccard strategy
        c1 = scoring["trajectory_jaccard"]["c1_threat_context"]
        assert c1["target"] == "trajectory"
        assert c1["expected_items"] == ["T1059"]

    # ----- Target assignments -----
    def test_submission_gets_target_submission(self) -> None:
        task = _make_task(
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["a"]},
                "scoring": {"max_score": 1.0},
            },
        )
        result = migrate_task(task)
        assert result["scoring"]["static"]["submission"]["target"] == "submission"

    def test_checkpoints_get_target_trajectory(self) -> None:
        task = _make_task(
            submission_evaluation_config={
                "strategy": "llm_judge",
                "criteria": {"model": "m"},
                "scoring": {"max_score": 1.0},
            },
            subtasks=[
                {
                    "subtask_id": "cp1",
                    "title": "CP",
                    "description": "d",
                    "step_evaluation_config": {
                        "strategy": "llm_judge",
                        "criteria": {"model": "m"},
                        "scoring": {"max_score": 0.5, "weight": 1.0},
                    },
                },
            ],
        )
        result = migrate_task(task)
        assert result["scoring"]["llm_judge"]["cp1"]["target"] == "trajectory"

    # ----- Scoring aggregation -----
    def test_scoring_aggregation_added(self) -> None:
        """Scoring aggregation block is added when requested."""
        task = _make_task(
            submission_evaluation_config={
                "strategy": "llm_judge",
                "criteria": {"model": "m"},
                "scoring": {"max_score": 1.0},
            },
            subtasks=[
                {
                    "subtask_id": "checkpoint_1",
                    "title": "CP1",
                    "description": "d1",
                    "step_evaluation_config": {
                        "strategy": "llm_judge",
                        "criteria": {"model": "m"},
                        "scoring": {"max_score": 0.5, "weight": 1.0},
                    },
                },
                {
                    "subtask_id": "checkpoint_2",
                    "title": "CP2",
                    "description": "d2",
                    "step_evaluation_config": {
                        "strategy": "llm_judge",
                        "criteria": {"model": "m"},
                        "scoring": {"max_score": 0.5, "weight": 1.0},
                    },
                },
            ],
        )

        result = migrate_task(task, add_aggregation="max")

        agg = result["scoring_aggregation"]
        assert "max" in agg
        assert agg["max"]["scores"][0] == "submission"
        assert set(agg["max"]["scores"][1]) == {"checkpoint_1", "checkpoint_2"}

    # ----- Idempotency -----
    def test_already_migrated_task_unchanged(self) -> None:
        """A task with 'scoring:' block and no old keys is returned unchanged."""
        already_migrated = _make_task(
            scoring={
                "static": {
                    "submission": {
                        "target": "submission",
                        "expected_answers": ["42"],
                        "max_score": 1.0,
                    },
                },
            },
        )

        result = migrate_task(already_migrated)

        assert result == already_migrated


# =========================================================================
# TestMigrateFile
# =========================================================================


class TestMigrateFile:
    """Tests for migrate_file() which handles full YAML file I/O."""

    def test_multi_task_file(self, tmp_path: Path) -> None:
        """File with multiple tasks in 'tasks:' list."""
        content = textwrap.dedent("""\
            tasks:
              - task_id: t1
                title: Task 1
                description: First
                submission_evaluation_config:
                  strategy: static
                  criteria:
                    expected_answers: ["a"]
                  scoring:
                    max_score: 1.0
              - task_id: t2
                title: Task 2
                description: Second
                submission_evaluation_config:
                  strategy: llm_judge
                  criteria:
                    model: m
                    golden_answer: b
                  scoring:
                    max_score: 1.0
        """)
        p = tmp_path / "tasks.yaml"
        p.write_text(content)

        changed = migrate_file(p)

        assert changed is True
        data = yaml.safe_load(p.read_text())
        for t in data["tasks"]:
            assert "submission_evaluation_config" not in t
            assert "scoring" in t

    def test_dry_run_doesnt_write(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            tasks:
              - task_id: t1
                title: T
                description: D
                submission_evaluation_config:
                  strategy: static
                  criteria:
                    expected_answers: ["x"]
                  scoring:
                    max_score: 1.0
        """)
        p = tmp_path / "task.yaml"
        p.write_text(content)
        original = p.read_text()

        changed = migrate_file(p, dry_run=True)

        assert changed is True
        assert p.read_text() == original

    def test_already_migrated_file_skipped(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            tasks:
              - task_id: t1
                title: T
                description: D
                scoring:
                  static:
                    submission:
                      target: submission
                      expected_answers: ["x"]
                      max_score: 1.0
        """)
        p = tmp_path / "task.yaml"
        p.write_text(content)

        changed = migrate_file(p)

        assert changed is False


# =========================================================================
# TestMigrateGlobalYaml
# =========================================================================


class TestMigrateGlobalYaml:
    """Tests for migrate_global_yaml()."""

    def test_unwrap_scoring_config(self, tmp_path: Path) -> None:
        """scoring_config.aggregation → aggregation at global_defaults level."""
        content = textwrap.dedent("""\
            domain: cybench
            global_defaults:
              max_steps: 30
              scoring_config:
                aggregation: max
        """)
        p = tmp_path / "global.yaml"
        p.write_text(content)

        changed = migrate_global_yaml(p)

        assert changed is True
        data = yaml.safe_load(p.read_text())
        gd = data["global_defaults"]
        assert "scoring_config" not in gd
        assert gd["aggregation"] == "max"

    def test_no_scoring_config_unchanged(self, tmp_path: Path) -> None:
        """File without scoring_config is left unchanged."""
        content = textwrap.dedent("""\
            domain: test
            global_defaults:
              max_steps: 10
              aggregation: max
        """)
        p = tmp_path / "global.yaml"
        p.write_text(content)

        changed = migrate_global_yaml(p)

        assert changed is False
