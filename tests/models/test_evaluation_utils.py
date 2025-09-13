"""
Unit tests for step evaluation utilities.

Tests the shared utilities for parsing and processing step evaluation results
from judge responses, following SABER's fail-fast principles.
"""

import pytest
from typing import List

from saber.models.evaluation_utils import (
    parse_step_evaluations,
    calculate_step_evaluation_score,
    build_step_evaluation_explanation,
)
from saber.models.rest.evaluation import StepEvaluation


class TestParseStepEvaluations:
    """Test step evaluation parsing from judge responses."""

    def test_parse_normal_step_evaluations(self):
        """Test parsing normal step evaluations with mixed subtasks and task completion."""
        judge_response = """
        The agent performed a thorough investigation:

        STEP_EVALUATIONS:
        [5: checkpoint_1] - Query revealed account SID involved in C2 activity
        [12: checkpoint_2] - Identified Manatee Tempest group association with IP
        [18: incident_5_task_1] - Successfully identified final answer: 198.43.121.209

        The investigation was successful.
        """

        step_evals = parse_step_evaluations(judge_response, "incident_5_task_1")

        assert len(step_evals) == 3

        # Check first subtask evaluation
        assert step_evals[0].step_number == 5
        assert step_evals[0].objective_id == "checkpoint_1"
        assert step_evals[0].objective_type == "subtask"
        assert step_evals[0].completed is True

        # Check second subtask evaluation
        assert step_evals[1].step_number == 12
        assert step_evals[1].objective_id == "checkpoint_2"
        assert step_evals[1].objective_type == "subtask"
        assert step_evals[1].completed is True

        # Check task completion
        assert step_evals[2].step_number == 18
        assert step_evals[2].objective_id == "incident_5_task_1"
        assert step_evals[2].objective_type == "task"
        assert step_evals[2].completed is True

    def test_parse_subtasks_only(self):
        """Test parsing when only subtasks are completed, no main task."""
        judge_response = """
        STEP_EVALUATIONS:
        [3: checkpoint_1] - Found initial evidence in database
        [7: checkpoint_2] - Discovered threat actor information
        [10: checkpoint_3] - Identified malware signature
        """

        step_evals = parse_step_evaluations(judge_response, "main_task_id")

        assert len(step_evals) == 3

        # All should be subtasks
        for step_eval in step_evals:
            assert step_eval.objective_type == "subtask"
            assert step_eval.completed is True

        # Check specific values
        assert step_evals[0].step_number == 3
        assert step_evals[0].objective_id == "checkpoint_1"
        assert step_evals[1].step_number == 7
        assert step_evals[1].objective_id == "checkpoint_2"
        assert step_evals[2].step_number == 10
        assert step_evals[2].objective_id == "checkpoint_3"

    def test_parse_task_only(self):
        """Test parsing when only main task is completed."""
        judge_response = """
        Agent jumped directly to the solution:

        STEP_EVALUATIONS:
        [15: final_task] - Successfully identified the target IP address

        Task completed efficiently.
        """

        step_evals = parse_step_evaluations(judge_response, "final_task")

        assert len(step_evals) == 1
        assert step_evals[0].step_number == 15
        assert step_evals[0].objective_id == "final_task"
        assert step_evals[0].objective_type == "task"
        assert step_evals[0].completed is True

    def test_parse_no_completions_explicit(self):
        """Test parsing when judge explicitly indicates no completions."""
        judge_response = """
        The agent attempted several queries but failed to complete any objectives.

        STEP_EVALUATIONS:
        [NO_COMPLETIONS]

        No progress was made toward the investigation goals.
        """

        step_evals = parse_step_evaluations(judge_response, "any_task_id")

        assert len(step_evals) == 0

    def test_parse_no_completions_empty_section(self):
        """Test parsing when STEP_EVALUATIONS section exists but is empty."""
        judge_response = """
        STEP_EVALUATIONS:

        No objectives were completed during this episode.
        """

        step_evals = parse_step_evaluations(judge_response, "task_id")

        assert len(step_evals) == 0

    def test_parse_with_whitespace_and_formatting(self):
        """Test parsing with various whitespace and formatting variations."""
        judge_response = """
        STEP_EVALUATIONS:
        [ 2 : checkpoint_alpha ] - Step with extra spaces
        [99:checkpoint_beta] - Step without spaces
        [ 1001 : final_task_id ] - Large step number
        """

        step_evals = parse_step_evaluations(judge_response, "final_task_id")

        assert len(step_evals) == 3

        # Check that whitespace is handled correctly
        assert step_evals[0].step_number == 2
        assert step_evals[0].objective_id == "checkpoint_alpha"
        assert step_evals[0].objective_type == "subtask"

        assert step_evals[1].step_number == 99
        assert step_evals[1].objective_id == "checkpoint_beta"
        assert step_evals[1].objective_type == "subtask"

        assert step_evals[2].step_number == 1001
        assert step_evals[2].objective_id == "final_task_id"
        assert step_evals[2].objective_type == "task"

    def test_parse_missing_section_fails(self):
        """Test that missing STEP_EVALUATIONS section raises RuntimeError."""
        judge_response = """
        This is a judge response without the required section.
        The agent did some work but no step evaluations are provided.
        """

        with pytest.raises(RuntimeError) as exc_info:
            parse_step_evaluations(judge_response, "task_id")

        error_msg = str(exc_info.value)
        assert "STEP_EVALUATIONS section not found" in error_msg
        assert "Expected format" in error_msg

    def test_parse_invalid_step_number_fails(self):
        """Test that malformed step evaluation entries are handled properly."""
        judge_response = """
        STEP_EVALUATIONS:
        [1: checkpoint_1] - Valid step
        This is malformed text without proper brackets
        """

        # This should parse the valid entry and ignore malformed ones
        step_evals = parse_step_evaluations(judge_response, "task_id")

        # Should only get the valid entry
        assert len(step_evals) == 1
        assert step_evals[0].step_number == 1
        assert step_evals[0].objective_id == "checkpoint_1"

    def test_parse_multiline_section(self):
        """Test parsing with section followed by multiple paragraphs."""
        judge_response = """
        Analysis results:

        STEP_EVALUATIONS:
        [1: initial_scan] - Database connection established
        [5: data_query] - Suspicious activity identified
        [8: correlation] - Threat actor linked to known group

        Additional analysis shows that the agent followed proper procedure.

        The investigation was thorough and methodical.
        """

        step_evals = parse_step_evaluations(judge_response, "correlation")

        assert len(step_evals) == 3
        assert step_evals[2].objective_type == "task"  # correlation matches task_id

    def test_parse_case_insensitive_section(self):
        """Test that section header is case insensitive."""
        judge_response = """
        step_evaluations:
        [1: test_objective] - Case insensitive section header
        """

        step_evals = parse_step_evaluations(judge_response, "other_task")

        assert len(step_evals) == 1
        assert step_evals[0].objective_id == "test_objective"


class TestCalculateStepEvaluationScore:
    """Test score calculation from step evaluations."""

    def test_calculate_score_task_completed(self):
        """Test score calculation when main task is completed."""
        step_evals = [
            StepEvaluation(step_number=5, objective_id="checkpoint_1", objective_type="subtask"),
            StepEvaluation(step_number=10, objective_id="checkpoint_2", objective_type="subtask"),
            StepEvaluation(step_number=15, objective_id="main_task", objective_type="task"),
        ]

        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "main_task", max_score=1.0
        )

        assert score == 1.0
        assert is_correct is True
        assert task_step == 15
        assert subtasks == ["checkpoint_1", "checkpoint_2"]

    def test_calculate_score_task_not_completed(self):
        """Test score calculation when main task is not completed."""
        step_evals = [
            StepEvaluation(step_number=3, objective_id="checkpoint_1", objective_type="subtask"),
            StepEvaluation(step_number=7, objective_id="checkpoint_2", objective_type="subtask"),
        ]

        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "main_task", max_score=2.5
        )

        assert score == 0.0
        assert is_correct is False
        assert task_step is None
        assert subtasks == ["checkpoint_1", "checkpoint_2"]

    def test_calculate_score_no_evaluations(self):
        """Test score calculation with no step evaluations."""
        step_evals: List[StepEvaluation] = []

        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "any_task", max_score=1.0
        )

        assert score == 0.0
        assert is_correct is False
        assert task_step is None
        assert subtasks == []

    def test_calculate_score_task_only(self):
        """Test score calculation when only main task is completed."""
        step_evals = [
            StepEvaluation(step_number=20, objective_id="target_task", objective_type="task"),
        ]

        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "target_task", max_score=5.0
        )

        assert score == 5.0
        assert is_correct is True
        assert task_step == 20
        assert subtasks == []

    def test_calculate_score_custom_max_score(self):
        """Test score calculation with custom maximum score."""
        step_evals = [
            StepEvaluation(step_number=8, objective_id="test_task", objective_type="task"),
        ]

        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "test_task", max_score=10.0
        )

        assert score == 10.0
        assert is_correct is True
        assert task_step == 8
        assert subtasks == []

    def test_calculate_score_multiple_task_completions(self):
        """Test score calculation when task appears multiple times (should use first occurrence)."""
        step_evals = [
            StepEvaluation(step_number=5, objective_id="task_1", objective_type="task"),
            StepEvaluation(step_number=10, objective_id="checkpoint_1", objective_type="subtask"),
            StepEvaluation(step_number=15, objective_id="task_1", objective_type="task"),  # Duplicate
        ]

        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "task_1", max_score=1.0
        )

        assert score == 1.0
        assert is_correct is True
        assert task_step == 5  # First occurrence
        assert subtasks == ["checkpoint_1"]


class TestBuildStepEvaluationExplanation:
    """Test building human-readable explanations."""

    def test_build_explanation_task_completed_with_subtasks(self):
        """Test explanation when task is completed with subtasks."""
        explanation = build_step_evaluation_explanation(
            is_correct=True,
            task_completed_at_step=15,
            subtasks_completed=["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        )

        expected = "Step evaluation: Task completed at step 15, completed subtasks: checkpoint_1, checkpoint_2, checkpoint_3"
        assert explanation == expected

    def test_build_explanation_task_completed_no_subtasks(self):
        """Test explanation when task is completed without subtasks."""
        explanation = build_step_evaluation_explanation(
            is_correct=True,
            task_completed_at_step=8,
            subtasks_completed=[]
        )

        expected = "Step evaluation: Task completed at step 8"
        assert explanation == expected

    def test_build_explanation_task_not_completed_with_subtasks(self):
        """Test explanation when task is not completed but subtasks are."""
        explanation = build_step_evaluation_explanation(
            is_correct=False,
            task_completed_at_step=None,
            subtasks_completed=["partial_1", "partial_2"]
        )

        expected = "Step evaluation: Main task not completed, but completed subtasks: partial_1, partial_2"
        assert explanation == expected

    def test_build_explanation_nothing_completed(self):
        """Test explanation when nothing is completed."""
        explanation = build_step_evaluation_explanation(
            is_correct=False,
            task_completed_at_step=None,
            subtasks_completed=[]
        )

        expected = "Step evaluation: Main task not completed"
        assert explanation == expected

    def test_build_explanation_single_subtask(self):
        """Test explanation with single subtask completed."""
        explanation = build_step_evaluation_explanation(
            is_correct=False,
            task_completed_at_step=None,
            subtasks_completed=["only_one"]
        )

        expected = "Step evaluation: Main task not completed, but completed subtasks: only_one"
        assert explanation == expected


class TestIntegrationScenarios:
    """Test complete workflows combining all utilities."""

    def test_complete_successful_evaluation_workflow(self):
        """Test complete workflow for successful evaluation."""
        judge_response = """
        Comprehensive security analysis:

        STEP_EVALUATIONS:
        [2: initial_recon] - Established database connection and basic queries
        [7: threat_identification] - Identified suspicious network activity
        [12: actor_attribution] - Linked activity to known threat group
        [18: security_incident_final] - Successfully determined IP address: 198.43.121.209

        The investigation was thorough and successful.
        """

        # Parse evaluations
        step_evals = parse_step_evaluations(judge_response, "security_incident_final")
        assert len(step_evals) == 4

        # Calculate score
        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "security_incident_final", max_score=1.0
        )
        assert score == 1.0
        assert is_correct is True
        assert task_step == 18
        assert len(subtasks) == 3

        # Build explanation
        explanation = build_step_evaluation_explanation(is_correct, task_step, subtasks)
        expected = "Step evaluation: Task completed at step 18, completed subtasks: initial_recon, threat_identification, actor_attribution"
        assert explanation == expected

    def test_complete_partial_evaluation_workflow(self):
        """Test complete workflow for partial evaluation (subtasks only)."""
        judge_response = """
        STEP_EVALUATIONS:
        [4: database_access] - Successfully connected to security database
        [9: initial_queries] - Ran basic reconnaissance queries
        """

        # Parse evaluations
        step_evals = parse_step_evaluations(judge_response, "main_investigation")
        assert len(step_evals) == 2

        # Calculate score
        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "main_investigation", max_score=2.0
        )
        assert score == 0.0
        assert is_correct is False
        assert task_step is None
        assert subtasks == ["database_access", "initial_queries"]

        # Build explanation
        explanation = build_step_evaluation_explanation(is_correct, task_step, subtasks)
        expected = "Step evaluation: Main task not completed, but completed subtasks: database_access, initial_queries"
        assert explanation == expected

    def test_complete_failed_evaluation_workflow(self):
        """Test complete workflow for failed evaluation."""
        judge_response = """
        STEP_EVALUATIONS:
        [NO_COMPLETIONS]
        """

        # Parse evaluations
        step_evals = parse_step_evaluations(judge_response, "failed_task")
        assert len(step_evals) == 0

        # Calculate score
        score, is_correct, task_step, subtasks = calculate_step_evaluation_score(
            step_evals, "failed_task", max_score=1.0
        )
        assert score == 0.0
        assert is_correct is False
        assert task_step is None
        assert subtasks == []

        # Build explanation
        explanation = build_step_evaluation_explanation(is_correct, task_step, subtasks)
        expected = "Step evaluation: Main task not completed"
        assert explanation == expected
