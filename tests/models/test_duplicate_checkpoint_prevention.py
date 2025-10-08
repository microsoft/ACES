"""Tests for duplicate checkpoint deduplication in scoring."""

import pytest
from saber.models.evaluation_utils import calculate_step_evaluation_score
from saber.models.rest.evaluation import StepEvaluation


def test_duplicate_subtask_scoring_prevention():
    """Test that duplicate subtask IDs don't cause multiple score additions."""
    # Create step evaluations with duplicate subtask completions
    step_evaluations = [
        StepEvaluation(
            step_number=5,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning="First completion of checkpoint_1"
        ),
        StepEvaluation(
            step_number=10,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning="Duplicate completion of checkpoint_1"
        ),
        StepEvaluation(
            step_number=15,
            objective_type="subtask",
            objective_id="checkpoint_2",
            completed=True,
            reasoning="Completion of checkpoint_2"
        ),
        StepEvaluation(
            step_number=18,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning="Another duplicate completion of checkpoint_1"
        ),
    ]

    # Define subtask scores
    subtasks_with_scores = {
        "checkpoint_1": 0.5,
        "checkpoint_2": 0.3,
    }

    # Calculate score
    score_value, is_correct, task_completed_at_step, subtasks_completed = calculate_step_evaluation_score(
        step_evaluations=step_evaluations,
        task_id="test_task",
        max_score=1.0,
        subtasks_with_scores=subtasks_with_scores
    )

    # Task not completed (no task-level evaluation)
    assert is_correct is False
    assert task_completed_at_step is None

    # Score should be 0.5 + 0.3 = 0.8 (NOT 0.5 + 0.5 + 0.5 + 0.3 = 1.8)
    assert score_value == 0.8, f"Expected 0.8, got {score_value} (duplicate subtask likely counted multiple times)"

    # Subtasks completed should contain unique IDs only
    assert set(subtasks_completed) == {"checkpoint_1", "checkpoint_2"}
    assert len(subtasks_completed) == 2  # Should be 2, not 4


def test_duplicate_subtask_with_task_completion():
    """Test deduplication when task is also completed."""
    step_evaluations = [
        StepEvaluation(
            step_number=5,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning="First completion of checkpoint_1"
        ),
        StepEvaluation(
            step_number=10,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning="Duplicate completion of checkpoint_1"
        ),
        StepEvaluation(
            step_number=20,
            objective_type="task",
            objective_id="test_task",
            completed=True,
            reasoning="Task completed"
        ),
    ]

    subtasks_with_scores = {
        "checkpoint_1": 0.5,
    }

    score_value, is_correct, task_completed_at_step, subtasks_completed = calculate_step_evaluation_score(
        step_evaluations=step_evaluations,
        task_id="test_task",
        max_score=1.0,
        subtasks_with_scores=subtasks_with_scores
    )

    # Task completed
    assert is_correct is True
    assert task_completed_at_step == 20

    # Score should be 1.0 (task) + 0.5 (checkpoint_1 once) = 1.5
    # NOT 1.0 + 0.5 + 0.5 = 2.0
    assert score_value == 1.5, f"Expected 1.5, got {score_value}"

    # Subtasks completed should be deduplicated
    assert subtasks_completed == ["checkpoint_1"]
    assert len(subtasks_completed) == 1


def test_multiple_different_subtasks_no_duplicates():
    """Test that different subtasks are all counted (baseline test)."""
    step_evaluations = [
        StepEvaluation(
            step_number=5,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning="checkpoint_1"
        ),
        StepEvaluation(
            step_number=10,
            objective_type="subtask",
            objective_id="checkpoint_2",
            completed=True,
            reasoning="checkpoint_2"
        ),
        StepEvaluation(
            step_number=15,
            objective_type="subtask",
            objective_id="checkpoint_3",
            completed=True,
            reasoning="checkpoint_3"
        ),
    ]

    subtasks_with_scores = {
        "checkpoint_1": 0.2,
        "checkpoint_2": 0.3,
        "checkpoint_3": 0.5,
    }

    score_value, is_correct, task_completed_at_step, subtasks_completed = calculate_step_evaluation_score(
        step_evaluations=step_evaluations,
        task_id="test_task",
        max_score=1.0,
        subtasks_with_scores=subtasks_with_scores
    )

    # Score should be 0.2 + 0.3 + 0.5 = 1.0
    assert score_value == 1.0

    # All subtasks should be present
    assert set(subtasks_completed) == {"checkpoint_1", "checkpoint_2", "checkpoint_3"}
    assert len(subtasks_completed) == 3


def test_extreme_duplicate_case():
    """Test extreme case with many duplicates of same checkpoint."""
    step_evaluations = [
        StepEvaluation(
            step_number=i,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning=f"Attempt {i}"
        )
        for i in range(1, 21)  # 20 duplicates of checkpoint_1
    ]

    subtasks_with_scores = {
        "checkpoint_1": 0.5,
    }

    score_value, is_correct, task_completed_at_step, subtasks_completed = calculate_step_evaluation_score(
        step_evaluations=step_evaluations,
        task_id="test_task",
        max_score=1.0,
        subtasks_with_scores=subtasks_with_scores
    )

    # Score should be 0.5 (once), NOT 0.5 * 20 = 10.0
    assert score_value == 0.5, f"Expected 0.5, got {score_value} (likely added 20 times)"

    # Should have only one unique subtask
    assert subtasks_completed == ["checkpoint_1"]
    assert len(subtasks_completed) == 1


def test_chunked_evaluation_duplicate_prevention():
    """
    Test that chunked evaluation doesn't create duplicates when same
    checkpoint appears in multiple chunks.
    """
    # Simulate evaluations from chunk 1 (steps 1-25)
    chunk1_evaluations = [
        StepEvaluation(
            step_number=10,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning="Checkpoint 1 in chunk 1"
        ),
    ]

    # Simulate evaluations from chunk 2 (steps 26-50)
    # Judge might re-identify checkpoint_1 when reviewing later steps
    chunk2_evaluations = [
        StepEvaluation(
            step_number=30,
            objective_type="subtask",
            objective_id="checkpoint_1",
            completed=True,
            reasoning="Checkpoint 1 mentioned again in chunk 2"
        ),
        StepEvaluation(
            step_number=40,
            objective_type="subtask",
            objective_id="checkpoint_2",
            completed=True,
            reasoning="Checkpoint 2 in chunk 2"
        ),
    ]

    # Aggregate all evaluations (as chunked evaluation does)
    all_evaluations = chunk1_evaluations + chunk2_evaluations

    subtasks_with_scores = {
        "checkpoint_1": 0.5,
        "checkpoint_2": 0.3,
    }

    score_value, is_correct, task_completed_at_step, subtasks_completed = calculate_step_evaluation_score(
        step_evaluations=all_evaluations,
        task_id="test_task",
        max_score=1.0,
        subtasks_with_scores=subtasks_with_scores
    )

    # Score should be 0.5 + 0.3 = 0.8 (checkpoint_1 counted once despite appearing in 2 chunks)
    assert score_value == 0.8, f"Expected 0.8, got {score_value}"

    # Should have 2 unique subtasks
    assert set(subtasks_completed) == {"checkpoint_1", "checkpoint_2"}
    assert len(subtasks_completed) == 2


def test_step_zero_is_valid():
    """Test that step 0 is valid since steps are 0-indexed."""
    from saber.models.evaluation_utils import parse_step_evaluations

    judge_response = """
    STEP_EVALUATIONS:
    [0: checkpoint_1] - First step identifies checkpoint 1
    [0: checkpoint_2] - First step also identifies checkpoint 2
    [5: checkpoint_3] - Later step identifies checkpoint 3
    """

    step_evaluations = parse_step_evaluations(judge_response, "test_task")

    # Should have 3 evaluations including step 0
    assert len(step_evaluations) == 3

    # Check step 0 evaluations are included
    step_0_evals = [e for e in step_evaluations if e.step_number == 0]
    assert len(step_0_evals) == 2
    assert {e.objective_id for e in step_0_evals} == {"checkpoint_1", "checkpoint_2"}

    # Check step 5 evaluation
    step_5_evals = [e for e in step_evaluations if e.step_number == 5]
    assert len(step_5_evals) == 1
    assert step_5_evals[0].objective_id == "checkpoint_3"
