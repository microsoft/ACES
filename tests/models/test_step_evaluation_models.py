"""
Unit tests for step evaluation data models.

Tests the Pydantic models used for step evaluation, including
validation, serialization, and deserialization.
"""

import pytest
from typing import List
from pydantic import ValidationError

from saber.models.rest.evaluation import (
    StepEvaluation,
    StepEvaluationResult,
    TaskEvaluationContext,
)


class TestStepEvaluation:
    """Test StepEvaluation model."""

    def test_step_evaluation_creation(self):
        """Test creating StepEvaluation with valid data."""
        step_eval = StepEvaluation(
            step_number=5,
            objective_id="checkpoint_1",
            objective_type="subtask",
            completed=True
        )

        assert step_eval.step_number == 5
        assert step_eval.objective_id == "checkpoint_1"
        assert step_eval.objective_type == "subtask"
        assert step_eval.completed is True

    def test_step_evaluation_task_type(self):
        """Test creating StepEvaluation with task objective type."""
        step_eval = StepEvaluation(
            step_number=10,
            objective_id="main_task_id",
            objective_type="task",
            completed=True
        )

        assert step_eval.objective_type == "task"

    def test_step_evaluation_defaults(self):
        """Test StepEvaluation default values."""
        step_eval = StepEvaluation(
            step_number=1,
            objective_id="test_objective",
            objective_type="subtask"
        )

        # Default completed should be True
        assert step_eval.completed is True

    def test_step_evaluation_serialization(self):
        """Test StepEvaluation serialization to dict."""
        step_eval = StepEvaluation(
            step_number=7,
            objective_id="security_check",
            objective_type="subtask",
            completed=False
        )

        serialized = step_eval.model_dump()
        expected = {
            "step_number": 7,
            "objective_id": "security_check",
            "objective_type": "subtask",
            "completed": False
        }
        assert serialized == expected

    def test_step_evaluation_json_serialization(self):
        """Test StepEvaluation JSON serialization."""
        step_eval = StepEvaluation(
            step_number=3,
            objective_id="test_id",
            objective_type="task"
        )

        json_data = step_eval.model_dump_json()
        assert isinstance(json_data, str)
        assert '"step_number":3' in json_data
        assert '"objective_id":"test_id"' in json_data
        assert '"objective_type":"task"' in json_data
        assert '"completed":true' in json_data

    def test_step_evaluation_deserialization(self):
        """Test StepEvaluation deserialization from dict."""
        data = {
            "step_number": 15,
            "objective_id": "final_step",
            "objective_type": "task",
            "completed": True
        }

        step_eval = StepEvaluation(**data)
        assert step_eval.step_number == 15
        assert step_eval.objective_id == "final_step"
        assert step_eval.objective_type == "task"
        assert step_eval.completed is True

    def test_step_evaluation_invalid_step_number(self):
        """Test validation error for invalid step number."""
        with pytest.raises(ValidationError) as exc_info:
            StepEvaluation(
                step_number=-1,  # Invalid: negative
                objective_id="test",
                objective_type="subtask"
            )

        error = exc_info.value.errors()[0]
        assert error["type"] == "greater_than_equal"
        assert error["loc"] == ("step_number",)

    def test_step_evaluation_zero_step_number(self):
        """Test that zero step number is valid (0-indexed)."""
        # Should not raise - step_number is 0-indexed
        step_eval = StepEvaluation(
            step_number=0,  # Valid: 0-indexed
            objective_id="test",
            objective_type="subtask"
        )
        assert step_eval.step_number == 0
        assert step_eval.objective_id == "test"

    def test_step_evaluation_empty_objective_id(self):
        """Test that empty objective ID is allowed."""
        step_eval = StepEvaluation(
            step_number=1,
            objective_id="",  # Empty string is allowed
            objective_type="subtask"
        )

        assert step_eval.objective_id == ""
        assert step_eval.step_number == 1

    def test_step_evaluation_invalid_objective_type(self):
        """Test that any objective type string is allowed."""
        step_eval = StepEvaluation(
            step_number=1,
            objective_id="test",
            objective_type="invalid_type"  # Any string is allowed
        )

        assert step_eval.objective_type == "invalid_type"


class TestStepEvaluationResult:
    """Test StepEvaluationResult model."""

    def test_step_evaluation_result_creation(self):
        """Test creating StepEvaluationResult with step evaluations."""
        step_evals = [
            StepEvaluation(step_number=1, objective_id="step1", objective_type="subtask"),
            StepEvaluation(step_number=5, objective_id="step2", objective_type="subtask"),
            StepEvaluation(step_number=10, objective_id="main_task", objective_type="task"),
        ]

        result = StepEvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            submission="test submission",
            score=0.8,
            max_score=1.0,
            success=True,
            strategy="test_strategy",
            step_evaluations=step_evals,
            task_completed_at_step=10,
            subtasks_completed=["step1", "step2"]
        )

        assert len(result.step_evaluations) == 3
        assert result.success is True
        assert result.task_completed_at_step == 10
        assert result.subtasks_completed == ["step1", "step2"]
        assert result.episode_id == "test_episode"
        assert result.task_id == "test_task"

    def test_step_evaluation_result_empty_steps(self):
        """Test StepEvaluationResult with no step evaluations."""
        result = StepEvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            submission="test submission",
            score=0.0,
            max_score=1.0,
            success=False,
            strategy="test_strategy",
            step_evaluations=[],
            task_completed_at_step=None,
            subtasks_completed=[]
        )

        assert len(result.step_evaluations) == 0
        assert result.success is False
        assert result.task_completed_at_step is None
        assert result.subtasks_completed == []

    def test_step_evaluation_result_serialization(self):
        """Test StepEvaluationResult serialization."""
        step_eval = StepEvaluation(
            step_number=3,
            objective_id="test_step",
            objective_type="subtask"
        )

        result = StepEvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            submission="test submission",
            score=0.5,
            max_score=1.0,
            success=False,
            strategy="test_strategy",
            step_evaluations=[step_eval],
            task_completed_at_step=None,
            subtasks_completed=["test_step"]
        )

        serialized = result.model_dump()

        assert len(serialized["step_evaluations"]) == 1
        assert serialized["step_evaluations"][0]["step_number"] == 3
        assert serialized["step_evaluations"][0]["objective_id"] == "test_step"
        assert serialized["success"] is False
        assert serialized["task_completed_at_step"] is None
        assert serialized["subtasks_completed"] == ["test_step"]
        assert serialized["episode_id"] == "test_episode"
        assert serialized["task_id"] == "test_task"

    def test_step_evaluation_result_deserialization(self):
        """Test StepEvaluationResult deserialization."""
        data = {
            "episode_id": "test_episode",
            "task_id": "test_task",
            "submission": "test submission",
            "score": 0.8,
            "max_score": 1.0,
            "success": True,
            "strategy": "test_strategy",
            "step_evaluations": [
                {
                    "step_number": 2,
                    "objective_id": "checkpoint_alpha",
                    "objective_type": "subtask",
                    "completed": True
                },
                {
                    "step_number": 8,
                    "objective_id": "main_objective",
                    "objective_type": "task",
                    "completed": True
                }
            ],
            "task_completed_at_step": 8,
            "subtasks_completed": ["checkpoint_alpha"]
        }

        result = StepEvaluationResult(**data)

        assert len(result.step_evaluations) == 2
        assert result.step_evaluations[0].step_number == 2
        assert result.step_evaluations[0].objective_id == "checkpoint_alpha"
        assert result.step_evaluations[1].step_number == 8
        assert result.step_evaluations[1].objective_id == "main_objective"
        assert result.success is True
        assert result.task_completed_at_step == 8
        assert result.subtasks_completed == ["checkpoint_alpha"]

    def test_step_evaluation_result_invalid_step_number_combination(self):
        """Test that task_completed_at_step validation is handled properly."""
        # This should be valid - task completed with step number
        result = StepEvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            submission="test submission",
            score=1.0,
            max_score=1.0,
            success=True,
            strategy="test_strategy",
            step_evaluations=[],
            task_completed_at_step=5,
            subtasks_completed=[]
        )
        assert result.task_completed_at_step == 5

        # This should also be valid - task not completed, no step number
        result = StepEvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            submission="test submission",
            score=0.0,
            max_score=1.0,
            success=False,
            strategy="test_strategy",
            step_evaluations=[],
            task_completed_at_step=None,
            subtasks_completed=[]
        )
        assert result.task_completed_at_step is None


class TestTaskEvaluationContext:
    """Test TaskEvaluationContext model."""

    def test_task_evaluation_context_creation(self):
        """Test creating TaskEvaluationContext with subtasks."""
        subtasks_data = [
            {"id": "checkpoint_1", "description": "Initial reconnaissance"},
            {"id": "checkpoint_2", "description": "Threat identification"},
        ]

        context = TaskEvaluationContext(
            task_id="security_incident_1",
            title="Security Incident",
            description="Investigate security incident",
            domain="security",
            subtasks=subtasks_data
        )

        assert context.task_id == "security_incident_1"
        assert context.title == "Security Incident"
        assert context.description == "Investigate security incident"
        assert context.domain == "security"
        assert len(context.subtasks) == 2
        assert context.subtasks[0]["id"] == "checkpoint_1"
        assert context.subtasks[1]["description"] == "Threat identification"

    def test_task_evaluation_context_no_subtasks(self):
        """Test TaskEvaluationContext without subtasks."""
        context = TaskEvaluationContext(
            task_id="simple_task",
            title="Simple Task",
            description="Simple task without subtasks",
            domain="security",
            subtasks=[]
        )

        assert len(context.subtasks) == 0
        assert context.task_id == "simple_task"

    def test_task_evaluation_context_optional_subtasks(self):
        """Test TaskEvaluationContext with default subtasks."""
        context = TaskEvaluationContext(
            task_id="task_1",
            title="Test Task",
            description="Test task",
            domain="security"
        )

        # Default subtasks should be an empty list
        assert context.subtasks == []

    def test_task_evaluation_context_serialization(self):
        """Test TaskEvaluationContext serialization."""
        subtasks = [{"id": "sub1", "desc": "First subtask"}]

        context = TaskEvaluationContext(
            task_id="test_task",
            title="Test Task",
            description="Test description",
            domain="security",
            subtasks=subtasks
        )

        serialized = context.model_dump()

        assert serialized["task_id"] == "test_task"
        assert serialized["title"] == "Test Task"
        assert serialized["description"] == "Test description"
        assert serialized["domain"] == "security"
        assert serialized["subtasks"] == subtasks

    def test_task_evaluation_context_validation(self):
        """Test TaskEvaluationContext validation."""
        # Any string values are allowed - no validation constraints
        context = TaskEvaluationContext(
            task_id="task",
            title="title",
            description="desc",
            domain="security",
            subtasks=[]
        )
        assert context.task_id == "task"

    def test_task_evaluation_context_empty_strings(self):
        """Test that empty string fields are allowed."""
        context = TaskEvaluationContext(
            task_id="",  # Empty strings are allowed
            title="title",
            description="desc",
            domain="security",
            subtasks=[]
        )

        assert context.task_id == ""

        context = TaskEvaluationContext(
            task_id="task",
            title="title",
            description="",  # Empty strings are allowed
            domain="security",
            subtasks=[]
        )

        assert context.description == ""


class TestModelIntegration:
    """Test integration between different models."""

    def test_step_evaluation_in_result_model(self):
        """Test using StepEvaluation within StepEvaluationResult."""
        step_evals = [
            StepEvaluation(step_number=1, objective_id="init", objective_type="subtask"),
            StepEvaluation(step_number=3, objective_id="process", objective_type="subtask"),
            StepEvaluation(step_number=7, objective_id="final", objective_type="task"),
        ]

        result = StepEvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            submission="test submission",
            score=1.0,
            max_score=1.0,
            success=True,
            strategy="test_strategy",
            step_evaluations=step_evals,
            task_completed_at_step=7,
            subtasks_completed=["init", "process"]
        )

        # Verify the step evaluations are properly nested
        assert len(result.step_evaluations) == 3

        # Check first step
        first_step = result.step_evaluations[0]
        assert isinstance(first_step, StepEvaluation)
        assert first_step.step_number == 1
        assert first_step.objective_id == "init"
        assert first_step.objective_type == "subtask"

        # Check task completion step
        task_step = result.step_evaluations[2]
        assert task_step.step_number == 7
        assert task_step.objective_id == "final"
        assert task_step.objective_type == "task"

    def test_full_evaluation_workflow_models(self):
        """Test complete workflow using all models together."""
        # Create task context
        context = TaskEvaluationContext(
            task_id="incident_response",
            title="Incident Response",
            description="Respond to security incident",
            domain="security",
            subtasks=[
                {"id": "detect", "description": "Detect threat"},
                {"id": "analyze", "description": "Analyze impact"},
                {"id": "contain", "description": "Contain threat"}
            ]
        )

        # Create step evaluations
        step_evals = [
            StepEvaluation(step_number=2, objective_id="detect", objective_type="subtask"),
            StepEvaluation(step_number=5, objective_id="analyze", objective_type="subtask"),
            StepEvaluation(step_number=8, objective_id="incident_response", objective_type="task"),
        ]

        # Create evaluation result
        result = StepEvaluationResult(
            episode_id="test_episode",
            task_id="incident_response",
            submission="Security incident handled successfully",
            score=1.0,
            max_score=1.0,
            success=True,
            strategy="test_strategy",
            step_evaluations=step_evals,
            task_completed_at_step=8,
            subtasks_completed=["detect", "analyze"]
        )

        # Verify the complete workflow
        assert context.task_id == "incident_response"
        assert len(context.subtasks) == 3
        assert result.success is True
        assert len(result.step_evaluations) == 3

        # Verify step evaluation details
        detect_step = result.step_evaluations[0]
        assert detect_step.objective_id == "detect"
        assert detect_step.objective_type == "subtask"

        final_step = result.step_evaluations[2]
        assert final_step.objective_id == context.task_id
        assert final_step.objective_type == "task"

    def test_json_round_trip_serialization(self):
        """Test complete JSON serialization and deserialization."""
        # Create complex nested structure
        step_evals = [
            StepEvaluation(step_number=1, objective_id="step1", objective_type="subtask"),
            StepEvaluation(step_number=5, objective_id="final", objective_type="task", completed=True),
        ]

        result = StepEvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            submission="Complete success submission",
            score=1.0,
            max_score=1.0,
            success=True,
            strategy="test_strategy",
            step_evaluations=step_evals,
            task_completed_at_step=5,
            subtasks_completed=["step1"]
        )

        # Serialize to JSON
        json_data = result.model_dump_json()
        assert isinstance(json_data, str)

        # Deserialize from JSON
        import json
        parsed_data = json.loads(json_data)
        reconstructed = StepEvaluationResult(**parsed_data)

        # Verify reconstruction
        assert len(reconstructed.step_evaluations) == 2
        assert reconstructed.success is True
        assert reconstructed.task_completed_at_step == 5
        assert reconstructed.subtasks_completed == ["step1"]
        assert reconstructed.episode_id == "test_episode"

        # Verify nested StepEvaluation objects
        first_step = reconstructed.step_evaluations[0]
        assert first_step.step_number == 1
        assert first_step.objective_id == "step1"
        assert first_step.objective_type == "subtask"

        final_step = reconstructed.step_evaluations[1]
        assert final_step.step_number == 5
        assert final_step.objective_id == "final"
        assert final_step.objective_type == "task"
