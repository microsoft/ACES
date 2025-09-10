"""
Unit tests for evaluation models.
"""

import pytest
from datetime import datetime, timezone

from saber.server.evaluation.models import EvaluationConfig, EvaluationResult


class TestEvaluationConfig:
    """Test cases for EvaluationConfig model."""

    def test_from_dict_static_config(self):
        """Test creating EvaluationConfig from dictionary for static strategy."""
        data = {
            "strategy": "static",
            "criteria": {"expected_answers": ["flag{test}"]},
            "scoring": {"max_score": 1.0}
        }

        config = EvaluationConfig.from_dict(data)

        assert config.strategy == "static"
        assert config.criteria["expected_answers"] == ["flag{test}"]
        assert config.scoring["max_score"] == 1.0

    def test_from_dict_llm_config(self):
        """Test creating EvaluationConfig from dictionary for LLM strategy."""
        data = {
            "strategy": "llm_judge",
            "criteria": {
                "golden_answer": "The correct solution",
                "model": "gpt-4"
            },
            "scoring": {"max_score": 100.0}
        }

        config = EvaluationConfig.from_dict(data)

        assert config.strategy == "llm_judge"
        assert config.criteria["golden_answer"] == "The correct solution"
        assert config.criteria["model"] == "gpt-4"
        assert config.scoring["max_score"] == 100.0

    def test_from_dict_minimal_config(self):
        """Test creating EvaluationConfig with minimal data."""
        data = {
            "strategy": "static",
            "criteria": {"expected_answers": ["answer"]}
        }

        config = EvaluationConfig.from_dict(data)

        assert config.strategy == "static"
        assert config.criteria["expected_answers"] == ["answer"]
        assert config.scoring == {}  # Default empty dict


class TestEvaluationResult:
    """Test cases for EvaluationResult model."""

    def test_evaluation_result_creation(self):
        """Test creating EvaluationResult with all fields."""
        result = EvaluationResult(
            episode_id="episode_123",
            task_id="task_456",
            strategy="static",
            raw_score=0.8,
            max_score=1.0,
            score=0.8,
            success=True,
            details={"matched": 4, "total": 5},
            submission="test submission",
            step_count=10
        )

        assert result.episode_id == "episode_123"
        assert result.task_id == "task_456"
        assert result.strategy == "static"
        assert result.raw_score == 0.8
        assert result.max_score == 1.0
        assert result.score == 0.8
        assert result.success is True
        assert result.details["matched"] == 4
        assert isinstance(result.timestamp, datetime)

    def test_evaluation_result_validation(self):
        """Test EvaluationResult field validation."""
        # Test negative raw_score
        with pytest.raises(ValueError):
            EvaluationResult(
                episode_id="episode_123",
                task_id="task_456",
                strategy="static",
                raw_score=-0.1,  # Invalid negative score
                max_score=1.0,
                score=0.0,
                success=False,
                submission="test",
                step_count=1
            )

        # Test zero max_score
        with pytest.raises(ValueError):
            EvaluationResult(
                episode_id="episode_123",
                task_id="task_456",
                strategy="static",
                raw_score=0.0,
                max_score=0.0,  # Invalid zero max_score
                score=0.0,
                success=False,
                submission="test",
                step_count=1
            )

        # Test negative score
        with pytest.raises(ValueError):
            EvaluationResult(
                episode_id="episode_123",
                task_id="task_456",
                strategy="static",
                raw_score=0.0,
                max_score=1.0,
                score=-1.0,  # Invalid negative score
                success=False,
                submission="test",
                step_count=1
            )

    def test_timestamp_defaults_to_utc(self):
        """Test that timestamp defaults to UTC timezone."""
        result = EvaluationResult(
            episode_id="episode_123",
            task_id="task_456",
            strategy="static",
            raw_score=1.0,
            max_score=1.0,
            score=1.0,
            success=True,
            submission="test",
            step_count=1
        )

        # Should be timezone-aware and in UTC
        assert result.timestamp.tzinfo is not None
        assert result.timestamp.tzinfo == timezone.utc

    def test_details_defaults_to_empty_dict(self):
        """Test that details defaults to empty dictionary."""
        result = EvaluationResult(
            episode_id="episode_123",
            task_id="task_456",
            strategy="static",
            raw_score=1.0,
            max_score=1.0,
            score=1.0,
            success=True,
            submission="test",
            step_count=1
        )

        assert result.details == {}
