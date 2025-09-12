"""Additional tests for LLMEvaluator edge cases introduced post-review."""

import json
import os
import asyncio
from unittest.mock import AsyncMock, patch, Mock
import pytest

from saber.server.evaluation.evaluators.llm_evaluator import LLMEvaluator
from saber.server.evaluation.models import EvaluationConfig, EpisodeEvaluationData
from saber.server.evaluation.constants import EVAL_STRATEGY_LLM_JUDGE
from saber.server.evaluation.exceptions import EvaluationError
from saber.server.evaluation.evaluation_manager import EvaluationManager
from saber.server.evaluation.store import JsonFileEvaluationStore
from saber.server.benchmarks.prompt_generator import JudgePromptPayload


class _MockEpisode:
    def __init__(self, episode_id: str, task_id: str, submission: str, session_id: str = "test_session"):
        self.episode_id = episode_id
        self.task_id = task_id
        self.submission = submission
        self.session_id = session_id
        self.is_complete = True
        self.steps = []
        self.completion_reason = "done"
    def get_executed_commands(self):
        return []


class _MockTask:
    def __init__(self, task_id: str, description: str, evaluation_config: dict):
        self.task_id = task_id
        self.description = description
        self.evaluation_config = evaluation_config
        # Add judge_prompt_renderer if not already present
        if "judge_prompt_renderer" not in self.evaluation_config:
            self.evaluation_config["judge_prompt_renderer"] = self._mock_judge_prompt_renderer

    def _mock_judge_prompt_renderer(self, submission, episode_id):
        """Mock judge prompt renderer function."""
        return JudgePromptPayload(
            messages=[
                {"role": "system", "content": "Evaluate this submission."},
                {"role": "user", "content": f"Submission: {submission}\nEpisode ID: {episode_id}\nIs this correct?"}
            ],
            model="gpt-3.5-turbo",
            task_id=self.task_id,
            episode_id=episode_id
        )


@pytest.mark.asyncio
async def test_llm_empty_message_content():
    evaluator = LLMEvaluator()
    config = EvaluationConfig(
        strategy=EVAL_STRATEGY_LLM_JUDGE,
        criteria={
            "golden_answer": "abc",
            "model": "gpt-test",
            "judge_system_template": "test_system.md",
            "judge_user_template": "test_user.md"
        },
        scoring={"max_score": 1.0},
    )
    episode_data = EpisodeEvaluationData(
        episode_id="e1",
        task_id="t1",
        submission="test",
        step_count=1
    )

    # Patch environment & _call_llm_json to simulate empty content edge triggered earlier (handled in _call_llm_json path)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "x"}):
        # Patch lower-level sync call by patching _call_llm_json to raise EvaluationError as produced by content None branch
        async def fake_call(judge_payload):
            raise EvaluationError("LLM response message content is empty")
        with patch.object(evaluator, "_call_llm_json", new=fake_call):
            with pytest.raises(EvaluationError, match="content is empty"):
                await evaluator.evaluate(episode_data, config, _MockTask("t1", "desc", {}))


@pytest.mark.asyncio
async def test_latency_and_golden_answer_persistence(tmp_path):
    # Setup manager with temp store dir
    store_dir = tmp_path / "evaluations"
    # Pass base directory path so EvaluationManager builds JsonFileEvaluationStore internally
    mgr = EvaluationManager(store=str(store_dir))

    task = _MockTask(
        "taskA",
        "Describe X",
        {
            "strategy": EVAL_STRATEGY_LLM_JUDGE,
            "criteria": {
                "golden_answer": "correct answer",
                "model": "gpt-test",
                "judge_system_template": "test_system.md",
                "judge_user_template": "test_user.md"
            },
            "scoring": {"max_score": 1.0},
        },
    )
    mgr.configure_for_task(task)
    episode = _MockEpisode("epA", task.task_id, submission="candidate answer")

    # Patch evaluator's _call_llm_json to return deterministic JSON
    evaluator = mgr.evaluators[EVAL_STRATEGY_LLM_JUDGE]
    with patch.dict(os.environ, {"OPENAI_API_KEY": "x"}):
        async def fake_call(judge_payload):
            await asyncio.sleep(0.01)
            return json.dumps({"analysis": "Some reasoning", "is_correct": True})
        with patch.object(evaluator, "_call_llm_json", new=fake_call):
            result = await mgr.evaluate_episode(episode, task)
            assert "latency_ms" in result.details
            assert result.details["golden_answer"] == "correct answer"

    # Verify persisted artifact contains golden_answer
    artifact = store_dir / episode.session_id / task.task_id / f"{episode.episode_id}.json"
    with artifact.open() as f:
        data = json.load(f)
    assert data["golden_answer"] == "correct answer"
