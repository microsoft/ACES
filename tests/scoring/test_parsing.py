"""Tests for saber.scoring.parsing — response parsing functions."""

from __future__ import annotations

import pytest

from saber.config.models import LLMJudgeResponseFormat
from saber.scoring.parsing import (
    _parse_binary,
    _parse_continuous,
    parse_checkpoint_score,
    parse_judge_response,
    parse_llm_step_evaluations,
)

# ── parse_judge_response — BINARY ──────────────────────────────────


class TestParseJudgeResponseBinary:
    """BINARY format: CORRECT/INCORRECT keyword matching."""

    fmt = LLMJudgeResponseFormat.BINARY

    def test_correct(self) -> None:
        assert parse_judge_response("CORRECT", self.fmt, 1.0) == 1.0

    def test_incorrect(self) -> None:
        assert parse_judge_response("INCORRECT", self.fmt, 1.0) == 0.0

    def test_correct_substring(self) -> None:
        assert parse_judge_response("The answer is CORRECT", self.fmt, 1.0) == 1.0

    def test_incorrect_present(self) -> None:
        assert parse_judge_response("INCORRECT because...", self.fmt, 1.0) == 0.0

    def test_empty_string(self) -> None:
        assert parse_judge_response("", self.fmt, 1.0) == 0.0

    def test_case_insensitive(self) -> None:
        assert parse_judge_response("correct", self.fmt, 1.0) == 1.0

    def test_incorrectly_contains_incorrect(self) -> None:
        assert parse_judge_response("INCORRECTLY", self.fmt, 1.0) == 0.0

    def test_mixed_text_with_correct(self) -> None:
        assert parse_judge_response("I think this is CORRECT overall", self.fmt, 1.0) == 1.0

    def test_max_score_five(self) -> None:
        assert parse_judge_response("CORRECT", self.fmt, 5.0) == 5.0

    def test_max_score_five_incorrect(self) -> None:
        assert parse_judge_response("INCORRECT", self.fmt, 5.0) == 0.0


# ── parse_judge_response — CONTINUOUS ──────────────────────────────


class TestParseJudgeResponseContinuous:
    """CONTINUOUS format: JSON {"score": float} or bare float."""

    fmt = LLMJudgeResponseFormat.CONTINUOUS

    def test_json_score(self) -> None:
        assert parse_judge_response('{"score": 0.75}', self.fmt, 1.0) == pytest.approx(0.75)

    def test_json_score_one(self) -> None:
        assert parse_judge_response('{"score": 1.0}', self.fmt, 1.0) == 1.0

    def test_json_score_zero(self) -> None:
        assert parse_judge_response('{"score": 0.0}', self.fmt, 1.0) == 0.0

    def test_bare_float_fallback(self) -> None:
        assert parse_judge_response("0.5", self.fmt, 1.0) == pytest.approx(0.5)

    def test_json_score_clamped(self) -> None:
        # 1.5 * max_score would exceed max_score — must clamp
        assert parse_judge_response('{"score": 1.5}', self.fmt, 1.0) == 1.0

    def test_invalid_json(self) -> None:
        assert parse_judge_response("invalid json", self.fmt, 1.0) == 0.0

    def test_empty_json_object(self) -> None:
        assert parse_judge_response("{}", self.fmt, 1.0) == 0.0

    def test_score_not_a_number(self) -> None:
        assert parse_judge_response('{"score": "not_a_number"}', self.fmt, 1.0) == 0.0

    def test_json_scaled_by_max(self) -> None:
        assert parse_judge_response('{"score": 0.5}', self.fmt, 10.0) == pytest.approx(5.0)

    def test_bare_float_scaled_by_max(self) -> None:
        # Bare float is treated as a 0-1 ratio, NOT an absolute value
        assert parse_judge_response("0.8", self.fmt, 10.0) == pytest.approx(8.0)

    def test_negative_json_score_clamped_to_zero(self) -> None:
        assert parse_judge_response('{"score": -0.5}', self.fmt, 10.0) == 0.0

    def test_negative_bare_float_clamped_to_zero(self) -> None:
        assert parse_judge_response("-0.5", self.fmt, 10.0) == 0.0

    def test_nan_string_returns_zero(self) -> None:
        assert parse_judge_response("NaN", self.fmt, 1.0) == 0.0

    def test_inf_string_clamped(self) -> None:
        # float("inf") * max_score = inf → clamped to max_score
        assert parse_judge_response("inf", self.fmt, 1.0) == 1.0


# ── parse_judge_response — STEP_EVALUATIONS ────────────────────────


class TestParseJudgeResponseStepEvaluations:
    """STEP_EVALUATIONS format: JSON with step_evaluations list."""

    fmt = LLMJudgeResponseFormat.STEP_EVALUATIONS

    def test_partial_completion(self) -> None:
        response = (
            '{"step_evaluations": ['
            '{"step_number": 1, "completed": true},'
            '{"step_number": 2, "completed": true},'
            '{"step_number": 3, "completed": false}'
            "]}"
        )
        assert parse_judge_response(response, self.fmt, 1.0) == pytest.approx(2.0 / 3.0)

    def test_all_completed(self) -> None:
        response = '{"step_evaluations": [{"step_number": 1, "completed": true},{"step_number": 2, "completed": true}]}'
        assert parse_judge_response(response, self.fmt, 1.0) == 1.0

    def test_none_completed(self) -> None:
        response = (
            '{"step_evaluations": [{"step_number": 1, "completed": false},{"step_number": 2, "completed": false}]}'
        )
        assert parse_judge_response(response, self.fmt, 1.0) == 0.0

    def test_empty_list(self) -> None:
        assert parse_judge_response('{"step_evaluations": []}', self.fmt, 1.0) == 0.0

    def test_invalid_json(self) -> None:
        assert parse_judge_response("not valid json", self.fmt, 1.0) == 0.0

    def test_missing_key(self) -> None:
        assert parse_judge_response('{"other_key": 1}', self.fmt, 1.0) == 0.0

    def test_scaled_by_max(self) -> None:
        response = (
            '{"step_evaluations": [{"step_number": 1, "completed": true},{"step_number": 2, "completed": false}]}'
        )
        assert parse_judge_response(response, self.fmt, 10.0) == pytest.approx(5.0)


# ── parse_llm_step_evaluations ─────────────────────────────────────


class TestParseLlmStepEvaluations:
    """Parse [step_number: checkpoint_id] entries."""

    def test_single_match(self) -> None:
        assert parse_llm_step_evaluations("[3: checkpoint_1]") == ["checkpoint_1"]

    def test_multiple_matches(self) -> None:
        assert parse_llm_step_evaluations("[1: cp_1] [5: cp_2]") == ["cp_1", "cp_2"]

    def test_no_completions(self) -> None:
        assert parse_llm_step_evaluations("[NO_COMPLETIONS]") == []

    def test_no_brackets(self) -> None:
        assert parse_llm_step_evaluations("no completions") == []

    def test_empty_string(self) -> None:
        assert parse_llm_step_evaluations("") == []

    def test_embedded_in_text(self) -> None:
        result = parse_llm_step_evaluations("text [2: ck_1] more text [7: ck_3]")
        assert result == ["ck_1", "ck_3"]


# ── parse_checkpoint_score ──────────────────────────────────────────


class TestParseCheckpointScore:
    """Parse a single-checkpoint LLM judge response."""

    def test_checkpoint_found(self) -> None:
        assert parse_checkpoint_score("[5: my_checkpoint]", "my_checkpoint", 1.0) == 1.0

    def test_no_completions(self) -> None:
        assert parse_checkpoint_score("[NO_COMPLETIONS]", "my_checkpoint", 1.0) == 0.0

    def test_fallback_correct(self) -> None:
        assert parse_checkpoint_score("CORRECT", "my_checkpoint", 1.0) == 1.0

    def test_fallback_incorrect(self) -> None:
        assert parse_checkpoint_score("INCORRECT", "my_checkpoint", 1.0) == 0.0

    def test_different_checkpoint(self) -> None:
        # Checkpoint not found → falls back to binary → no CORRECT keyword → 0.0
        assert parse_checkpoint_score("[5: other_cp]", "my_checkpoint", 1.0) == 0.0

    def test_max_score_scaling(self) -> None:
        assert parse_checkpoint_score("[1: cp_a]", "cp_a", 5.0) == 5.0


# ── _parse_continuous — NaN guard ───────────────────────────────────


class TestParseContinuousNaN:
    """NaN values must be treated as zero."""

    def test_nan_json_score_returns_zero(self) -> None:
        assert _parse_continuous('{"score": "NaN"}', 10.0) == 0.0

    def test_nan_bare_float_returns_zero(self) -> None:
        assert _parse_continuous("NaN", 10.0) == 0.0


# ── _parse_continuous — markdown-wrapped JSON ───────────────────────


class TestParseContinuousMarkdownWrappedJson:
    """Some judges return the score JSON inside a ```json ... ``` fence."""

    def test_markdown_json_fence(self) -> None:
        response = '```json\n{"score": 0.8}\n```'
        assert _parse_continuous(response, 1.0) == pytest.approx(0.8)

    def test_markdown_plain_fence(self) -> None:
        response = "```\n{\"score\": 0.5}\n```"
        assert _parse_continuous(response, 1.0) == pytest.approx(0.5)

    def test_json_with_leading_prose(self) -> None:
        response = "Here is the score: {\"score\": 0.42}"
        assert _parse_continuous(response, 1.0) == pytest.approx(0.42)

    def test_json_with_trailing_prose(self) -> None:
        response = '{"score": 0.6} — based on the rubric above.'
        assert _parse_continuous(response, 1.0) == pytest.approx(0.6)

    def test_markdown_json_scaled_by_max(self) -> None:
        response = '```json\n{"score": 0.5}\n```'
        assert _parse_continuous(response, 10.0) == pytest.approx(5.0)

    def test_markdown_json_clamped(self) -> None:
        response = '```json\n{"score": 1.5}\n```'
        assert _parse_continuous(response, 1.0) == 1.0


# ── _parse_binary — word boundary ───────────────────────────────────


class TestParseBinaryWordBoundary:
    """CORRECT must match as a whole word, not as a substring."""

    def test_correctly_not_matched(self) -> None:
        assert _parse_binary("The agent correctly failed", 1.0) == 0.0

    def test_corrected_not_matched(self) -> None:
        assert _parse_binary("The issue was CORRECTED", 1.0) == 0.0

    def test_correct_still_matches(self) -> None:
        assert _parse_binary("CORRECT", 1.0) == 1.0

    def test_correct_in_sentence(self) -> None:
        assert _parse_binary("The answer is CORRECT.", 1.0) == 1.0
