"""Standalone response-parsing functions for LLM judge outputs.

Pure functions with no dependencies on templates, strategies, or scoring context.
They parse raw LLM response strings into numeric scores.
"""

from __future__ import annotations

import json
import math
import re

from saber.config.models import LLMJudgeResponseFormat


def parse_judge_response(
    response: str,
    response_format: LLMJudgeResponseFormat,
    max_score: float,
) -> float:
    """Parse judge response based on format.

    - BINARY: Look for CORRECT/INCORRECT keywords → max_score or 0.
    - CONTINUOUS: Parse JSON ``{"score": 0.0-1.0}`` → scaled to max_score.
      Falls back to parsing bare float.
    - STEP_EVALUATIONS: Parse JSON ``{"step_evaluations": [{"step_number": N,
      "completed": bool}, ...]}`` → completed_count / total_count * max_score.

    Args:
        response: Raw LLM response text.
        response_format: Expected format.
        max_score: Maximum score value.

    Returns:
        Score value in [0.0, max_score].
    """
    if response_format is LLMJudgeResponseFormat.BINARY:
        return _parse_binary(response, max_score)
    if response_format is LLMJudgeResponseFormat.CONTINUOUS:
        return _parse_continuous(response, max_score)
    if response_format is LLMJudgeResponseFormat.STEP_EVALUATIONS:
        return _parse_step_evaluations(response, max_score)
    return 0.0  # pragma: no cover – exhaustive enum


def parse_llm_step_evaluations(response: str) -> list[str]:
    """Parse ``[step_number: checkpoint_id]`` entries from LLM batch response.

    Format: ``[3: checkpoint_1]`` means step 3 completed *checkpoint_1*.
    ``[NO_COMPLETIONS]`` means nothing was completed.

    Args:
        response: Raw LLM response text.

    Returns:
        List of completed checkpoint IDs.
    """
    if "[NO_COMPLETIONS]" in response:
        return []
    return [m.group(2) for m in re.finditer(r"\[(\d+):\s*(\S+)\]", response)]


def parse_llm_not_completed_explanations(response: str) -> dict[str, str]:
    """Parse ``[checkpoint_id] - explanation`` from a NOT_COMPLETED block.

    Format::

        NOT_COMPLETED:
        [checkpoint_1] - The agent never queried the UrlClickEvents table...
        [checkpoint_2] - No evidence of the specific timestamp...

    Args:
        response: Raw LLM response text.

    Returns:
        Mapping of checkpoint_id → explanation string.
    """
    explanations: dict[str, str] = {}
    in_block = False
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("NOT_COMPLETED"):
            in_block = True
            continue
        if in_block:
            m = re.match(r"\[([\w]+)\]\s*[-–—]\s*(.+)", stripped)
            if m:
                explanations[m.group(1)] = m.group(2).strip()
            elif stripped.startswith("```") or stripped.startswith("STEP_EVALUATION"):
                in_block = False
    return explanations


def parse_checkpoint_score(
    response: str,
    checkpoint_id: str,
    max_score: float,
) -> float:
    """Parse a single-checkpoint LLM judge response.

    First tries ``[step: checkpoint_id]`` format via
    :func:`parse_llm_step_evaluations`.  Falls back to CORRECT/INCORRECT
    binary parsing.

    Args:
        response: Raw LLM response.
        checkpoint_id: The checkpoint to look for.
        max_score: Maximum score value.

    Returns:
        *max_score* if checkpoint found/correct, 0.0 otherwise.
    """
    completed = parse_llm_step_evaluations(response)
    if completed:
        return max_score if checkpoint_id in completed else 0.0
    # No step-format matches — fall back to binary
    return _parse_binary(response, max_score)


# ── Private helpers ─────────────────────────────────────────────────


def _parse_binary(response: str, max_score: float) -> float:
    """Return *max_score* if CORRECT is present as a whole word and INCORRECT is not."""
    upper = response.strip().upper()
    if re.search(r"\bCORRECT\b", upper) and not re.search(r"\bINCORRECT\b", upper):
        return max_score
    return 0.0


def _parse_continuous(response: str, max_score: float) -> float:
    """Parse JSON ``{"score": <float>}`` or bare float, scaled by *max_score*.

    Tolerates markdown-wrapped JSON (```json ... ```) by extracting the first
    ``{...}`` object substring before parsing. Both the JSON path and bare-float
    fallback assume the value is a 0-1 ratio that gets multiplied by *max_score*.
    The result is clamped to ``[0.0, max_score]``.
    """
    text = response.strip()

    # Extract first JSON object (handles markdown ```json ... ``` wrapping)
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict) and "score" in data:
                value = float(data["score"])
                if math.isnan(value):
                    return 0.0
                return max(0.0, min(value * max_score, max_score))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: bare float (also treated as 0-1 ratio)
    try:
        value = float(text)
        if math.isnan(value):
            return 0.0
        return max(0.0, min(value * max_score, max_score))
    except ValueError:
        return 0.0


def _parse_step_evaluations(response: str, max_score: float) -> float:
    """Parse JSON step_evaluations list and compute score."""
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not isinstance(data, dict):
        return 0.0

    steps: list[dict[str, object]] = data.get("step_evaluations", [])
    if not steps:
        return 0.0

    total = len(steps)
    completed = sum(1 for s in steps if s.get("completed") is True)
    return completed / total * max_score
