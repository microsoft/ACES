# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for per-scorer ``submission_source`` resolution.

Covers :class:`saber.config.models.SubmissionSource` schema validation and
:func:`saber.scoring.factory._resolve_submission` behavior under default,
file-source, fallback, and error paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.config.models import (
    ScorerConfig,
    ScorerTarget,
    StaticCriteria,
    SubmissionFallback,
    SubmissionSource,
    SubmissionSourceType,
)
from saber.scoring.factory import (
    SubmissionSourceError,
    _resolve_submission,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_scorer(
    *,
    submission_source: SubmissionSource | None = None,
    strategy: str = "static",
) -> ScorerConfig:
    """Minimal ScorerConfig fixture with optional submission_source."""
    return ScorerConfig(
        scorer_name="test_scorer",
        strategy=strategy,
        target=ScorerTarget.TRAJECTORY,
        submission_source=submission_source,
        criteria=StaticCriteria(expected_answers=["answer"]),
    )


def _make_state(completion_text: str = "") -> MagicMock:
    """Minimal TaskState fixture with state.output.completion."""
    state = MagicMock()
    state.output.completion = completion_text
    return state


# ── Schema validation ────────────────────────────────────────────────


class TestSubmissionSourceSchema:
    def test_default_is_completion_type(self) -> None:
        src = SubmissionSource()
        assert src.type == SubmissionSourceType.COMPLETION
        assert src.path is None
        assert src.fallback == SubmissionFallback.ERROR

    def test_file_type_requires_path(self) -> None:
        with pytest.raises(ValueError, match="path is required"):
            SubmissionSource(type=SubmissionSourceType.FILE)

    def test_file_type_accepts_path(self) -> None:
        src = SubmissionSource(
            type=SubmissionSourceType.FILE,
            path="/workspace/shared/report.md",
        )
        assert src.path == "/workspace/shared/report.md"
        assert src.encoding == "utf-8"

    def test_explicit_completion_type_path_is_optional(self) -> None:
        # path is ignored when type=completion, no error
        src = SubmissionSource(type=SubmissionSourceType.COMPLETION)
        assert src.path is None


# ── Resolution: default behavior (existing 5 RSA agents) ─────────────


class TestResolveSubmissionDefault:
    @pytest.mark.asyncio
    async def test_no_submission_source_returns_completion(self) -> None:
        state = _make_state("the chat msg")
        scorer = _make_scorer()  # no submission_source set
        result = await _resolve_submission(state, scorer)
        assert result == "the chat msg"

    @pytest.mark.asyncio
    async def test_explicit_completion_source_returns_completion(self) -> None:
        state = _make_state("the chat msg")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.COMPLETION,
            )},
        )
        result = await _resolve_submission(state, scorer)
        assert result == "the chat msg"

    @pytest.mark.asyncio
    async def test_no_output_returns_empty_string(self) -> None:
        state = MagicMock()
        state.output = None
        scorer = _make_scorer()
        result = await _resolve_submission(state, scorer)
        assert result == ""


# ── Resolution: file source happy path ───────────────────────────────


class TestResolveSubmissionFile:
    @pytest.mark.asyncio
    async def test_reads_file_from_sandbox(self) -> None:
        state = _make_state("brief ack — should be ignored")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/report.md",
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                return_value="full 10kb structured report"
            )
            result = await _resolve_submission(state, scorer)

        assert result == "full 10kb structured report"
        mock_sandbox.return_value.read_file.assert_awaited_once_with(
            "/workspace/shared/report.md"
        )


# ── Resolution: file missing — fallback semantics ────────────────────


class TestResolveSubmissionFileFallback:
    @pytest.mark.asyncio
    async def test_missing_file_with_completion_fallback_uses_chat_msg(self) -> None:
        state = _make_state("brief ack")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/missing.md",
                fallback=SubmissionFallback.COMPLETION,
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                side_effect=FileNotFoundError
            )
            result = await _resolve_submission(state, scorer)
        assert result == "brief ack"

    @pytest.mark.asyncio
    async def test_missing_file_with_error_fallback_raises(self) -> None:
        state = _make_state("")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/missing.md",
                fallback=SubmissionFallback.ERROR,
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                side_effect=FileNotFoundError
            )
            with pytest.raises(SubmissionSourceError, match="not found"):
                await _resolve_submission(state, scorer)

    @pytest.mark.asyncio
    async def test_missing_file_default_fallback_is_error(self) -> None:
        # Verify that omitting fallback keyword defaults to ERROR (not silent
        # fallback) — protects against future schema drift hiding misconfig.
        state = _make_state("brief ack")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/missing.md",
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                side_effect=FileNotFoundError
            )
            with pytest.raises(SubmissionSourceError):
                await _resolve_submission(state, scorer)


# ── Resolution: I/O errors other than FileNotFoundError ──────────────


class TestResolveSubmissionIOErrors:
    """Verify PermissionError, OSError, UnicodeDecodeError route through fallback."""

    @pytest.mark.asyncio
    async def test_permission_error_with_completion_fallback(self) -> None:
        state = _make_state("fallback msg")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/secret.md",
                fallback=SubmissionFallback.COMPLETION,
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                side_effect=PermissionError("access denied")
            )
            result = await _resolve_submission(state, scorer)
        assert result == "fallback msg"

    @pytest.mark.asyncio
    async def test_permission_error_with_error_fallback_raises(self) -> None:
        state = _make_state("")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/secret.md",
                fallback=SubmissionFallback.ERROR,
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                side_effect=PermissionError("access denied")
            )
            with pytest.raises(SubmissionSourceError, match="read failed"):
                await _resolve_submission(state, scorer)

    @pytest.mark.asyncio
    async def test_unicode_decode_error_with_completion_fallback(self) -> None:
        state = _make_state("fallback msg")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/binary.bin",
                fallback=SubmissionFallback.COMPLETION,
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
            )
            result = await _resolve_submission(state, scorer)
        assert result == "fallback msg"

    @pytest.mark.asyncio
    async def test_os_error_with_error_fallback_raises(self) -> None:
        state = _make_state("")
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/bad_mount.md",
                fallback=SubmissionFallback.ERROR,
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                side_effect=OSError("I/O error")
            )
            with pytest.raises(SubmissionSourceError, match="read failed"):
                await _resolve_submission(state, scorer)

    @pytest.mark.asyncio
    async def test_file_read_failure_with_none_output_returns_empty(self) -> None:
        """state.output is None AND file read fails with fallback=completion."""
        state = MagicMock()
        state.output = None
        scorer = _make_scorer().model_copy(
            update={"submission_source": SubmissionSource(
                type=SubmissionSourceType.FILE,
                path="/workspace/shared/missing.md",
                fallback=SubmissionFallback.COMPLETION,
            )},
        )
        with patch("inspect_ai.util.sandbox") as mock_sandbox:
            mock_sandbox.return_value.read_file = AsyncMock(
                side_effect=FileNotFoundError
            )
            result = await _resolve_submission(state, scorer)
        assert result == ""


# ── Regression: existing scorers without submission_source unchanged ─


class TestRegressionExistingScorers:
    """Verify the 5 RSA-domain agents' scoring path is identical to pre-change.

    None of hardening, detection, red_team, investigation_planner, or
    posture-analysis (pre-migration) set ``submission_source`` — all should
    receive ``state.output.completion`` exactly as before.
    """

    @pytest.mark.asyncio
    async def test_static_scorer_unchanged(self) -> None:
        scorer = _make_scorer(strategy="static")
        assert scorer.submission_source is None
        state = _make_state("agent's final message")
        result = await _resolve_submission(state, scorer)
        assert result == "agent's final message"

    @pytest.mark.asyncio
    async def test_llm_judge_scorer_unchanged(self) -> None:
        from saber.config.models import LLMJudgeCriteria

        scorer = _make_scorer(strategy="llm_judge").model_copy(
            update={"criteria": LLMJudgeCriteria(
                model="openai/gpt-4",
                judge_user_template="judge/test.j2",
                judge_system_template="judge/test_system.j2",
            )},
        )
        assert scorer.submission_source is None
        state = _make_state("agent emitted this as final msg")
        result = await _resolve_submission(state, scorer)
        assert result == "agent emitted this as final msg"
