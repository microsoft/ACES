# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for bridge generation tracking filter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from saber.agents.bridge_tracking import _has_active_delegation, create_tracking_filter
from saber.agents.bridge_tracking_models import DELEGATION_TOOL_NAMES


def _make_messages(n: int) -> list[MagicMock]:
    """Create n mock messages for testing."""
    msgs = []
    for _ in range(n):
        m = MagicMock()
        m.tool_calls = None
        msgs.append(m)
    return msgs


def _make_model(name: str = "test-model") -> MagicMock:
    """Create a mock model with a name attribute."""
    model = MagicMock()
    model.name = name
    return model


@pytest.mark.asyncio
class TestCreateTrackingFilter:
    """Tests for create_tracking_filter()."""

    async def test_first_call_classified_as_main(self) -> None:
        """First generation with N messages -> CallType.MAIN."""
        filter_fn, get_summary = create_tracking_filter()
        await filter_fn(_make_model(), _make_messages(5), [], None, MagicMock())

        summary = get_summary()
        assert summary.main_generations == 1
        assert summary.subagent_generations == 0

    async def test_shorter_history_classified_as_subagent(self) -> None:
        """After main(10 msgs), call with 3 msgs -> SUBAGENT."""
        filter_fn, get_summary = create_tracking_filter()
        await filter_fn(_make_model(), _make_messages(10), [], None, MagicMock())
        await filter_fn(_make_model(), _make_messages(3), [], None, MagicMock())

        summary = get_summary()
        assert summary.main_generations == 1
        assert summary.subagent_generations == 1

    async def test_growing_history_stays_main(self) -> None:
        """5, 8, 12 messages -> all MAIN."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()
        for count in (5, 8, 12):
            await filter_fn(model, _make_messages(count), [], None, MagicMock())

        summary = get_summary()
        assert summary.main_generations == 3
        assert summary.subagent_generations == 0

    async def test_subagent_then_main_resumes(self) -> None:
        """Main(10) -> Subagent(3) -> Main(13) -> correct classification."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()
        await filter_fn(model, _make_messages(10), [], None, MagicMock())
        await filter_fn(model, _make_messages(3), [], None, MagicMock())
        await filter_fn(model, _make_messages(13), [], None, MagicMock())

        summary = get_summary()
        assert summary.main_generations == 2
        assert summary.subagent_generations == 1

    async def test_equal_length_classified_as_subagent(self) -> None:
        """Same count as previous -> SUBAGENT (not > high_water)."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()
        await filter_fn(model, _make_messages(5), [], None, MagicMock())
        await filter_fn(model, _make_messages(5), [], None, MagicMock())

        summary = get_summary()
        assert summary.main_generations == 1
        assert summary.subagent_generations == 1

    async def test_generation_index_increments(self) -> None:
        """Each call increments generation_index."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()
        for i in range(4):
            await filter_fn(model, _make_messages(i + 1), [], None, MagicMock())

        summary = get_summary()
        assert summary.total_generations == 4

    async def test_model_name_captured(self) -> None:
        """Model mock with name attribute -> captured in metadata."""
        filter_fn, get_summary = create_tracking_filter()
        await filter_fn(_make_model("gpt-4o"), _make_messages(5), [], None, MagicMock())

        summary = get_summary()
        assert "gpt-4o" in summary.models_used

    async def test_multiple_models_tracked(self) -> None:
        """Different model objects -> models_used contains all."""
        filter_fn, get_summary = create_tracking_filter()
        await filter_fn(_make_model("gpt-4o"), _make_messages(5), [], None, MagicMock())
        await filter_fn(_make_model("claude-3"), _make_messages(8), [], None, MagicMock())

        summary = get_summary()
        assert set(summary.models_used) == {"gpt-4o", "claude-3"}

    async def test_get_summary_returns_correct_counts(self) -> None:
        """3 main + 2 subagent -> summary has correct totals."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()
        # 3 main calls (growing)
        for count in (5, 10, 15):
            await filter_fn(model, _make_messages(count), [], None, MagicMock())
        # 2 subagent calls (shorter)
        for count in (3, 7):
            await filter_fn(model, _make_messages(count), [], None, MagicMock())

        summary = get_summary()
        assert summary.total_generations == 5
        assert summary.main_generations == 3
        assert summary.subagent_generations == 2

    async def test_filter_returns_none(self) -> None:
        """Always returns None (observation only)."""
        filter_fn, _ = create_tracking_filter()
        result = await filter_fn(_make_model(), _make_messages(5), [], None, MagicMock())
        assert result is None

    async def test_info_event_recorded_in_transcript(self) -> None:
        """Mock transcript().info() -> receives correct data dict."""
        filter_fn, _ = create_tracking_filter()

        mock_transcript = MagicMock()
        with patch(
            "saber.agents.bridge_tracking.transcript",
            return_value=mock_transcript,
        ):
            await filter_fn(_make_model("gpt-4o"), _make_messages(5), [], None, MagicMock())

        mock_transcript.info.assert_called_once()
        call_args = mock_transcript.info.call_args
        data = call_args[0][0]
        assert data["call_type"] == "main"
        assert data["model_name"] == "gpt-4o"
        assert data["message_count"] == 5
        assert data["generation_index"] == 0
        assert call_args[1]["source"] == "saber.bridge_tracking"

    async def test_info_event_graceful_without_transcript(self) -> None:
        """When transcript() raises -> filter still runs without error."""
        filter_fn, get_summary = create_tracking_filter()

        with patch(
            "saber.agents.bridge_tracking.transcript",
            side_effect=RuntimeError("no transcript context"),
        ):
            await filter_fn(_make_model(), _make_messages(5), [], None, MagicMock())

        # Filter should still have recorded the generation
        summary = get_summary()
        assert summary.total_generations == 1


def _make_delegation_messages(
    n: int, delegation_at: int | None = None, function_name: str = "task"
) -> list[MagicMock]:
    """Create n mock messages with optional delegation tool result at index."""
    msgs: list[MagicMock] = []
    for i in range(n):
        m = MagicMock()
        if delegation_at is not None and i == delegation_at:
            m.role = "tool"
            m.function = function_name
            m.tool_calls = None
        else:
            m.role = "user" if i % 2 == 0 else "assistant"
            m.function = None
            m.tool_calls = None
        msgs.append(m)
    return msgs


@pytest.mark.asyncio
class TestDelegationToolDetection:
    """Tests for dual-heuristic delegation detection (H2)."""

    async def test_copilot_monotonic_with_delegation_detected_as_subagent(
        self,
    ) -> None:
        """Growing msg counts with task tool result → H2 detects SUBAGENT."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()

        # Gen 0: 5 msgs, no delegation → MAIN
        await filter_fn(model, _make_delegation_messages(5), [], None, MagicMock())
        # Gen 1: 10 msgs, no delegation → MAIN
        await filter_fn(model, _make_delegation_messages(10), [], None, MagicMock())
        # Gen 2: 15 msgs, delegation at last index (14) → SUBAGENT via H2
        await filter_fn(
            model,
            _make_delegation_messages(15, delegation_at=14),
            [],
            None,
            MagicMock(),
        )

        summary = get_summary()
        assert summary.main_generations == 2
        assert summary.subagent_generations == 1

    async def test_multiple_delegations_multiple_subagents(self) -> None:
        """Multiple gens with delegation tool results → multiple SUBAGENTs."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()

        # Gen 0: 5 msgs → MAIN
        await filter_fn(model, _make_delegation_messages(5), [], None, MagicMock())
        # Gen 1: 10 msgs, delegation at last index (9) → SUBAGENT
        await filter_fn(
            model,
            _make_delegation_messages(10, delegation_at=9),
            [],
            None,
            MagicMock(),
        )
        # Gen 2: 15 msgs, delegation at last index (14) → SUBAGENT
        await filter_fn(
            model,
            _make_delegation_messages(15, delegation_at=14),
            [],
            None,
            MagicMock(),
        )

        summary = get_summary()
        assert summary.main_generations == 1
        assert summary.subagent_generations == 2

    async def test_delegation_coexists_with_message_drop(self) -> None:
        """H1 still works: Claude Code pattern with message count drop."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()

        # Gen 0: 10 msgs → MAIN
        await filter_fn(model, _make_delegation_messages(10), [], None, MagicMock())
        # Gen 1: 3 msgs (dropped) → SUBAGENT via H1
        await filter_fn(model, _make_delegation_messages(3), [], None, MagicMock())
        # Gen 2: 13 msgs → MAIN (resumes)
        await filter_fn(model, _make_delegation_messages(13), [], None, MagicMock())

        summary = get_summary()
        assert summary.main_generations == 2
        assert summary.subagent_generations == 1

    async def test_no_false_positive_for_non_delegation_tools(self) -> None:
        """bash/python tool results don't trigger H2."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()

        # Gen 0: 5 msgs → MAIN
        await filter_fn(model, _make_delegation_messages(5), [], None, MagicMock())
        # Gen 1: 10 msgs with bash tool at index 8 → MAIN (not a delegation tool)
        await filter_fn(
            model,
            _make_delegation_messages(10, delegation_at=8, function_name="bash"),
            [],
            None,
            MagicMock(),
        )

        summary = get_summary()
        assert summary.main_generations == 2
        assert summary.subagent_generations == 0

    async def test_delegation_metadata_flag(self) -> None:
        """delegation_detected is True for H2, False for H1 and normal MAIN."""
        filter_fn, _ = create_tracking_filter()
        model = _make_model()

        recorded: list[dict[str, object]] = []
        mock_transcript = MagicMock()
        mock_transcript.info.side_effect = lambda data, **kw: recorded.append(data)

        with patch(
            "saber.agents.bridge_tracking.transcript",
            return_value=mock_transcript,
        ):
            # Gen 0: 5 msgs → MAIN (delegation_detected=False)
            await filter_fn(
                model, _make_delegation_messages(5), [], None, MagicMock()
            )
            # Gen 1: 3 msgs → SUBAGENT via H1 (delegation_detected=False)
            await filter_fn(
                model, _make_delegation_messages(3), [], None, MagicMock()
            )
            # Gen 2: 10 msgs → MAIN (delegation_detected=False)
            await filter_fn(
                model, _make_delegation_messages(10), [], None, MagicMock()
            )
            # Gen 3: 15 msgs, delegation at 14 → SUBAGENT via H2 (delegation_detected=True)
            await filter_fn(
                model,
                _make_delegation_messages(15, delegation_at=14),
                [],
                None,
                MagicMock(),
            )

        assert recorded[0]["delegation_detected"] is False  # MAIN
        assert recorded[1]["delegation_detected"] is False  # H1 SUBAGENT
        assert recorded[2]["delegation_detected"] is False  # MAIN
        assert recorded[3]["delegation_detected"] is True  # H2 SUBAGENT

    async def test_read_agent_also_detected(self) -> None:
        """function='read_agent' triggers H2 just like 'task'."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()

        # Gen 0: 5 msgs → MAIN
        await filter_fn(model, _make_delegation_messages(5), [], None, MagicMock())
        # Gen 1: 10 msgs, read_agent at index 9 → SUBAGENT via H2
        await filter_fn(
            model,
            _make_delegation_messages(10, delegation_at=9, function_name="read_agent"),
            [],
            None,
            MagicMock(),
        )

        summary = get_summary()
        assert summary.main_generations == 1
        assert summary.subagent_generations == 1

    async def test_delegation_does_not_update_high_water_mark(self) -> None:
        """After H2 subagent, main thread with higher count still classified as MAIN."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()

        # Gen 0: 5 msgs → MAIN (hwm=5)
        await filter_fn(model, _make_delegation_messages(5), [], None, MagicMock())
        # Gen 1: 10 msgs, delegation at 9 → SUBAGENT via H2 (hwm stays 5)
        await filter_fn(
            model,
            _make_delegation_messages(10, delegation_at=9),
            [],
            None,
            MagicMock(),
        )
        # Gen 2: 8 msgs, no delegation → MAIN (8 > 5, hwm=8)
        await filter_fn(model, _make_delegation_messages(8), [], None, MagicMock())

        summary = get_summary()
        assert summary.main_generations == 2
        assert summary.subagent_generations == 1

    def test_has_active_delegation_with_task_at_tail_end(self) -> None:
        """_has_active_delegation returns True when task is last message."""
        msgs = _make_delegation_messages(10, delegation_at=9)
        assert _has_active_delegation(msgs) is True

    def test_has_active_delegation_false_when_assistant_follows(self) -> None:
        """_has_active_delegation returns False when assistant msg follows delegation."""
        msgs = _make_delegation_messages(10, delegation_at=7)
        # Index 8 = user, index 9 = assistant — both follow delegation result
        assert _has_active_delegation(msgs) is False

    def test_has_active_delegation_without_delegation(self) -> None:
        """_has_active_delegation returns False when no delegation tools."""
        msgs = _make_delegation_messages(10)
        assert _has_active_delegation(msgs) is False

    def test_has_active_delegation_empty_messages(self) -> None:
        """_has_active_delegation returns False for empty list."""
        assert _has_active_delegation([]) is False

    def test_has_active_delegation_outside_tail(self) -> None:
        """Delegation at index 0 of 10 msgs → outside last 5 → False."""
        msgs = _make_delegation_messages(10, delegation_at=0)
        assert _has_active_delegation(msgs) is False

    async def test_main_resumes_after_delegation_with_result_in_tail(
        self,
    ) -> None:
        """After subagent, main thread with task result still in tail → MAIN."""
        filter_fn, get_summary = create_tracking_filter()
        model = _make_model()

        # Gen 0: 5 msgs → MAIN (hwm=5)
        await filter_fn(model, _make_delegation_messages(5), [], None, MagicMock())
        # Gen 1: 10 msgs, delegation at 9 → SUBAGENT via H2
        await filter_fn(
            model,
            _make_delegation_messages(10, delegation_at=9),
            [],
            None,
            MagicMock(),
        )
        # Gen 2: 12 msgs, task result at 9 but assistant msg follows at 11
        # → delegation result is NOT at tail end → MAIN
        msgs = _make_delegation_messages(12, delegation_at=9)
        # Explicit for clarity — helper already produces user/assistant at
        # even/odd indices, but we spell it out to document the intent.
        msgs[10].role = "user"
        msgs[10].function = None
        msgs[11].role = "assistant"
        msgs[11].function = None
        await filter_fn(model, msgs, [], None, MagicMock())

        summary = get_summary()
        assert summary.main_generations == 2
        assert summary.subagent_generations == 1

    def test_delegation_tool_names_is_frozenset(self) -> None:
        """DELEGATION_TOOL_NAMES is a frozenset containing expected tools."""
        assert isinstance(DELEGATION_TOOL_NAMES, frozenset)
        assert "task" in DELEGATION_TOOL_NAMES
        assert "read_agent" in DELEGATION_TOOL_NAMES
