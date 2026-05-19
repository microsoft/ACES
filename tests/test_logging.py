# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for saber.logging – thin wrapper delegating to inspect_ai's logger."""

from __future__ import annotations

import logging
import re

import pytest

import saber.logging as saber_logging
from saber.logging import _has_inspect_handler, _resolve_level, configure_logging, display_progress, get_logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLogHandler(logging.Handler):
    """Mimics inspect_ai's LogHandler for testing (class name matters)."""

    class_name_alias = "LogHandler"

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        pass


# We need the class name to be "LogHandler" for _has_inspect_handler()
_FakeLogHandler.__name__ = "LogHandler"


@pytest.fixture(autouse=True)
def _reset_logging_state() -> None:
    """Reset module-level sentinel and saber logger between tests."""
    saber_logger = logging.getLogger("saber")
    original_handlers = list(saber_logger.handlers)
    original_level = saber_logger.level
    original_propagate = saber_logger.propagate

    saber_logging._CONFIGURED = False

    yield

    # Remove any handlers added during the test
    for handler in saber_logger.handlers[:]:
        if handler not in original_handlers:
            saber_logger.removeHandler(handler)
    saber_logger.setLevel(original_level)
    saber_logger.propagate = original_propagate
    saber_logging._CONFIGURED = False


@pytest.fixture()
def _with_inspect_handler() -> None:
    """Temporarily add a fake LogHandler to the root logger."""
    handler = _FakeLogHandler()
    logging.getLogger().addHandler(handler)
    yield
    logging.getLogger().removeHandler(handler)


# ---------------------------------------------------------------------------
# _has_inspect_handler
# ---------------------------------------------------------------------------


class TestHasInspectHandler:
    """Tests for the inspect_ai handler detection helper."""

    def test_returns_false_without_handler(self) -> None:
        assert _has_inspect_handler() is False

    @pytest.mark.usefixtures("_with_inspect_handler")
    def test_returns_true_with_handler(self) -> None:
        assert _has_inspect_handler() is True


# ---------------------------------------------------------------------------
# _resolve_level
# ---------------------------------------------------------------------------


class TestResolveLevel:
    """Tests for log level resolution."""

    def test_explicit_level_wins(self) -> None:
        assert _resolve_level(logging.WARNING) == logging.WARNING

    def test_env_var_used_when_no_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SABER_LOG_LEVEL", "error")
        assert _resolve_level(None) == logging.ERROR

    def test_defaults_to_debug(self) -> None:
        assert _resolve_level(None) == logging.DEBUG


# ---------------------------------------------------------------------------
# configure_logging — standalone mode (no inspect_ai handler)
# ---------------------------------------------------------------------------


class TestConfigureLoggingStandalone:
    """When no inspect_ai LogHandler is present, saber adds its own handler."""

    def test_adds_stream_handler(self) -> None:
        saber_logger = logging.getLogger("saber")
        initial_count = len(saber_logger.handlers)

        configure_logging(logging.INFO)

        assert len(saber_logger.handlers) == initial_count + 1
        assert isinstance(saber_logger.handlers[-1], logging.StreamHandler)

    def test_disables_propagation(self) -> None:
        configure_logging()

        saber_logger = logging.getLogger("saber")
        assert saber_logger.propagate is False

    def test_sets_level(self) -> None:
        configure_logging(logging.WARNING)

        saber_logger = logging.getLogger("saber")
        assert saber_logger.level == logging.WARNING

    def test_idempotent(self) -> None:
        saber_logger = logging.getLogger("saber")
        initial_count = len(saber_logger.handlers)

        configure_logging(logging.INFO)
        configure_logging(logging.INFO)

        assert len(saber_logger.handlers) == initial_count + 1

    def test_first_call_wins(self) -> None:
        configure_logging(logging.INFO)
        configure_logging(logging.DEBUG)

        saber_logger = logging.getLogger("saber")
        assert saber_logger.level == logging.INFO

    def test_child_loggers_inherit(self, capfd: pytest.CaptureFixture[str]) -> None:
        configure_logging(logging.DEBUG)

        child = logging.getLogger("saber.lifecycle")
        child.debug("test message from child")

        captured = capfd.readouterr()
        assert "test message from child" in captured.err

    def test_does_not_affect_root_logger(self) -> None:
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)

        configure_logging()

        assert root_logger.handlers == original_handlers

    def test_format_output(self, capfd: pytest.CaptureFixture[str]) -> None:
        configure_logging()

        test_logger = logging.getLogger("saber.test")
        test_logger.info("hello world")

        captured = capfd.readouterr()
        pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[INFO\] saber\.test: hello world"
        assert re.search(pattern, captured.err), f"Output did not match expected format: {captured.err!r}"


# ---------------------------------------------------------------------------
# configure_logging — under inspect_ai (LogHandler on root)
# ---------------------------------------------------------------------------


class TestConfigureLoggingUnderInspectAI:
    """When inspect_ai's LogHandler is present, saber delegates to it."""

    @pytest.mark.usefixtures("_with_inspect_handler")
    def test_no_handler_added(self) -> None:
        saber_logger = logging.getLogger("saber")
        initial_count = len(saber_logger.handlers)

        configure_logging()

        assert len(saber_logger.handlers) == initial_count

    @pytest.mark.usefixtures("_with_inspect_handler")
    def test_propagation_enabled(self) -> None:
        configure_logging()

        saber_logger = logging.getLogger("saber")
        assert saber_logger.propagate is True

    @pytest.mark.usefixtures("_with_inspect_handler")
    def test_level_set_to_debug_by_default(self) -> None:
        configure_logging()

        saber_logger = logging.getLogger("saber")
        assert saber_logger.level == logging.DEBUG

    @pytest.mark.usefixtures("_with_inspect_handler")
    def test_messages_reach_root_handler(self) -> None:
        """Verify saber messages propagate to root's LogHandler."""
        configure_logging()

        # Collect records that reach the root handler
        received: list[logging.LogRecord] = []
        for h in logging.getLogger().handlers:
            if type(h).__name__ == "LogHandler":
                original_emit = h.emit
                h.emit = lambda record: received.append(record)
                break

        logging.getLogger("saber.task").info("propagated message")

        assert len(received) == 1
        assert received[0].getMessage() == "propagated message"
        assert received[0].name == "saber.task"

        # Restore
        h.emit = original_emit  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for the get_logger() namespace helper."""

    def test_saber_prefix_passes_through(self) -> None:
        lgr = get_logger("saber.task")
        assert lgr.name == "saber.task"

    def test_bare_saber_passes_through(self) -> None:
        lgr = get_logger("saber")
        assert lgr.name == "saber"

    def test_unprefixed_name_gets_saber_prefix(self) -> None:
        lgr = get_logger("domains.crsbench.setup")
        assert lgr.name == "saber.domains.crsbench.setup"

    def test_dunder_name_from_saber_package(self) -> None:
        """Simulates ``get_logger(__name__)`` from saber.task."""
        lgr = get_logger("saber.scoring.factory")
        assert lgr.name == "saber.scoring.factory"

    def test_returns_logging_logger(self) -> None:
        lgr = get_logger("saber.test")
        assert isinstance(lgr, logging.Logger)

    def test_child_logger_inherits_config(self, capfd: pytest.CaptureFixture[str]) -> None:
        """A logger obtained via get_logger() inherits from the saber root."""
        configure_logging(logging.DEBUG)

        lgr = get_logger("saber.child.test")
        lgr.debug("child message via get_logger")

        captured = capfd.readouterr()
        assert "child message via get_logger" in captured.err


# ---------------------------------------------------------------------------
# display_progress
# ---------------------------------------------------------------------------


class TestDisplayProgress:
    """Tests for display_progress() — user-visible progress messages."""

    @pytest.mark.usefixtures("_with_inspect_handler")
    def test_writes_to_stderr_under_inspect_ai(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Under inspect_ai, message is written directly to stderr."""
        display_progress("Starting services...")

        captured = capfd.readouterr()
        assert "[SABER] Starting services..." in captured.err

    @pytest.mark.usefixtures("_with_inspect_handler")
    def test_logs_at_info_under_inspect_ai(self) -> None:
        """Under inspect_ai, message is also logged at INFO level."""
        # configure_logging detects the inspect handler and sets saber
        # logger level to DEBUG so INFO messages propagate.
        configure_logging()

        received: list[logging.LogRecord] = []
        for h in logging.getLogger().handlers:
            if type(h).__name__ == "LogHandler":
                original_emit = h.emit
                h.emit = lambda record: received.append(record)
                break

        display_progress("test info log")

        assert any(r.getMessage() == "test info log" for r in received)
        assert any(r.levelno == logging.INFO for r in received)

        h.emit = original_emit  # type: ignore[possibly-undefined]

    def test_standalone_no_stderr_prefix(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Without inspect_ai handler, no [SABER] prefix written to stderr."""
        configure_logging(logging.DEBUG)

        display_progress("standalone message")

        captured = capfd.readouterr()
        # The message should appear via the saber StreamHandler (in stderr),
        # but NOT with the [SABER] prefix (that's only for inspect_ai bypass).
        assert "[SABER] standalone message" not in captured.err
        # The logger INFO message should still appear via the StreamHandler.
        assert "standalone message" in captured.err

    def test_standalone_logs_at_info(self) -> None:
        """Without inspect_ai handler, message is logged at INFO."""
        configure_logging(logging.DEBUG)

        received: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = lambda record: received.append(record)  # type: ignore[assignment]
        logging.getLogger("saber").addHandler(handler)

        try:
            display_progress("standalone info")
            assert any(
                r.getMessage() == "standalone info" and r.levelno == logging.INFO for r in received
            )
        finally:
            logging.getLogger("saber").removeHandler(handler)
