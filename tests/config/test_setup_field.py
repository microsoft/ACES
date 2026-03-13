"""Tests for the setup field on TaskConfig, GlobalDefaults, and converter mapping."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

from inspect_ai.dataset import Sample

from saber.config.converter import tasks_to_samples
from saber.config.loader import deep_merge
from saber.config.models import (
    GlobalDefaults,
    PromptPaths,
    TaskConfig,
)
from saber.prompts.renderer import PromptRenderer

# ── Helpers ─────────────────────────────────────────────────────────


def _minimal_prompts() -> PromptPaths:
    return PromptPaths(instruction="instructions/demo.j2")


def _minimal_task(**overrides: object) -> TaskConfig:
    defaults: dict[str, object] = {
        "task_id": "t1",
        "title": "T",
        "description": "D",
        "prompts": _minimal_prompts(),
    }
    defaults.update(overrides)
    return TaskConfig(**defaults)  # type: ignore[arg-type]


def _mock_renderer() -> PromptRenderer:
    renderer = create_autospec(PromptRenderer, instance=True)

    def _render_all(prompts: object, task: object) -> dict[str, str]:
        result: dict[str, str] = {}
        for key in ("instruction", "assistant"):
            path = getattr(prompts, key, None)
            if path:
                result[key] = f"rendered:{path}"
        return result

    renderer.render_all_prompts.side_effect = _render_all
    return renderer


# ── TaskConfig.setup ────────────────────────────────────────────────


class TestTaskConfigSetup:
    """TaskConfig accepts an optional setup field."""

    def test_defaults_to_none(self) -> None:
        tc = _minimal_task()
        assert tc.setup is None

    def test_accepts_string(self) -> None:
        tc = _minimal_task(setup="git init /repo")
        assert tc.setup == "git init /repo"

    def test_accepts_none_explicitly(self) -> None:
        tc = _minimal_task(setup=None)
        assert tc.setup is None

    def test_empty_string_is_valid(self) -> None:
        tc = _minimal_task(setup="")
        assert tc.setup == ""


# ── GlobalDefaults.setup ────────────────────────────────────────────


class TestGlobalDefaultsSetup:
    """GlobalDefaults accepts an optional setup field."""

    def test_defaults_to_none(self) -> None:
        gd = GlobalDefaults()
        assert gd.setup is None

    def test_accepts_string(self) -> None:
        gd = GlobalDefaults(setup="echo hello")
        assert gd.setup == "echo hello"

    def test_accepts_none_explicitly(self) -> None:
        gd = GlobalDefaults(setup=None)
        assert gd.setup is None

    def test_empty_string_is_valid(self) -> None:
        gd = GlobalDefaults(setup="")
        assert gd.setup == ""


# ── deep_merge cascade for setup ────────────────────────────────────


class TestSetupCascade:
    """Setup field cascades correctly via deep_merge."""

    def test_global_has_setup_task_omits(self) -> None:
        """Task inherits setup from global when task doesn't specify it."""
        base = {"setup": "git init /repo"}
        override: dict[str, object] = {"task_id": "t1"}
        result = deep_merge(base, override)
        assert result["setup"] == "git init /repo"

    def test_task_overrides_global(self) -> None:
        """Task setup replaces global setup."""
        base = {"setup": "git init /repo"}
        override = {"setup": "custom_setup.sh"}
        result = deep_merge(base, override)
        assert result["setup"] == "custom_setup.sh"

    def test_task_null_does_not_erase(self) -> None:
        """None in override doesn't erase base (deep_merge rule)."""
        base = {"setup": "git init /repo"}
        override: dict[str, object] = {"setup": None}
        result = deep_merge(base, override)
        assert result["setup"] == "git init /repo"

    def test_empty_string_overrides_inherited(self) -> None:
        """Empty string in task disables inherited setup (opt-out mechanism).

        Since deep_merge skips None, tasks must use ``setup: ""`` to
        explicitly disable an inherited setup script.
        """
        base = {"setup": "git init /repo"}
        override = {"setup": ""}
        result = deep_merge(base, override)
        assert result["setup"] == ""


# ── Converter maps setup to Sample ──────────────────────────────────


class TestConverterSetup:
    """Converter maps TaskConfig.setup → Sample.setup."""

    def test_setup_present(self, tmp_path: Path) -> None:
        task = _minimal_task(setup="git init /repo")
        renderer = _mock_renderer()
        samples = tasks_to_samples([task], tmp_path, renderer)
        assert len(samples) == 1
        assert samples[0].setup == "git init /repo"

    def test_setup_absent(self, tmp_path: Path) -> None:
        task = _minimal_task()
        renderer = _mock_renderer()
        samples = tasks_to_samples([task], tmp_path, renderer)
        assert len(samples) == 1
        assert samples[0].setup is None

    def test_setup_empty_string(self, tmp_path: Path) -> None:
        """Empty string setup propagates (not converted to None)."""
        task = _minimal_task(setup="")
        renderer = _mock_renderer()
        samples = tasks_to_samples([task], tmp_path, renderer)
        assert len(samples) == 1
        assert samples[0].setup == ""

    def test_setup_multiline(self, tmp_path: Path) -> None:
        script = "git init /repo\ngit add .\ngit commit -m init"
        task = _minimal_task(setup=script)
        renderer = _mock_renderer()
        samples = tasks_to_samples([task], tmp_path, renderer)
        assert samples[0].setup == script
