# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for saber.prompts.renderer — PromptRenderer and path validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import UndefinedError
from jinja2.exceptions import TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

from saber.config.models import (
    InitialContext,
    PromptPaths,
    ScorerConfig,
    ScorerTarget,
    TaskConfig,
    ToolConfig,
)
from saber.prompts.renderer import (
    PromptRenderer,
    PromptTemplateContext,
    _assert_safe_template_path,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with test templates."""
    instructions = tmp_path / "instructions"
    instructions.mkdir()
    (instructions / "demo.j2").write_text("Hello {{ name }}!\n")

    (instructions / "task_ctx.j2").write_text("Task {{ task_id }}: {{ title }}\n")

    shared = instructions / "shared"
    shared.mkdir()
    (shared / "common_guidelines.j2").write_text("Follow the rules.\n")

    (instructions / "with_include.j2").write_text(
        "Intro\n{% include 'instructions/shared/common_guidelines.j2' %}\nDone\n"
    )

    (instructions / "nested_ctx.j2").write_text("DB: {{ initial_context.database }}\n")

    (instructions / "loop.j2").write_text("{% for t in tools %}Tool: {{ t }}\n{% endfor %}")

    (instructions / "conditional.j2").write_text("{% if show %}Visible{% else %}Hidden{% endif %}\n")

    assistants = tmp_path / "assistants"
    assistants.mkdir()
    (assistants / "inspect_assistant.j2").write_text("You are an assistant.\n")

    judge = tmp_path / "judge"
    judge.mkdir()
    (judge / "system.txt").write_text("Judge system prompt.\n")

    deep = tmp_path / "deep" / "nested"
    deep.mkdir(parents=True)
    (deep / "file.j2").write_text("Deep {{ value }}\n")

    (tmp_path / "plain.jinja").write_text("Plain {{ x }}\n")

    return tmp_path


@pytest.fixture()
def renderer(prompts_dir: Path) -> PromptRenderer:
    """PromptRenderer wired to the temporary prompts directory."""
    return PromptRenderer(prompts_dir)


def _minimal_task(
    prompts_dir: Path,  # noqa: ARG001 – used only to satisfy path existence
) -> TaskConfig:
    """Build a minimal TaskConfig for testing render_all_prompts."""
    return TaskConfig(
        task_id="test-task-1",
        title="Test Task",
        description="A task for testing",
        prompts=PromptPaths(
            instruction="instructions/task_ctx.j2",
            assistant="assistants/inspect_assistant.j2",
        ),
        initial_context={"incident": "breach"},
        tools={"run_command": ToolConfig(), "read_file": ToolConfig(timeout=60)},
        max_steps=30,
    )


# ── _assert_safe_template_path ──────────────────────────────────────


class TestAssertSafeTemplatePath:
    """Tests for the module-level path validation helper."""

    # -- valid paths --------------------------------------------------

    @pytest.mark.parametrize(
        "path",
        [
            "instructions/demo.j2",
            "judge/system.txt",
            "template.jinja",
            "deep/nested/file.j2",
            "simple.j2",
        ],
    )
    def test_valid_paths_accepted(self, path: str) -> None:
        _assert_safe_template_path(path)  # should not raise

    # -- null byte rejection ------------------------------------------

    def test_null_byte_rejected(self) -> None:
        with pytest.raises(ValueError, match="[Nn]ull"):
            _assert_safe_template_path("instructions/demo\x00.j2")

    def test_null_byte_at_start(self) -> None:
        with pytest.raises(ValueError, match="[Nn]ull"):
            _assert_safe_template_path("\x00demo.j2")

    # -- path traversal rejection -------------------------------------

    @pytest.mark.parametrize(
        "path",
        [
            "../etc/passwd",
            "foo/../bar.j2",
            "../../secret.j2",
            "a/b/../../c.j2",
        ],
    )
    def test_path_traversal_rejected(self, path: str) -> None:
        with pytest.raises(ValueError, match="[Tt]raversal|\\.\\."):
            _assert_safe_template_path(path)

    # -- absolute path rejection --------------------------------------

    @pytest.mark.parametrize(
        "path",
        [
            "/etc/passwd",
            "/tmp/test.j2",
        ],
    )
    def test_absolute_path_rejected(self, path: str) -> None:
        with pytest.raises(ValueError, match="[Aa]bsolute|[Rr]elative"):
            _assert_safe_template_path(path)

    # -- disallowed extension rejection --------------------------------

    @pytest.mark.parametrize(
        "path",
        [
            "script.py",
            "config.yaml",
            "page.html",
            "run.sh",
            "noextension",
        ],
    )
    def test_disallowed_extension_rejected(self, path: str) -> None:
        with pytest.raises(ValueError, match="[Ee]xtension"):
            _assert_safe_template_path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "instructions/demo.MD",
            "template.TXT",
            "file.Jinja",
            "nested/dir/file.J2",
        ],
    )
    def test_uppercase_extensions_accepted(self, path: str) -> None:
        """Extension check is case-insensitive."""
        _assert_safe_template_path(path)  # should not raise


# ── PromptRenderer.__init__ ─────────────────────────────────────────


class TestPromptRendererInit:
    """Construction / initialisation tests."""

    def test_creates_with_valid_directory(self, prompts_dir: Path) -> None:
        renderer = PromptRenderer(prompts_dir)
        assert renderer._prompts_dir == prompts_dir.resolve()

    def test_uses_sandboxed_environment(self, renderer: PromptRenderer) -> None:
        assert isinstance(renderer._env, SandboxedEnvironment)


# ── PromptRenderer.render ────────────────────────────────────────────


class TestPromptRendererRender:
    """Tests for rendering individual templates."""

    def test_simple_variable_substitution(self, renderer: PromptRenderer) -> None:
        result = renderer.render("instructions/demo.j2", {"name": "Alice"})
        assert result == "Hello Alice!\n"

    def test_nested_dict_access(self, renderer: PromptRenderer) -> None:
        ctx = {"initial_context": {"database": "postgres://db:5432"}}
        result = renderer.render("instructions/nested_ctx.j2", ctx)
        assert result == "DB: postgres://db:5432\n"

    def test_jinja2_include(self, renderer: PromptRenderer) -> None:
        result = renderer.render("instructions/with_include.j2", {})
        assert "Follow the rules." in result
        assert "Intro" in result
        assert "Done" in result

    def test_loop(self, renderer: PromptRenderer) -> None:
        ctx = {"tools": ["run_command", "read_file"]}
        result = renderer.render("instructions/loop.j2", ctx)
        assert "Tool: run_command" in result
        assert "Tool: read_file" in result

    def test_conditional(self, renderer: PromptRenderer) -> None:
        assert "Visible" in renderer.render("instructions/conditional.j2", {"show": True})
        assert "Hidden" in renderer.render("instructions/conditional.j2", {"show": False})

    def test_deep_nested_template(self, renderer: PromptRenderer) -> None:
        result = renderer.render("deep/nested/file.j2", {"value": 42})
        assert result == "Deep 42\n"

    def test_plain_jinja_extension(self, renderer: PromptRenderer) -> None:
        result = renderer.render("plain.jinja", {"x": "yes"})
        assert result == "Plain yes\n"

    def test_txt_extension(self, renderer: PromptRenderer) -> None:
        result = renderer.render("judge/system.txt", {})
        assert result == "Judge system prompt.\n"

    def test_missing_template_raises(self, renderer: PromptRenderer) -> None:
        with pytest.raises(TemplateNotFound):
            renderer.render("nonexistent.j2", {})

    def test_undefined_variable_raises(self, renderer: PromptRenderer) -> None:
        with pytest.raises(UndefinedError):
            renderer.render("instructions/demo.j2", {})  # missing 'name'

    def test_unsafe_path_raises_before_load(self, renderer: PromptRenderer) -> None:
        with pytest.raises(ValueError, match="[Tt]raversal|\\.\\."):
            renderer.render("../etc/passwd", {})

    def test_absolute_path_raises(self, renderer: PromptRenderer) -> None:
        with pytest.raises(ValueError, match="[Aa]bsolute|[Rr]elative"):
            renderer.render("/etc/passwd", {})

    def test_trailing_newline_preserved(self, renderer: PromptRenderer) -> None:
        result = renderer.render("instructions/demo.j2", {"name": "Bob"})
        assert result.endswith("\n")


# ── PromptRenderer.build_task_context ────────────────────────────────


class TestTaskContext:
    """Tests for building Jinja2 context from TaskConfig."""

    def test_returns_prompt_template_context(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        ctx = renderer.build_task_context(task)
        assert isinstance(ctx, PromptTemplateContext)

    def test_has_correct_fields(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        ctx = renderer.build_task_context(task)
        expected_fields = {
            "task_id",
            "title",
            "description",
            "initial_context",
            "scorers",
            "tools",
            "max_steps",
            "initial_files",
        }
        assert set(ctx.model_fields.keys()) == expected_fields

    def test_task_id_propagated(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        ctx = renderer.build_task_context(task)
        assert ctx.task_id == "test-task-1"

    def test_tools_listed_as_names(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        ctx = renderer.build_task_context(task)
        assert ctx.tools == ("run_command", "read_file")

    def test_max_steps(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        ctx = renderer.build_task_context(task)
        assert ctx.max_steps == 30

    def test_initial_context(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        ctx = renderer.build_task_context(task)
        assert isinstance(ctx.initial_context, InitialContext)
        assert ctx.initial_context.incident == "breach"  # type: ignore[attr-defined]


# ── PromptRenderer.render_all_prompts ────────────────────────────────


class TestRenderAllPrompts:
    """Tests for rendering all four prompt types for a task."""

    def test_returns_dict_with_expected_keys(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        result = renderer.render_all_prompts(task.prompts, task)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "instruction",
            "assistant",
        }
        for value in result.values():
            assert isinstance(value, str)

    def test_instruction_rendered_with_context(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        result = renderer.render_all_prompts(task.prompts, task)
        assert result["instruction"] == "Task test-task-1: Test Task\n"

    def test_all_prompts_content(self, prompts_dir: Path) -> None:
        """Use templates that reference task context variables."""
        # Overwrite instruction template to use task_id
        (prompts_dir / "instructions" / "ctx_demo.j2").write_text("Task: {{ task_id }} — {{ title }}\n")
        task = TaskConfig(
            task_id="ctx-task",
            title="Context Test",
            description="desc",
            prompts=PromptPaths(
                instruction="instructions/ctx_demo.j2",
                assistant="assistants/inspect_assistant.j2",
            ),
        )
        renderer = PromptRenderer(prompts_dir)
        result = renderer.render_all_prompts(task.prompts, task)
        assert result["instruction"] == "Task: ctx-task — Context Test\n"
        assert result["assistant"] == "You are an assistant.\n"
        assert "submit" not in result

    def test_context_includes_all_expected_keys(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        """Ensure render_all_prompts merges full task context."""
        (prompts_dir / "instructions" / "keys.j2").write_text("{{ task_id }},{{ max_steps }},{{ tools | length }}\n")
        task = TaskConfig(
            task_id="key-check",
            title="T",
            description="D",
            prompts=PromptPaths(
                instruction="instructions/keys.j2",
                assistant="assistants/inspect_assistant.j2",
            ),
            tools={"a": ToolConfig(), "b": ToolConfig()},
            max_steps=10,
        )
        renderer2 = PromptRenderer(prompts_dir)
        result = renderer2.render_all_prompts(task.prompts, task)
        assert result["instruction"] == "key-check,10,2\n"


# ── PromptTemplateContext with scorers ───────────────────────────────


class TestPromptContextNew:
    """Tests for PromptTemplateContext with scorers field."""

    def test_context_has_scorers_field(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = TaskConfig(
            task_id="scorer-task",
            title="Scorer Task",
            description="A task with scorers",
            prompts=PromptPaths(
                instruction="instructions/task_ctx.j2",
                assistant="assistants/inspect_assistant.j2",
            ),
            scorers=(
                ScorerConfig(
                    scorer_name="sub",
                    strategy="static",
                    target=ScorerTarget.SUBMISSION,
                ),
            ),
        )
        ctx = renderer.build_task_context(task)
        assert isinstance(ctx.scorers, tuple)
        assert len(ctx.scorers) == 1
        assert isinstance(ctx.scorers[0], ScorerConfig)
        assert ctx.scorers[0].scorer_name == "sub"

    def test_context_scorers_empty_by_default(self, renderer: PromptRenderer, prompts_dir: Path) -> None:
        task = _minimal_task(prompts_dir)
        ctx = renderer.build_task_context(task)
        assert ctx.scorers == ()
