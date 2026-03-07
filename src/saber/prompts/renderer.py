"""Secure Jinja2 prompt rendering for SABER benchmark tasks.

Uses ``SandboxedEnvironment`` to prevent template injection and
validates all template paths before loading.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from jinja2 import StrictUndefined, select_autoescape
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field

from saber.config.models import InitialContext, ScorerConfig, to_template_vars

if TYPE_CHECKING:
    from pathlib import Path

    from saber.config.models import PromptPaths, TaskConfig


class PromptTemplateContext(BaseModel):
    """Typed context for prompt template rendering (instruction/assistant)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    title: str
    description: str
    initial_context: InitialContext
    scorers: tuple[ScorerConfig, ...] = ()
    tools: tuple[str, ...]
    max_steps: int
    initial_files: dict[str, str] = Field(default_factory=dict)


# Extensions allowed for template files
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".j2", ".md", ".txt", ".jinja"})


def _assert_safe_template_path(template_path: str) -> None:
    """Validate that a template path is safe to load.

    Rejects:
    - Null bytes in the path
    - Path traversal (``..`` component)
    - Absolute paths
    - Disallowed file extensions

    Raises:
        ValueError: If the path is unsafe.
    """
    if "\x00" in template_path:
        msg = f"Null byte in template path: {template_path!r}"
        raise ValueError(msg)

    p = PurePosixPath(template_path)

    if ".." in p.parts:
        msg = f"Path traversal detected in template path: {template_path!r}"
        raise ValueError(msg)

    if p.is_absolute():
        msg = f"Absolute paths are not allowed, must be relative: {template_path!r}"
        raise ValueError(msg)

    suffix = p.suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        msg = (
            f"Disallowed extension {suffix!r} in template path: {template_path!r}. "
            f"Allowed extensions: {sorted(_ALLOWED_EXTENSIONS)}"
        )
        raise ValueError(msg)


class PromptRenderer:
    """Renders Jinja2 prompt templates with task context.

    Uses ``SandboxedEnvironment`` for security and ``StrictUndefined``
    to fail fast on missing variables.
    """

    def __init__(self, prompts_dir: Path) -> None:
        from jinja2 import FileSystemLoader

        self._prompts_dir = prompts_dir.resolve()
        self._env = SandboxedEnvironment(
            loader=FileSystemLoader(str(self._prompts_dir)),
            undefined=StrictUndefined,
            autoescape=select_autoescape([]),
            keep_trailing_newline=True,
        )

    def render(self, template_path: str, context: dict[str, object]) -> str:
        """Render a single template file with the given context.

        Args:
            template_path: Relative path to the template inside the prompts
                directory (e.g. ``"instructions/demo.j2"``).
            context: Variable mapping passed to Jinja2.

        Returns:
            The rendered template string.

        Raises:
            ValueError: If *template_path* fails safety validation.
            jinja2.TemplateNotFound: If the template file does not exist.
            jinja2.UndefinedError: If a required variable is missing.
        """
        _assert_safe_template_path(template_path)
        template = self._env.get_template(template_path)
        return str(template.render(context))

    def render_all_prompts(self, prompts: PromptPaths, task: TaskConfig) -> dict[str, str]:
        """Render all four prompt types for a task.

        Args:
            prompts: Paths to the four prompt template files.
            task: The task configuration providing template context.

        Returns:
            A dict mapping prompt name to rendered string with keys:
            ``instruction``, ``assistant``.
        """
        ctx = self.build_task_context(task)
        ctx_dict = to_template_vars(ctx)

        _PROMPT_ATTR_MAP: dict[str, str] = {
            "instruction": "instruction",
            "assistant": "assistant",
        }

        result: dict[str, str] = {}
        for render_key, attr_name in _PROMPT_ATTR_MAP.items():
            path = getattr(prompts, attr_name, None)
            if path:
                result[render_key] = self.render(path, ctx_dict)
        return result

    def build_task_context(self, task: TaskConfig) -> PromptTemplateContext:
        """Build a typed Jinja2 context from a task configuration.

        Args:
            task: The task configuration.

        Returns:
            A :class:`PromptTemplateContext` for template rendering.
        """
        return PromptTemplateContext(
            task_id=task.task_id,
            title=task.title,
            description=task.description,
            initial_context=task.initial_context,
            scorers=tuple(task.scorers),
            tools=tuple(task.tools.keys()),
            max_steps=task.max_steps,
            initial_files=task.initial_files,
        )
