"""Judge prompt template renderer for scoring strategies."""

from __future__ import annotations

from pathlib import Path

from jinja2 import BaseLoader, DictLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment


class TemplateRenderer:
    """Renders Jinja2 judge prompt templates for scoring strategies.

    Supports two modes:

    - **File-based**: loads templates from a directory (production).
    - **Dict-based**: uses in-memory template strings (testing).
    """

    def __init__(
        self,
        templates_dir: Path | None = None,
        templates: dict[str, str] | None = None,
    ) -> None:
        if templates is not None:
            loader: BaseLoader = DictLoader(templates)
        elif templates_dir is not None:
            from jinja2 import FileSystemLoader

            loader = FileSystemLoader(str(templates_dir.resolve()))
        else:
            loader = DictLoader({})

        self._env = SandboxedEnvironment(
            loader=loader,
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )

    def render(self, template_name: str, context: dict[str, object]) -> str:
        """Render a template with the given context.

        Args:
            template_name: Template name/path.
            context: Variables for the template.

        Returns:
            Rendered template string.

        Raises:
            jinja2.TemplateNotFound: If template doesn't exist.
            jinja2.UndefinedError: If a required variable is missing.
        """
        template = self._env.get_template(template_name)
        return str(template.render(context))
