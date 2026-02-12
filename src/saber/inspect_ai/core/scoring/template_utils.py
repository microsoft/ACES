"""Template utilities for SABER scoring system.

Provides helpers for fetching and rendering Jinja2 templates for evaluation.
"""

from typing import Any

from jinja2 import BaseLoader, Environment, TemplateError

from ....logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


class TemplateStringLoader(BaseLoader):
    """Load Jinja2 templates from string content."""

    def __init__(self, templates: dict[str, str]):
        """Initialize template loader.

        Args:
            templates: Dict mapping template names to template content strings
        """
        self.templates = templates

    def get_source(self, environment: Any, template: str) -> tuple:
        """Get template source.

        Args:
            environment: Jinja2 environment
            template: Template name

        Returns:
            Tuple of (source, filename, uptodate_function)

        Raises:
            TemplateError: If template not found
        """
        if template in self.templates:
            return self.templates[template], None, lambda: True
        raise TemplateError(f"Template not found: {template}")


async def resolve_template_content(session_manager: Any, template_value: str) -> str:
    """Resolve a template value that may be a file path or inline content.

    The SABER server only auto-resolves template file paths for the built-in
    ``llm_judge`` strategy.  Custom strategies receive the raw YAML value,
    which is typically a relative file path (e.g. ``judge/state_diff_system.md``).

    This helper detects whether the value is already template content (contains
    Jinja2 syntax or newlines) or a file path, and fetches the content from the
    server when needed.

    Args:
        session_manager: Client session manager with ``get_template_content``
        template_value: Either inline template content or a server-relative path

    Returns:
        Resolved template content string
    """
    # If it contains Jinja2 syntax or newlines, treat as inline content
    if "{{" in template_value or "\n" in template_value:
        return template_value

    # Otherwise treat as a file path and fetch from the server
    try:
        content: str = await session_manager.get_template_content(template_value)
        return content
    except Exception as exc:
        logger.warning(f"Could not fetch template '{template_value}': {exc}")
        return template_value


async def fetch_and_render_template(
    session_manager: Any,
    state: Any,
    template_path: str,
    context: dict[str, Any],
) -> str:
    """Fetch template from server and render with Jinja2.

    Args:
        session_manager: Client session manager for fetching templates
        state: Task state
        template_path: Path to template on server
        context: Template rendering context

    Returns:
        Rendered template string
    """
    from inspect_ai.util import store

    task_store = store()
    session_id = task_store.get("saber_session_id")

    # Fetch template content from server
    template_content = await session_manager.get_template_content(session_id, template_path)

    # Create Jinja2 environment and render
    env = Environment(loader=TemplateStringLoader({template_path: template_content}))
    template = env.get_template(template_path)
    rendered: str = template.render(**context)

    return rendered
