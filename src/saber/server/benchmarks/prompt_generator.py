"""Prompt generation service for SABER benchmark tasks.

Responsibilities:
    * Load and render Jinja2 templates for task-specific prompts.
    * Fail fast on: missing template, unsafe template name, missing include/extends
      dependency, undefined variable, or missing required task configuration.
    * Enforce SABER principles: no silent fallbacks, explicit configuration, clear errors.

Priority Fixes Implemented:
    1. Sandboxed Jinja2 environment + StrictUndefined.
    2. Path safety validation (reject '..' or absolute paths) to prevent traversal.
    3. Recursive dependency validation for includes / extends at validation time.
    4. Removal of silent defaults for timeout, max_steps, allowed_executors (now mandatory for prompt generation).
    5. Clear, aggregated errors for context + template validation.
    6. Distinguish missing root template vs missing included template.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from jinja2 import FileSystemLoader, StrictUndefined, TemplateError, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

from .task import Task

logger = logging.getLogger(__name__)


class PromptGenerationError(Exception):
    """Raised when prompt generation fails (render phase or context construction)."""


class TemplateValidationError(Exception):
    """Raised when template validation fails during startup or explicit validation."""


class PromptContextError(PromptGenerationError):
    """Raised when required task configuration for prompt context is missing."""


@dataclass
class PromptContext:
    """Data container for template rendering context."""

    domain: str
    task_id: str
    task_title: str
    task_description: str
    timeout_seconds: int
    max_steps: int
    environment: str
    subtasks: List[Dict[str, Any]]
    allowed_executors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Jinja2 template rendering."""
        return {
            "domain": self.domain,
            "task_id": self.task_id,
            "task_title": self.task_title,
            "task_description": self.task_description,
            "timeout_seconds": self.timeout_seconds,
            "max_steps": self.max_steps,
            "environment": self.environment,
            "subtasks": self.subtasks,
            "allowed_executors": self.allowed_executors,
        }


class PromptGenerator:
    """
    Handles loading and rendering of prompt templates for tasks.

    Validates all templates at startup and fails fast on missing or invalid templates.
    Uses Jinja2 templating engine with FileSystemLoader for template management.
    """

    def __init__(self, prompts_dir: str):
        """
        Initialize PromptGenerator with template directory.

        Args:
            prompts_dir: Directory containing Jinja2 template files

        Raises:
            TemplateValidationError: If prompts directory doesn't exist
        """
        self.prompts_dir = Path(prompts_dir)

        # Fail fast if prompts directory doesn't exist
        if not self.prompts_dir.exists():
            raise TemplateValidationError(f"Prompts directory does not exist: {self.prompts_dir}")

        if not self.prompts_dir.is_dir():
            raise TemplateValidationError(f"Prompts path is not a directory: {self.prompts_dir}")

        # Initialize Jinja2 environment
        self.jinja_env = SandboxedEnvironment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=False,  # Deliberately off: prompts often require raw control characters
            undefined=StrictUndefined,
            trim_blocks=False,
            lstrip_blocks=False,
        )

        logger.info(f"PromptGenerator initialized with templates from: {self.prompts_dir}")

    def render_prompt_for_task(self, task: Task) -> str:
        """
        Render prompt for a specific task using its template file.

        Args:
            task: Task object with prompt_template_file specified

        Returns:
            Rendered prompt string ready for PolicyManager

        Raises:
            PromptGenerationError: If template rendering fails
            TemplateValidationError: If template file is missing or invalid
        """
        if not task.prompt_template_file:
            raise PromptGenerationError(f"Task '{task.task_id}' missing required prompt_template_file")

        self._assert_safe_template_name(task.prompt_template_file)

        try:
            # Load template - fail fast if not found
            template = self.jinja_env.get_template(task.prompt_template_file)

            # Build rendering context from task
            context = self._build_context_from_task(task)

            # Render template with context
            rendered_prompt = str(template.render(context.to_dict()))

            logger.debug(
                f"Successfully rendered prompt for task '{task.task_id}' using template '{task.prompt_template_file}'"
            )
            return rendered_prompt

        except TemplateNotFound as e:
            # Could be root template or an included template
            missing_name = getattr(e, "name", task.prompt_template_file)
            location = "included template" if missing_name != task.prompt_template_file else "template file"
            raise TemplateValidationError(
                f"{location} not found for task '{task.task_id}': {missing_name}. " f"Expected under {self.prompts_dir}"
            ) from e

        except TemplateError as e:
            raise PromptGenerationError(
                f"Template rendering failed for task '{task.task_id}' "
                f"using template '{task.prompt_template_file}': {e}"
            ) from e

        except Exception as e:
            raise PromptGenerationError(f"Unexpected error rendering prompt for task '{task.task_id}': {e}") from e

    def validate_template(self, template_file: str) -> bool:
        """
        Validate template syntax and file existence.

        Args:
            template_file: Template filename to validate

        Returns:
            True if template is valid

        Raises:
            TemplateValidationError: If template is invalid or missing
        """
        self._assert_safe_template_name(template_file)
        try:
            loader = self.jinja_env.loader
            if loader is None:
                raise TemplateValidationError("No loader configured for template environment")

            visited: Set[str] = set()
            missing: List[str] = []

            def _collect(name: str) -> Tuple[str, str]:
                self._assert_safe_template_name(name)
                source, _, _ = loader.get_source(self.jinja_env, name)
                return name, source

            def _walk(name: str) -> None:
                if name in visited:
                    return
                visited.add(name)
                try:
                    _, src = _collect(name)
                except TemplateNotFound:
                    missing.append(name)
                    return
                deps = self._extract_template_dependencies(src)
                for dep in deps:
                    _walk(dep)

            _walk(template_file)
            if missing:
                raise TemplateValidationError(
                    f"Template '{template_file}' has missing dependencies (recursive): {missing}"
                )
            # Compile root template only (Jinja2 will parse dependency syntax during traversal above)
            self.jinja_env.get_template(template_file)
            logger.debug("Template validation successful: %s (deps=%d)", template_file, len(visited) - 1)
            return True

        except TemplateNotFound as e:
            raise TemplateValidationError(f"Template file not found: {template_file}") from e

        except TemplateError as e:
            raise TemplateValidationError(f"Template syntax error in {template_file}: {e}") from e

    def validate_all_task_templates(self, tasks: List[Task]) -> None:
        """
        Validate all templates referenced by tasks at startup.

        Args:
            tasks: List of tasks to validate templates for

        Raises:
            TemplateValidationError: If any template is missing or invalid
        """
        logger.info(f"Validating templates for {len(tasks)} tasks...")

        missing_templates = []
        invalid_templates = []

        for task in tasks:
            if not task.prompt_template_file:
                missing_templates.append(f"Task '{task.task_id}' missing prompt_template_file")
                continue

            try:
                self.validate_template(task.prompt_template_file)
            except TemplateValidationError as e:
                invalid_templates.append(str(e))

        # Collect all errors and fail fast with complete list
        errors = missing_templates + invalid_templates
        if errors:
            error_msg = "Template validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise TemplateValidationError(error_msg)

        logger.info(f"All {len(tasks)} task templates validated successfully")

    def _build_context_from_task(self, task: Task) -> PromptContext:
        """
        Build rendering context from task object.

        Args:
            task: Task object to extract context from

        Returns:
            PromptContext with all variables needed for template rendering
        """
        missing: List[str] = []
        # timeout must exist
        if not (task.execution_config and "timeout" in task.execution_config):
            missing.append("execution_config.timeout")
        # max_steps must exist
        if not (task.episode_config and "max_steps" in task.episode_config):
            missing.append("episode_config.max_steps")

        # allowed executors must be explicit (task.allowed_executors is authoritative)
        if not task.allowed_executors:
            missing.append("allowed_executors")

        if missing:
            raise PromptContextError(
                f"Task '{task.task_id}' missing required configuration fields for prompt context: {missing}"
            )

        timeout_seconds = task.execution_config["timeout"]
        max_steps = task.episode_config["max_steps"]

        # Convert environment to string representation
        environment_str = str(task.environment) if task.environment else "default"

        # Convert subtasks to serializable format
        subtasks_data = []
        if task.subtasks:
            for subtask in task.subtasks:
                subtasks_data.append(
                    {
                        "title": subtask.title,
                        "description": subtask.description,
                        "objective": subtask.objective,
                    }
                )
        # Allowed executors now required (validated above)
        allowed_executors = task.allowed_executors or []

        return PromptContext(
            domain=task.domain,
            task_id=task.task_id,
            task_title=task.title,
            task_description=task.description,
            timeout_seconds=int(timeout_seconds),
            max_steps=int(max_steps),
            environment=environment_str,
            subtasks=subtasks_data,
            allowed_executors=allowed_executors,
        )

    # -------------------- Private Helpers --------------------
    _INCLUDE_RE = re.compile(r"{%\s*include\s*['\"]([^'\"]+)['\"]\s*%}")
    _EXTENDS_RE = re.compile(r"{%\s*extends\s*['\"]([^'\"]+)['\"]\s*%}")

    def _extract_template_dependencies(self, source: str) -> List[str]:
        """Extract direct include/extends dependencies from a template source string."""
        deps: Set[str] = set()
        for regex in (self._INCLUDE_RE, self._EXTENDS_RE):
            for match in regex.findall(source):
                deps.add(match)
        return list(deps)

    def _assert_safe_template_name(self, name: str) -> None:
        """Validate template filename to prevent path traversal or symlink escape.

        Enforces:
          * Not absolute
          * No segment == '..'
          * Resolved path within prompts_dir (symlink safe)
          * Only markdown / text-like extensions (defensive narrowing)
        """
        if name.startswith(("/", "\\")):
            raise TemplateValidationError(f"Unsafe template path (absolute not allowed): {name}")
        parts = name.split("/")
        if any(p == ".." for p in parts):
            raise TemplateValidationError(f"Unsafe template path contains '..': {name}")
        candidate = (self.prompts_dir / name).resolve()
        root = self.prompts_dir.resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise TemplateValidationError(f"Template path escapes root: {name}")
        # Basic extension allow-list
        if not candidate.suffix.lower() in {".md", ".txt", ".jinja", ".j2"}:
            raise TemplateValidationError(
                f"Disallowed template file extension for '{name}'. Allowed: .md,.txt,.jinja,.j2"
            )

    # Public helper (optional usage in future wiring)
    def validate_task_context(self, task: Task) -> None:
        """Validate that task has all required config for prompt context (fail-fast pre-render)."""
        try:
            self._build_context_from_task(task)
        except PromptContextError as e:
            raise TemplateValidationError(str(e)) from e
