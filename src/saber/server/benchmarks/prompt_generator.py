"""Prompt generation service for SABER benchmark tasks.

Responsibilities:
    * Load and render Jinja2 templates for both agent prompts and judge prompts
    * Agent prompts: Task-specific prompts for AI agents during episode execution
    * Judge prompts: LLM evaluation prompts for scoring agent submissions
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
from typing import Any, Dict, List, Optional, Set, Tuple

from jinja2 import FileSystemLoader, StrictUndefined, TemplateError, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

from ..base import Episode
from .task import Task

logger = logging.getLogger(__name__)


class PromptGenerationError(Exception):
    """Raised when prompt generation fails (render phase or context construction)."""


class TemplateValidationError(Exception):
    """Raised when template validation fails during startup or explicit validation."""


class PromptContextError(PromptGenerationError):
    """Raised when required task configuration for prompt context is missing."""


@dataclass
class JudgePromptPayload:
    """Complete LLM prompt payload for judge evaluation."""

    messages: List[Dict[str, str]]  # OpenAI messages format: [{"role": "system", "content": "..."}, ...]
    model: str  # LLM model to use
    task_id: str  # Task identifier for logging
    episode_id: str  # Episode identifier for logging

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for OpenAI API call."""
        return {
            "messages": self.messages,
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 500,
        }


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
    initial_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Jinja2 template rendering."""
        result = {
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

        # Add initial_context if provided
        if self.initial_context:
            result["initial_context"] = self.initial_context

        return result


@dataclass
@dataclass
class JudgePromptContext:
    """Data container for judge template rendering context."""

    question: str
    golden_answer: Optional[str]  # Make optional for defensive tasks
    episode: Episode  # Full episode object instead of just submission
    task: Task
    evaluation_config: Dict[str, Any]
    model: str
    domain: str
    task_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Jinja2 template rendering."""
        return {
            "question": self.question,
            "golden_answer": self.golden_answer,
            "submission": self.episode.submission,  # Extract submission from episode
            "episode": {
                "episode_id": self.episode.episode_id,
                "start_time": self.episode.start_time,
                "end_time": self.episode.end_time,
                "duration": self.episode.duration,
                "state": self.episode.state.value,  # Convert enum to string
                "completion_reason": self.episode.completion_reason,
                "submission": self.episode.submission,
                "steps": [
                    {
                        "step_number": step.step_number,
                        "timestamp": step.timestamp,
                        "action": {
                            "tool_name": step.action.tool_name,
                            "parameters": step.action.parameters,
                        },
                        "response": step.response,
                        "done": step.done,
                    }
                    for step in self.episode.steps
                ],
                "metadata": self.episode.metadata,
                "context": self.episode.context,
                "max_steps": self.episode.max_steps,
                # Helper methods for template convenience
                "get_last_n_steps": lambda n: (
                    self.episode.steps[-n:] if len(self.episode.steps) >= n else self.episode.steps
                ),
                "get_first_n_steps": lambda n: self.episode.steps[:n],
                "get_failed_steps": lambda: [
                    step for step in self.episode.steps if step.response.get("exit_code", 0) != 0
                ],
                "get_commands_summary": lambda max_length=500: self._get_commands_summary(max_length),
                "get_step_count": lambda: len(self.episode.steps),
            },
            "episode_id": self.episode.episode_id,  # Backward compatibility
            "model": self.model,
            "domain": self.domain,
            "task_id": self.task_id,
            "task": {
                "task_id": self.task.task_id,
                "title": self.task.title,
                "description": self.task.description,
                "domain": self.task.domain,
                "subtasks": (
                    [
                        {
                            "subtask_id": subtask.subtask_id,
                            "title": subtask.title,
                            "description": subtask.description,
                            "objective": subtask.objective,
                            **({"hint": subtask.hint} if hasattr(subtask, "hint") and subtask.hint else {}),
                        }
                        for subtask in self.task.subtasks
                    ]
                    if hasattr(self.task, "subtasks") and self.task.subtasks
                    else []
                ),
            },
            "evaluation_config": self.evaluation_config,
        }

    def _get_commands_summary(self, max_length: int) -> str:
        """Generate a truncated summary of all commands executed."""
        commands = []
        for step in self.episode.steps:
            if step.action.parameters.get("arguments"):
                cmd = step.action.parameters["arguments"]
                commands.append(f"Step {step.step_number}: {cmd}")

        summary = "\n".join(commands)
        if len(summary) > max_length:
            # Truncate and add ellipsis
            summary = summary[: max_length - 3] + "..."

        return summary


class PromptGenerator:
    """
    Handles loading and rendering of both agent and judge prompt templates for tasks.

    Agent Templates:
    - Located in prompts_dir/ (e.g., agent_template.md)
    - Used for generating task prompts for AI agents during episode execution
    - Context includes task details, execution config, environment info

    Judge Templates:
    - Located in prompts_dir/judge/ subdirectory (e.g., default_judge.md)
    - Used for generating LLM evaluation prompts for scoring agent submissions
    - Context includes question, golden answer, submission, evaluation config

    Validates all templates at startup and fails fast on missing or invalid templates.
    Uses Jinja2 templating engine with FileSystemLoader for template management.
    """

    def __init__(self, prompts_dir: str):
        """
        Initialize PromptGenerator with template directory.

        Args:
            prompts_dir: Directory containing Jinja2 template files
                        - Agent templates: directly in prompts_dir/
                        - Judge templates: in prompts_dir/judge/ subdirectory

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

    def render_agent_prompt_for_task(self, task: Task) -> str:
        """
        Render agent prompt for a specific task using its template file.

        Args:
            task: Task object with prompt_template_file specified

        Returns:
            Rendered agent prompt string ready for PolicyManager

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
                f"Successfully rendered agent prompt for task '{task.task_id}' "
                f"using template '{task.prompt_template_file}'"
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
            raise PromptGenerationError(
                f"Unexpected error rendering agent prompt for task '{task.task_id}': {e}"
            ) from e

    def render_judge_prompt_for_episode(self, task: Task, episode: Episode) -> "JudgePromptPayload":
        """
        Render judge prompts for a specific task using episode-based template rendering.

        Args:
            task: Task object containing judge system and user template configuration
            episode: Complete episode object containing execution history and submission

        Returns:
            JudgePromptPayload with complete messages array ready for LLM API

        Raises:
            PromptGenerationError: If template rendering fails
            TemplateValidationError: If template files are missing or invalid
            EvaluationConfigError: If task not configured for LLM judge evaluation
        """
        # Validate episode completeness first - fail fast on incomplete episodes
        if not episode.is_complete:
            from ..evaluation.exceptions import EvaluationConfigError

            raise EvaluationConfigError(
                f"Cannot generate judge prompt for incomplete episode '{episode.episode_id}'. "
                f"Episode state: {episode.state.value}. Episodes must be COMPLETED or FAILED before evaluation."
            )

        # Validate that task uses llm_judge strategy
        eval_config = task.evaluation_config
        if not eval_config or eval_config.get("strategy") != "llm_judge":
            from ..evaluation.exceptions import EvaluationConfigError

            raise EvaluationConfigError(
                f"Task '{task.task_id}' is not configured for LLM judge evaluation. "
                f"Current strategy: {eval_config.get('strategy') if eval_config else 'None'}"
            )

        # Get judge template filenames and model from task configuration
        judge_system_template = eval_config["criteria"]["judge_system_template"]
        judge_user_template = eval_config["criteria"]["judge_user_template"]
        model = eval_config["criteria"]["model"]

        # Extract golden_answer if present (optional for defensive tasks)
        golden_answer = eval_config["criteria"].get("golden_answer")

        # Build judge prompt context with episode data
        context = JudgePromptContext(
            question=task.description,
            golden_answer=golden_answer,  # Now optional
            episode=episode,  # Pass full episode object
            task=task,
            evaluation_config=eval_config["criteria"],
            model=model,
            domain=task.domain,
            task_id=task.task_id,
        )

        # Render both system and user prompts from templates
        system_prompt = self.render_judge_prompt(judge_system_template, context)
        user_prompt = self.render_judge_prompt(judge_user_template, context)

        # Build OpenAI messages format - templates control everything
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        return JudgePromptPayload(messages=messages, model=model, task_id=task.task_id, episode_id=episode.episode_id)

    def render_judge_prompt(self, template_file: str, context: JudgePromptContext) -> str:
        """
        Render judge prompt for LLM evaluation using specified template file.

        Args:
            template_file: Judge template filename (relative to prompts_dir/judge/)
            context: JudgePromptContext with evaluation data

        Returns:
            Rendered judge prompt string ready for LLM evaluator

        Raises:
            PromptGenerationError: If template rendering fails
            TemplateValidationError: If template file is missing or invalid
        """
        if not template_file:
            raise PromptGenerationError(f"Judge template file required for task '{context.task_id}'")

        # Construct full template path within judge subdirectory
        judge_template_path = f"judge/{template_file}"
        self._assert_safe_template_name(judge_template_path)

        try:
            # Load template - fail fast if not found
            template = self.jinja_env.get_template(judge_template_path)

            # Render template with judge context
            rendered_prompt = str(template.render(context.to_dict()))

            logger.debug(
                f"Successfully rendered judge prompt for task '{context.task_id}' "
                f"using template '{judge_template_path}'"
            )
            return rendered_prompt

        except TemplateNotFound as e:
            # Could be root template or an included template
            missing_name = getattr(e, "name", judge_template_path)
            location = "included template" if missing_name != judge_template_path else "judge template file"
            raise TemplateValidationError(
                f"{location} not found for task '{context.task_id}': {missing_name}. "
                f"Expected under {self.prompts_dir}/judge/"
            ) from e

        except TemplateError as e:
            raise PromptGenerationError(
                f"Judge template rendering failed for task '{context.task_id}' "
                f"using template '{judge_template_path}': {e}"
            ) from e

        except Exception as e:
            raise PromptGenerationError(
                f"Unexpected error rendering judge prompt for task '{context.task_id}': {e}"
            ) from e

    def validate_template(self, template_file: str) -> bool:
        """
        Validate agent template syntax and file existence.

        Args:
            template_file: Agent template filename to validate

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

    def validate_judge_template(self, template_file: str) -> bool:
        """
        Validate judge template syntax and file existence.

        Args:
            template_file: Judge template filename (relative to judge/ subdirectory)

        Returns:
            True if template is valid

        Raises:
            TemplateValidationError: If template is invalid or missing
        """
        if not template_file:
            raise TemplateValidationError("Judge template filename cannot be empty")

        # Construct full template path within judge subdirectory
        judge_template_path = f"judge/{template_file}"
        self._assert_safe_template_name(judge_template_path)

        try:
            loader = self.jinja_env.loader
            if loader is None:
                raise TemplateValidationError("No loader configured for template environment")

            # Check if judge directory exists
            judge_dir = self.prompts_dir / "judge"
            if not judge_dir.exists():
                raise TemplateValidationError(f"Judge templates directory does not exist: {judge_dir}")

            # Validate template dependencies recursively (same as agent templates)
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

            _walk(judge_template_path)
            if missing:
                raise TemplateValidationError(
                    f"Judge template '{template_file}' has missing dependencies (recursive): {missing}"
                )

            # Compile root template to validate syntax
            self.jinja_env.get_template(judge_template_path)
            logger.debug("Judge template validation successful: %s (deps=%d)", template_file, len(visited) - 1)
            return True

        except TemplateNotFound as e:
            raise TemplateValidationError(f"Judge template file not found: {template_file}") from e

        except TemplateError as e:
            raise TemplateValidationError(f"Judge template syntax error in {template_file}: {e}") from e

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
                subtask_dict = {
                    "subtask_id": subtask.subtask_id,
                    "title": subtask.title,
                    "description": subtask.description,
                    "objective": subtask.objective,
                }
                # Include hint field if it exists
                if hasattr(subtask, "hint") and subtask.hint:
                    subtask_dict["hint"] = subtask.hint
                subtasks_data.append(subtask_dict)
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
            initial_context=task.initial_context,  # Include initial_context from task
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
