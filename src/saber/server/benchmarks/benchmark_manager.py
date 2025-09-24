"""BenchmarkManager implementation for task definition management.

Logging category: TASK_MANAGER.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...models import BenchmarkInfo, TaskInfo
from ..base import Episode
from .benchmark_config_loader import BenchmarkConfigLoader
from .exceptions import SubTaskNotFoundException, TaskNotFoundException
from .prompt_generator import PromptGenerator, TemplateValidationError
from .subtask import SubTask
from .task import Task

logger = get_saber_logger(LogCategory.TASK_MANAGER, __name__)


class BenchmarkManager:
    """
    Domain task definition manager for client-side orchestration.

    Manages task definitions and configurations from YAML files.
    No longer handles server-side orchestration - that's now client responsibility.
    """

    def __init__(self, domain: str, config_dir: str):
        """
        Initialize BenchmarkManager with domain and configuration.

        Args:
            domain: Security domain for benchmarks (e.g., 'webapp_pentest')
            config_dir: Directory path containing task configuration files
        """
        self.domain = domain
        self.config_dir = Path(config_dir)
        self.tasks_dir_path = self.config_dir / "tasks"
        self.tasks: Dict[str, Task] = {}
        self.benchmark_config: Dict[str, Any] = {}

        # Initialize specialized components
        self.config_loader = BenchmarkConfigLoader(domain)

        # Initialize prompt generator
        prompts_dir = self.config_dir / "prompts"
        self.prompt_generator = PromptGenerator(str(prompts_dir))

        logger.info(
            "BenchmarkManager initialization",
            extra={
                "event": "benchmark_manager_init",
                "domain": self.domain,
                "config_dir": str(self.config_dir),
                "tasks_dir": str(self.tasks_dir_path),
            },
        )

        # Load tasks and benchmark configuration
        self.load_tasks_from_directory()

        # Validate all task templates at startup (fail-fast)
        self.validate_all_task_templates()

    def load_tasks_from_directory(self) -> None:
        """
        Load and parse YAML task definitions and benchmark configuration using BenchmarkConfigLoader.

        Raises:
            InvalidTaskDefinitionException: If YAML is invalid or malformed
        """
        operation_context = {
            "domain": self.domain,
            "tasks_dir": str(self.tasks_dir_path),
        }
        log_operation_start(logger, "benchmark_manager_load_tasks", **operation_context)

        try:
            self.tasks = self.config_loader.load_tasks_from_directory(str(self.tasks_dir_path))
            self.benchmark_config = self.config_loader.load_benchmark_config()

            # Inject judge prompt renderer functions for llm_judge tasks
            self._inject_judge_prompt_renderers()

            log_operation_success(
                logger,
                "benchmark_manager_load_tasks",
                task_count=len(self.tasks),
                benchmark_config_keys=list(self.benchmark_config.keys()),
                **operation_context,
            )
        except Exception as exc:  # pragma: no cover - fail fast on loader errors
            log_operation_failure(logger, "benchmark_manager_load_tasks", exc, **operation_context)
            raise

    def _inject_judge_prompt_renderers(self) -> None:
        """
        Inject judge prompt renderer functions into task evaluation configs.

        For tasks using llm_judge strategy, adds a judge_prompt_renderer function
        that can be called by the EvaluationManager without cross-manager dependencies.
        """
        for task in self.tasks.values():
            eval_config = task.evaluation_config
            if eval_config and eval_config.get("strategy") == "llm_judge":
                # Create a closure that captures the task and prompt generator
                def create_renderer(task_ref: Task) -> Callable[[Episode], Any]:
                    def judge_prompt_renderer(episode: Episode) -> Any:
                        return self.prompt_generator.render_judge_prompt_for_episode(task_ref, episode)

                    return judge_prompt_renderer

                # Inject the renderer function into the evaluation config
                eval_config["judge_prompt_renderer"] = create_renderer(task)
                logger.debug(
                    "Injected judge prompt renderer",
                    extra={
                        "event": "benchmark_judge_prompt_renderer_injected",
                        "task_id": task.task_id,
                    },
                )

    def get_benchmark_info(self) -> BenchmarkInfo:
        """
        Get complete benchmark information for client-side orchestration.

        Returns:
            BenchmarkInfo object with all tasks and their episode attempt configurations
        """
        task_infos = []
        total_episodes = 0

        for task in self.tasks.values():
            episode_attempts = task.get_episode_attempts()
            episode_config = task.episode_config or {}
            max_steps = episode_config.get("max_steps", 100)

            # Render all three prompts for this task
            rendered_prompts = self.prompt_generator.render_agent_prompts_for_task(task)

            task_info = TaskInfo(
                task_id=task.task_id,
                title=task.title,
                description=task.description,
                episode_attempts=episode_attempts,
                subtask_count=len(task.subtasks),
                max_steps=max_steps,
                # NEW: Three distinct prompts (replaces initial_prompt)
                instruction_prompt=rendered_prompts["instruction"],
                assistant_prompt=rendered_prompts["assistant"],
                submit_prompt=rendered_prompts["submit"],
            )
            task_infos.append(task_info)
            total_episodes += episode_attempts

        return BenchmarkInfo(
            domain=self.domain, tasks=task_infos, total_tasks=len(task_infos), total_episodes=total_episodes
        )

    def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            Task instance

        Raises:
            TaskNotFoundException: If task is not found
        """
        if task_id not in self.tasks:
            raise TaskNotFoundException(task_id)
        return self.tasks[task_id]

    def get_subtask(self, task_id: str, subtask_id: str) -> SubTask:
        """
        Get a subtask by task ID and subtask ID.

        Args:
            task_id: ID of the parent task
            subtask_id: ID of the subtask

        Returns:
            SubTask instance

        Raises:
            TaskNotFoundException: If task is not found
            SubTaskNotFoundException: If subtask is not found
        """
        task = self.get_task(task_id)
        subtask = task.get_subtask_by_id(subtask_id)

        if subtask is None:
            raise SubTaskNotFoundException(task_id, subtask_id)

        return subtask

    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available tasks (legacy method).

        Returns:
            List of task information dictionaries
        """
        return [
            {
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "subtask_count": len(task.subtasks),
            }
            for task in self.tasks.values()
        ]

    def get_benchmark_config(self) -> Dict[str, Any]:
        """
        Get the loaded benchmark configuration.

        Returns:
            Domain-level benchmark configuration
        """
        return self.benchmark_config.copy()

    def get_episode_config(self, task_id: str) -> Dict[str, Any]:
        """
        Get episode configuration for a specific task.

        Args:
            task_id: ID of the task to get episode config for

        Returns:
            Episode configuration including max_steps, timeouts, and other settings
        """
        task = self.get_task(task_id)
        episode_config = task.episode_config or {}
        execution_config = task.execution_config or {}

        return {
            "max_steps": episode_config.get("max_steps", 100),
            "timeout_seconds": episode_config.get("timeout_seconds", execution_config.get("timeout", 300)),
            "task_id": task_id,
            "task_title": task.title,
            "task_description": task.description,
            "domain": task.domain,
            "subtask_count": len(task.subtasks),
        }

    def list_benchmark_tasks(self) -> List[Dict[str, Any]]:
        """
        Get a list of all tasks with their benchmark configuration (legacy method).

        Returns:
            List of task information dictionaries including benchmark settings
        """
        return [
            {
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "subtask_count": len(task.subtasks),
                "episode_attempts": task.get_episode_attempts(),
                "benchmark_config": task.benchmark_config,
            }
            for task in self.tasks.values()
        ]

    def validate_all_task_templates(self) -> None:
        """
        Validate all task templates at startup.

        This method enforces SABER's fail-fast principle by validating:
        1. All task templates exist and are syntactically valid
        2. All included dependencies (shared partials) exist
        3. All task contexts can be built with required configuration

        Raises:
            TemplateValidationError: If any template or context validation fails
        """
        operation_context: Dict[str, Any] = {
            "domain": self.domain,
            "task_count": len(self.tasks),
        }
        log_operation_start(logger, "benchmark_template_validation", **operation_context)

        validation_errors: List[Dict[str, Any]] = []

        for task_id, task in self.tasks.items():
            task_context = {
                "task_id": task_id,
            }
            try:
                # Validate all three prompt templates exist and syntax is correct
                for prompt_type, template_file in task.prompts.items():
                    self.prompt_generator.validate_template(template_file)
                    task_context[f"{prompt_type}_template"] = template_file
                    logger.debug(
                        "Template validation passed",
                        extra={
                            "event": "template_validation_success",
                            "task_id": task_id,
                            "prompt_type": prompt_type,
                            "template_file": template_file,
                        }
                    )

                # Validate that we can build context for this task (ensures required config is present)
                self.prompt_generator.validate_task_context(task)

                eval_config = task.evaluation_config
                if eval_config and eval_config.get("strategy") == "llm_judge":
                    criteria = eval_config.get("criteria", {})
                    judge_system_template = criteria.get("judge_system_template")
                    judge_user_template = criteria.get("judge_user_template")

                    self.prompt_generator.validate_judge_template(judge_system_template)
                    self.prompt_generator.validate_judge_template(judge_user_template)

                    logger.debug(
                        "Judge templates validated",
                        extra={
                            "event": "benchmark_judge_templates_valid",
                            "task_id": task_id,
                            "judge_system_template": judge_system_template,
                            "judge_user_template": judge_user_template,
                        },
                    )

                logger.debug(
                    "Task template validation passed",
                    extra={
                        "event": "benchmark_task_template_valid",
                        **task_context,
                    },
                )

            except Exception as exc:
                error_payload = {
                    "task_id": task_id,
                    "error": str(exc),
                }
                validation_errors.append(error_payload)
                logger.error(
                    "Task template validation failed",
                    extra={
                        "event": "benchmark_task_template_invalid",
                        **task_context,
                        "error": str(exc),
                    },
                )

        if validation_errors:
            error_lines = [f"Task '{err['task_id']}': {err['error']}" for err in validation_errors]
            summary_text = f"Template validation failed for domain '{self.domain}'. Errors:\n" + "\n".join(
                f"  - {line}" for line in error_lines
            )
            exception = TemplateValidationError(summary_text)
            log_operation_failure(
                logger,
                "benchmark_template_validation",
                exception,
                **operation_context,
                validation_error_count=len(validation_errors),
                validation_errors=validation_errors,
                error_summary=summary_text,
            )
            raise exception

        log_operation_success(
            logger,
            "benchmark_template_validation",
            validated_task_count=len(self.tasks),
            **operation_context,
        )

    def render_judge_prompt_for_episode(self, task_id: str, episode: Episode) -> Any:
        """
        Generate judge prompt for a specific task using episode-based template rendering.

        Args:
            task_id: ID of the task to generate judge prompt for
            episode: Complete episode object containing execution history and submission

        Returns:
            JudgePromptPayload with complete messages array ready for LLM API

        Raises:
            TaskNotFoundException: If task is not found
            PromptGenerationError: If prompt generation fails
            EvaluationConfigError: If task not configured for LLM judge evaluation
        """
        task = self.get_task(task_id)
        return self.prompt_generator.render_judge_prompt_for_episode(task, episode)

    def get_dependency_config(self) -> Dict[str, float]:
        """
        Get dependency resolution configuration from global configuration.

        Returns:
            Dict containing dependency configuration:
            - wait_seconds: Maximum time to wait for dependencies
            - retry_interval: Initial retry interval
            - max_retry_interval: Maximum retry interval
        """
        return self.config_loader.get_dependency_config()
