"""BenchmarkManager implementation for task definition management."""

from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List

from ...models import BenchmarkInfo, TaskInfo
from ..base import Episode
from .benchmark_config_loader import BenchmarkConfigLoader
from .exceptions import SubTaskNotFoundException, TaskNotFoundException
from .prompt_generator import PromptGenerator, TemplateValidationError
from .subtask import SubTask
from .task import Task

logger = getLogger(__name__)


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

        logger.info(f"Initializing BenchmarkManager for domain '{domain}' with config dir: {config_dir}")
        logger.info(f"Tasks directory: {self.tasks_dir_path}")

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
        self.tasks = self.config_loader.load_tasks_from_directory(str(self.tasks_dir_path))
        self.benchmark_config = self.config_loader.load_benchmark_config()

        # Inject judge prompt renderer functions for llm_judge tasks
        self._inject_judge_prompt_renderers()

        logger.info(
            f"BenchmarkManager initialization complete. Loaded {len(self.tasks)} tasks for domain '{self.domain}'"
        )
        logger.info(f"Benchmark config: {self.benchmark_config}")

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
                logger.debug(f"Injected judge prompt renderer for task '{task.task_id}'")

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

            task_info = TaskInfo(
                task_id=task.task_id,
                title=task.title,
                description=task.description,
                episode_attempts=episode_attempts,
                subtask_count=len(task.subtasks),
                max_steps=max_steps,
                initial_prompt=self.get_task_prompt(task.task_id),
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
        logger.info(f"Validating all task templates for domain '{self.domain}'")

        validation_errors = []

        for task_id, task in self.tasks.items():
            try:
                # Validate template exists and syntax is correct
                self.prompt_generator.validate_template(task.prompt_template_file)

                # Validate that we can build context for this task (ensures required config is present)
                self.prompt_generator.validate_task_context(task)

                # Validate judge templates for llm_judge tasks
                eval_config = task.evaluation_config
                if eval_config and eval_config.get("strategy") == "llm_judge":
                    judge_system_template = eval_config["criteria"]["judge_system_template"]
                    judge_user_template = eval_config["criteria"]["judge_user_template"]

                    self.prompt_generator.validate_judge_template(judge_system_template)
                    self.prompt_generator.validate_judge_template(judge_user_template)

                    logger.debug(
                        f"Judge template validation passed for task '{task_id}': "
                        f"system='{judge_system_template}', user='{judge_user_template}'"
                    )

                logger.debug(f"Template validation passed for task '{task_id}'")

            except Exception as e:
                error_msg = f"Task '{task_id}': {str(e)}"
                validation_errors.append(error_msg)
                logger.error(f"Template validation failed for task '{task_id}': {e}")

        if validation_errors:
            error_summary = f"Template validation failed for domain '{self.domain}'. Errors:\n" + "\n".join(
                f"  - {err}" for err in validation_errors
            )
            logger.error(error_summary)
            raise TemplateValidationError(error_summary)

        logger.info(f"All {len(self.tasks)} task templates validated successfully for domain '{self.domain}'")

    def get_task_prompt(self, task_id: str) -> str:
        """
        Generate prompt for a specific task using template rendering.

        Args:
            task_id: ID of the task to generate prompt for

        Returns:
            Rendered prompt string

        Raises:
            TaskNotFoundException: If task is not found
            PromptGenerationError: If prompt generation fails
        """
        task = self.get_task(task_id)
        return self.prompt_generator.render_agent_prompt_for_task(task)

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
