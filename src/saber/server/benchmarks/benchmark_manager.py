"""BenchmarkManager implementation for task definition management.

Logging category: TASK_MANAGER.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...models import (
    BenchmarkInfo,
    BenchmarkTask,
    DependencyGraph,
    OrchestratedTask,
    OrchestrationStrategy,
    SingleEpisodeTask,
    SubTaskDefinition,
)
from ...models.constants import StepEvaluationStrategy, SubmissionEvaluationStrategy
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
        self._dependency_graph: Optional[DependencyGraph] = None  # Cached dependency graph

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

            # Build dependency graph once after all tasks loaded
            self._dependency_graph = self._build_dependency_graph()

            log_operation_success(
                logger,
                "benchmark_manager_load_tasks",
                task_count=len(self.tasks),
                benchmark_config_keys=list(self.benchmark_config.keys()),
                dependency_roots=len(self._dependency_graph.dependency_roots),
                **operation_context,
            )
        except Exception as exc:  # pragma: no cover - fail fast on loader errors
            log_operation_failure(logger, "benchmark_manager_load_tasks", exc, **operation_context)
            raise

    def _inject_judge_prompt_renderers(self) -> None:
        """
        Inject judge prompt renderer functions into task evaluation configs.

        CLIENT-SIDE EVALUATION: This method is now a no-op.
        Judge prompt rendering is done client-side using the Inspect AI saber_scorer.
        Judge templates are served as raw files via REST API endpoints.
        """
        # No-op: Judge rendering moved to client-side
        pass

    def get_benchmark_info(self) -> BenchmarkInfo:
        """
        Get complete benchmark information for client-side orchestration.

        Classifies tasks into SingleEpisodeTask or OrchestratedTask based on
        dependency relationships. Uses cached dependency graph for efficiency.

        Returns:
            BenchmarkInfo object with polymorphic BenchmarkTask list
        """
        # Use cached dependency graph to assemble benchmark tasks
        benchmark_tasks = self._assemble_benchmark_tasks()
        total_episodes = sum(bt.get_total_episodes() for bt in benchmark_tasks)

        return BenchmarkInfo(
            domain=self.domain,
            tasks=benchmark_tasks,  # type: ignore[arg-type]
            total_tasks=len(benchmark_tasks),
            total_episodes=total_episodes,
        )

    def _build_dependency_graph(self) -> DependencyGraph:
        """Build dependency graph from loaded tasks (called once at init).

        Returns:
            DependencyGraph with all task relationships mapped

        Raises:
            ValueError: If circular dependencies or nested orchestrations detected
        """
        graph = DependencyGraph()

        # Add all tasks to graph
        for task_id in self.tasks:
            graph.add_task(task_id)

        # Add dependency relationships
        for task in self.tasks.values():
            if task.depends_on_task_id:
                graph.add_dependency(task.task_id, task.depends_on_task_id)  # type: ignore
                logger.debug(
                    f"Task dependency registered: {task.task_id} -> {task.depends_on_task_id}",
                    extra={
                        "dependent": task.task_id,
                        "target": task.depends_on_task_id,
                    },
                )

        # Validate graph structure
        try:
            graph.validate_acyclic()
            graph.validate_no_nested_orchestrations()
        except ValueError as e:
            logger.error(
                f"Invalid task dependency graph: {e}",
                extra={"error": str(e)},
            )
            raise

        logger.info(
            "Dependency graph built successfully",
            extra={
                "total_tasks": len(graph.all_tasks),
                "dependency_roots": len(graph.dependency_roots),
                "dependent_tasks": len(graph.dependencies),
            },
        )

        return graph

    def _assemble_benchmark_tasks(self) -> List[BenchmarkTask]:
        """Classify and assemble tasks into BenchmarkTask objects.

        Uses the cached dependency graph to identify orchestrations and
        create appropriate SingleEpisodeTask or OrchestratedTask instances.

        Returns:
            List of BenchmarkTask objects (polymorphic)
        """
        benchmark_tasks: List[BenchmarkTask] = []
        processed: set[str] = set()

        # Create orchestrated tasks for dependency groups
        if self._dependency_graph is None:
            return benchmark_tasks
        for group in self._dependency_graph.get_orchestrated_groups():
            orchestrated = self._create_orchestrated_task(group)
            benchmark_tasks.append(orchestrated)
            processed.update(orchestrated.get_task_ids())

            logger.debug(
                f"Created orchestrated task: {orchestrated.benchmark_task_id}",
                extra={
                    "orchestration_id": orchestrated.benchmark_task_id,
                    "sub_task_count": len(orchestrated.sub_tasks),
                    "task_ids": orchestrated.get_task_ids(),
                },
            )

        # Create single-episode tasks for independent tasks
        for task_id in self._dependency_graph.get_independent_tasks():
            if task_id not in processed:
                single = self.create_single_episode_task(self.tasks[task_id])
                benchmark_tasks.append(single)
                processed.add(task_id)

                logger.debug(
                    f"Created single-episode task: {task_id}",
                    extra={"task_id": task_id},
                )

        # Warn about any orphaned dependents
        for task_id in self.tasks:
            if task_id not in processed:
                logger.warning(
                    f"Orphaned dependent task: {task_id} (dependency target not found)",
                    extra={"task_id": task_id},
                )

        return benchmark_tasks

    def _create_orchestrated_task(self, group: List[str]) -> OrchestratedTask:
        """Create OrchestratedTask from a group of task IDs.

        Args:
            group: List of task IDs where first task is the dependency target
                   and remaining tasks depend on it. Roles are user-defined.

        Returns:
            OrchestratedTask with all sub-tasks configured
        """
        root_id = group[0]
        root_task = self.tasks[root_id]

        # Validate first task (dependency target) has a role
        if not root_task.role:
            raise ValueError(
                f"Task '{root_id}' must have a 'role' defined for orchestration. "
                f"Add 'role: <role_name>' to the task YAML file (e.g., role: blue)."
            )

        # Build sub-task definitions
        sub_tasks: List[SubTaskDefinition] = []

        # Add root task
        root_prompts = self.prompt_generator.render_agent_prompts_for_task(root_task)
        root_episode_config = root_task.episode_config or {}

        sub_tasks.append(
            SubTaskDefinition(
                task_id=root_id,
                role=root_task.role,
                order=0,
                depends_on_role=None,
                domain=root_task.domain,
                title=root_task.title,
                description=root_task.description,
                episode_attempts=root_task.get_episode_attempts(),
                subtask_count=len(root_task.subtasks),
                max_steps=root_episode_config.get("max_steps", 100),
                instruction_prompt=root_prompts["instruction"],
                assistant_prompt=root_prompts["assistant"],
                submit_prompt=root_prompts["submit"],
                continue_prompt=root_prompts["continue"],
                transcript_config=root_episode_config.get("transcript_config"),
            )
        )

        # Add dependent tasks
        for idx, dependent_id in enumerate(group[1:], start=1):
            dependent_task = self.tasks[dependent_id]
            dependent_prompts = self.prompt_generator.render_agent_prompts_for_task(dependent_task)
            dependent_episode_config = dependent_task.episode_config or {}

            sub_tasks.append(
                SubTaskDefinition(
                    task_id=dependent_id,
                    role=dependent_task.role,
                    order=idx,
                    depends_on_role=root_task.role,
                    domain=dependent_task.domain,
                    title=dependent_task.title,
                    description=dependent_task.description,
                    episode_attempts=dependent_task.get_episode_attempts(),
                    subtask_count=len(dependent_task.subtasks),
                    max_steps=dependent_episode_config.get("max_steps", 100),
                    instruction_prompt=dependent_prompts["instruction"],
                    assistant_prompt=dependent_prompts["assistant"],
                    submit_prompt=dependent_prompts["submit"],
                    continue_prompt=dependent_prompts["continue"],
                    transcript_config=dependent_episode_config.get("transcript_config"),
                )
            )

        # Use minimum episode_attempts across all sub-tasks
        min_attempts = min(st.episode_attempts for st in sub_tasks)

        return OrchestratedTask(
            benchmark_task_id=f"{root_id}_orchestrated",
            episode_attempts=min_attempts,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=sub_tasks,
            orchestration_config={
                "root_task_id": root_id,
                "creation_order": "root_first",
                "shared_semaphore_slot": True,
            },
        )

    def create_single_episode_task(self, task: Task) -> SingleEpisodeTask:
        """Create SingleEpisodeTask from a Task definition.

        Args:
            task: Task definition

        Returns:
            SingleEpisodeTask with all configuration including rendered prompts
        """
        episode_config = task.episode_config or {}
        rendered_prompts = self.prompt_generator.render_agent_prompts_for_task(task)

        return SingleEpisodeTask(
            benchmark_task_id=task.task_id,
            episode_attempts=task.get_episode_attempts(),
            task_id=task.task_id,
            domain=task.domain,
            title=task.title,
            description=task.description,
            subtask_count=len(task.subtasks),
            max_steps=episode_config.get("max_steps", 100),
            initial_context=task.initial_context,
            instruction_prompt=rendered_prompts["instruction"],
            assistant_prompt=rendered_prompts["assistant"],
            submit_prompt=rendered_prompts["submit"],
            continue_prompt=rendered_prompts["continue"],
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

    def get_single_episode_task(self, task_id: str) -> SingleEpisodeTask:
        """Get a SingleEpisodeTask by ID with rendered prompts.

        This is the preferred method for getting task data needed for episode
        creation, as it returns a SingleEpisodeTask with fully rendered prompts.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            SingleEpisodeTask with rendered prompts

        Raises:
            TaskNotFoundException: If task is not found
        """
        task = self.get_task(task_id)
        return self.create_single_episode_task(task)

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
                        },
                    )

                # Validate that we can build context for this task (ensures required config is present)
                self.prompt_generator.validate_task_context(task)

                # Check both submission and step evaluation configs for llm_judge validation
                eval_configs = []
                if hasattr(task, "submission_evaluation_config") and task.submission_evaluation_config:
                    eval_configs.append(("submission", task.submission_evaluation_config))
                if hasattr(task, "step_evaluation_config") and task.step_evaluation_config:
                    eval_configs.append(("step", task.step_evaluation_config))

                for eval_type, eval_config in eval_configs:
                    # Check LLM strategy for both submission and step evaluation
                    is_llm_strategy = (
                        eval_type == "submission"
                        and eval_config.get("strategy") == SubmissionEvaluationStrategy.LLM_JUDGE
                    ) or (eval_type == "step" and eval_config.get("strategy") == StepEvaluationStrategy.LLM_JUDGE)
                    if eval_config and is_llm_strategy:
                        criteria = eval_config.get("criteria", {})
                        judge_system_template = criteria.get("judge_system_template")
                        judge_user_template = criteria.get("judge_user_template")

                        if judge_system_template:
                            self.prompt_generator.validate_template(judge_system_template)
                        if judge_user_template:
                            self.prompt_generator.validate_template(judge_user_template)

                        logger.debug(
                            "Judge templates validated",
                            extra={
                                "event": "benchmark_judge_templates_valid",
                                "task_id": task_id,
                                "evaluation_type": eval_type,
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

        CLIENT-SIDE EVALUATION: This method is deprecated and raises an error.
        Judge prompt rendering is now done client-side using the Inspect AI saber_scorer.
        Judge templates are served as raw files via REST API endpoints.

        Args:
            task_id: ID of the task to generate judge prompt for
            episode: Complete episode object containing execution history and submission

        Raises:
            NotImplementedError: This method is no longer supported
        """
        raise NotImplementedError(
            "Server-side judge prompt rendering has been removed. "
            "Judge prompts are now rendered client-side using the Inspect AI saber_scorer."
        )

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

    def get_task_prompt(self, task_id: str, prompt_type: str = "instruction") -> str:
        """
        Get a specific rendered prompt for a task (backwards compatibility method).

        Args:
            task_id: ID of the task
            prompt_type: Type of prompt to get ("instruction", "assistant", or "submit")

        Returns:
            Rendered prompt string

        Raises:
            TaskNotFoundException: If task is not found
            PromptGenerationError: If prompt rendering fails
        """
        task = self.get_task(task_id)
        rendered_prompts = self.prompt_generator.render_agent_prompts_for_task(task)
        return rendered_prompts[prompt_type]

    def get_template_content(self, template_path: str) -> str:
        """
        Get raw template content by path for client-side evaluation.

        Args:
            template_path: Relative template path (e.g., 'judge/submission/system.md')

        Returns:
            Raw template content string

        Raises:
            FileNotFoundError: If template file doesn't exist
            TemplateError: If template cannot be read
        """
        template_file_path = self.config_dir / "prompts" / template_path

        if not template_file_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        try:
            with open(template_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise TemplateValidationError(f"Failed to read template {template_path}: {e}") from e
