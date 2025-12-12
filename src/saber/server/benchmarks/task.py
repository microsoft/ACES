"""Task implementation for task management system.

Logging category: ``LogCategory.TASK_MANAGER``.
"""

from typing import Any, Dict, List, Optional, Set, Union

from ...logging_config import LogCategory, get_saber_logger
from .subtask import SubTask

logger = get_saber_logger(LogCategory.TASK_MANAGER, __name__)


class Task:
    """
    Represents a high-level scenario with metadata and subtasks with necessary configurations for other SABER modules
    """

    def __init__(
        self,
        task_id: str,
        domain: str,
        title: str,
        description: str,
        prompts: Dict[str, str],
        subtasks: Optional[List[SubTask]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        environment: Optional[Union[str, Dict[str, Any]]] = None,
        allowed_executors: Optional[List[str]] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        episode_config: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
        submission_evaluation_config: Optional[Dict[str, Any]] = None,
        step_evaluation_config: Optional[Dict[str, Any]] = None,
        # Note - step evaluation config is also captured in the subtasks themselves
        depends_on_task_id: Optional[str] = None,
        role: Optional[str] = None,
        initial_files: Optional[Dict[str, str]] = None,
        is_template: bool = False,
        dependency_template: Optional[str] = None,
    ):
        """
        Initialize a task.

        Args:
            task_id: Unique identifier for the task
            domain: Security domain this task belongs to
            title: Human-readable title
            description: Detailed description of the task
            prompts: Dictionary with three required prompt types:
                {"instruction": "file.md", "assistant": "file.md", "submit": "file.md"}
            subtasks: List of subtasks
            initial_context: Initial context provided when the task starts
            environment: Environment specification that can be:
                - str: Template reference (e.g., "dvwa_pentest")
                - dict with "base_template": Hybrid approach with template + additions
                - dict with granular config: Full environment specification
            allowed_executors: List of allowed executor types for this task
            execution_config: Direct execution configuration (timeouts, limits, etc.)
            episode_config: Episode-specific configuration (max_steps, timeouts, etc.)
            benchmark_config: Benchmark-specific configuration (episode_attempts, etc.)
            submission_evaluation_config: Submission-level evaluation config (strategy, criteria, scoring)
            step_evaluation_config: Step-level evaluation config (strategy, criteria, subtasks)
            depends_on_task_id: DEPRECATED - Do not use. Use dependency_template instead.
            role: Role identifier for model/agent assignment in orchestrated tasks. Use domain-
                appropriate naming (e.g., cyber='blue'/'red', commerce='buyer'/'seller',
                network='client'/'server'). Required when dependency_template is set. Not
                needed for standalone tasks or templates.
            initial_files: Dictionary mapping destination paths in container to source paths
                relative to server/data directory. Example:
                {"/root/pom.xml": "sandbox_files/challenge/pom.xml"}
            is_template: If True, this task is a template and won't be executed directly.
                Templates are blueprints for instantiation via dependency_template references.
            dependency_template: Reference to a template task_id. Creates a fresh instance of
                the template and establishes a dependency on it. Mutually exclusive with
                depends_on_task_id.
        """
        # Validate prompts dictionary - fail fast
        if not isinstance(prompts, dict):
            raise ValueError(f"Task '{task_id}': prompts must be a dictionary")

        required_prompt_types = ["instruction", "assistant", "submit", "continue"]
        for prompt_type in required_prompt_types:
            if prompt_type not in prompts:
                raise ValueError(f"Task '{task_id}': missing required prompt type '{prompt_type}'")
            if not isinstance(prompts[prompt_type], str) or not prompts[prompt_type].strip():
                raise ValueError(f"Task '{task_id}': prompt type '{prompt_type}' must be a non-empty string")

        # Validate template constraints - fail fast
        if is_template and depends_on_task_id:
            raise ValueError(
                f"Task '{task_id}': Templates cannot have 'depends_on_task_id'. "
                f"Templates are blueprints and cannot depend on other tasks."
            )

        if is_template and dependency_template:
            raise ValueError(
                f"Task '{task_id}': Templates cannot have 'dependency_template'. "
                f"Templates cannot depend on other templates."
            )

        if dependency_template and depends_on_task_id:
            raise ValueError(
                f"Task '{task_id}': Cannot specify both 'dependency_template' and 'depends_on_task_id'. "
                f"Use dependency_template for template-based orchestrations."
            )

        if depends_on_task_id:
            raise ValueError(
                f"Task '{task_id}': 'depends_on_task_id' is deprecated and no longer supported. "
                f"Use 'dependency_template' instead. Mark the dependency target with 'is_template: true' "
                f"and reference it via 'dependency_template: {depends_on_task_id}'."
            )

        # Validate role configuration - fail fast
        if dependency_template and not role:
            raise ValueError(
                f"Task '{task_id}': 'role' is required when 'dependency_template' is set. "
                f"Add 'role: <role_name>' to the task YAML file."
            )

        # Validate templates must have role
        if is_template and not role:
            raise ValueError(
                f"Task '{task_id}': Templates must have a 'role' defined. "
                f"Add 'role: <role_name>' to the template task in the YAML file."
            )

        self.task_id = task_id
        self.domain = domain
        self.title = title
        self.description = description
        self.prompts = prompts
        self.subtasks = subtasks or []
        self.initial_context = initial_context or {}
        self.environment = environment
        self.allowed_executors = allowed_executors
        self.execution_config = execution_config or {}
        self.episode_config = episode_config or {}
        self.benchmark_config = benchmark_config or {}
        # NEW FORMAT ONLY: Client-side evaluation configs
        self.submission_evaluation_config = submission_evaluation_config or {}
        self.step_evaluation_config = step_evaluation_config
        # Runtime dependency tracking - set by template expander after expansion
        # Should NOT be set in YAML (validation rejects user-provided values)
        self.depends_on_task_id = depends_on_task_id
        self.role = role
        self.initial_files = initial_files or {}
        self.is_template = is_template
        self.dependency_template = dependency_template

        # Create lookup map for efficient subtask access
        self._subtask_map = {st.subtask_id: st for st in self.subtasks}

    def add_subtask(self, subtask: SubTask) -> None:
        """
        Add a subtask to this task.

        Args:
            subtask: The subtask to add
        """
        subtask.task_id = self.task_id  # Ensure consistency
        self.subtasks.append(subtask)
        self._subtask_map[subtask.subtask_id] = subtask
        logger.debug(
            "Subtask registered",
            extra={
                "event": "task_subtask_registered",
                "task_id": self.task_id,
                "subtask_id": subtask.subtask_id,
            },
        )

    def get_subtask_by_id(self, subtask_id: str) -> Optional[SubTask]:
        """
        Get a subtask by its ID.

        Args:
            subtask_id: The ID of the subtask to retrieve

        Returns:
            The subtask if found, None otherwise
        """
        return self._subtask_map.get(subtask_id)

    def get_all_subtask_ids(self) -> Set[str]:
        """
        Get all subtask IDs for this task.

        Returns:
            Set of all subtask IDs
        """
        return {subtask.subtask_id for subtask in self.subtasks}

    def is_complete(self) -> bool:
        """
        Check if this task is complete.

        Note: This is a placeholder implementation. Task completion is typically
        determined by external evaluation components based on episode state.

        Returns:
            Boolean indicating if the task is complete
        """
        # Placeholder implementation - tasks are not self-completing
        # Completion is determined by evaluation logic external to the task
        return False

    def get_episode_attempts(self) -> int:
        """
        Get the number of episode attempts configured for this task.

        Returns:
            Number of episode attempts (guaranteed to be configured by BenchmarkConfigLoader)

        Raises:
            KeyError: If episode_attempts is not configured (should not happen with proper validation)
        """
        if "episode_attempts" not in self.benchmark_config:
            logger.error(
                "Task missing episode_attempts configuration",
                extra={
                    "event": "task_missing_episode_attempts",
                    "task_id": self.task_id,
                },
            )
            raise KeyError(
                f"Task '{self.task_id}' does not have episode_attempts configured. "
                "This indicates a validation error in BenchmarkConfigLoader."
            )
        return int(self.benchmark_config["episode_attempts"])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary representation for serialization.

        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "title": self.title,
            "description": self.description,
            "prompts": self.prompts,
            "subtasks": [subtask.to_dict() if hasattr(subtask, "to_dict") else subtask for subtask in self.subtasks],
            "initial_context": self.initial_context,
            "environment": self.environment,
            "allowed_executors": self.allowed_executors,
            "execution_config": self.execution_config,
            "episode_config": self.episode_config,
            "benchmark_config": self.benchmark_config,
            "submission_evaluation_config": self.submission_evaluation_config,
            "step_evaluation_config": self.step_evaluation_config,
            "depends_on_task_id": self.depends_on_task_id,
            "initial_files": self.initial_files,
            "subtask_count": len(self.subtasks),
            "episode_attempts": self.get_episode_attempts(),
        }
