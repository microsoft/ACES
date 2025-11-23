"""Benchmark task models for orchestrated execution.

This module provides polymorphic task abstractions for SABER's orchestrated task execution.
Tasks can be either single-episode (traditional) or orchestrated (multi-episode coordinated).

The key design insight: dependency pairing should be a first-class concept represented by
object types rather than runtime flags. This makes orchestration explicit in the data model.

Logging category: TASK_MANAGER
"""

import fnmatch
import graphlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal

from ..logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.TASK_MANAGER, __name__)

# Configuration constants
MAX_SUBTASKS_PER_ORCHESTRATION = 10  # Maximum sub-tasks allowed in a single orchestration


class TaskExecutionMode(str, Enum):
    """Execution mode for benchmark tasks."""

    SINGLE = "single"  # Single episode execution
    ORCHESTRATED = "orchestrated"  # Multiple episodes coordinated together


class OrchestrationStrategy(str, Enum):
    """Orchestration execution strategies.

    NOTE: Currently only SEQUENTIAL_PAIRED is implemented in task_handlers.py.
    The strategy field is validated but not yet used for branching logic.
    See Issue #XXX for parallel orchestration implementation.
    """

    SEQUENTIAL_PAIRED = "sequential_paired"  # Create episodes in dependency order, wait for each
    PARALLEL = "parallel"  # Create all episodes concurrently (NOT YET IMPLEMENTED)
    CONDITIONAL = "conditional"  # Conditional execution based on results (NOT YET IMPLEMENTED)


class SubTaskDefinition(BaseModel):
    """Sub-task within an orchestrated task."""

    task_id: str = Field(..., description="Unique task identifier")
    role: Optional[str] = Field(
        None,
        description=(
            "Role identifier for model/agent assignment. Domain-specific naming: "
            "cyber='blue'/'red', commerce='buyer'/'seller', network='client'/'server'. "
            "Required when task is part of an orchestration."
        ),
    )

    @model_validator(mode="after")
    def validate_task_id_format(self) -> "SubTaskDefinition":
        """Validate task_id contains only safe characters."""
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", self.task_id):
            raise ValueError(
                f"task_id '{self.task_id}' contains invalid characters. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
            )
        return self

    order: int = Field(..., description="Execution order (0-based)")
    depends_on_role: Optional[str] = Field(
        None, description="Role identifier that this sub-task depends on (establishes execution ordering)"
    )

    # Full task definition embedded (for self-contained orchestration)
    domain: str = Field(..., description="Security domain")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Task description")
    episode_attempts: int = Field(..., description="Number of episode attempts")
    subtask_count: int = Field(default=0, description="Number of subtasks")
    max_steps: int = Field(..., description="Maximum steps per episode")

    # Prompts
    instruction_prompt: str = Field(..., description="Agent instruction prompt")
    assistant_prompt: str = Field(..., description="Agent assistant prompt")
    submit_prompt: str = Field(..., description="Agent submit prompt")


class BenchmarkTask(BaseModel, ABC):
    """Abstract base for all benchmark tasks.

    This polymorphic abstraction enables type-safe representation of different
    execution patterns (single vs orchestrated) with shared interfaces.
    """

    task_type: str = Field(..., description="Discriminator field for polymorphic deserialization")
    benchmark_task_id: Optional[str] = Field(
        None, description="Unique benchmark task identifier (auto-generated if not provided)"
    )
    execution_mode: TaskExecutionMode = Field(..., description="How this task executes")
    episode_attempts: int = Field(..., description="Number of attempts for this benchmark task")

    model_config = ConfigDict(use_enum_values=True)

    @abstractmethod
    def get_task_ids(self) -> List[str]:
        """Get all underlying SABER task IDs.

        Returns:
            List of task IDs that will be executed
        """
        pass

    @abstractmethod
    def get_total_episodes(self) -> int:
        """Calculate total episodes this task creates.

        Returns:
            Total number of episodes across all attempts
        """
        pass

    @abstractmethod
    def matches_filter(self, pattern: str) -> bool:
        """Check if this task matches the given filter pattern.

        Polymorphic filtering: Each task type knows how to match itself.
        - SingleEpisodeTask: Match against task_id
        - OrchestratedTask: Match against orchestration ID or any sub-task ID

        Args:
            pattern: Exact match or glob pattern

        Returns:
            True if this task should be included when filter is applied
        """
        pass


class SingleEpisodeTask(BenchmarkTask):
    """Traditional task - one task, one episode per attempt.

    This represents the existing SABER behavior where each task definition
    creates a single episode per attempt.
    """

    task_type: Literal["single"] = Field(default="single", description="Task type discriminator")
    execution_mode: TaskExecutionMode = Field(default=TaskExecutionMode.SINGLE, description="Single episode mode")

    # Full task information
    task_id: str = Field(..., description="Unique task identifier")

    @model_validator(mode="before")
    @classmethod
    def validate_task_id_format(cls, data: Any) -> Any:
        """Validate task_id contains only safe characters."""
        import re

        if isinstance(data, dict) and "task_id" in data:
            task_id = data["task_id"]
            if not re.match(r"^[a-zA-Z0-9_-]+$", task_id):
                raise ValueError(
                    f"task_id '{task_id}' contains invalid characters. "
                    "Only alphanumeric characters, underscores, and hyphens are allowed."
                )
        return data

    domain: str = Field(..., description="Security domain")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Task description")
    subtask_count: int = Field(default=0, description="Number of subtasks")
    max_steps: int = Field(..., description="Maximum steps per episode")

    # Prompts
    instruction_prompt: str = Field(..., description="Agent instruction prompt")
    assistant_prompt: str = Field(..., description="Agent assistant prompt")
    submit_prompt: str = Field(..., description="Agent submit prompt")

    @model_validator(mode="after")
    def auto_generate_benchmark_task_id(self) -> "SingleEpisodeTask":
        """Auto-generate benchmark_task_id from task_id if not provided."""
        if not self.benchmark_task_id:
            self.benchmark_task_id = self.task_id
        return self

    def get_task_ids(self) -> List[str]:
        """Return single task ID."""
        return [self.task_id]

    def get_total_episodes(self) -> int:
        """Return episode attempts (one episode per attempt)."""
        return self.episode_attempts

    def matches_filter(self, pattern: str) -> bool:
        """Match against single task_id.

        Args:
            pattern: Exact match or glob pattern

        Returns:
            True if task_id matches pattern
        """
        return self.task_id == pattern or fnmatch.fnmatch(self.task_id, pattern)


class OrchestratedTask(BenchmarkTask):
    """Orchestrated task - multiple tasks execute as a coordinated group.

    This represents paired or multi-task scenarios where multiple episodes
    must be created and managed together (e.g., blue+red teams, client+server).

    WARNING: orchestration_strategy is validated but currently ignored in execution.
    All orchestrations use sequential_paired logic regardless of this field.
    """

    task_type: Literal["orchestrated"] = Field(default="orchestrated", description="Task type discriminator")
    execution_mode: TaskExecutionMode = Field(
        default=TaskExecutionMode.ORCHESTRATED, description="Orchestrated execution mode"
    )
    orchestration_strategy: OrchestrationStrategy = Field(
        default=OrchestrationStrategy.SEQUENTIAL_PAIRED,
        description="Orchestration strategy (validated but currently only sequential_paired is implemented)",
    )
    sub_tasks: List[SubTaskDefinition] = Field(..., description="Sub-tasks in this orchestration")
    orchestration_config: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific configuration")

    @model_validator(mode="after")
    def validate_subtask_constraints(self) -> "OrchestratedTask":
        """Validate sub-task count and generate benchmark_task_id if needed."""
        # Validate sub-task count
        if len(self.sub_tasks) == 0:
            raise ValueError("Orchestration must have at least one sub-task")
        if len(self.sub_tasks) > MAX_SUBTASKS_PER_ORCHESTRATION:
            raise ValueError(
                f"Orchestration has {len(self.sub_tasks)} sub-tasks, "
                f"maximum allowed is {MAX_SUBTASKS_PER_ORCHESTRATION}"
            )

        # Auto-generate benchmark_task_id if not provided
        if not self.benchmark_task_id:
            # Generate from first sub-task ID with suffix
            first_task_id = self.sub_tasks[0].task_id
            self.benchmark_task_id = f"{first_task_id}_orchestrated"

        return self

    def get_task_ids(self) -> List[str]:
        """Return all sub-task IDs."""
        return [sub.task_id for sub in self.sub_tasks]

    def get_total_episodes(self) -> int:
        """Return episode attempts (one attempt creates episodes for ALL sub-tasks)."""
        return self.episode_attempts

    def matches_filter(self, pattern: str) -> bool:
        """Match against orchestration ID or any sub-task ID.

        If filter matches ANY sub-task, the entire orchestration is included.
        This ensures dependent episodes always run together.

        Args:
            pattern: Exact match or glob pattern

        Returns:
            True if orchestration or any sub-task matches pattern
        """
        # Return False if pattern is None or empty
        if not pattern:
            return False

        # Check orchestration ID itself
        if self.benchmark_task_id == pattern or fnmatch.fnmatch(
            self.benchmark_task_id, pattern
        ):  # type: ignore[type-var]
            return True

        # Check any sub-task ID (entire orchestration included if any match)
        for subtask in self.sub_tasks:
            if subtask.task_id == pattern or fnmatch.fnmatch(subtask.task_id, pattern):
                logger.info(
                    f"Task filter matched sub-task '{subtask.task_id}' - "
                    f"including entire orchestration '{self.benchmark_task_id}'",
                    extra={
                        "pattern": pattern,
                        "subtask_id": subtask.task_id,
                        "orchestration_id": self.benchmark_task_id,
                    },
                )
                return True

        return False


class DependencyGraph:
    """Analyzes task dependency relationships and validates orchestration structure.

    This helper class builds and validates the dependency graph from task definitions,
    identifying dependency roots and their dependents for orchestration assembly.
    """

    def __init__(self) -> None:
        """Initialize empty dependency graph."""
        self.dependencies: Dict[str, str] = {}  # dependent -> target
        self.dependents: Dict[str, List[str]] = {}  # target -> [dependents]
        self.dependency_roots: Set[str] = set()
        self.all_tasks: Set[str] = set()

    def add_task(self, task_id: str) -> None:
        """Add a task to the graph.

        Args:
            task_id: Task identifier to add
        """
        self.all_tasks.add(task_id)

    def add_dependency(self, dependent: str, target: str) -> None:
        """Add a dependency relationship.

        Args:
            dependent: Task that depends on target
            target: Task that is depended upon
        """
        self.dependencies[dependent] = target
        if target not in self.dependents:
            self.dependents[target] = []
        self.dependents[target].append(dependent)
        self.dependency_roots.add(target)
        self.all_tasks.add(dependent)
        self.all_tasks.add(target)

    def is_root(self, task_id: str) -> bool:
        """Check if task is a dependency root (has dependents).

        Args:
            task_id: Task to check

        Returns:
            True if task has other tasks depending on it
        """
        return task_id in self.dependency_roots

    def is_dependent(self, task_id: str) -> bool:
        """Check if task is dependent on another task.

        Args:
            task_id: Task to check

        Returns:
            True if task depends on another task
        """
        return task_id in self.dependencies

    def get_dependents(self, task_id: str) -> List[str]:
        """Get list of tasks that depend on the given task.

        Args:
            task_id: Root task to get dependents for

        Returns:
            List of dependent task IDs
        """
        return self.dependents.get(task_id, [])

    def get_dependency_target(self, task_id: str) -> Optional[str]:
        """Get the task that the given task depends on.

        Args:
            task_id: Dependent task

        Returns:
            Target task ID, or None if task has no dependencies
        """
        return self.dependencies.get(task_id)

    def validate_acyclic(self) -> None:
        """Validate that the dependency graph is acyclic (no circular dependencies).

        Uses Python's graphlib.TopologicalSorter for cycle detection.

        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build graph for topological sort
        graph: Dict[str, List[str]] = {task_id: [] for task_id in self.all_tasks}
        for dependent, target in self.dependencies.items():
            graph[target].append(dependent)

        try:
            sorter = graphlib.TopologicalSorter(graph)
            # Prepare will raise CycleError if cycles exist
            sorter.prepare()
        except graphlib.CycleError as e:
            raise ValueError(f"Circular dependency detected in task graph: {e}")

    def validate_no_nested_orchestrations(self) -> None:
        """Validate that orchestrated tasks don't contain other orchestrated tasks.

        Nested orchestrations are explicitly forbidden in the initial implementation
        to avoid complexity in semaphore accounting and error handling.

        Raises:
            ValueError: If a dependency root is itself a dependent (nested orchestration)
        """
        for root_id in self.dependency_roots:
            if root_id in self.dependencies:
                target_id = self.dependencies[root_id]
                raise ValueError(
                    f"Nested orchestrations not supported: '{root_id}' is a dependency root "
                    f"but also depends on '{target_id}'. Each orchestration must be independent."
                )

    def get_orchestrated_groups(self) -> List[List[str]]:
        """Get groups of tasks that should be orchestrated together.

        Each group contains a root task and its direct dependents, ordered by
        dependency (root first, then dependents).

        Returns:
            List of task ID groups, where each group is [root, dependent1, dependent2, ...]
        """
        groups = []
        for root_id in sorted(self.dependency_roots):  # Sort for deterministic ordering
            group = [root_id] + sorted(self.get_dependents(root_id))
            groups.append(group)
        return groups

    def get_independent_tasks(self) -> List[str]:
        """Get tasks that are not part of any orchestration.

        Returns:
            List of task IDs that are neither roots nor dependents
        """
        orchestrated = self.dependency_roots | set(self.dependencies.keys())
        return sorted(self.all_tasks - orchestrated)


__all__ = [
    "TaskExecutionMode",
    "OrchestrationStrategy",
    "BenchmarkTask",
    "SingleEpisodeTask",
    "OrchestratedTask",
    "SubTaskDefinition",
    "DependencyGraph",
]
