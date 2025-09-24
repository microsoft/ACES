"""
Task Agent Resolver

Pure mapping resolution service for direct agent creation architecture.
Maps task IDs to agent assignments without managing agent instances.

BREAKING CHANGE: No backwards compatibility - simplified for direct agent pattern.
"""

from typing import Any, Dict, List, Set

from ...logging_config import get_agent_logger
from ..exceptions import AgentConfigurationError
from ..models import AgentAssignment, SABERConfig

logger = get_agent_logger(__name__)


class TaskAgentResolver:
    """
    Pure mapping resolver for task-to-agent assignment without instance management.

    Responsibilities:
    - Parse agent assignments from configuration
    - Validate task coverage during initialization
    - Provide task-to-agent mapping for direct agent creation
    - Support wildcard and explicit task assignments
    - No agent instance management or creation

    Used by agent manager to determine which agent class to instantiate per task.

    BREAKING CHANGE: Simplified for direct agent architecture - no agent instances.
    """

    def __init__(self, config: SABERConfig):
        """
        Initialize resolver with SABER configuration.

        Args:
            config: SABER configuration with agents assignments

        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            raise ValueError("SABERConfig is required")
        if not config.agents:
            raise ValueError("No agent assignments found in configuration")

        self.config = config
        self._task_to_assignment: Dict[str, AgentAssignment] = {}
        self._wildcard_assignment: AgentAssignment | None = None
        self._available_task_ids: Set[str] = set()

        logger.debug(
            "Task agent resolver initialized",
            extra={
                "event": "task_agent_resolver_initialized",
                "assignment_count": len(config.agents),
            },
        )

    def initialize(self, available_task_ids: List[str]) -> None:
        """
        Initialize resolver with available task IDs and validate coverage.

        Args:
            available_task_ids: List of task IDs available from server

        Raises:
            AgentConfigurationError: If task coverage validation fails
        """
        if not available_task_ids:
            logger.error(
                "Task resolver initialization missing task identifiers",
                extra={"event": "task_agent_resolver_missing_tasks"},
            )
            raise AgentConfigurationError("No available task IDs provided for resolution")

        self._available_task_ids = set(available_task_ids)
        logger.info(
            "Initializing task resolver",
            extra={
                "event": "task_agent_resolver_initializing",
                "available_task_count": len(available_task_ids),
            },
        )

        # Build task-to-assignment mappings
        self._build_task_mappings()

        # Validate task coverage
        self._validate_task_coverage()

        logger.info(
            "Task resolver initialized",
            extra={
                "event": "task_agent_resolver_ready",
                "explicit_mapping_count": len(self._task_to_assignment),
                "has_wildcard": bool(self._wildcard_assignment),
            },
        )
        if self._wildcard_assignment:
            logger.info(
                "Task resolver wildcard assignment",
                extra={
                    "event": "task_agent_resolver_wildcard",
                    "agent_id": self._wildcard_assignment.id,
                },
            )

    def _build_task_mappings(self) -> None:
        """Build internal mappings from agent assignments."""
        self._task_to_assignment.clear()
        self._wildcard_assignment = None

        for assignment in self.config.agents:
            for task_pattern in assignment.tasks:
                if task_pattern == "*":
                    if self._wildcard_assignment is not None:
                        logger.error(
                            "Multiple wildcard agent assignments detected",
                            extra={
                                "event": "task_agent_resolver_duplicate_wildcard",
                                "existing_agent": self._wildcard_assignment.id,
                                "conflicting_agent": assignment.id,
                            },
                        )
                        raise AgentConfigurationError(
                            f"Multiple wildcard assignments found: {self._wildcard_assignment.id} and {assignment.id}"
                        )
                    self._wildcard_assignment = assignment
                    logger.debug(
                        "Wildcard assignment registered",
                        extra={
                            "event": "task_agent_resolver_wildcard_registered",
                            "agent_id": assignment.id,
                        },
                    )
                else:
                    # Explicit task assignment
                    if task_pattern in self._task_to_assignment:
                        existing = self._task_to_assignment[task_pattern]
                        logger.error(
                            "Duplicate task assignment detected",
                            extra={
                                "event": "task_agent_resolver_duplicate_assignment",
                                "task_id": task_pattern,
                                "existing_agent": existing.id,
                                "conflicting_agent": assignment.id,
                            },
                        )
                        raise AgentConfigurationError(
                            f"Duplicate task assignment for '{task_pattern}': {existing.id} and {assignment.id}"
                        )
                    self._task_to_assignment[task_pattern] = assignment
                    logger.debug(
                        "Explicit task assignment registered",
                        extra={
                            "event": "task_agent_resolver_assignment_registered",
                            "task_id": task_pattern,
                            "agent_id": assignment.id,
                        },
                    )

    def _validate_task_coverage(self) -> None:
        """Validate that all available tasks have agent assignments."""
        uncovered_tasks = []

        for task_id in self._available_task_ids:
            if not self.has_assignment_for_task(task_id):
                uncovered_tasks.append(task_id)

        if uncovered_tasks:
            raise AgentConfigurationError(
                f"No agent assignments found for tasks: {uncovered_tasks}. "
                f"Available assignments: {list(self._task_to_assignment.keys())} "
                f"+ wildcard: {'Yes' if self._wildcard_assignment else 'No'}"
            )

        logger.info(
            "All tasks covered by agent assignments",
            extra={"event": "task_agent_resolver_coverage_complete"},
        )

    def get_assignment_for_task(self, task_id: str) -> AgentAssignment:
        """
        Get agent assignment for a specific task ID.

        Args:
            task_id: Task identifier

        Returns:
            AgentAssignment for the task

        Raises:
            AgentConfigurationError: If no assignment found for task
        """
        # First check explicit mapping
        if task_id in self._task_to_assignment:
            assignment = self._task_to_assignment[task_id]
            logger.debug(
                "Resolved explicit task assignment",
                extra={
                    "event": "task_agent_resolver_explicit_match",
                    "task_id": task_id,
                    "agent_id": assignment.id,
                },
            )
            return assignment

        # Fall back to wildcard if available
        if self._wildcard_assignment is not None:
            logger.debug(
                "Resolved wildcard task assignment",
                extra={
                    "event": "task_agent_resolver_wildcard_match",
                    "task_id": task_id,
                    "agent_id": self._wildcard_assignment.id,
                },
            )
            return self._wildcard_assignment

        # No assignment found
        logger.error(
            "Task assignment missing",
            extra={
                "event": "task_agent_resolver_task_missing",
                "task_id": task_id,
                "explicit_assignments": sorted(self._task_to_assignment.keys()),
                "has_wildcard": bool(self._wildcard_assignment),
            },
        )
        available_assignments = list(self._task_to_assignment.keys())
        wildcard_status = "Yes" if self._wildcard_assignment else "No"
        raise AgentConfigurationError(
            f"No agent assignment found for task '{task_id}'. "
            f"Available explicit assignments: {available_assignments}, "
            f"Wildcard available: {wildcard_status}"
        )

    def has_assignment_for_task(self, task_id: str) -> bool:
        """
        Check if task has an agent assignment (explicit or wildcard).

        Args:
            task_id: Task identifier

        Returns:
            True if assignment exists, False otherwise
        """
        return task_id in self._task_to_assignment or self._wildcard_assignment is not None

    def get_all_assigned_task_ids(self) -> List[str]:
        """
        Get all task IDs that have explicit assignments.

        Returns:
            List of task IDs with explicit agent assignments
        """
        return list(self._task_to_assignment.keys())

    def get_coverage_summary(self) -> Dict[str, Any]:
        """
        Get summary of task coverage for debugging/monitoring.

        Returns:
            Dictionary with coverage statistics and details
        """
        explicit_tasks = set(self._task_to_assignment.keys())
        covered_tasks = set()

        for task_id in self._available_task_ids:
            if self.has_assignment_for_task(task_id):
                covered_tasks.add(task_id)

        wildcard_covered = self._available_task_ids - explicit_tasks if self._wildcard_assignment else set()

        return {
            "total_available_tasks": len(self._available_task_ids),
            "total_covered_tasks": len(covered_tasks),
            "explicit_assignments": len(explicit_tasks),
            "wildcard_covered": len(wildcard_covered),
            "has_wildcard": self._wildcard_assignment is not None,
            "coverage_complete": len(covered_tasks) == len(self._available_task_ids),
            "uncovered_tasks": list(self._available_task_ids - covered_tasks),
            "wildcard_agent": self._wildcard_assignment.id if self._wildcard_assignment else None,
        }
