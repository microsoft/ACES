"""
Task Agent Resolver

Pure mapping resolution service for direct agent creation architecture.
Maps task IDs to agent assignments without managing agent instances.

BREAKING CHANGE: No backwards compatibility - simplified for direct agent pattern.
"""

import logging
from typing import Any, Dict, List, Set

from ..exceptions import AgentConfigurationError
from ..models import AgentAssignment, SABERConfig

logger = logging.getLogger(__name__)


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

        logger.debug(f"Initialized TaskAgentResolver with {len(config.agents)} agent assignments")

    def initialize(self, available_task_ids: List[str]) -> None:
        """
        Initialize resolver with available task IDs and validate coverage.

        Args:
            available_task_ids: List of task IDs available from server

        Raises:
            AgentConfigurationError: If task coverage validation fails
        """
        if not available_task_ids:
            raise AgentConfigurationError("No available task IDs provided for resolution")

        self._available_task_ids = set(available_task_ids)
        logger.info(f"Initializing resolver for {len(available_task_ids)} available tasks")

        # Build task-to-assignment mappings
        self._build_task_mappings()

        # Validate task coverage
        self._validate_task_coverage()

        logger.info(f"Task agent resolver initialized with {len(self._task_to_assignment)} explicit mappings")
        if self._wildcard_assignment:
            logger.info(f"Wildcard assignment available: {self._wildcard_assignment.id}")

    def _build_task_mappings(self) -> None:
        """Build internal mappings from agent assignments."""
        self._task_to_assignment.clear()
        self._wildcard_assignment = None

        for assignment in self.config.agents:
            for task_pattern in assignment.tasks:
                if task_pattern == "*":
                    if self._wildcard_assignment is not None:
                        raise AgentConfigurationError(
                            f"Multiple wildcard assignments found: "
                            f"{self._wildcard_assignment.id} and {assignment.id}"
                        )
                    self._wildcard_assignment = assignment
                    logger.debug(f"Found wildcard assignment: {assignment.id}")
                else:
                    # Explicit task assignment
                    if task_pattern in self._task_to_assignment:
                        existing = self._task_to_assignment[task_pattern]
                        raise AgentConfigurationError(
                            f"Duplicate task assignment for '{task_pattern}': " f"{existing.id} and {assignment.id}"
                        )
                    self._task_to_assignment[task_pattern] = assignment
                    logger.debug(f"Mapped task '{task_pattern}' to agent '{assignment.id}'")

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

        logger.info("All available tasks have valid agent assignments")

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
            logger.debug(f"Found explicit assignment for task '{task_id}': {assignment.id}")
            return assignment

        # Fall back to wildcard if available
        if self._wildcard_assignment is not None:
            logger.debug(f"Using wildcard assignment for task '{task_id}': {self._wildcard_assignment.id}")
            return self._wildcard_assignment

        # No assignment found
        raise AgentConfigurationError(
            f"No agent assignment found for task '{task_id}'. "
            f"Available explicit assignments: {list(self._task_to_assignment.keys())}, "
            f"Wildcard available: {'Yes' if self._wildcard_assignment else 'No'}"
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
