"""Template expansion logic for task orchestrations.

This module handles the expansion of template-based task definitions into
concrete orchestrated task groups. Templates allow a single task definition
to be instantiated multiple times with unique IDs, enabling scenarios like
multiple attack vectors against fresh instances of a defense system.

Logging category: TASK_MANAGER
"""

from copy import deepcopy
from typing import Dict, List

from ...logging_config import LogCategory, get_saber_logger
from .exceptions import InvalidTaskDefinitionException
from .task import Task

logger = get_saber_logger(LogCategory.TASK_MANAGER, __name__)


class TemplateExpander:
    """Expands template references into concrete task instances for orchestration.

    This class handles the transformation of template-based task definitions into
    concrete task instances that can be orchestrated. Each task with a dependency_template
    field gets a fresh instance of the referenced template.

    Example:
        Template: blue_agent (is_template=True)
        Dependents: red_attack_1, red_attack_2 (both reference blue_agent)

        Result: Two separate task groups:
            - [blue_agent_instance_1, red_attack_1]
            - [blue_agent_instance_2, red_attack_2]
    """

    def __init__(self, domain: str):
        """Initialize template expander.

        Args:
            domain: Security domain for logging context
        """
        self.domain = domain

    def expand_templates(self, tasks: Dict[str, Task]) -> Dict[str, Task]:
        """Expand template references into concrete task instances.

        This is the main entry point for template expansion. It:
        1. Identifies template tasks
        2. Identifies tasks with template dependencies
        3. Creates fresh template instances for each dependent
        4. Removes original template tasks (they're not executed directly)

        Args:
            tasks: Dictionary mapping task_id to Task objects (may include templates)

        Returns:
            Dictionary with template instances replacing templates, template tasks removed

        Raises:
            InvalidTaskDefinitionException: If template references are invalid
        """
        logger.info(
            "Template expansion started",
            extra={
                "event": "template_expansion_start",
                "domain": self.domain,
                "total_tasks": len(tasks),
            },
        )

        # Phase 1: Identify templates and template-dependent tasks
        templates: Dict[str, Task] = {}
        template_dependents: List[Task] = []
        concrete_tasks: Dict[str, Task] = {}

        for task_id, task in tasks.items():
            if task.is_template:
                templates[task_id] = task
                logger.debug(
                    f"Identified template task: {task_id}",
                    extra={
                        "event": "template_identified",
                        "template_id": task_id,
                    },
                )
            elif task.dependency_template:
                template_dependents.append(task)
                logger.debug(
                    f"Identified template-dependent task: {task_id} -> {task.dependency_template}",
                    extra={
                        "event": "template_dependent_identified",
                        "dependent_id": task_id,
                        "template_id": task.dependency_template,
                    },
                )
            else:
                # Standalone task (no template involvement)
                concrete_tasks[task_id] = task

        # Phase 2: Validate template references
        self._validate_template_references(template_dependents, templates)

        # Phase 3: Expand template instances for each dependent
        for dependent_task in template_dependents:
            template_id = dependent_task.dependency_template
            if not template_id:  # Type guard (should never happen after validation)
                continue
            template = templates[template_id]  # Guaranteed to exist after validation

            # Generate instance ID (user-controlled naming via task_id)
            instance_id = f"{template_id}_{dependent_task.task_id}"

            # Create template instance
            instance = self._create_template_instance(
                template=template,
                instance_id=instance_id,
                dependent_task=dependent_task,
            )

            # Store instance
            concrete_tasks[instance_id] = instance

            # Update dependent to use concrete dependency
            # After template expansion, depends_on_task_id points to the concrete instance
            dependent_task.depends_on_task_id = str(instance_id)  # type: ignore[assignment]
            # Clear dependency_template - it's been resolved to a concrete dependency
            dependent_task.dependency_template = None
            concrete_tasks[dependent_task.task_id] = dependent_task

            logger.info(
                f"Template instance created: {instance_id}",
                extra={
                    "event": "template_instance_created",
                    "template_id": template_id,
                    "instance_id": instance_id,
                    "dependent_id": dependent_task.task_id,
                },
            )

        logger.info(
            "Template expansion completed",
            extra={
                "event": "template_expansion_complete",
                "domain": self.domain,
                "original_task_count": len(tasks),
                "template_count": len(templates),
                "instance_count": len(template_dependents),
                "final_task_count": len(concrete_tasks),
            },
        )

        return concrete_tasks

    def _validate_template_references(self, template_dependents: List[Task], templates: Dict[str, Task]) -> None:
        """Validate that all template references are valid.

        Args:
            template_dependents: Tasks that reference templates
            templates: Available template tasks

        Raises:
            InvalidTaskDefinitionException: If any template reference is invalid
        """
        for dependent_task in template_dependents:
            template_id = dependent_task.dependency_template

            if template_id not in templates:
                raise InvalidTaskDefinitionException(
                    f"Task '{dependent_task.task_id}' references non-existent template '{template_id}'. "
                    f"Available templates: {list(templates.keys())}"
                )

            logger.debug(
                f"Validated template reference: {dependent_task.task_id} -> {template_id}",
                extra={
                    "event": "template_reference_validated",
                    "dependent_id": dependent_task.task_id,
                    "template_id": template_id,
                },
            )

    def _create_template_instance(
        self,
        template: Task,
        instance_id: str,
        dependent_task: Task,
    ) -> Task:
        """Create a concrete instance from a template.

        Deep copies the template and customizes it with a unique ID. The instance
        is a fully independent task that can be orchestrated.

        Args:
            template: Template task to instantiate
            instance_id: Unique ID for this instance
            dependent_task: Task that depends on this instance (for logging context)

        Returns:
            New Task instance with customized ID
        """
        # Deep copy template to avoid modifying original
        instance = Task(
            task_id=instance_id,
            domain=template.domain,
            title=template.title,
            description=template.description,
            prompts=deepcopy(template.prompts),
            subtasks=deepcopy(template.subtasks),
            initial_context=deepcopy(template.initial_context),
            environment=deepcopy(template.environment),
            allowed_executors=template.allowed_executors.copy() if template.allowed_executors else None,
            execution_config=deepcopy(template.execution_config),
            episode_config=deepcopy(template.episode_config),
            benchmark_config=deepcopy(template.benchmark_config),
            submission_evaluation_config=deepcopy(template.submission_evaluation_config),
            step_evaluation_config=deepcopy(template.step_evaluation_config),
            depends_on_task_id=None,  # Template instances are roots
            role=template.role,  # Preserve template role
            initial_files=deepcopy(template.initial_files),
            is_template=False,  # Instance is concrete, not a template
            dependency_template=None,
        )

        logger.debug(
            f"Template instance constructed: {instance_id}",
            extra={
                "event": "template_instance_constructed",
                "instance_id": instance_id,
                "template_id": template.task_id,
                "instance_role": instance.role,
            },
        )

        return instance

    def validate_templates(self, tasks: Dict[str, Task]) -> None:
        """Validate template definitions for consistency.

        Ensures that:
        - Templates don't have depends_on_task_id
        - Templates don't have dependency_template
        - Templates have required role defined

        Args:
            tasks: Dictionary of all tasks to validate

        Raises:
            InvalidTaskDefinitionException: If template validation fails
        """
        for task_id, task in tasks.items():
            if task.is_template:
                # Templates validated in Task.__init__, but double-check critical constraints
                if task.depends_on_task_id:
                    raise InvalidTaskDefinitionException(
                        f"Template '{task_id}' cannot have 'depends_on_task_id'. "
                        f"Templates are blueprints and cannot depend on other tasks."
                    )

                if task.dependency_template:
                    raise InvalidTaskDefinitionException(
                        f"Template '{task_id}' cannot have 'dependency_template'. "
                        f"Templates cannot depend on other templates."
                    )

                if not task.role:
                    raise InvalidTaskDefinitionException(
                        f"Template '{task_id}' must have a 'role' defined. "
                        f"Templates used in orchestrations require role assignment."
                    )

                logger.debug(
                    f"Template validation passed: {task_id}",
                    extra={
                        "event": "template_validated",
                        "template_id": task_id,
                        "role": task.role,
                    },
                )
