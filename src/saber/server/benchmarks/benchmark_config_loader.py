"""Benchmark configuration loader for benchmark task definitions.

Logging category: CONFIG.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...models.constants import StepEvaluationStrategy  # noqa: F401
from ...models.constants import SubmissionEvaluationStrategy  # noqa: F401
from ._validation import validate_step_evaluation_config, validate_submission_evaluation_config
from .exceptions import InvalidTaskDefinitionException
from .subtask import SubTask
from .task import Task

logger = get_saber_logger(LogCategory.CONFIG, __name__)


# Configuration field names
FIELD_DOMAIN = "domain"
FIELD_EXECUTORS = "executors"
FIELD_TASKS = "tasks"
FIELD_GLOBAL_DEFAULTS = "global_defaults"
FIELD_BENCHMARK_CONFIG = "benchmark_config"
FIELD_PERMANENT_ENVIRONMENT = "permanent_environment"
FIELD_EXECUTION_CONFIG = "execution_config"
FIELD_EPISODE_CONFIG = "episode_config"
FIELD_DEPENDENCY_CONFIG = "dependency_config"
FIELD_PROMPTS = "prompts"
FIELD_TIMEOUT = "timeout"
FIELD_MAX_STEPS = "max_steps"
FIELD_EPISODE_ATTEMPTS = "episode_attempts"
FIELD_ALLOWED_EXECUTORS = "allowed_executors"
FIELD_TRANSCRIPT_CONFIG = "transcript_config"

# Task field names
FIELD_TASK_ID = "task_id"
FIELD_TITLE = "title"
FIELD_DESCRIPTION = "description"
FIELD_INITIAL_CONTEXT = "initial_context"
FIELD_ENVIRONMENT = "environment"
FIELD_SANDBOX_ENVIRONMENT = "sandbox_environment"
FIELD_DEPENDS_ON_TASK_ID = "depends_on_task_id"
FIELD_ROLE = "role"
FIELD_INITIAL_FILES = "initial_files"
FIELD_SUBTASKS = "subtasks"
FIELD_INHERIT_SHARED = "inherit_shared"

# Subtask field names
FIELD_SUBTASK_ID = "subtask_id"
FIELD_OBJECTIVE = "objective"
FIELD_HINTS = "hints"
FIELD_SCORING = "scoring"
FIELD_MAX_SCORE = "max_score"
FIELD_WEIGHT = "weight"

# Evaluation config field names
FIELD_SUBMISSION_EVALUATION_CONFIG = "submission_evaluation_config"
FIELD_STEP_EVALUATION_CONFIG = "step_evaluation_config"
FIELD_STRATEGY = "strategy"
FIELD_CRITERIA = "criteria"
FIELD_MODEL = "model"
FIELD_EXPECTED_ANSWERS = "expected_answers"
FIELD_JUDGE_SYSTEM_TEMPLATE = "judge_system_template"
FIELD_JUDGE_USER_TEMPLATE = "judge_user_template"
FIELD_STEPS_PER_MESSAGE = "steps_per_message"

# Dependency config field names
FIELD_WAIT_SECONDS = "wait_seconds"
FIELD_RETRY_INTERVAL = "retry_interval"
FIELD_MAX_RETRY_INTERVAL = "max_retry_interval"

# Prompt types
PROMPT_TYPE_INSTRUCTION = "instruction"
PROMPT_TYPE_ASSISTANT = "assistant"
PROMPT_TYPE_SUBMIT = "submit"
REQUIRED_PROMPT_TYPES = [PROMPT_TYPE_INSTRUCTION, PROMPT_TYPE_ASSISTANT, PROMPT_TYPE_SUBMIT]

# File names
FILENAME_GLOBAL_CONFIG = "global.yaml"
FILENAME_GLOBAL_CONFIG_YML = "global.yml"
FILENAME_SHARED_CONFIG = "shared.yaml"
FILENAME_SHARED_CONFIG_YML = "shared.yml"
EXCLUDED_CONFIG_FILES = [
    FILENAME_GLOBAL_CONFIG,
    FILENAME_GLOBAL_CONFIG_YML,
    FILENAME_SHARED_CONFIG,
    FILENAME_SHARED_CONFIG_YML,
]

# Default scoring weights
DEFAULT_WEIGHT = 1.0
DEFAULT_MAX_SCORE = 1.0


def deep_merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Args:
        base: Base dictionary (e.g., global defaults)
        override: Override dictionary (e.g., task specific config)

    Returns:
        New dictionary with deep merged values
    """
    result = base.copy()

    # Handle None override (e.g., when YAML has empty/null value)
    if override is None:
        return result

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # Override takes precedence for non-dict values or new keys
            result[key] = value

    return result


class BenchmarkConfigLoader:
    """
    Handles loading and parsing YAML task definitions and benchmark configurations into Task objects.
    Enhanced to support benchmark-specific configuration including episode_attempts.
    """

    def __init__(self, domain: str):
        """
        Initialize BenchmarkConfigLoader for a specific domain.

        Args:
            domain: The security domain (e.g., 'webapp_pentest')
        """
        self.domain = domain
        self.allowed_executors: Optional[list[str]] = None
        self.benchmark_config: Dict[str, Any] = {}
        self.global_defaults: Dict[str, Any] = {}
        self.permanent_environment: Optional[str] = None
        self.yaml_data: Optional[Dict[str, Any]] = None

    def load_tasks_from_directory(self, tasks_dir_path: str) -> Dict[str, Task]:
        """
        Load and parse YAML task definitions from directory structure into Task objects.

        Expected structure:
        tasks/
        ├── global.yaml              # Global config + defaults
        ├── task_file1.yaml          # 1-N tasks
        ├── task_file2.yaml          # 1-N tasks
        └── category/                # Optional subdirectories
            └── more_tasks.yaml      # 1-N tasks

        Args:
            tasks_dir_path: Path to the tasks directory containing global.yaml and task files

        Returns:
            Dictionary mapping task_id to Task objects

        Raises:
            InvalidTaskDefinitionException: If YAML is invalid or malformed
        """
        tasks_dir = Path(tasks_dir_path)
        operation_context = {
            "domain": self.domain,
            "tasks_dir": str(tasks_dir),
        }
        log_operation_start(logger, "benchmark_tasks_directory_load", **operation_context)

        try:
            if not tasks_dir.exists():
                error = InvalidTaskDefinitionException(f"Tasks directory not found: {tasks_dir}", str(tasks_dir))
                log_operation_failure(logger, "benchmark_tasks_directory_load", error, **operation_context)
                raise error

            if not tasks_dir.is_dir():
                error = InvalidTaskDefinitionException(f"Tasks path is not a directory: {tasks_dir}", str(tasks_dir))
                log_operation_failure(logger, "benchmark_tasks_directory_load", error, **operation_context)
                raise error

            # Load global configuration first
            global_config_path = tasks_dir / "global.yaml"
            if not global_config_path.exists():
                error = InvalidTaskDefinitionException(
                    f"Missing required global.yaml in tasks directory: {global_config_path}",
                    str(global_config_path),
                )
                log_operation_failure(logger, "benchmark_tasks_directory_load", error, **operation_context)
                raise error

            self._load_global_config(global_config_path)

            # Discover all task files (exclude global.yaml)
            task_files = self._discover_task_files(tasks_dir)
            if not task_files:
                error = InvalidTaskDefinitionException(
                    f"No task files found in directory: {tasks_dir}",
                    str(tasks_dir),
                )
                log_operation_failure(logger, "benchmark_tasks_directory_load", error, **operation_context)
                raise error

            logger.info(
                "Discovered task files",
                extra={
                    "event": "benchmark_task_files_discovered",
                    "domain": self.domain,
                    "tasks_dir": str(tasks_dir),
                    "task_file_count": len(task_files),
                    "task_files": [str(path.relative_to(tasks_dir)) for path in task_files],
                },
            )

            # Load tasks from all files
            all_tasks: Dict[str, Task] = {}
            for task_file in task_files:
                logger.info(
                    "Loading benchmark task file",
                    extra={
                        "event": "benchmark_task_file_loading",
                        "task_file": str(task_file),
                        "domain": self.domain,
                    },
                )
                file_tasks = self._load_tasks_from_single_file(task_file)

                # Check for duplicate task IDs across files
                for task_id in file_tasks:
                    if task_id in all_tasks:
                        error_message = (
                            f"Duplicate task_id '{task_id}' found in {task_file}. "
                            "Previously defined in another task file."
                        )
                        error = InvalidTaskDefinitionException(error_message, str(task_file))
                        log_operation_failure(
                            logger,
                            "benchmark_tasks_directory_load",
                            error,
                            duplicate_task_id=task_id,
                            conflicting_file=str(task_file),
                            **operation_context,
                        )
                        raise error

                all_tasks.update(file_tasks)

            # Validate role configuration for orchestrated tasks
            self._validate_dependency_roles(all_tasks)

            log_operation_success(
                logger,
                "benchmark_tasks_directory_load",
                task_count=len(all_tasks),
                task_file_count=len(task_files),
                **operation_context,
            )
            return all_tasks

        except InvalidTaskDefinitionException:
            raise
        except Exception as exc:  # pragma: no cover - fail fast for unexpected issues
            log_operation_failure(logger, "benchmark_tasks_directory_load", exc, **operation_context)
            raise

    def load_tasks_from_file(self, tasks_file_path: str) -> Dict[str, Task]:
        """
        DEPRECATED: Load and parse YAML task definitions from single file into Task objects.

        This method is deprecated. Use load_tasks_from_directory() instead.

        Args:
            tasks_file_path: Path to the YAML tasks definition file

        Returns:
            Dictionary mapping task_id to Task objects

        Raises:
            InvalidTaskDefinitionException: If YAML is invalid or malformed
        """
        tasks_path = Path(tasks_file_path)
        operation_context = {
            "domain": self.domain,
            "tasks_file": str(tasks_path),
        }
        log_operation_start(logger, "benchmark_tasks_file_load", **operation_context)

        try:
            if not tasks_path.exists():
                error = InvalidTaskDefinitionException(f"Tasks file not found: {tasks_path}", str(tasks_path))
                log_operation_failure(logger, "benchmark_tasks_file_load", error, **operation_context)
                raise error

            with open(tasks_path, "r", encoding="utf-8") as file:
                self.yaml_data = yaml.safe_load(file)

            if not isinstance(self.yaml_data, dict):
                error = InvalidTaskDefinitionException("YAML root must be a dictionary", str(tasks_path))
                log_operation_failure(logger, "benchmark_tasks_file_load", error, **operation_context)
                raise error

            yaml_domain = self.yaml_data.get(FIELD_DOMAIN)
            if yaml_domain != self.domain:
                error = InvalidTaskDefinitionException(
                    f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'",
                    str(tasks_path),
                )
                log_operation_failure(logger, "benchmark_tasks_file_load", error, **operation_context)
                raise error

            self.permanent_environment = self.yaml_data.get(FIELD_PERMANENT_ENVIRONMENT)
            logger.info(
                "Permanent environment configuration processed",
                extra={
                    "event": "benchmark_permanent_environment_detected",
                    "domain": self.domain,
                    "tasks_file": str(tasks_path),
                    "permanent_environment": self.permanent_environment,
                    "configured": bool(self.permanent_environment),
                },
            )

            self._parse_global_defaults()
            self._parse_benchmark_config()

            tasks_data = self.yaml_data.get(FIELD_TASKS, [])
            if not isinstance(tasks_data, list):
                error = InvalidTaskDefinitionException("Tasks must be a list", str(tasks_path))
                log_operation_failure(logger, "benchmark_tasks_file_load", error, **operation_context)
                raise error

            executors_data = self.yaml_data.get(FIELD_EXECUTORS)
            if executors_data is not None:
                if not isinstance(executors_data, list):
                    error = InvalidTaskDefinitionException("Executors must be a list", str(tasks_path))
                    log_operation_failure(logger, "benchmark_tasks_file_load", error, **operation_context)
                    raise error

                self.allowed_executors = executors_data
            else:
                self.allowed_executors = None

            logger.info(
                "Executors configuration processed",
                extra={
                    "event": "benchmark_executors_configuration_processed",
                    "tasks_file": str(tasks_path),
                    "domain": self.domain,
                    "executor_count": len(self.allowed_executors or []),
                    "restricts_executors": self.allowed_executors is not None,
                },
            )

            tasks: Dict[str, Task] = {}
            logger.info(
                "Parsing tasks from YAML file",
                extra={
                    "event": "benchmark_tasks_parse_start",
                    "tasks_file": str(tasks_path),
                    "domain": self.domain,
                    "task_count": len(tasks_data),
                },
            )

            for index, task_data in enumerate(tasks_data):
                logger.debug(
                    "Parsing benchmark task entry",
                    extra={
                        "event": "benchmark_task_entry_parsing",
                        "tasks_file": str(tasks_path),
                        "domain": self.domain,
                        "task_index": index,
                        "total_tasks": len(tasks_data),
                    },
                )
                task = self._parse_task(task_data)
                tasks[task.task_id] = task

            # Validate dependency role configuration
            self._validate_dependency_roles(tasks)

            log_operation_success(
                logger,
                "benchmark_tasks_file_load",
                task_count=len(tasks),
                **operation_context,
            )
            return tasks

        except yaml.YAMLError as exc:
            error = InvalidTaskDefinitionException(f"YAML parsing error: {exc}", str(tasks_path))
            log_operation_failure(logger, "benchmark_tasks_file_load", error, **operation_context)
            raise error
        except InvalidTaskDefinitionException:
            raise
        except Exception as exc:  # pragma: no cover - unexpected errors should fail fast
            log_operation_failure(logger, "benchmark_tasks_file_load", exc, **operation_context)
            raise InvalidTaskDefinitionException(f"Error loading tasks: {exc}", str(tasks_path))

    def _load_global_config(self, global_config_path: Path) -> None:
        """
        Load global configuration from global.yaml file.

        Args:
            global_config_path: Path to global.yaml file

        Raises:
            InvalidTaskDefinitionException: If global config is invalid
        """
        operation_context = {
            "domain": self.domain,
            "global_config_path": str(global_config_path),
        }
        log_operation_start(logger, "benchmark_global_config_load", **operation_context)

        try:
            with open(global_config_path, "r", encoding="utf-8") as file:
                global_data = yaml.safe_load(file)

            if not isinstance(global_data, dict):
                error = InvalidTaskDefinitionException("Global YAML root must be a dictionary", str(global_config_path))
                log_operation_failure(logger, "benchmark_global_config_load", error, **operation_context)
                raise error

            yaml_domain = global_data.get(FIELD_DOMAIN)
            if yaml_domain != self.domain:
                error = InvalidTaskDefinitionException(
                    f"Domain mismatch in global.yaml: expected '{self.domain}', got '{yaml_domain}'",
                    str(global_config_path),
                )
                log_operation_failure(logger, "benchmark_global_config_load", error, **operation_context)
                raise error

            self.yaml_data = global_data

            self.permanent_environment = global_data.get(FIELD_PERMANENT_ENVIRONMENT)
            logger.info(
                "Global permanent environment processed",
                extra={
                    "event": "benchmark_global_permanent_environment",
                    "domain": self.domain,
                    "global_config_path": str(global_config_path),
                    "permanent_environment": self.permanent_environment,
                    "configured": bool(self.permanent_environment),
                },
            )

            self._parse_global_defaults()
            self._parse_benchmark_config()

            log_operation_success(
                logger,
                "benchmark_global_config_load",
                **operation_context,
            )

        except yaml.YAMLError as exc:
            error = InvalidTaskDefinitionException(f"YAML parsing error in global.yaml: {exc}", str(global_config_path))
            log_operation_failure(logger, "benchmark_global_config_load", error, **operation_context)
            raise error
        except InvalidTaskDefinitionException:
            raise
        except Exception as exc:  # pragma: no cover - unexpected file failures
            log_operation_failure(logger, "benchmark_global_config_load", exc, **operation_context)
            raise InvalidTaskDefinitionException(f"Error loading global.yaml: {exc}", str(global_config_path))

    def _discover_task_files(self, tasks_dir: Path) -> List[Path]:
        """
        Discover all YAML task files in directory, excluding global.yaml and shared.yaml files.
        Supports both flat and hierarchical organization.

        Args:
            tasks_dir: Path to tasks directory

        Returns:
            Sorted list of task file paths
        """
        task_files = []

        # Find all .yaml and .yml files recursively, excluding global.yaml and shared.yaml
        for pattern in ["**/*.yaml", "**/*.yml"]:
            for yaml_file in tasks_dir.glob(pattern):
                if yaml_file.name not in EXCLUDED_CONFIG_FILES:
                    task_files.append(yaml_file)

        # Sort for deterministic loading order
        task_files.sort()

        logger.debug(
            "Discovered task files",
            extra={
                "event": "benchmark_task_files_discovered_debug",
                "domain": self.domain,
                "tasks_dir": str(tasks_dir),
                "task_file_count": len(task_files),
                "task_files": [str(f.relative_to(tasks_dir)) for f in task_files],
            },
        )
        return task_files

    def _load_tasks_from_single_file(self, task_file_path: Path) -> Dict[str, Task]:
        """
        Load tasks from a single task file. Also loads subtasks and their criteria.

        Args:
            task_file_path: Path to task file

        Returns:
            Dictionary mapping task_id to Task objects

        Raises:
            InvalidTaskDefinitionException: If task file is invalid
        """
        operation_context = {
            "domain": self.domain,
            "task_file": str(task_file_path),
        }
        log_operation_start(logger, "benchmark_task_file_parse", **operation_context)

        try:
            with open(task_file_path, "r", encoding="utf-8") as file:
                file_data = yaml.safe_load(file)

            if not isinstance(file_data, dict):
                error = InvalidTaskDefinitionException("Task file YAML root must be a dictionary", str(task_file_path))
                log_operation_failure(logger, "benchmark_task_file_parse", error, **operation_context)
                raise error

            # Load shared configuration for this directory if it exists
            shared_config = self._load_shared_config(task_file_path.parent)

            # Parse tasks from this file
            tasks_data = file_data.get(FIELD_TASKS, [])
            if not isinstance(tasks_data, list):
                error = InvalidTaskDefinitionException(
                    "'tasks' section must be a list of task objects", str(task_file_path)
                )
                log_operation_failure(logger, "benchmark_task_file_parse", error, **operation_context)
                raise error

            if not tasks_data:
                logger.warning(
                    "Task file contained no tasks",
                    extra={
                        "event": "benchmark_task_file_empty",
                        "domain": self.domain,
                        "task_file": str(task_file_path),
                    },
                )
                log_operation_success(
                    logger,
                    "benchmark_task_file_parse",
                    task_count=0,
                    **operation_context,
                )
                return {}

            tasks = {}
            logger.debug(
                "Benchmark task entries discovered",
                extra={
                    "event": "benchmark_task_entries_discovered",
                    "domain": self.domain,
                    "task_file": str(task_file_path),
                    "task_count": len(tasks_data),
                },
            )

            for i, task_data in enumerate(tasks_data):
                logger.debug(
                    "Parsing task from file",
                    extra={
                        "event": "benchmark_task_parsing",
                        "domain": self.domain,
                        "task_file": str(task_file_path),
                        "task_index": i,
                        "total_tasks": len(tasks_data),
                    },
                )

                # Merge shared config into task data (task-specific takes precedence)
                merged_task_data = self._merge_shared_config(task_data, shared_config)

                task = self._parse_task(merged_task_data)
                tasks[task.task_id] = task
                logger.info(
                    "Benchmark task parsed",
                    extra={
                        "event": "benchmark_task_parsed",
                        "domain": self.domain,
                        "task_file": str(task_file_path),
                        "task_id": task.task_id,
                        "subtask_count": len(task.subtasks),
                    },
                )

            log_operation_success(
                logger,
                "benchmark_task_file_parse",
                task_count=len(tasks),
                **operation_context,
            )
            return tasks

        except yaml.YAMLError as exc:
            error = InvalidTaskDefinitionException(
                f"YAML parsing error in {task_file_path}: {exc}", str(task_file_path)
            )
            log_operation_failure(logger, "benchmark_task_file_parse", error, **operation_context)
            raise error
        except InvalidTaskDefinitionException:
            raise
        except Exception as exc:  # pragma: no cover - fail fast on unexpected errors
            log_operation_failure(logger, "benchmark_task_file_parse", exc, **operation_context)
            raise InvalidTaskDefinitionException(
                f"Error loading task file {task_file_path}: {exc}", str(task_file_path)
            )

    def _load_shared_config(self, directory: Path) -> Dict[str, Any]:
        """
        Load shared configuration from shared.yaml in the given directory.

        Args:
            directory: Directory to check for shared.yaml

        Returns:
            Shared configuration dictionary (empty if no shared.yaml found)
        """
        shared_file = directory / FILENAME_SHARED_CONFIG

        if not shared_file.exists():
            logger.debug(
                "No shared configuration found",
                extra={
                    "event": "benchmark_shared_config_missing",
                    "directory": str(directory),
                },
            )
            return {}

        try:
            with open(shared_file, "r", encoding="utf-8") as file:
                shared_data = yaml.safe_load(file)

            if not isinstance(shared_data, dict):
                raise InvalidTaskDefinitionException("Shared config YAML root must be a dictionary", str(shared_file))

            logger.info(
                "Shared configuration loaded",
                extra={
                    "event": "benchmark_shared_config_loaded",
                    "shared_file": str(shared_file),
                    "keys": list(shared_data.keys()),
                },
            )
            return shared_data

        except yaml.YAMLError as e:
            raise InvalidTaskDefinitionException(f"YAML parsing error in {shared_file}: {e}", str(shared_file))
        except Exception as e:
            raise InvalidTaskDefinitionException(f"Error loading shared config {shared_file}: {e}", str(shared_file))

    def _merge_shared_config(self, task_data: Dict[str, Any], shared_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge shared configuration into task data with proper precedence.

        Task-specific configuration takes precedence over shared configuration.
        Shared config is dynamically merged for all compatible fields.

        Requires explicit opt-in via 'inherit_shared: true' in task_data to use shared config.

        Args:
            task_data: Original task configuration
            shared_config: Shared configuration from directory

        Returns:
            Merged task configuration

        Raises:
            InvalidTaskDefinitionException: If shared config exists but inherit_shared is not true
        """
        if not shared_config:
            return task_data

        # Check if task explicitly opts into shared config inheritance
        inherit_shared = task_data.get(FIELD_INHERIT_SHARED, False)
        task_id = task_data.get(FIELD_TASK_ID, "unknown")

        # If shared config exists but task doesn't explicitly inherit, skip merging
        if not inherit_shared:
            if shared_config:
                logger.debug(
                    "Task not inheriting from shared config (inherit_shared not set)",
                    extra={
                        "event": "benchmark_shared_config_not_inherited",
                        "task_id": task_id,
                        "available_shared_fields": list(shared_config.keys()),
                    },
                )
            return task_data

        # Task explicitly inherits from shared config
        logger.debug(
            "Task inheriting from shared configuration",
            extra={
                "event": "benchmark_shared_config_inheritance_enabled",
                "task_id": task_id,
                "shared_fields": list(shared_config.keys()),
            },
        )

        # Create a deep copy to avoid modifying original data
        merged_data = task_data.copy()

        # Dynamically merge all fields from shared config
        for field_name, shared_field_value in shared_config.items():
            if field_name in merged_data:
                # Both shared and task-specific values exist - deep merge if both are dicts
                task_field_value = merged_data[field_name]

                if isinstance(shared_field_value, dict) and isinstance(task_field_value, dict):
                    # Deep merge: shared first, then task-specific (task overrides shared)
                    merged_data[field_name] = deep_merge_dicts(shared_field_value, task_field_value)
                    logger.debug(
                        f"Merged shared {field_name}",
                        extra={
                            "event": f"benchmark_{field_name}_merged",
                            "task_id": task_id,
                            "shared_keys": list(shared_field_value.keys()),
                            "task_keys": list(task_field_value.keys()),
                            "merged_keys": list(merged_data[field_name].keys()),
                        },
                    )
                else:
                    # Task-specific value takes precedence for non-dict values
                    logger.debug(
                        f"Task-specific {field_name} takes precedence",
                        extra={
                            "event": f"benchmark_{field_name}_override",
                            "task_id": task_id,
                        },
                    )
            else:
                # No task-specific value, use shared entirely
                merged_data[field_name] = (
                    shared_field_value.copy() if isinstance(shared_field_value, dict) else shared_field_value
                )
                logger.debug(
                    f"Applied shared {field_name}",
                    extra={
                        "event": f"benchmark_{field_name}_applied",
                        "task_id": task_id,
                        "shared_keys": (
                            list(shared_field_value.keys()) if isinstance(shared_field_value, dict) else None
                        ),
                    },
                )

        return merged_data

    def get_allowed_executors(self) -> Optional[list[str]]:
        """
        Get the list of allowed executors from the configuration.

        Returns:
            List of allowed executor names, or None if no restriction is configured
        """
        return self.allowed_executors

    def load_benchmark_config(self) -> Dict[str, Any]:
        """
        Get the loaded benchmark configuration.

        Returns:
            Domain-level benchmark configuration
        """
        return self.benchmark_config.copy()

    def get_permanent_environment(self) -> Optional[str]:
        """
        Get the permanent environment configuration.

        Returns:
            Permanent environment name, or None if not configured
        """
        return self.permanent_environment

    def get_global_defaults(self) -> Dict[str, Any]:
        """
        Get the loaded global defaults configuration.

        Returns:
            Global defaults configuration for all tasks
        """
        return self.global_defaults.copy()

    def get_dependency_config(self) -> Dict[str, float]:
        """
        Get dependency resolution configuration from global defaults.

        Returns:
            Dict containing dependency configuration with fallback defaults:
            - wait_seconds: Maximum time to wait for dependencies (default: 10.0)
            - retry_interval: Initial retry interval (default: 0.5)
            - max_retry_interval: Maximum retry interval (default: 2.0)
        """
        dependency_config = self.global_defaults.get(FIELD_DEPENDENCY_CONFIG, {})

        # Provide sensible defaults
        return {
            FIELD_WAIT_SECONDS: dependency_config.get(FIELD_WAIT_SECONDS, 10.0),
            FIELD_RETRY_INTERVAL: dependency_config.get(FIELD_RETRY_INTERVAL, 0.5),
            FIELD_MAX_RETRY_INTERVAL: dependency_config.get(FIELD_MAX_RETRY_INTERVAL, 2.0),
        }

    def _parse_global_defaults(self) -> None:
        """
        Parse global defaults configuration from YAML data.

        Global defaults provide fallback values for execution_config, episode_config,
        and benchmark_config when not specified at task level.
        """
        if self.yaml_data is None:
            return

        global_defaults_data = self.yaml_data.get(FIELD_GLOBAL_DEFAULTS)

        if global_defaults_data is None:
            # No global defaults specified - use empty dict
            self.global_defaults = {}
            logger.info(
                "No global defaults configured",
                extra={
                    "event": "benchmark_global_defaults_missing",
                    "domain": self.domain,
                },
            )
            return

        if not isinstance(global_defaults_data, dict):
            raise InvalidTaskDefinitionException("global_defaults must be a dictionary")

        # Parse global prompts defaults (optional)
        if FIELD_PROMPTS in global_defaults_data:
            prompts_config = global_defaults_data[FIELD_PROMPTS]
            if not isinstance(prompts_config, dict):
                raise InvalidTaskDefinitionException("global_defaults.prompts must be a dictionary")

            # Validate prompt types if provided - partial prompts are allowed in global defaults
            for prompt_type, template_file in prompts_config.items():
                if prompt_type not in REQUIRED_PROMPT_TYPES:
                    raise InvalidTaskDefinitionException(
                        f"global_defaults.prompts contains invalid prompt type '{prompt_type}'. "
                        f"Valid types: {', '.join(REQUIRED_PROMPT_TYPES)}"
                    )
                if not isinstance(template_file, str) or not template_file.strip():
                    raise InvalidTaskDefinitionException(
                        f"global_defaults.prompts.{prompt_type} must be a non-empty string"
                    )

            logger.info(f"Loaded global prompts configuration: {prompts_config}")
        else:
            logger.info("No global prompts defaults specified")

        # Validate structure of global defaults
        valid_sections = [
            FIELD_EXECUTION_CONFIG,
            FIELD_EPISODE_CONFIG,
            FIELD_BENCHMARK_CONFIG,
            FIELD_DEPENDENCY_CONFIG,
            FIELD_PROMPTS,
        ]
        for section_name in global_defaults_data:
            if section_name not in valid_sections:
                raise InvalidTaskDefinitionException(
                    f"Invalid section '{section_name}' in global_defaults. " f"Valid sections are: {valid_sections}"
                )

            if section_name != FIELD_PROMPTS and not isinstance(global_defaults_data[section_name], dict):
                raise InvalidTaskDefinitionException(f"global_defaults.{section_name} must be a dictionary")

        # Store global defaults
        self.global_defaults = global_defaults_data.copy()
        logger.info(
            "Loaded global defaults configuration",
            extra={
                "event": "benchmark_global_defaults_loaded",
                "domain": self.domain,
                "sections": list(self.global_defaults.keys()),
            },
        )

    def _parse_benchmark_config(self) -> None:
        """
        Parse benchmark configuration from YAML data.

        Uses global defaults as fallback if no explicit benchmark_config is provided.
        If global_defaults.benchmark_config exists, it will be used when domain-level
        benchmark_config is missing.

        Raises:
            InvalidTaskDefinitionException: If neither benchmark_config nor global defaults provide episode_attempts
        """
        if self.yaml_data is None:
            raise InvalidTaskDefinitionException("No YAML data loaded")

        benchmark_data = self.yaml_data.get(FIELD_BENCHMARK_CONFIG)
        global_benchmark_defaults = self.global_defaults.get(FIELD_BENCHMARK_CONFIG, {})

        # Merge global defaults with domain-level configuration
        merged_config = {}

        # Start with global defaults
        if global_benchmark_defaults:
            merged_config.update(global_benchmark_defaults)

        # Override with domain-level configuration if provided
        if benchmark_data is not None:
            if not isinstance(benchmark_data, dict):
                raise InvalidTaskDefinitionException("benchmark_config must be a dictionary")
            merged_config.update(benchmark_data)

        # Ensure we have episode_attempts configured
        if FIELD_EPISODE_ATTEMPTS not in merged_config:
            raise InvalidTaskDefinitionException(
                "Missing required 'episode_attempts' in benchmark configuration. "
                "You must specify episode_attempts either in benchmark_config or global_defaults.benchmark_config."
            )

        episode_attempts = merged_config[FIELD_EPISODE_ATTEMPTS]
        if not isinstance(episode_attempts, int) or episode_attempts < 1:
            raise InvalidTaskDefinitionException(
                f"episode_attempts must be a positive integer, got: {episode_attempts}"
            )

        # Store final benchmark configuration
        self.benchmark_config = merged_config
        logger.info(
            "Loaded benchmark configuration",
            extra={
                "event": "benchmark_config_loaded",
                "domain": self.domain,
                "keys": list(self.benchmark_config.keys()),
                "episode_attempts": self.benchmark_config.get(FIELD_EPISODE_ATTEMPTS),
            },
        )

    def _validate_dependency_roles(self, tasks: Dict[str, Task]) -> None:
        """
        Validate role configuration for orchestrated tasks with dependencies.

        Enforces:
        1. Root tasks (tasks with dependents) must have a role defined
        2. Dependent tasks must have a role defined
        3. Role references in dependency chains are valid

        Args:
            tasks: Dictionary mapping task_id to Task objects

        Raises:
            InvalidTaskDefinitionException: If role configuration is invalid
        """
        # Build dependency graph
        task_dependents: Dict[str, List[str]] = {}  # Maps task_id -> list of dependent task_ids

        for task_id, task in tasks.items():
            if task.depends_on_task_id:
                if task.depends_on_task_id not in task_dependents:
                    task_dependents[task.depends_on_task_id] = []
                task_dependents[task.depends_on_task_id].append(task_id)

        # Validate root tasks have roles
        for root_task_id, dependent_task_ids in task_dependents.items():
            root_task = tasks.get(root_task_id)
            if not root_task:
                raise InvalidTaskDefinitionException(
                    f"Task '{dependent_task_ids[0]}' depends on non-existent task '{root_task_id}'"
                )

            if not root_task.role:
                raise InvalidTaskDefinitionException(
                    f"Root task '{root_task_id}' must have a 'role' defined "
                    f"(has {len(dependent_task_ids)} dependent tasks)"
                )

        # Validate dependent tasks have roles (enforced by Task.__init__ but double-check)
        for task_id, task in tasks.items():
            if task.depends_on_task_id and not task.role:
                raise InvalidTaskDefinitionException(f"Dependent task '{task_id}' must have a 'role' defined")

    def _parse_task(self, task_data: Dict[str, Any]) -> Task:
        """
        Parse a single task from YAML data.
        Parses subtasks and subtask criteria
        and sets task.subtasks to that parsed data

        Args:
            task_data: Dictionary containing task definition

        Returns:
            Task instance
        """
        candidate_task_id = task_data.get("task_id", "unknown")
        operation_context = {
            "domain": self.domain,
            "task_id": candidate_task_id,
        }
        failure_context: Dict[str, Any] = {}
        log_operation_start(logger, "benchmark_task_parse", **operation_context)

        try:
            # Updated required fields - removed prompt_template_file, prompts are handled separately
            required_fields = [FIELD_TASK_ID, FIELD_TITLE, FIELD_DESCRIPTION]
            for field in required_fields:
                if field not in task_data:
                    failure_context = {"missing_field": field}
                    raise InvalidTaskDefinitionException(f"Missing required field: {field}")

            task_id = task_data[FIELD_TASK_ID]
            operation_context[FIELD_TASK_ID] = task_id
            title = task_data[FIELD_TITLE]
            description = task_data[FIELD_DESCRIPTION]
            initial_context = task_data.get(FIELD_INITIAL_CONTEXT, {})

            logger.debug(
                "Parsing benchmark task",
                extra={
                    "event": "benchmark_task_parsing_start",
                    "domain": self.domain,
                    "task_id": task_id,
                    "title": title,
                },
            )

            # NEW: Parse prompts with global defaults inheritance
            if FIELD_PROMPTS in task_data:
                task_prompts = task_data[FIELD_PROMPTS]
                if not isinstance(task_prompts, dict):
                    raise InvalidTaskDefinitionException(f"Task '{task_id}' prompts must be a dictionary")
            else:
                task_prompts = {}

            # Inherit from global defaults, allow task-level overrides
            final_prompts = {}
            global_prompts = self.global_defaults.get(FIELD_PROMPTS, {})

            for prompt_type in REQUIRED_PROMPT_TYPES:
                if prompt_type in task_prompts:
                    final_prompts[prompt_type] = task_prompts[prompt_type]
                elif prompt_type in global_prompts:
                    final_prompts[prompt_type] = global_prompts[prompt_type]
                else:
                    raise InvalidTaskDefinitionException(
                        f"Task '{task_id}' missing '{prompt_type}' prompt and no global default provided"
                    )

            # Validate template files exist
            for prompt_type, template_file in final_prompts.items():
                if not isinstance(template_file, str) or not template_file.strip():
                    raise InvalidTaskDefinitionException(
                        f"Task '{task_id}' {prompt_type} template file must be a non-empty string, got: {template_file}"
                    )
                # Note: Template file existence will be validated by PromptGenerator during startup

            logger.debug(
                "Task prompts resolved",
                extra={
                    "event": "benchmark_task_prompts_resolved",
                    "task_id": task_id,
                    "resolved_prompts": final_prompts,
                },
            )

            sandbox_environment = task_data.get(FIELD_ENVIRONMENT) or task_data.get(FIELD_SANDBOX_ENVIRONMENT)
            logger.debug(
                "Resolved task environment settings",
                extra={
                    "event": "benchmark_task_environment_resolved",
                    "task_id": task_id,
                    "resolved_environment": sandbox_environment,
                    "environment_field": task_data.get(FIELD_ENVIRONMENT),
                    "sandbox_environment_field": task_data.get(FIELD_SANDBOX_ENVIRONMENT),
                },
            )

            task_execution_config = task_data.get(FIELD_EXECUTION_CONFIG, {})
            global_execution_defaults = self.global_defaults.get(FIELD_EXECUTION_CONFIG, {})

            execution_config = deep_merge_dicts(global_execution_defaults, task_execution_config)

            # Validate executors section exists and has at least one executor configured
            if FIELD_EXECUTORS not in execution_config or not execution_config[FIELD_EXECUTORS]:
                failure_context = {"missing_field": f"{FIELD_EXECUTION_CONFIG}.{FIELD_EXECUTORS}"}
                message = f"Task '{task_id}' missing required {FIELD_EXECUTION_CONFIG}.{FIELD_EXECUTORS} section"
                raise InvalidTaskDefinitionException(message)

            if not isinstance(execution_config[FIELD_EXECUTORS], dict):
                failure_context = {
                    "invalid_field": f"{FIELD_EXECUTION_CONFIG}.{FIELD_EXECUTORS}",
                    "value": execution_config[FIELD_EXECUTORS],
                }
                message = f"Task '{task_id}' {FIELD_EXECUTION_CONFIG}.{FIELD_EXECUTORS} must be a dictionary"
                raise InvalidTaskDefinitionException(message)

            # Validate each executor has a timeout configured
            for executor_type, executor_config in execution_config[FIELD_EXECUTORS].items():
                if not isinstance(executor_config, dict):
                    failure_context = {
                        "invalid_field": f"{FIELD_EXECUTION_CONFIG}.{FIELD_EXECUTORS}.{executor_type}",
                        "value": executor_config,
                    }
                    message = f"Task '{task_id}' executor config for '{executor_type}' must be a dictionary"
                    raise InvalidTaskDefinitionException(message)

                if FIELD_TIMEOUT not in executor_config:
                    failure_context = {
                        "missing_field": f"{FIELD_EXECUTION_CONFIG}.{FIELD_EXECUTORS}.{executor_type}.{FIELD_TIMEOUT}"
                    }
                    message = f"Task '{task_id}' executor '{executor_type}' missing required {FIELD_TIMEOUT}"
                    raise InvalidTaskDefinitionException(message)

                timeout = executor_config[FIELD_TIMEOUT]
                if not isinstance(timeout, int) or timeout <= 0:
                    failure_context = {
                        "invalid_field": (
                            f"{FIELD_EXECUTION_CONFIG}.{FIELD_EXECUTORS}." f"{executor_type}.{FIELD_TIMEOUT}"
                        ),
                        "value": timeout,
                    }
                    message = (
                        f"Task '{task_id}' executor '{executor_type}' {FIELD_TIMEOUT} "
                        f"must be positive int, got: {timeout}"
                    )
                    raise InvalidTaskDefinitionException(message)

            task_episode_config = task_data.get(FIELD_EPISODE_CONFIG, {})
            global_episode_defaults = self.global_defaults.get(FIELD_EPISODE_CONFIG, {})

            episode_config = deep_merge_dicts(global_episode_defaults, task_episode_config)

            if FIELD_MAX_STEPS not in episode_config:
                failure_context = {"missing_field": f"{FIELD_EPISODE_CONFIG}.{FIELD_MAX_STEPS}"}
                message = (
                    f"Task '{task_id}' missing required {FIELD_EPISODE_CONFIG}.{FIELD_MAX_STEPS} "
                    "(no implicit default)"
                )
                raise InvalidTaskDefinitionException(message)
            if not isinstance(episode_config[FIELD_MAX_STEPS], int) or episode_config[FIELD_MAX_STEPS] <= 0:
                failure_context = {
                    "invalid_field": f"{FIELD_EPISODE_CONFIG}.{FIELD_MAX_STEPS}",
                    "value": episode_config[FIELD_MAX_STEPS],
                }
                message = (
                    f"Task '{task_id}' {FIELD_EPISODE_CONFIG}.{FIELD_MAX_STEPS} must be positive int, "
                    f"got: {episode_config[FIELD_MAX_STEPS]}"
                )
                raise InvalidTaskDefinitionException(message)

            # Validate transcript_config if present
            if FIELD_TRANSCRIPT_CONFIG in episode_config:
                transcript_config = episode_config[FIELD_TRANSCRIPT_CONFIG]

                if not isinstance(transcript_config, dict):
                    failure_context = {
                        "invalid_field": f"{FIELD_EPISODE_CONFIG}.{FIELD_TRANSCRIPT_CONFIG}",
                        "value": transcript_config,
                    }
                    raise InvalidTaskDefinitionException(
                        f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG} must be a dictionary"
                    )

                # Validate websocket section if present
                if "websocket" in transcript_config:
                    websocket_config = transcript_config["websocket"]
                    if not isinstance(websocket_config, dict):
                        raise InvalidTaskDefinitionException(
                            f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket must be a dictionary"
                        )

                    # Validate top-level connection fields
                    for field_name, field_type in [
                        ("connection_timeout", (int, float)),
                        ("ping_interval", (int, float)),
                        ("pong_timeout", (int, float)),
                        ("initial_reconnect_delay", (int, float)),
                        ("max_reconnect_delay", (int, float)),
                        ("reconnect_backoff_multiplier", (int, float)),
                        ("max_reconnect_attempts", int),
                    ]:
                        if field_name in websocket_config:
                            value = websocket_config[field_name]
                            if not isinstance(value, field_type) or value <= 0:  # type: ignore[arg-type]
                                raise InvalidTaskDefinitionException(
                                    f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket.{field_name} "
                                    f"must be positive number"
                                )

                    if "reconnect_enabled" in websocket_config:
                        if not isinstance(websocket_config["reconnect_enabled"], bool):
                            raise InvalidTaskDefinitionException(
                                f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket.reconnect_enabled "
                                f"must be boolean"
                            )

                    # Validate push configuration if present
                    if "push" in websocket_config:
                        push_config = websocket_config["push"]
                        if not isinstance(push_config, dict):
                            raise InvalidTaskDefinitionException(
                                f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket.push must be a dictionary"
                            )

                        for field_name, field_type in [
                            ("confirmation_timeout", (int, float)),
                            ("max_retry_attempts", int),
                            ("retry_backoff_multiplier", (int, float)),
                        ]:
                            if field_name in push_config:
                                value = push_config[field_name]
                                if not isinstance(value, field_type) or value <= 0:  # type: ignore[arg-type]
                                    raise InvalidTaskDefinitionException(
                                        f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket.push.{field_name} "
                                        f"must be positive number"
                                    )

                        for bool_field in ["enabled", "retry_enabled"]:
                            if bool_field in push_config:
                                if not isinstance(push_config[bool_field], bool):
                                    raise InvalidTaskDefinitionException(
                                        f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket.push.{bool_field} "
                                        f"must be boolean"
                                    )

                    # Validate pull configuration if present
                    if "pull" in websocket_config:
                        pull_config = websocket_config["pull"]
                        if not isinstance(pull_config, dict):
                            raise InvalidTaskDefinitionException(
                                f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket.pull must be a dictionary"
                            )

                        for field_name, field_type in [
                            ("event_timeout", (int, float)),
                            ("sync_timeout", (int, float)),
                            ("event_queue_max_size", int),
                        ]:
                            if field_name in pull_config:
                                value = pull_config[field_name]
                                if not isinstance(value, field_type) or value <= 0:  # type: ignore[arg-type]
                                    raise InvalidTaskDefinitionException(
                                        f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket.pull.{field_name} "
                                        f"must be positive number"
                                    )

                        for bool_field in ["enabled", "blocking"]:
                            if bool_field in pull_config:
                                if not isinstance(pull_config[bool_field], bool):
                                    raise InvalidTaskDefinitionException(
                                        f"Task '{task_id}' {FIELD_TRANSCRIPT_CONFIG}.websocket.pull.{bool_field} "
                                        f"must be boolean"
                                    )

            task_benchmark_config = task_data.get(FIELD_BENCHMARK_CONFIG, {})
            global_benchmark_defaults = self.global_defaults.get(FIELD_BENCHMARK_CONFIG, {})

            if not isinstance(task_benchmark_config, dict):
                failure_context = {"invalid_field": FIELD_BENCHMARK_CONFIG, "value": task_benchmark_config}
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}' {FIELD_BENCHMARK_CONFIG} must be a dictionary if provided"
                )

            if FIELD_EPISODE_ATTEMPTS in task_benchmark_config:
                episode_attempts = task_benchmark_config[FIELD_EPISODE_ATTEMPTS]
                if not isinstance(episode_attempts, int) or episode_attempts < 1:
                    failure_context = {
                        "invalid_field": f"{FIELD_BENCHMARK_CONFIG}.{FIELD_EPISODE_ATTEMPTS}",
                        "value": episode_attempts,
                    }
                    raise InvalidTaskDefinitionException(
                        f"Task '{task_id}' {FIELD_EPISODE_ATTEMPTS} must be a positive integer, got: {episode_attempts}"
                    )

            merged_benchmark_config: Dict[str, Any] = {}
            merged_benchmark_config = deep_merge_dicts(merged_benchmark_config, global_benchmark_defaults)
            merged_benchmark_config = deep_merge_dicts(merged_benchmark_config, self.benchmark_config)
            merged_benchmark_config = deep_merge_dicts(merged_benchmark_config, task_benchmark_config)
            if (
                FIELD_EPISODE_ATTEMPTS not in merged_benchmark_config
                or merged_benchmark_config[FIELD_EPISODE_ATTEMPTS] < 1
            ):
                failure_context = {"missing_field": f"{FIELD_BENCHMARK_CONFIG}.{FIELD_EPISODE_ATTEMPTS}"}
                message = (
                    f"Task '{task_id}' does not have valid {FIELD_EPISODE_ATTEMPTS} configuration. "
                    f"Each task must have {FIELD_EPISODE_ATTEMPTS} either from domain-level {FIELD_BENCHMARK_CONFIG} "
                    "or task-level override."
                )
                raise InvalidTaskDefinitionException(message)

            # NEW FORMAT ONLY: Load submission_evaluation_config and step_evaluation_config
            # NO backward compatibility with old evaluation_config
            submission_evaluation_config = task_data.get(FIELD_SUBMISSION_EVALUATION_CONFIG)
            step_evaluation_config = task_data.get(FIELD_STEP_EVALUATION_CONFIG)

            if not submission_evaluation_config:
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}' missing required {FIELD_SUBMISSION_EVALUATION_CONFIG} section. "
                    "All tasks MUST have submission evaluation configuration."
                )

            # Validate submission config
            validate_submission_evaluation_config(submission_evaluation_config, task_id)

            # step_evaluation_config is optional
            if step_evaluation_config:
                validate_step_evaluation_config(step_evaluation_config, task_id)

            # Parse initial_files if present
            initial_files = task_data.get(FIELD_INITIAL_FILES)
            if initial_files is not None:
                if not isinstance(initial_files, dict):
                    failure_context = {"invalid_field": FIELD_INITIAL_FILES, "value": initial_files}
                    raise InvalidTaskDefinitionException(
                        f"Task '{task_id}' {FIELD_INITIAL_FILES} must be a dictionary if provided"
                    )
                # Validate that all values are strings (paths)
                for dest_path, source_path in initial_files.items():
                    if not isinstance(dest_path, str) or not isinstance(source_path, str):
                        failure_context = {
                            "invalid_field": FIELD_INITIAL_FILES,
                            "dest_path": dest_path,
                            "source_path": source_path,
                        }
                        raise InvalidTaskDefinitionException(
                            f"Task '{task_id}' {FIELD_INITIAL_FILES} entries must be string -> string mappings"
                        )

            subtasks_data = task_data.get(FIELD_SUBTASKS, [])
            subtasks = []

            logger.debug(
                "Parsing task subtasks",
                extra={
                    "event": "benchmark_task_subtasks_parsing",
                    "task_id": task_id,
                    "subtask_count": len(subtasks_data),
                },
            )
            for index, subtask_data in enumerate(subtasks_data):
                logger.debug(
                    "Parsing subtask",
                    extra={
                        "event": "benchmark_subtask_parsing",
                        "task_id": task_id,
                        "subtask_index": index,
                        "total_subtasks": len(subtasks_data),
                    },
                )

                # Parse subtask
                subtask = self._parse_subtask(subtask_data, task_id, step_evaluation_config)
                subtasks.append(subtask)
                logger.debug(
                    "Subtask parsed",
                    extra={
                        "event": "benchmark_subtask_parsed",
                        "task_id": task_id,
                        "subtask_id": subtask.subtask_id,
                    },
                )

            # Create Task with new prompts parameter (instead of prompt_template_file)
            # Derive allowed_executors from executors config keys
            allowed_executors = (
                list(execution_config[FIELD_EXECUTORS].keys()) if FIELD_EXECUTORS in execution_config else None
            )

            # Read role from YAML (optional, required for orchestrated tasks)
            role = task_data.get(FIELD_ROLE)
            depends_on_task_id = task_data.get(FIELD_DEPENDS_ON_TASK_ID)

            task = Task(
                task_id=task_id,
                domain=self.domain,
                title=title,
                description=description,
                prompts=final_prompts,  # Use the multi-prompt structure
                subtasks=subtasks,
                initial_context=initial_context,
                environment=sandbox_environment,
                allowed_executors=allowed_executors,
                execution_config=execution_config,
                episode_config=episode_config,
                benchmark_config=merged_benchmark_config,
                submission_evaluation_config=submission_evaluation_config,
                step_evaluation_config=step_evaluation_config,
                depends_on_task_id=depends_on_task_id,
                role=role,
                initial_files=initial_files,
            )

            log_operation_success(
                logger,
                "benchmark_task_parse",
                subtask_count=len(subtasks),
                **operation_context,
            )
            return task

        except InvalidTaskDefinitionException as exc:
            log_operation_failure(
                logger,
                "benchmark_task_parse",
                exc,
                **operation_context,
                **failure_context,
            )
            raise
        except Exception as exc:  # pragma: no cover - fail fast for unexpected parsing errors
            log_operation_failure(logger, "benchmark_task_parse", exc, **operation_context)
            raise

    def _parse_subtask(
        self, subtask_data: Dict[str, Any], task_id: str, step_evaluation_config: Optional[Dict[str, Any]]
    ) -> SubTask:
        """
        Parse a single subtask from YAML data. Parse criteria, weight, max_score.

        NEW FORMAT ONLY: Supports scoring: { max_score: X } structure.
        NO backward compatibility with old direct max_score field.

        Args:
            subtask_data: Dictionary containing subtask definition
            task_id: ID of the parent task

        Returns:
            SubTask instance
        """
        required_fields = [FIELD_SUBTASK_ID, FIELD_TITLE, FIELD_DESCRIPTION, FIELD_OBJECTIVE]
        for field in required_fields:
            if field not in subtask_data:
                raise InvalidTaskDefinitionException(f"Missing required subtask field: {field}")

        # Handle subtask_strategy and criteria validation
        subtask_defined_step_evaluation_config = subtask_data.get(FIELD_STEP_EVALUATION_CONFIG)

        if subtask_defined_step_evaluation_config:
            validate_step_evaluation_config(subtask_defined_step_evaluation_config, task_id)
            subtask_strategy = subtask_defined_step_evaluation_config.get(FIELD_STRATEGY)
            subtask_criteria = subtask_defined_step_evaluation_config.get(FIELD_CRITERIA, {})
            subtask_scoring = subtask_defined_step_evaluation_config.get(FIELD_SCORING, {})

        else:
            # if strategy is not defined per subtask, then use the step_evaluation_config
            if step_evaluation_config:
                subtask_strategy = step_evaluation_config.get(FIELD_STRATEGY)
                subtask_criteria = step_evaluation_config.get(FIELD_CRITERIA, {})
                subtask_scoring = step_evaluation_config.get(FIELD_SCORING, {})
            else:
                # no step evaluation config defined at task or subtask level
                subtask_strategy = None
                subtask_criteria = {}
                subtask_scoring = {}

        # Support direct scoring field in subtask (merge/override with step_evaluation_config scoring)
        direct_scoring = subtask_data.get(FIELD_SCORING)
        if direct_scoring:
            if not isinstance(direct_scoring, dict):
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}': subtask '{subtask_data.get(FIELD_SUBTASK_ID)}' "
                    f"{FIELD_SCORING} must be a dictionary"
                )
            # Merge direct scoring with config-based scoring (direct takes precedence)
            subtask_scoring = {**subtask_scoring, **direct_scoring}

        return SubTask(
            subtask_id=subtask_data[FIELD_SUBTASK_ID],
            task_id=task_id,
            title=subtask_data[FIELD_TITLE],
            description=subtask_data[FIELD_DESCRIPTION],
            objective=subtask_data[FIELD_OBJECTIVE],
            hints=subtask_data.get(FIELD_HINTS),
            subtask_strategy=subtask_strategy,  # Use validated strategy
            subtask_criteria=subtask_criteria,  # Use validated criteria
            subtask_weight=subtask_scoring.get(FIELD_WEIGHT, DEFAULT_WEIGHT),
            subtask_max_score=subtask_scoring.get(FIELD_MAX_SCORE, DEFAULT_MAX_SCORE),
        )
