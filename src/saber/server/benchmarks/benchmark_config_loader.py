"""BenchmarkConfigLoader for loading and parsing YAML task definitions and benchmark configurations."""

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .exceptions import InvalidTaskDefinitionException
from .subtask import SubTask
from .task import Task

logger = getLogger(__name__)


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Args:
        base: Base dictionary (e.g., global defaults)
        override: Override dictionary (e.g., task specific config)

    Returns:
        New dictionary with deep merged values
    """
    result = base.copy()

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
        logger.info(f"Loading tasks from directory: {tasks_dir}")

        if not tasks_dir.exists():
            raise InvalidTaskDefinitionException(f"Tasks directory not found: {tasks_dir}", str(tasks_dir))

        if not tasks_dir.is_dir():
            raise InvalidTaskDefinitionException(f"Tasks path is not a directory: {tasks_dir}", str(tasks_dir))

        # Load global configuration first
        global_config_path = tasks_dir / "global.yaml"
        if not global_config_path.exists():
            raise InvalidTaskDefinitionException(
                f"Missing required global.yaml in tasks directory: {global_config_path}", str(global_config_path)
            )

        self._load_global_config(global_config_path)

        # Discover all task files (exclude global.yaml)
        task_files = self._discover_task_files(tasks_dir)
        if not task_files:
            raise InvalidTaskDefinitionException(f"No task files found in directory: {tasks_dir}", str(tasks_dir))

        logger.info(f"Found {len(task_files)} task files to load")

        # Load tasks from all files
        all_tasks = {}
        for task_file in task_files:
            logger.info(f"Loading tasks from: {task_file}")
            file_tasks = self._load_tasks_from_single_file(task_file)

            # Check for duplicate task IDs across files
            for task_id in file_tasks:
                if task_id in all_tasks:
                    raise InvalidTaskDefinitionException(
                        f"Duplicate task_id '{task_id}' found in {task_file}. "
                        f"Previously defined in another task file.",
                        str(task_file),
                    )

            all_tasks.update(file_tasks)

        logger.info(f"Successfully loaded {len(all_tasks)} tasks from {len(task_files)} files")
        return all_tasks

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
        logger.info(f"Loading tasks from YAML file: {tasks_path}")

        try:
            if not tasks_path.exists():
                logger.error(f"Tasks file not found: {tasks_path}")
                raise InvalidTaskDefinitionException(f"Tasks file not found: {tasks_path}", str(tasks_path))

            with open(tasks_path, "r", encoding="utf-8") as file:
                self.yaml_data = yaml.safe_load(file)
                logger.debug(f"Successfully loaded YAML data from {tasks_path}")

            if not isinstance(self.yaml_data, dict):
                logger.error("YAML root must be a dictionary (got sequence or scalar)")
                raise InvalidTaskDefinitionException("YAML root must be a dictionary", str(tasks_path))

            # Validate domain consistency
            yaml_domain = self.yaml_data.get("domain")
            if yaml_domain != self.domain:
                logger.error(f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'")
                raise InvalidTaskDefinitionException(
                    f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'",
                    str(tasks_path),
                )

            # Parse permanent environment configuration (optional)
            self.permanent_environment = self.yaml_data.get("permanent_environment")
            if self.permanent_environment:
                logger.info(f"Found permanent environment configuration: {self.permanent_environment}")
            else:
                logger.info("No permanent environment configuration found")

            # Parse global defaults configuration (optional)
            self._parse_global_defaults()

            # Parse benchmark configuration (optional)
            self._parse_benchmark_config()

            # Parse tasks
            tasks_data = self.yaml_data.get("tasks", [])
            if not isinstance(tasks_data, list):
                logger.error("Tasks section must be a list of task objects")
                raise InvalidTaskDefinitionException("Tasks must be a list", str(tasks_path))

            # Parse executors configuration (optional)
            executors_data = self.yaml_data.get("executors")
            if executors_data is not None:
                if not isinstance(executors_data, list):
                    logger.error("executors field must be a list")
                    raise InvalidTaskDefinitionException("Executors must be a list", str(tasks_path))

                self.allowed_executors = executors_data
                logger.info(f"Loaded executor configuration: {self.allowed_executors}")
            else:
                self.allowed_executors = None
                logger.info("No executor configuration found, all executors will be available")

            tasks = {}
            logger.info(f"Found {len(tasks_data)} tasks to load")

            for i, task_data in enumerate(tasks_data):
                logger.debug(f"Parsing task {i+1}/{len(tasks_data)}")
                task = self._parse_task(task_data)
                tasks[task.task_id] = task
                logger.info(f"Successfully loaded task '{task.task_id}' " f"with {len(task.subtasks)} subtasks")

            logger.info(f"BenchmarkConfigLoader completed. Loaded {len(tasks)} tasks " f"for domain '{self.domain}'")

            return tasks

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise InvalidTaskDefinitionException(f"YAML parsing error: {e}", str(tasks_path))
        except Exception as e:
            if isinstance(e, InvalidTaskDefinitionException):
                raise
            logger.error(f"Unexpected error loading tasks: {e}")
            raise InvalidTaskDefinitionException(f"Error loading tasks: {e}", str(tasks_path))

    def _load_global_config(self, global_config_path: Path) -> None:
        """
        Load global configuration from global.yaml file.

        Args:
            global_config_path: Path to global.yaml file

        Raises:
            InvalidTaskDefinitionException: If global config is invalid
        """
        logger.info(f"Loading global configuration from: {global_config_path}")

        try:
            with open(global_config_path, "r", encoding="utf-8") as file:
                global_data = yaml.safe_load(file)
                logger.debug(f"Successfully loaded global YAML data from {global_config_path}")

            if not isinstance(global_data, dict):
                raise InvalidTaskDefinitionException("Global YAML root must be a dictionary", str(global_config_path))

            # Validate domain consistency
            yaml_domain = global_data.get("domain")
            if yaml_domain != self.domain:
                raise InvalidTaskDefinitionException(
                    f"Domain mismatch in global.yaml: expected '{self.domain}', got '{yaml_domain}'",
                    str(global_config_path),
                )

            # Store global data for processing
            self.yaml_data = global_data

            # Parse permanent environment configuration (optional)
            self.permanent_environment = global_data.get("permanent_environment")
            if self.permanent_environment:
                logger.info(f"Found permanent environment configuration: {self.permanent_environment}")
            else:
                logger.info("No permanent environment configuration found")

            # Parse global defaults configuration (optional)
            self._parse_global_defaults()

            # Parse benchmark configuration (optional)
            self._parse_benchmark_config()

            logger.info("Global configuration loaded successfully")

        except yaml.YAMLError as e:
            raise InvalidTaskDefinitionException(f"YAML parsing error in global.yaml: {e}", str(global_config_path))
        except Exception as e:
            if isinstance(e, InvalidTaskDefinitionException):
                raise
            raise InvalidTaskDefinitionException(f"Error loading global.yaml: {e}", str(global_config_path))

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
                if yaml_file.name not in ["global.yaml", "global.yml", "shared.yaml", "shared.yml"]:
                    task_files.append(yaml_file)

        # Sort for deterministic loading order
        task_files.sort()

        logger.debug(f"Discovered {len(task_files)} task files: {[str(f.relative_to(tasks_dir)) for f in task_files]}")
        return task_files

    def _load_tasks_from_single_file(self, task_file_path: Path) -> Dict[str, Task]:
        """
        Load tasks from a single task file.

        Args:
            task_file_path: Path to task file

        Returns:
            Dictionary mapping task_id to Task objects

        Raises:
            InvalidTaskDefinitionException: If task file is invalid
        """
        try:
            with open(task_file_path, "r", encoding="utf-8") as file:
                file_data = yaml.safe_load(file)
                logger.debug(f"Successfully loaded YAML data from {task_file_path}")

            if not isinstance(file_data, dict):
                raise InvalidTaskDefinitionException("Task file YAML root must be a dictionary", str(task_file_path))

            # Load shared configuration for this directory if it exists
            shared_config = self._load_shared_config(task_file_path.parent)

            # Parse tasks from this file
            tasks_data = file_data.get("tasks", [])
            if not isinstance(tasks_data, list):
                raise InvalidTaskDefinitionException(
                    "'tasks' section must be a list of task objects", str(task_file_path)
                )

            if not tasks_data:
                logger.warning(f"No tasks found in file: {task_file_path}")
                return {}

            tasks = {}
            logger.debug(f"Found {len(tasks_data)} tasks in {task_file_path}")

            for i, task_data in enumerate(tasks_data):
                logger.debug(f"Parsing task {i+1}/{len(tasks_data)} from {task_file_path}")

                # Merge shared config into task data (task-specific takes precedence)
                merged_task_data = self._merge_shared_config(task_data, shared_config)

                task = self._parse_task(merged_task_data)
                tasks[task.task_id] = task
                logger.info(f"Successfully loaded task '{task.task_id}' from {task_file_path.name}")

            return tasks

        except yaml.YAMLError as e:
            raise InvalidTaskDefinitionException(f"YAML parsing error in {task_file_path}: {e}", str(task_file_path))
        except Exception as e:
            if isinstance(e, InvalidTaskDefinitionException):
                raise
            raise InvalidTaskDefinitionException(f"Error loading task file {task_file_path}: {e}", str(task_file_path))

    def _load_shared_config(self, directory: Path) -> Dict[str, Any]:
        """
        Load shared configuration from shared.yaml in the given directory.

        Args:
            directory: Directory to check for shared.yaml

        Returns:
            Shared configuration dictionary (empty if no shared.yaml found)
        """
        shared_file = directory / "shared.yaml"

        if not shared_file.exists():
            logger.debug(f"No shared.yaml found in {directory}")
            return {}

        try:
            with open(shared_file, "r", encoding="utf-8") as file:
                shared_data = yaml.safe_load(file)
                logger.debug(f"Successfully loaded shared config from {shared_file}")

            if not isinstance(shared_data, dict):
                raise InvalidTaskDefinitionException("Shared config YAML root must be a dictionary", str(shared_file))

            logger.info(f"Loaded shared configuration from {shared_file}")
            return shared_data

        except yaml.YAMLError as e:
            raise InvalidTaskDefinitionException(f"YAML parsing error in {shared_file}: {e}", str(shared_file))
        except Exception as e:
            raise InvalidTaskDefinitionException(f"Error loading shared config {shared_file}: {e}", str(shared_file))

    def _merge_shared_config(self, task_data: Dict[str, Any], shared_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge shared configuration into task data with proper precedence.

        Task-specific configuration takes precedence over shared configuration.
        Shared config is applied to initial_context and any other compatible fields.

        Args:
            task_data: Original task configuration
            shared_config: Shared configuration from directory

        Returns:
            Merged task configuration
        """
        if not shared_config:
            return task_data

        # Create a deep copy to avoid modifying original data
        merged_data = task_data.copy()

        # Merge initial_context if both exist
        if "initial_context" in shared_config and "initial_context" in merged_data:
            # Task-specific initial_context takes precedence, but we deep merge with shared
            shared_initial_context = shared_config["initial_context"]
            task_initial_context = merged_data["initial_context"]

            # Deep merge: shared first, then task-specific (task overrides shared)
            merged_data["initial_context"] = deep_merge_dicts(shared_initial_context, task_initial_context)

            logger.debug(f"Deep merged shared initial_context for task {task_data.get('task_id', 'unknown')}")

        elif "initial_context" in shared_config:
            # No task-specific initial_context, use shared
            merged_data["initial_context"] = shared_config["initial_context"].copy()
            logger.debug(f"Applied shared initial_context for task {task_data.get('task_id', 'unknown')}")

        # Could extend this to merge other configuration sections as needed
        # For now, we focus on initial_context as that's where database_connection lives

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
        dependency_config = self.global_defaults.get("dependency_config", {})

        # Provide sensible defaults
        return {
            "wait_seconds": dependency_config.get("wait_seconds", 10.0),
            "retry_interval": dependency_config.get("retry_interval", 0.5),
            "max_retry_interval": dependency_config.get("max_retry_interval", 2.0),
        }

    def _parse_global_defaults(self) -> None:
        """
        Parse global defaults configuration from YAML data.

        Global defaults provide fallback values for execution_config, episode_config,
        and benchmark_config when not specified at task level.
        """
        if self.yaml_data is None:
            return

        global_defaults_data = self.yaml_data.get("global_defaults")

        if global_defaults_data is None:
            # No global defaults specified - use empty dict
            self.global_defaults = {}
            logger.info("No global_defaults section found, using empty defaults")
            return

        if not isinstance(global_defaults_data, dict):
            raise InvalidTaskDefinitionException("global_defaults must be a dictionary")

        # Parse global prompts defaults (optional)
        if "prompts" in global_defaults_data:
            prompts_config = global_defaults_data["prompts"]
            if not isinstance(prompts_config, dict):
                raise InvalidTaskDefinitionException("global_defaults.prompts must be a dictionary")

            # Validate prompt types if provided - partial prompts are allowed in global defaults
            for prompt_type, template_file in prompts_config.items():
                if prompt_type not in ["instruction", "assistant", "submit"]:
                    raise InvalidTaskDefinitionException(
                        f"global_defaults.prompts contains invalid prompt type '{prompt_type}'. "
                        f"Valid types: instruction, assistant, submit"
                    )
                if not isinstance(template_file, str) or not template_file.strip():
                    raise InvalidTaskDefinitionException(
                        f"global_defaults.prompts.{prompt_type} must be a non-empty string"
                    )

            logger.info(f"Loaded global prompts configuration: {prompts_config}")
        else:
            logger.info("No global prompts defaults specified")

        # Validate structure of global defaults
        valid_sections = ["execution_config", "episode_config", "benchmark_config", "dependency_config", "prompts"]
        for section_name in global_defaults_data:
            if section_name not in valid_sections:
                raise InvalidTaskDefinitionException(
                    f"Invalid section '{section_name}' in global_defaults. " f"Valid sections are: {valid_sections}"
                )

            if section_name != "prompts" and not isinstance(global_defaults_data[section_name], dict):
                raise InvalidTaskDefinitionException(f"global_defaults.{section_name} must be a dictionary")

        # Store global defaults
        self.global_defaults = global_defaults_data.copy()
        logger.info(f"Loaded global defaults configuration: {self.global_defaults}")

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

        benchmark_data = self.yaml_data.get("benchmark_config")
        global_benchmark_defaults = self.global_defaults.get("benchmark_config", {})

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
        if "episode_attempts" not in merged_config:
            raise InvalidTaskDefinitionException(
                "Missing required 'episode_attempts' in benchmark configuration. "
                "You must specify episode_attempts either in benchmark_config or global_defaults.benchmark_config."
            )

        episode_attempts = merged_config["episode_attempts"]
        if not isinstance(episode_attempts, int) or episode_attempts < 1:
            raise InvalidTaskDefinitionException(
                f"episode_attempts must be a positive integer, got: {episode_attempts}"
            )

        # Store final benchmark configuration
        self.benchmark_config = merged_config

        logger.info(f"Loaded benchmark configuration: {self.benchmark_config}")

    def _parse_task(self, task_data: Dict[str, Any]) -> Task:
        """
        Parse a single task from YAML data.

        Args:
            task_data: Dictionary containing task definition

        Returns:
            Task instance
        """
        required_fields = ["task_id", "title", "description"]
        for field in required_fields:
            if field not in task_data:
                logger.error(f"Missing required field '{field}' in task definition")
                raise InvalidTaskDefinitionException(f"Missing required field: {field}")

        task_id = task_data["task_id"]
        title = task_data["title"]
        description = task_data["description"]
        initial_context = task_data.get("initial_context", {})

        logger.debug(f"Parsing task '{task_id}': {title}")

        # NEW: Parse prompts with global defaults inheritance
        if "prompts" in task_data:
            task_prompts = task_data["prompts"]
            if not isinstance(task_prompts, dict):
                raise InvalidTaskDefinitionException(f"Task '{task_id}' prompts must be a dictionary")
        else:
            task_prompts = {}

        # Inherit from global defaults, allow task-level overrides
        final_prompts = {}
        global_prompts = self.global_defaults.get("prompts", {})

        for prompt_type in ["instruction", "assistant", "submit"]:
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

        logger.debug(f"Task '{task_id}' resolved prompts: {final_prompts}")

        # Get sandbox environment string (resolution happens in execution layer)
        # Support both 'environment' and 'sandbox_environment' for flexibility
        sandbox_environment = task_data.get("environment") or task_data.get("sandbox_environment")

        # 🔍 DEBUG: Log what environment was parsed
        logger.info(f"🔍 DEBUG: Task '{task_id}' environment from YAML: {sandbox_environment}")
        logger.info(f"🔍 DEBUG: Raw task_data environment field: {task_data.get('environment')}")
        logger.info(f"🔍 DEBUG: Raw task_data sandbox_environment field: {task_data.get('sandbox_environment')}")

        # Get execution configuration with global defaults fallback
        task_execution_config = task_data.get("execution_config", {})
        global_execution_defaults = self.global_defaults.get("execution_config", {})

        # Deep merge global defaults with task-specific config (task-specific takes precedence)
        execution_config = deep_merge_dicts(global_execution_defaults, task_execution_config)
        # Enforce required execution_config.timeout after merge (explicit or via global defaults)
        if "timeout" not in execution_config:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' missing required execution_config.timeout (no implicit default)"
            )
        if not isinstance(execution_config["timeout"], int) or execution_config["timeout"] <= 0:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' execution_config.timeout must be positive int, got: {execution_config['timeout']}"
            )

        # Resolve allowed_executors precedence: task-level explicit > global defaults > domain-level executors
        if "allowed_executors" not in execution_config and self.allowed_executors is not None:
            execution_config["allowed_executors"] = self.allowed_executors
        if "allowed_executors" not in execution_config:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' missing required allowed_executors (no implicit default)"
            )
        if not isinstance(execution_config["allowed_executors"], list) or not execution_config["allowed_executors"]:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' allowed_executors must be a non-empty list, "
                f"got: {execution_config['allowed_executors']}"
            )

        # Get episode configuration with global defaults fallback
        task_episode_config = task_data.get("episode_config", {})
        global_episode_defaults = self.global_defaults.get("episode_config", {})

        # Deep merge global defaults with task-specific config (task-specific takes precedence)
        episode_config = deep_merge_dicts(global_episode_defaults, task_episode_config)

        # Enforce required episode_config.max_steps after merge
        if "max_steps" not in episode_config:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' missing required episode_config.max_steps (no implicit default)"
            )
        if not isinstance(episode_config["max_steps"], int) or episode_config["max_steps"] <= 0:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' episode_config.max_steps must be positive int, got: {episode_config['max_steps']}"
            )

        # Get task-level benchmark configuration with global defaults fallback
        task_benchmark_config = task_data.get("benchmark_config", {})
        global_benchmark_defaults = self.global_defaults.get("benchmark_config", {})

        if not isinstance(task_benchmark_config, dict):
            raise InvalidTaskDefinitionException(f"Task '{task_id}' benchmark_config must be a dictionary if provided")

        # Validate task-level episode_attempts if provided
        if "episode_attempts" in task_benchmark_config:
            episode_attempts = task_benchmark_config["episode_attempts"]
            if not isinstance(episode_attempts, int) or episode_attempts < 1:
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}' episode_attempts must be a positive integer, got: {episode_attempts}"
                )

        # Deep merge configurations in order of precedence:
        # 1. Global defaults (lowest priority)
        # 2. Domain-level benchmark_config
        # 3. Task-level benchmark_config (highest priority)
        merged_benchmark_config: Dict[str, Any] = {}
        merged_benchmark_config = deep_merge_dicts(merged_benchmark_config, global_benchmark_defaults)
        merged_benchmark_config = deep_merge_dicts(merged_benchmark_config, self.benchmark_config)
        merged_benchmark_config = deep_merge_dicts(
            merged_benchmark_config, task_benchmark_config
        )  # Ensure final config has valid episode_attempts
        if "episode_attempts" not in merged_benchmark_config or merged_benchmark_config["episode_attempts"] < 1:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' does not have valid episode_attempts configuration. "
                "Each task must have episode_attempts either from domain-level benchmark_config or task-level override."
            )

        # Validate evaluation configuration (REQUIRED - no backwards compatibility)
        evaluation_config = task_data.get("evaluation_config")
        if not evaluation_config:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' missing required evaluation_config section. "
                "All tasks MUST have evaluation configuration."
            )

        self._validate_evaluation_config(evaluation_config, task_id)

        # Parse subtasks
        subtasks_data = task_data.get("subtasks", [])
        subtasks = []

        logger.debug(f"Task '{task_id}' has {len(subtasks_data)} subtasks")
        for j, subtask_data in enumerate(subtasks_data):
            logger.debug(f"Parsing subtask {j+1}/{len(subtasks_data)} for task '{task_id}'")
            subtask = self._parse_subtask(subtask_data, task_id)
            subtasks.append(subtask)
            logger.debug(f"Successfully parsed subtask '{subtask.subtask_id}'")

        task = Task(
            task_id=task_id,
            domain=self.domain,
            title=title,
            description=description,
            prompts=final_prompts,
            subtasks=subtasks,
            initial_context=initial_context,
            environment=sandbox_environment,
            allowed_executors=execution_config.get("allowed_executors", self.allowed_executors),
            execution_config=execution_config,
            episode_config=episode_config,
            benchmark_config=merged_benchmark_config,
            evaluation_config=evaluation_config,
            depends_on_task_id=task_data.get("depends_on_task_id"),
        )

        logger.debug(f"Created task '{task_id}' with {len(subtasks)} subtasks")
        return task

    def _parse_subtask(self, subtask_data: Dict[str, Any], task_id: str) -> SubTask:
        """
        Parse a single subtask from YAML data.

        Args:
            subtask_data: Dictionary containing subtask definition
            task_id: ID of the parent task

        Returns:
            SubTask instance
        """
        required_fields = ["subtask_id", "title", "description", "objective"]
        for field in required_fields:
            if field not in subtask_data:
                raise InvalidTaskDefinitionException(f"Missing required subtask field: {field}")

        return SubTask(
            subtask_id=subtask_data["subtask_id"],
            task_id=task_id,
            title=subtask_data["title"],
            description=subtask_data["description"],
            objective=subtask_data["objective"],
            hint=subtask_data.get("hint"),
        )

    def _validate_evaluation_config(self, eval_config: Dict[str, Any], task_id: str) -> None:
        """
        Validate evaluation configuration for a task. Fails fast on invalid config.

        Args:
            eval_config: Evaluation configuration dictionary
            task_id: Task ID for error reporting

        Raises:
            InvalidTaskDefinitionException: If configuration is invalid
        """
        # Validate strategy
        strategy = eval_config.get("strategy")
        if strategy not in ("static", "llm_judge"):
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': Invalid or missing evaluation strategy. "
                f"Must be 'static' or 'llm_judge', got: {strategy}"
            )

        # Validate criteria section
        criteria = eval_config.get("criteria")
        if not isinstance(criteria, dict):
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': Missing or invalid criteria section in evaluation_config"
            )

        # Validate scoring section
        scoring = eval_config.get("scoring", {})
        if not isinstance(scoring, dict):
            raise InvalidTaskDefinitionException(f"Task '{task_id}': scoring must be a dictionary if provided")

        max_score = scoring.get("max_score", 1.0)
        if not isinstance(max_score, (int, float)) or max_score <= 0:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': max_score must be a positive number, got: {max_score}"
            )

        # Strategy-specific validation
        if strategy == "static":
            expected_answers = criteria.get("expected_answers")
            if not expected_answers or not isinstance(expected_answers, list):
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}': static strategy requires 'expected_answers' as a list in criteria"
                )

        elif strategy == "llm_judge":
            # golden_answer is optional for llm_judge strategy
            golden_answer = criteria.get("golden_answer")
            if golden_answer is not None and not isinstance(golden_answer, str):
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}': llm_judge strategy golden_answer must be a string if provided, "
                    f"got: {type(golden_answer)}"
                )

            model = criteria.get("model")
            if not model or not isinstance(model, str):
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}': llm_judge strategy requires 'model' as a string in criteria"
                )

            # BREAKING CHANGE: Require separate system and user templates
            judge_system_template = criteria.get("judge_system_template")
            if not judge_system_template or not isinstance(judge_system_template, str):
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}': llm_judge strategy requires 'judge_system_template' as a string in criteria"
                )

            judge_user_template = criteria.get("judge_user_template")
            if not judge_user_template or not isinstance(judge_user_template, str):
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}': llm_judge strategy requires 'judge_user_template' as a string in criteria"
                )

            # Check for deprecated single template field
            old_template = criteria.get("judge_prompt_template")
            if old_template:
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}': 'judge_prompt_template' is deprecated. "
                    f"Use 'judge_system_template' and 'judge_user_template' instead"
                )
