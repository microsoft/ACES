"""
Simple SABER Configuration Loader

Replaces the old HarnessConfigLoader with a simple YAML loader that works
with the new SABERConfig format and eval_async integration.

Following SABER's philosophy:
- Fail fast when configuration files are invalid or missing
- Clean validation of required fields
- No silent fallbacks that mask configuration issues
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .models import AgentAssignment, SABERConfig

logger = logging.getLogger(__name__)


class SABERConfigLoader:
    """Simple configuration loader for SABER YAML configs."""

    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> SABERConfig:
        """
        Load SABER configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            SABERConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
            yaml.YAMLError: If YAML parsing fails
        """

        # Convert string to Path if necessary
        if isinstance(config_path, str):
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading SABER configuration from: {config_path}")

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration: {e}") from e

        if not config_data:
            raise ValueError(f"Configuration file is empty: {config_path}")

        if not isinstance(config_data, dict):
            raise ValueError(f"Configuration must be a YAML dictionary, got: {type(config_data)}")

        # Convert YAML config to SABERConfig
        saber_config = SABERConfigLoader._convert_yaml_to_saber_config(config_data, config_path)

        logger.info("SABER configuration loaded successfully")
        if saber_config.session_config:
            logger.debug(f"Config: rest_url={saber_config.session_config.base_url}, model={saber_config.model}")
        else:
            logger.debug(f"Config: model={saber_config.model} (no session config)")

        return saber_config

    @staticmethod
    def _convert_yaml_to_saber_config(config_data: Dict[str, Any], config_path: Path) -> SABERConfig:
        """
        Convert YAML configuration dictionary to SABERConfig.

        Uses the new nested configuration format. No legacy support.

        Args:
            config_data: YAML configuration dictionary
            config_path: Path to config file (for relative path resolution)

        Returns:
            SABERConfig instance

        Raises:
            ValueError: If required configuration is missing or invalid
        """

        # Extract task configuration
        task_config = config_data.get("tasks", {})
        if not isinstance(task_config, dict):
            raise ValueError("'tasks' section must be a dictionary")

        task_ids = task_config.get("task_ids")
        if task_ids and not isinstance(task_ids, list):
            raise ValueError("'tasks.task_ids' must be a list")

        # Extract agent assignments to check for per-agent models
        agent_assignments_config = config_data.get("agents")
        if agent_assignments_config is None:
            raise ValueError("Configuration must specify 'agents' - legacy formats no longer supported")

        if not isinstance(agent_assignments_config, list):
            raise ValueError("'agents' must be a list")

        if not agent_assignments_config:
            raise ValueError("'agents' cannot be empty - at least one agent assignment is required")

        # Check if all agents have models specified
        agents_with_models = 0
        for assignment in agent_assignments_config:
            if isinstance(assignment, dict) and assignment.get("model"):
                agents_with_models += 1

        all_agents_have_models = agents_with_models == len(agent_assignments_config)

        # Extract model configuration (optional if all agents have models)
        model = config_data.get("model")
        if not model and not all_agents_have_models:
            raise ValueError("Configuration must specify 'model' field OR all agents must have 'model' field")

        # Handle both nested model format (legacy) and flat format (new)
        if model:
            if isinstance(model, dict):
                # Legacy nested format: model: {name: "...", args: {...}}
                model_name = model.get("name")
                if not model_name:
                    raise ValueError("Configuration model section must specify 'name' field")
                model_args = model.get("args", {})
                model = model_name  # Use the name as the model string
            else:
                # New flat format: model: "model_name", model_args: {...}
                model_args = config_data.get("model_args", {})
        else:
            # No global model, agents specify their own
            model_args = config_data.get("model_args", {})

        if not isinstance(model_args, dict):
            raise ValueError("'model_args' must be a dictionary")

        # Extract server configuration (required, nested format only)
        server_config = config_data.get("server")
        if not server_config or not isinstance(server_config, dict):
            raise ValueError("Configuration must specify 'server' section with rest_url and mcp_url")

        rest_url = server_config.get("rest_url")
        if not rest_url:
            raise ValueError("Configuration must specify 'server.rest_url'")

        mcp_url = server_config.get("mcp_url")
        if not mcp_url:
            raise ValueError("Configuration must specify 'server.mcp_url'")

        client_id = server_config.get("client_id", "saber-client")

        # Validate and parse agent assignments
        assignments = []
        for i, assignment in enumerate(agent_assignments_config):
            if not isinstance(assignment, dict):
                raise ValueError(f"agents[{i}] must be a dictionary")

            assignment_id = assignment.get("id")
            if not assignment_id:
                raise ValueError(f"agents[{i}] must have an 'id' field")

            tasks = assignment.get("tasks")
            if not tasks or not isinstance(tasks, list):
                raise ValueError(f"agents[{i}] must have a 'tasks' field as a list")

            model = assignment.get("model")  # Optional model field

            kwargs = assignment.get("kwargs", {})
            if not isinstance(kwargs, dict):
                raise ValueError(f"agents[{i}].kwargs must be a dictionary")

            # Create AgentAssignment object with validation
            assignment_obj = AgentAssignment(id=assignment_id, model=model, tasks=tasks, kwargs=kwargs)
            assignments.append(assignment_obj)

        # Extract execution configuration (check both top-level and execution section)
        execution_config = config_data.get("execution", {})

        log_level = config_data.get("log_level", execution_config.get("log_level", "INFO"))
        log_dir = config_data.get("log_dir", execution_config.get("log_dir"))
        ui_enabled = config_data.get("ui_enabled", execution_config.get("ui_enabled", True))
        container_timeout = config_data.get("container_timeout", execution_config.get("container_timeout", 300))

        # Extract eval_async configuration
        max_samples = config_data.get("max_samples")
        max_subprocesses = config_data.get("max_subprocesses", 1)
        parallel_execution = config_data.get("parallel_execution", True)
        max_parallel_tasks = config_data.get("max_parallel_tasks", 4)

        # Extract log upload configuration
        log_upload_config = config_data.get("log_upload", {})
        if not isinstance(log_upload_config, dict):
            raise ValueError("'log_upload' section must be a dictionary")

        log_upload_enabled = log_upload_config.get("enabled", True)
        log_upload_max_retries = log_upload_config.get("max_retries", 3)
        log_upload_timeout = log_upload_config.get("timeout", 30.0)
        log_upload_fail_on_error = log_upload_config.get("fail_on_error", False)

        # Validate log upload configuration
        if not isinstance(log_upload_enabled, bool):
            raise ValueError("'log_upload.enabled' must be a boolean")
        if not isinstance(log_upload_max_retries, int) or log_upload_max_retries < 0:
            raise ValueError("'log_upload.max_retries' must be a non-negative integer")
        if not isinstance(log_upload_timeout, (int, float)) or log_upload_timeout <= 0:
            raise ValueError("'log_upload.timeout' must be a positive number")
        if not isinstance(log_upload_fail_on_error, bool):
            raise ValueError("'log_upload.fail_on_error' must be a boolean")

        # Create SABERConfig using the new agents format
        saber_config = SABERConfig.create(
            model=model or "per-agent",  # Default fallback when all agents have models
            model_args=model_args,
            rest_url=rest_url,
            mcp_url=mcp_url,
            client_id=client_id,
            task_ids=task_ids,
            agents=assignments,
            log_level=log_level,
            log_dir=log_dir,
            ui_enabled=ui_enabled,
            container_timeout=container_timeout,
            max_samples=max_samples,
            max_subprocesses=max_subprocesses,
            parallel_execution=parallel_execution,
            max_parallel_tasks=max_parallel_tasks,
            log_upload_enabled=log_upload_enabled,
            log_upload_max_retries=log_upload_max_retries,
            log_upload_timeout=log_upload_timeout,
            log_upload_fail_on_error=log_upload_fail_on_error,
        )

        return saber_config
