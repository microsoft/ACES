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
from typing import Any, Dict, Optional, Union

import yaml

from .models import SABERConfig

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
        logger.debug(f"Config: saber_rest_url={saber_config.saber_rest_url}, model={saber_config.model}")

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

        # Extract model configuration (required)
        model = config_data.get("model")
        if not model:
            raise ValueError("Configuration must specify 'model' field")

        # Handle both nested model format (legacy) and flat format (new)
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

        # Extract task configuration
        task_config = config_data.get("tasks", {})
        if not isinstance(task_config, dict):
            raise ValueError("'tasks' section must be a dictionary")

        task_ids = task_config.get("task_ids")
        if task_ids and not isinstance(task_ids, list):
            raise ValueError("'tasks.task_ids' must be a list")

        # Extract agent configuration
        agent_config = config_data.get("agent", {})
        if not isinstance(agent_config, dict):
            raise ValueError("'agent' section must be a dictionary")

        # Handle agent_id or path - require one of them
        agent_id = agent_config.get("id")
        agent_path = agent_config.get("path")

        if not agent_id and not agent_path:
            raise ValueError("Agent configuration must specify either 'agent.id' or 'agent.path'")

        if agent_path:
            # Resolve relative paths relative to config file
            agent_path = SABERConfigLoader._resolve_agent_path(agent_path, config_path)

        # Extract agent parameters
        debug_mode = agent_config.get("debug_mode", False)

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

        # Create SABERConfig using the factory method
        saber_config = SABERConfig.create(
            model=model,
            model_args=model_args,
            rest_url=rest_url,
            mcp_url=mcp_url,
            client_id=client_id,
            task_ids=task_ids,
            agent_id=agent_id,
            agent_path=agent_path,
            debug_mode=debug_mode,
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

    @staticmethod
    def _resolve_agent_path(agent_path: str, config_path: Path) -> Optional[str]:
        """
        Resolve agent path relative to config file location.

        Args:
            agent_path: Agent path from config (may be relative)
            config_path: Path to config file

        Returns:
            Resolved absolute agent path, or None if not found
        """

        if not agent_path:
            return None

        agent_path_obj = Path(agent_path)

        # If already absolute, use as-is
        if agent_path_obj.is_absolute():
            if agent_path_obj.exists():
                return str(agent_path_obj)
            else:
                logger.warning(f"Agent path not found: {agent_path}")
                return str(agent_path_obj)  # Return anyway, let caller handle

        # Resolve relative to config file directory
        config_dir = config_path.parent
        resolved_path = config_dir / agent_path_obj

        if resolved_path.exists():
            return str(resolved_path.resolve())
        else:
            logger.warning(f"Agent path not found: {resolved_path}")
            return str(resolved_path)  # Return anyway, let caller handle
