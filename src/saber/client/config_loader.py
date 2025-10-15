"""
Simple SABER Configuration Loader

Replaces the old HarnessConfigLoader with a simple YAML loader that works
with the new SABERConfig format and eval_async integration.

Logging category: CONFIG

Following SABER's philosophy:
- Fail fast when configuration files are invalid or missing
- Clean validation of required fields
- No silent fallbacks that mask configuration issues
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from ..logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from .models import AgentAssignment, SABERConfig

logger = get_saber_logger(LogCategory.CONFIG, __name__)


class SABERConfigLoader:
    """Simple configuration loader for SABER YAML configs."""

    @staticmethod
    def load_from_file(config_path: Union[str, Path], domain: Optional[str] = None) -> SABERConfig:
        """
        Load SABER configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
            domain: Optional domain name for logging organization

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

        operation = "load_saber_config"
        operation_context = {"config_path": str(config_path)}
        log_operation_start(logger, operation, **operation_context)

        if not config_path.exists():
            log_operation_failure(
                logger,
                operation,
                "configuration_file_missing",
                **operation_context,
            )
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        stage = "open_file"
        try:
            with open(config_path, "r", encoding="utf-8") as file_handle:
                stage = "parse_yaml"
                config_data = yaml.safe_load(file_handle)

            stage = "validate_structure"
            if not config_data:
                raise ValueError("Configuration file is empty")

            if not isinstance(config_data, dict):
                raise ValueError(f"Configuration must be a YAML dictionary, got: {type(config_data)}")

            stage = "convert_to_model"
            saber_config = SABERConfigLoader._convert_yaml_to_saber_config(config_data, config_path, domain=domain)

        except yaml.YAMLError as exc:
            log_operation_failure(
                logger,
                operation,
                exc,
                stage=stage,
                error_type="YAMLError",
                **operation_context,
            )
            raise ValueError(f"Failed to parse YAML configuration: {exc}") from exc
        except Exception as exc:
            log_operation_failure(
                logger,
                operation,
                exc,
                stage=stage,
                error_type=exc.__class__.__name__,
                **operation_context,
            )
            raise

        log_operation_success(
            logger,
            operation,
            agent_count=len(saber_config.agents),
            has_session_config=bool(saber_config.session_config),
            **operation_context,
        )

        if saber_config.session_config:
            logger.debug(
                "Session configuration resolved",
                extra={
                    "rest_url": saber_config.session_config.base_url,
                    "model": saber_config.model,
                },
            )
        else:
            logger.debug(
                "Session configuration missing",
                extra={"model": saber_config.model},
            )

        return saber_config

    @staticmethod
    def load_config_inputs(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate config structure without requiring server URLs.

        This method allows loading configs with server.mode: auto or missing server URLs,
        returning raw configuration inputs that can be hydrated with runtime values.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dictionary containing validated config inputs

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration structure is invalid
            yaml.YAMLError: If YAML parsing fails
        """
        # Convert string to Path if necessary
        if isinstance(config_path, str):
            config_path = Path(config_path)

        operation = "load_saber_config_inputs"
        operation_context = {"config_path": str(config_path)}
        log_operation_start(logger, operation, **operation_context)

        if not config_path.exists():
            log_operation_failure(
                logger,
                operation,
                "configuration_file_missing",
                **operation_context,
            )
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        stage = "open_file"
        try:
            with open(config_path, "r", encoding="utf-8") as file_handle:
                stage = "parse_yaml"
                config_data = yaml.safe_load(file_handle)

            stage = "validate_structure"
            if not config_data:
                raise ValueError("Configuration file is empty")

            if not isinstance(config_data, dict):
                raise ValueError(f"Configuration must be a YAML dictionary, got: {type(config_data)}")

            # Validate basic structure without requiring server URLs
            stage = "validate_agents"
            agent_assignments_config = config_data.get("agents")
            if not agent_assignments_config or not isinstance(agent_assignments_config, list):
                raise ValueError("Configuration must specify 'agents' as a list")

            # Extract and validate task configuration
            stage = "validate_tasks"
            task_config = config_data.get("tasks", {})
            if not isinstance(task_config, dict):
                raise ValueError("'tasks' section must be a dictionary")

            # Server validation is relaxed - allow auto mode or missing URLs
            stage = "validate_server"
            server_config = config_data.get("server")
            if server_config and not isinstance(server_config, dict):
                raise ValueError("'server' section must be a dictionary")

        except yaml.YAMLError as exc:
            log_operation_failure(
                logger,
                operation,
                exc,
                stage=stage,
                error_type="YAMLError",
                **operation_context,
            )
            raise ValueError(f"Failed to parse YAML configuration: {exc}") from exc
        except Exception as exc:
            log_operation_failure(
                logger,
                operation,
                exc,
                stage=stage,
                error_type=exc.__class__.__name__,
                **operation_context,
            )
            raise

        log_operation_success(
            logger,
            operation,
            agent_count=len(agent_assignments_config),
            has_server_config=bool(server_config),
            **operation_context,
        )

        return config_data

    @staticmethod
    def _convert_yaml_to_saber_config(
        config_data: Dict[str, Any], config_path: Path, domain: Optional[str] = None
    ) -> SABERConfig:
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

        # Extract server configuration (optional for auto mode hydration)
        server_config = config_data.get("server")
        if not server_config or not isinstance(server_config, dict):
            raise ValueError("Configuration must specify 'server' section")

        # Server URLs are optional when mode=auto (will be hydrated at runtime)
        server_mode = server_config.get("mode")
        rest_url = server_config.get("rest_url")
        mcp_url = server_config.get("mcp_url")

        # Validate URLs are present unless in auto mode
        if server_mode != "auto":
            if not rest_url:
                raise ValueError("Configuration must specify 'server.rest_url' (or use server.mode: auto)")
            if not mcp_url:
                raise ValueError("Configuration must specify 'server.mcp_url' (or use server.mode: auto)")

        # If in auto mode but URLs aren't provided yet, they must be hydrated before use
        if server_mode == "auto" and (not rest_url or not mcp_url):
            # This is expected - URLs will be injected by the caller
            # We'll create the config without session_config and let it be set later
            rest_url = None
            mcp_url = None

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

        # Extract endpoint configuration for model API requests
        endpoint_config = config_data.get("endpoint_configuration", {})
        if not isinstance(endpoint_config, dict):
            raise ValueError("'endpoint_configuration' section must be a dictionary")

        endpoint_timeout = endpoint_config.get("timeout")
        endpoint_max_retries = endpoint_config.get("max_retries")
        endpoint_max_connections = endpoint_config.get("max_connections")

        # Validate endpoint configuration
        if endpoint_timeout is not None and (not isinstance(endpoint_timeout, (int, float)) or endpoint_timeout <= 0):
            raise ValueError("'endpoint_configuration.timeout' must be a positive number or null")
        if endpoint_max_retries is not None and (not isinstance(endpoint_max_retries, int) or endpoint_max_retries < 0):
            raise ValueError("'endpoint_configuration.max_retries' must be a non-negative integer or null")
        if endpoint_max_connections is not None and (
            not isinstance(endpoint_max_connections, int) or endpoint_max_connections <= 0
        ):
            raise ValueError("'endpoint_configuration.max_connections' must be a positive integer or null")

        # Convert timeout to int if it's a float (for type safety)
        if endpoint_timeout is not None and isinstance(endpoint_timeout, float):
            endpoint_timeout = int(endpoint_timeout)

        # Create SABERConfig using the new agents format
        # For auto mode without URLs, pass None and session_config will be None

        saber_config = SABERConfig.create(
            model=model or "per-agent",  # Default fallback when all agents have models
            model_args=model_args,
            rest_url=rest_url,  # May be None in auto mode
            mcp_url=mcp_url,  # May be None in auto mode
            client_id=client_id,
            task_ids=task_ids,
            agents=assignments,
            log_level=log_level,
            log_dir=log_dir,
            domain=domain,
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
            endpoint_timeout=endpoint_timeout,
            endpoint_max_retries=endpoint_max_retries,
            endpoint_max_connections=endpoint_max_connections,
        )

        return saber_config
