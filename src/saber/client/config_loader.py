#!/usr/bin/env python3
"""
SABER Harness Configuration Loader

Utilities for loading SABER harness configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .harness_models import SABERHarnessConfig


class HarnessConfigLoader:
    """Loads and validates SABER harness configuration from YAML files."""

    @staticmethod
    def load_from_file(config_path: Path) -> SABERHarnessConfig:
        """Load configuration from a YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return HarnessConfigLoader._parse_config(config_data)

    @staticmethod
    def _parse_config(config_data: Dict[str, Any]) -> SABERHarnessConfig:
        """Parse configuration data into SABERHarnessConfig."""
        # Server configuration
        server = config_data.get("server", {})
        server_url = server.get("url", "http://localhost:8000")
        mcp_url = server.get("mcp_url", "http://localhost:8001")
        client_id = server.get("client_id", "saber-client")
        request_timeout = server.get("request_timeout", 300.0)

        # Logging configuration
        logging = config_data.get("logging", {})
        log_level = logging.get("level", "INFO")
        client_log_dir = logging.get("client_log_dir")
        if client_log_dir:
            client_log_dir = Path(client_log_dir)
        enable_container_logging = logging.get("enable_container_logging", True)
        log_retention_days = logging.get("log_retention_days", 30)
        max_log_size_mb = logging.get("max_log_size_mb", 50)
        compress_old_logs = logging.get("compress_old_logs", True)

        # Task configuration
        tasks = config_data.get("tasks", {})
        task_ids = tasks.get("task_ids")
        if task_ids == []:  # Empty list means run all
            task_ids = None
        parallelism = tasks.get("parallelism", 1)

        # Safety configuration
        safety = config_data.get("safety", {})
        max_steps_client_safety = safety.get("max_steps_client_safety", 100)

        # LLM configuration
        llm = config_data.get("llm", {})
        llm_provider = llm.get("provider")
        llm_config = llm.get("config", {})

        # Agent configuration
        agent = config_data.get("agent", {})
        agent_config = {}
        if "image" in agent:
            agent_config["image"] = agent["image"]
        if "config" in agent:
            agent_config.update(agent["config"])

        # Extract agent path for separate handling
        agent_path = agent.get("path")

        # UI configuration
        ui = config_data.get("ui", {})
        ui_backend = ui.get("backend", "auto")
        ui_enabled = ui.get("enabled", True)
        ui_internal_only = ui.get("internal_only", True)
        ui_tool_detail_level = ui.get("tool_detail_level", "full")

        # Apply environment variable overrides
        HarnessConfigLoader._apply_env_overrides(locals())

        return SABERHarnessConfig(
            server_url=server_url,
            mcp_url=mcp_url,
            client_id=client_id,
            request_timeout=request_timeout,
            log_level=log_level,
            client_log_dir=client_log_dir,
            enable_container_logging=enable_container_logging,
            log_retention_days=log_retention_days,
            max_log_size_mb=max_log_size_mb,
            compress_old_logs=compress_old_logs,
            task_ids=task_ids,
            parallelism=parallelism,
            max_steps_client_safety=max_steps_client_safety,
            llm_provider=llm_provider,
            llm_config=llm_config,
            agent_config=agent_config,
            agent_path=agent_path,
            ui_backend=ui_backend,
            ui_enabled=ui_enabled,
            ui_internal_only=ui_internal_only,
            ui_tool_detail_level=ui_tool_detail_level,
        )

    @staticmethod
    def _apply_env_overrides(config_vars: Dict[str, Any]) -> None:
        """Apply environment variable overrides to configuration."""
        # Server URL overrides
        if "SABER_SERVER_URL" in os.environ:
            config_vars["server_url"] = os.environ["SABER_SERVER_URL"]
        if "SABER_MCP_URL" in os.environ:
            config_vars["mcp_url"] = os.environ["SABER_MCP_URL"]

        # Logging overrides
        if "SABER_LOG_LEVEL" in os.environ:
            config_vars["log_level"] = os.environ["SABER_LOG_LEVEL"]

        # Agent image override
        if "SABER_AGENT_IMAGE" in os.environ:
            if "agent_config" not in config_vars:
                config_vars["agent_config"] = {}
            config_vars["agent_config"]["image"] = os.environ["SABER_AGENT_IMAGE"]

    @staticmethod
    def create_default_config(config_path: Path) -> None:
        """Create a default configuration file."""
        default_config = {
            "server": {
                "url": "http://localhost:8000",
                "mcp_url": "http://localhost:8001",
                "client_id": "saber-client",
                "request_timeout": 60.0,
            },
            "logging": {
                "level": "INFO",
                "client_log_dir": "./logs",
                "enable_container_logging": True,
                "log_retention_days": 30,
                "max_log_size_mb": 50,
                "compress_old_logs": True,
            },
            "tasks": {
                "task_ids": [],
                "parallelism": 1,
            },
            "safety": {
                "max_steps_client_safety": 100,
            },
            "llm": {
                "provider": None,
                "config": {},
            },
            "agent": {
                "path": None,
                "image": "saber/agent-runner:latest",
                "config": {},
            },
            "ui": {
                "backend": "auto",
                "enabled": True,
                "internal_only": True,
                "tool_detail_level": "full",
            },
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
