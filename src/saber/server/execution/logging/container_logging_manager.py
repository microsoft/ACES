"""
Container Logging Manager for enhanced debugging capabilities.

This module provides centralized logging for all container-related activities
including docker-compose configurations and container logs. It ensures that
when containers crash or fail, debugging information is easily accessible.

Logging category: ``LogCategory.DOCKER``.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ....logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)

logger = get_saber_logger(LogCategory.DOCKER, __name__)


class ContainerLoggingManager:
    """
    Manages logging for all container types in SABER system.

    Provides centralized logging of:
    - Docker Compose configurations
    - Container runtime logs
    - Container lifecycle events
    - Debug information for troubleshooting
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Container Logging Manager.

        Args:
            config: Configuration dictionary containing:
                - logs_directory: Host directory path for log storage
                - domain: Domain name for organizing logs
                - enable_logging: Whether to enable container logging (default: True)
        """
        self.config = config
        self.domain = config.get("domain", "unknown")
        self.enable_logging = config.get("enable_logging", True)

        # Configure logs directory - should be mounted to host
        self.logs_directory = Path(config.get("logs_directory", "/app/logs"))

        if self.enable_logging:
            self._setup_logging_directories()

    def _setup_logging_directories(self) -> None:
        """Create the logging directory structure."""
        logger.debug(
            "Container logging setup starting",
            extra={
                "event": "container_logging_setup_start",
                "domain": self.domain,
                "logs_directory": str(self.logs_directory),
            },
        )
        try:
            # Create main logs directory
            self.logs_directory.mkdir(parents=True, exist_ok=True)

            # Create base subdirectories for different log types
            # Note: Subdirectories for environment types (permanent-environments, etc.)
            # are created automatically when needed under compose-configs/ and container-logs/
            subdirs = ["compose-configs", "container-logs"]

            for subdir in subdirs:
                (self.logs_directory / subdir).mkdir(exist_ok=True)

        except Exception as exc:
            self.enable_logging = False
            log_operation_failure(
                logger,
                "container_logging_setup",
                exc,
                domain=self.domain,
                logs_directory=str(self.logs_directory),
            )
            raise RuntimeError("Failed to setup logging directories") from exc
        else:
            logger.debug(
                "Container logging setup completed",
                extra={
                    "event": "container_logging_setup_complete",
                    "domain": self.domain,
                    "logs_directory": str(self.logs_directory),
                },
            )

    def log_compose_config(
        self,
        compose_config: Dict[str, Any],
        config_type: str,
        identifier: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Log a docker-compose configuration to persistent storage.

        Args:
            compose_config: Docker compose configuration dictionary
            config_type: Type of configuration (permanent, sandbox, server-client)
            identifier: Unique identifier for this configuration (session_id, project_name, etc.)
            additional_metadata: Additional metadata to include in the log

        Returns:
            Path to the logged configuration file, or None if logging disabled
        """
        if not self.enable_logging:
            logger.debug(
                "Compose configuration logging skipped",
                extra={
                    "event": "container_logging_disabled",
                    "config_type": config_type,
                    "identifier": identifier,
                },
            )
            return None

        logger.debug(
            "Compose configuration logging requested",
            extra={
                "event": "container_compose_config_log_start",
                "domain": self.domain,
                "config_type": config_type,
                "identifier": identifier,
            },
        )

        try:
            timestamp = datetime.now().isoformat()
            filename = f"{timestamp}_{config_type}_{identifier}.yml"

            # Determine subdirectory based on config type
            if config_type == "permanent":
                subdir = "permanent-environments"
            elif config_type == "sandbox":
                subdir = "sandbox-environments"
            else:
                subdir = "compose-configs"

            config_path = self.logs_directory / "compose-configs" / subdir / filename
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare enhanced configuration with metadata
            enhanced_config = {
                "metadata": {
                    "timestamp": timestamp,
                    "domain": self.domain,
                    "config_type": config_type,
                    "identifier": identifier,
                    "saber_version": "1.0.0",  # Could be made dynamic
                    **(additional_metadata or {}),
                },
                "docker_compose_config": compose_config,
            }

            # Write YAML configuration
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(enhanced_config, f, default_flow_style=False, sort_keys=False)

            logger.debug(
                "Compose configuration logged",
                extra={
                    "event": "container_compose_config_log_complete",
                    "domain": self.domain,
                    "config_type": config_type,
                    "identifier": identifier,
                    "config_path": str(config_path),
                },
            )
            return str(config_path)

        except Exception as exc:
            log_operation_failure(
                logger,
                "container_compose_config_log",
                exc,
                domain=self.domain,
                config_type=config_type,
                identifier=identifier,
            )
            return None

    def log_container_logs(
        self, container_name: str, project_name: str, config_type: str, follow: bool = False, tail_lines: int = 1000
    ) -> Optional[str]:
        """
        Collect and log container logs to persistent storage.

        Args:
            container_name: Name of the container/service
            project_name: Docker compose project name
            config_type: Type of configuration (permanent, sandbox, server-client)
            follow: Whether to continuously follow logs (for background collection)
            tail_lines: Number of recent log lines to collect

        Returns:
            Path to the logged container logs, or None if logging disabled
        """
        if not self.enable_logging:
            logger.debug(
                "Container log collection skipped",
                extra={
                    "event": "container_logging_disabled",
                    "container_name": container_name,
                    "project_name": project_name,
                    "config_type": config_type,
                },
            )
            return None

        logger.debug(
            "Container log collection requested",
            extra={
                "event": "container_logs_collect_start",
                "domain": self.domain,
                "container_name": container_name,
                "project_name": project_name,
                "config_type": config_type,
                "follow": follow,
                "tail_lines": tail_lines,
            },
        )

        try:
            timestamp = datetime.now().isoformat()
            log_filename = f"{timestamp}_{config_type}_{project_name}_{container_name}.log"

            # Determine subdirectory based on config type
            if config_type == "permanent":
                subdir = "permanent-environments"
            elif config_type == "sandbox":
                subdir = "sandbox-environments"
            else:
                subdir = "server-client-logs"

            log_path = self.logs_directory / "container-logs" / subdir / log_filename
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Collect container logs using docker compose logs
            cmd = ["docker", "compose", "-p", project_name, "logs", "--timestamps", f"--tail={tail_lines}"]

            if follow:
                cmd.append("--follow")

            cmd.append(container_name)

            # Execute command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30 if not follow else None)

            # Write logs with metadata header
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"# Container Logs for {container_name}\n")
                f.write(f"# Project: {project_name}\n")
                f.write(f"# Type: {config_type}\n")
                f.write(f"# Domain: {self.domain}\n")
                f.write(f"# Collected: {timestamp}\n")
                f.write(f"# Command: {' '.join(cmd)}\n")
                f.write("# " + "=" * 60 + "\n\n")

                if result.returncode == 0:
                    f.write(result.stdout)
                else:
                    f.write(f"Error collecting logs (exit code {result.returncode}):\n")
                    f.write(result.stderr)

            logger.debug(
                "Container log collection completed",
                extra={
                    "event": "container_logs_collect_complete",
                    "domain": self.domain,
                    "container_name": container_name,
                    "project_name": project_name,
                    "config_type": config_type,
                    "log_path": str(log_path),
                    "follow": follow,
                    "tail_lines": tail_lines,
                },
            )
            return str(log_path)

        except subprocess.TimeoutExpired:
            logger.warning(
                "Timeout collecting container logs",
                extra={
                    "event": "container_logs_timeout",
                    "container_name": container_name,
                    "project_name": project_name,
                    "config_type": config_type,
                    "follow": follow,
                    "tail_lines": tail_lines,
                    "timeout_seconds": 30 if not follow else None,
                },
            )
            return None
        except Exception as exc:
            log_operation_failure(
                logger,
                "container_logs_collect",
                exc,
                domain=self.domain,
                container_name=container_name,
                project_name=project_name,
                config_type=config_type,
                follow=follow,
                tail_lines=tail_lines,
            )
            return None

    def log_all_project_containers(self, project_name: str, config_type: str, tail_lines: int = 1000) -> List[str]:
        """
        Collect logs for all containers in a docker-compose project.

        Args:
            project_name: Docker compose project name
            config_type: Type of configuration (permanent, sandbox, server-client)
            tail_lines: Number of recent log lines to collect per container

        Returns:
            List of paths to logged container log files
        """
        if not self.enable_logging:
            return []

        logger.debug(
            "Project log collection requested",
            extra={
                "event": "container_project_logs_collect_start",
                "domain": self.domain,
                "project_name": project_name,
                "config_type": config_type,
                "tail_lines": tail_lines,
            },
        )

        log_paths: List[str] = []
        try:
            cmd = ["docker", "compose", "-p", project_name, "ps", "--services"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.warning(
                    "Failed to list project services",
                    extra={
                        "event": "container_project_services_list_failed",
                        "project_name": project_name,
                        "config_type": config_type,
                        "stderr": result.stderr.strip(),
                        "return_code": result.returncode,
                    },
                )
                return log_paths

            services = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
            for service in services:
                log_path = self.log_container_logs(
                    container_name=service,
                    project_name=project_name,
                    config_type=config_type,
                    tail_lines=tail_lines,
                )
                if log_path:
                    log_paths.append(log_path)

        except Exception as exc:
            log_operation_failure(
                logger,
                "container_project_logs_collect",
                exc,
                domain=self.domain,
                project_name=project_name,
                config_type=config_type,
                tail_lines=tail_lines,
            )
        else:
            logger.debug(
                "Project log collection completed",
                extra={
                    "event": "container_project_logs_collect_complete",
                    "domain": self.domain,
                    "project_name": project_name,
                    "config_type": config_type,
                    "tail_lines": tail_lines,
                    "collected_logs": len(log_paths),
                },
            )

        return log_paths

    def log_container_lifecycle_event(
        self, event_type: str, container_info: Dict[str, Any], additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log container lifecycle events (start, stop, crash, etc.).

        Args:
            event_type: Type of event (start, stop, crash, error)
            container_info: Information about the container
            additional_data: Additional event data
        """
        if not self.enable_logging:
            return

        try:
            timestamp = datetime.now().isoformat()

            event_data = {
                "timestamp": timestamp,
                "domain": self.domain,
                "event_type": event_type,
                "container_info": container_info,
                "additional_data": additional_data or {},
            }

            # Log to a daily events file
            date_str = datetime.now().strftime("%Y-%m-%d")
            events_file = self.logs_directory / f"container-events-{date_str}.jsonl"

            with open(events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")

            logger.info(
                "Container lifecycle event recorded",
                extra={
                    "event": "container_lifecycle_event_recorded",
                    "event_type": event_type,
                    "domain": self.domain,
                    "events_file": str(events_file),
                },
            )

        except Exception as exc:
            log_operation_failure(
                logger,
                "container_lifecycle_event_log",
                exc,
                domain=self.domain,
                event_type=event_type,
            )

    async def start_background_log_collection(
        self, project_name: str, config_type: str, services: Optional[List[str]] = None
    ) -> None:
        """
        Start background log collection for a project (non-blocking).

        Args:
            project_name: Docker compose project name
            config_type: Type of configuration
            services: Specific services to follow, or None for all
        """
        if not self.enable_logging:
            return

        operation = "container_background_log_collection_start"
        log_operation_start(
            logger,
            operation,
            domain=self.domain,
            project_name=project_name,
            config_type=config_type,
            services=services,
        )

        try:
            collected_logs = self.log_all_project_containers(project_name, config_type)

            log_operation_success(
                logger,
                operation,
                domain=self.domain,
                project_name=project_name,
                config_type=config_type,
                services=services,
                collected_log_count=len(collected_logs),
            )
        except Exception as exc:
            log_operation_failure(
                logger,
                operation,
                exc,
                domain=self.domain,
                project_name=project_name,
                config_type=config_type,
                services=services,
            )
            raise

    async def _background_log_collector(
        self, project_name: str, config_type: str, services: Optional[List[str]] = None
    ) -> None:
        """Background task for continuous log collection."""
        operation = "container_background_log_collection"
        log_operation_start(
            logger,
            operation,
            domain=self.domain,
            project_name=project_name,
            config_type=config_type,
            services=services,
        )
        try:
            # Collect initial logs
            self.log_all_project_containers(project_name, config_type)

            # In a full implementation, this would set up log following
            # For now, we'll just do periodic collection
            log_operation_success(
                logger,
                operation,
                domain=self.domain,
                project_name=project_name,
                config_type=config_type,
                services=services,
            )

        except Exception as exc:
            log_operation_failure(
                logger,
                operation,
                exc,
                domain=self.domain,
                project_name=project_name,
                config_type=config_type,
                services=services,
            )

    def cleanup_old_logs(self, days_to_keep: int = 7) -> None:
        """
        Clean up old log files to prevent disk space issues.

        Args:
            days_to_keep: Number of days of logs to retain
        """
        if not self.enable_logging or not self.logs_directory.exists():
            return

        operation = "container_logs_cleanup"
        log_operation_start(
            logger,
            operation,
            domain=self.domain,
            logs_directory=str(self.logs_directory),
            days_to_keep=days_to_keep,
        )

        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            removed_files = 0

            for log_file in self.logs_directory.rglob("*"):
                if log_file.is_file():
                    if log_file.stat().st_mtime < cutoff_time:
                        try:
                            log_file.unlink()
                            removed_files += 1
                            logger.debug(
                                "Old log file removed",
                                extra={
                                    "event": "container_log_removed",
                                    "path": str(log_file),
                                },
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to remove old log file",
                                extra={
                                    "event": "container_log_remove_failed",
                                    "path": str(log_file),
                                    "error": str(exc),
                                },
                            )

            log_operation_success(
                logger,
                operation,
                domain=self.domain,
                logs_directory=str(self.logs_directory),
                days_to_keep=days_to_keep,
                removed_files=removed_files,
            )

        except Exception as exc:
            log_operation_failure(
                logger,
                operation,
                exc,
                domain=self.domain,
                logs_directory=str(self.logs_directory),
                days_to_keep=days_to_keep,
            )

    def get_logs_summary(self) -> Dict[str, Any]:
        """
        Get a summary of collected logs for debugging purposes.

        Returns:
            Dictionary containing logs summary information
        """
        if not self.enable_logging or not self.logs_directory.exists():
            return {"enabled": False}

        try:
            summary = {
                "enabled": True,
                "logs_directory": str(self.logs_directory),
                "domain": self.domain,
                "subdirectories": {},
                "recent_files": [],
            }

            # Count files in each subdirectory
            for subdir in self.logs_directory.iterdir():
                if subdir.is_dir():
                    file_count = len(list(subdir.rglob("*")))
                    summary["subdirectories"][subdir.name] = file_count

            # Get recent files (last 10)
            all_files = []
            for log_file in self.logs_directory.rglob("*"):
                if log_file.is_file():
                    all_files.append((log_file.stat().st_mtime, str(log_file)))

            all_files.sort(reverse=True)
            summary["recent_files"] = [f[1] for f in all_files[:10]]

            return summary

        except Exception as exc:
            log_operation_failure(
                logger,
                "container_logs_summary",
                exc,
                domain=self.domain,
                logs_directory=str(self.logs_directory),
            )
            return {"enabled": True, "error": str(exc)}
