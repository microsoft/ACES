"""
Container Logging Manager for enhanced debugging capabilities.

This module provides centralized logging for all container-related activities
including docker-compose configurations and container logs. It ensures that
when containers crash or fail, debugging information is easily accessible.
"""

import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ....logging_config import get_execution_logger

logger = get_execution_logger(__name__)


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
        try:
            # Create main logs directory
            self.logs_directory.mkdir(parents=True, exist_ok=True)

            # Create base subdirectories for different log types
            # Note: Subdirectories for environment types (permanent-environments, etc.)
            # are created automatically when needed under compose-configs/ and container-logs/
            subdirs = ["compose-configs", "container-logs"]

            for subdir in subdirs:
                (self.logs_directory / subdir).mkdir(exist_ok=True)

            logger.info(f"Container logging initialized - logs directory: {self.logs_directory}")

        except Exception as e:
            logger.error(f"Failed to setup logging directories: {e}")
            self.enable_logging = False

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
            return None

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

            logger.info(f"Docker compose config logged: {config_path}")
            return str(config_path)

        except Exception as e:
            logger.error(f"Failed to log docker compose config: {e}")
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
            return None

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

            logger.info(f"Container logs collected: {log_path}")
            return str(log_path)

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout collecting logs for container {container_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to collect container logs for {container_name}: {e}")
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

        log_paths = []

        try:
            # Get list of containers in the project
            cmd = ["docker", "compose", "-p", project_name, "ps", "--services"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                services = result.stdout.strip().split("\n")
                services = [s.strip() for s in services if s.strip()]

                # Collect logs for each service
                for service in services:
                    log_path = self.log_container_logs(
                        container_name=service,
                        project_name=project_name,
                        config_type=config_type,
                        tail_lines=tail_lines,
                    )
                    if log_path:
                        log_paths.append(log_path)
            else:
                logger.warning(f"Failed to list services for project {project_name}: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to collect project logs for {project_name}: {e}")

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

        except Exception as e:
            logger.error(f"Failed to log container lifecycle event: {e}")

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

        # This would typically run in a background task
        # For now, we'll just do an initial collection
        await asyncio.create_task(self._background_log_collector(project_name, config_type, services))

    async def _background_log_collector(
        self, project_name: str, config_type: str, services: Optional[List[str]] = None
    ) -> None:
        """Background task for continuous log collection."""
        try:
            # Collect initial logs
            self.log_all_project_containers(project_name, config_type)

            # In a full implementation, this would set up log following
            # For now, we'll just do periodic collection
            logger.info(f"Background log collection started for project {project_name}")

        except Exception as e:
            logger.error(f"Background log collection failed for {project_name}: {e}")

    def cleanup_old_logs(self, days_to_keep: int = 7) -> None:
        """
        Clean up old log files to prevent disk space issues.

        Args:
            days_to_keep: Number of days of logs to retain
        """
        if not self.enable_logging or not self.logs_directory.exists():
            return

        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)

            for log_file in self.logs_directory.rglob("*"):
                if log_file.is_file():
                    if log_file.stat().st_mtime < cutoff_time:
                        try:
                            log_file.unlink()
                            logger.debug(f"Cleaned up old log file: {log_file}")
                        except Exception as e:
                            logger.warning(f"Failed to remove old log file {log_file}: {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")

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

        except Exception as e:
            logger.error(f"Failed to generate logs summary: {e}")
            return {"enabled": True, "error": str(e)}
