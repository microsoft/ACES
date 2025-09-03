"""
Container Log Manager

Manages container logging using Docker's built-in logging drivers.
Collects logs after container completion for reliability and simplicity.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import docker

from .config import ClientLoggingConfig, SessionLoggingContext
from .docker_log_collector import DockerLogCollector

logger = logging.getLogger(__name__)


class ContainerLogManager:
    """Manages container logging using Docker's built-in logging drivers."""

    def __init__(
        self,
        config: ClientLoggingConfig,
        session_context: SessionLoggingContext,
        session_log_dir: Optional[Path] = None,
    ):
        """Initialize container log manager."""
        self.config = config
        self.session_context = session_context

        # Use provided session log directory or create one based on session ID
        if session_log_dir:
            self.session_log_dir = session_log_dir
            self._provided_session_log_dir = True
        else:
            base_dir = Path(config.base_log_dir) if isinstance(config.base_log_dir, str) else config.base_log_dir
            self.session_log_dir = base_dir / session_context.session_id
            self._provided_session_log_dir = False

        # Docker log collector
        self.log_collector = DockerLogCollector(config, session_context)

        # Track containers that need log collection
        self._registered_containers: Dict[str, Dict[str, str]] = {}

        # Ensure session directory exists
        self._setup_session_directory()

    def _setup_session_directory(self) -> None:
        """Set up the session logging directory structure."""
        try:
            # If no session_log_dir was provided, create timestamped directory to prevent overwrites
            if not self._provided_session_log_dir:
                timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
                base_dir = (
                    Path(self.config.base_log_dir)
                    if isinstance(self.config.base_log_dir, str)
                    else self.config.base_log_dir
                )
                timestamped_session_dir = base_dir / f"{timestamp}_{self.session_context.session_id}"
                self.session_log_dir = timestamped_session_dir

            # Ensure the session log directory exists
            self.session_log_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories following server pattern
            subdirs = [
                "container-logs/sidecar-containers",
                "container-logs/agent-containers",
                "container-events",
            ]

            for subdir in subdirs:
                (self.session_log_dir / subdir).mkdir(parents=True, exist_ok=True)

            # Write session metadata
            self._write_session_metadata()

            logger.info(f"📁 Session log directory: {self.session_log_dir}")

        except Exception as e:
            logger.error(f"❌ Failed to setup session directory: {e}")
            # Fail fast - don't continue with broken logging setup
            raise

    def _write_session_metadata(self) -> None:
        """Write session metadata to meta.json."""
        try:
            import json

            metadata = {
                "session_id": self.session_context.session_id,
                "client_id": self.session_context.client_id,
                "task_id": self.session_context.task_id,
                "episode_id": self.session_context.episode_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "log_collection_method": "docker_post_completion",
                "config": {
                    "log_format": self.config.log_format,
                    "log_level": self.config.log_level,
                    "include_metadata": self.config.include_metadata,
                },
            }

            meta_file = self.session_log_dir / "meta.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"❌ Failed to write session metadata: {e}")
            # Don't fail the entire process for metadata issues
            pass

    def get_docker_logging_config(self) -> Dict[str, Any]:
        """
        Get Docker logging configuration to be used when creating containers.

        Returns:
            Dict with logging configuration for Docker containers
        """
        return {
            "log_driver": "json-file",
            "log_config": {
                "max-size": "50m",  # 50MB max size per log file
                "max-file": "3",  # Keep 3 rotated files max
            },
        }

    def register_container(
        self,
        container: docker.models.containers.Container,
        log_file_name: str,
        component_type: str = "unknown",
    ) -> None:
        """
        Register a container for log collection after it completes.

        Args:
            container: Docker container to register
            log_file_name: Name for the log file
            component_type: Type of component (client, sidecar, agent)
        """
        try:
            container_id = container.id
            container_name = container.name

            # Determine subdirectory based on component type
            if component_type == "sidecar":
                subdir = "container-logs/sidecar-containers"
            elif component_type == "agent":
                subdir = "container-logs/agent-containers"
            else:
                subdir = "container-logs"

            # Create timestamped filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            timestamped_filename = f"{timestamp}_{log_file_name}"

            # Store registration info
            self._registered_containers[container_id] = {
                "container_name": container_name,
                "log_file_name": timestamped_filename,
                "component_type": component_type,
                "subdir": subdir,
                "registered_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                f"📝 Registered container {container_name} for log collection -> {subdir}/{timestamped_filename}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to register container {container.name}: {e}")
            # Fail fast on registration failures
            raise

    def collect_container_logs(self, container_id: str) -> bool:
        """
        Collect logs for a registered container after it has completed.

        Args:
            container_id: Container ID (can be short form)

        Returns:
            bool: True if logs collected successfully
        """
        try:
            # Find the container registration
            registration = None
            for cid, reg in self._registered_containers.items():
                if cid.startswith(container_id) or container_id.startswith(cid):
                    registration = reg
                    container_id = cid  # Use full ID
                    break

            if not registration:
                logger.error(f"❌ Container {container_id} not registered for log collection")
                return False

            # Get container object
            docker_client = docker.from_env()  # type: ignore[attr-defined]

            try:
                container = docker_client.containers.get(container_id)

                # Determine output file path
                output_file = self.session_log_dir / registration["subdir"] / registration["log_file_name"]

                # Collect logs using the DockerLogCollector
                success = self.log_collector.collect_container_logs(
                    container=container, output_file=output_file, component_type=registration["component_type"]
                )

                if success:
                    logger.info(f"✅ Collected logs for container {container.name}")
                else:
                    logger.warning(f"⚠️ Failed to collect logs for container {container.name}")

                return success

            except docker.errors.NotFound:
                # Container has been removed - this is normal for ephemeral containers
                # Check if logs were already collected via Docker logging driver during execution
                output_file = self.session_log_dir / registration["subdir"] / registration["log_file_name"]

                if output_file.exists() and output_file.stat().st_size > 0:
                    # Logs already collected via Docker logging driver during container execution
                    logger.info(
                        f"✅ Logs already collected for removed container {container_id[:12]} via Docker logging driver"
                    )
                    return True
                else:
                    # Try to collect logs using fallback method
                    logger.info(f"📝 Container {container_id[:12]} removed, attempting log recovery...")

                    result = self._collect_logs_from_docker_path(container_id, registration)

                    if result:
                        logger.info(f"✅ Successfully recovered logs for removed container {container_id[:12]}")
                    else:
                        logger.info(f"ℹ️  No additional logs available for removed container {container_id[:12]}")

                    return result

        except Exception as e:
            logger.error(f"❌ Error collecting logs for container {container_id}: {e}")
            # Fail fast on collection errors
            raise
            raise

    def _collect_logs_from_docker_path(self, container_id: str, registration: Dict[str, str]) -> bool:
        """
        Fallback method to collect logs directly from Docker's internal log files.

        Args:
            container_id: Full container ID
            registration: Container registration info

        Returns:
            bool: True if logs collected successfully
        """
        try:
            output_file = self.session_log_dir / registration["subdir"] / registration["log_file_name"]

            success = self.log_collector.collect_logs_from_docker_path(
                container_id=container_id, output_file=output_file, component_type=registration["component_type"]
            )

            return success

        except Exception as e:
            logger.error(f"❌ Error collecting logs from Docker path for {container_id}: {e}")
            return False

    def collect_all_registered_logs(self) -> Dict[str, bool]:
        """
        Collect logs for all registered containers.

        Returns:
            Dict mapping container IDs to collection success status
        """
        results = {}

        logger.info(f"📝 Collecting logs for {len(self._registered_containers)} registered containers")

        for container_id in self._registered_containers.keys():
            try:
                results[container_id] = self.collect_container_logs(container_id)
            except Exception as e:
                logger.error(f"❌ Failed to collect logs for {container_id}: {e}")
                results[container_id] = False

        successful = sum(1 for success in results.values() if success)
        logger.info(f"✅ Successfully collected logs for {successful}/{len(results)} containers")

        return results

    def get_registered_containers(self) -> Dict[str, Dict[str, str]]:
        """Get information about registered containers."""
        return self._registered_containers.copy()

    def clear_registrations(self) -> None:
        """Clear all container registrations."""
        self._registered_containers.clear()
        logger.info("🧹 Cleared all container registrations")

    async def cleanup_and_finalize(self) -> None:
        """
        Cleanup and finalize all container logging.

        Collects logs from all registered containers and finalizes the logging session.
        """
        try:
            logger.info("🧹 Finalizing container log collection")

            # Collect logs from all registered containers
            results = self.collect_all_registered_logs()

            # Report results
            successful = sum(1 for success in results.values() if success)
            total = len(results)

            if successful == total:
                logger.info(f"✅ Successfully collected logs from all {total} containers")
            else:
                failed = total - successful
                logger.warning(f"⚠️ Log collection completed with {failed}/{total} failures")

            # Clear registrations
            self.clear_registrations()

            logger.info("✅ Container log collection finalized")

        except Exception as e:
            logger.error(f"❌ Error during log cleanup and finalization: {e}")
            # Fail fast on cleanup errors
            raise

    def log_container_lifecycle_event(
        self, event_type: str, container_info: Dict, additional_data: Optional[Dict] = None
    ) -> None:
        """
        Log container lifecycle events (start, stop, crash, etc.).

        Args:
            event_type: Type of event (start, stop, crash, error)
            container_info: Information about the container
            additional_data: Additional event data
        """
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"

            event_data = {
                "timestamp": timestamp,
                "session_id": self.session_context.session_id,
                "client_id": self.session_context.client_id,
                "task_id": self.session_context.task_id,
                "episode_id": self.session_context.episode_id,
                "event_type": event_type,
                "container_info": container_info,
            }

            if additional_data:
                event_data.update(additional_data)

            # Write to container events log
            events_dir = self.session_log_dir / "container-events"
            events_dir.mkdir(parents=True, exist_ok=True)

            events_file = events_dir / f"container-events-{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"

            import json

            with open(events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")

            logger.debug(f"📝 Logged container event: {event_type} for {container_info.get('name', 'unknown')}")

        except Exception as e:
            logger.error(f"❌ Failed to log container lifecycle event: {e}")
            # Don't fail fast on lifecycle logging errors - they're not critical

    # Legacy compatibility methods (simplified versions)
    async def start_logging_container(
        self,
        container: docker.models.containers.Container,
        log_file_name: str,
        component_type: str = "unknown",
    ) -> bool:
        """
        Legacy compatibility method that registers container for post-completion log collection.

        Note: This method no longer starts real-time logging but registers the container
        for log collection after completion.
        """
        try:
            self.register_container(container, log_file_name, component_type)
            logger.info(f"📝 Container {container.name} registered for post-completion log collection")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to register container {container.name}: {e}")
            return False

    async def stop_logging_container(self, container_id: str) -> bool:
        """
        Legacy compatibility method that collects logs for a container.

        Note: This method now collects logs after container completion instead of stopping streaming.
        """
        try:
            return self.collect_container_logs(container_id)
        except Exception as e:
            logger.error(f"❌ Failed to collect logs for container {container_id}: {e}")
            return False

    async def stop_all_logging(self) -> None:
        """
        Legacy compatibility method that collects all registered container logs.
        """
        try:
            self.collect_all_registered_logs()
            logger.info("✅ All container logs collected")
        except Exception as e:
            logger.error(f"❌ Failed to collect all logs: {e}")
