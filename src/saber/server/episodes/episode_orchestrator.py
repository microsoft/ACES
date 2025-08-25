#!/usr/bin/env python3
"""
SABER Episode Container Orchestrator

This is a dedicated sidecar container that monitors episode status and manages
the lifecycle of all containers associated with a SABER episode.

When an episode becomes inactive, this orchestrator:
1. Detects the episode termination via polling
2. Identifies all containers belonging to the episode
3. Performs graceful shutdown of all episode containers
4. Cleans up Docker resources (networks, volumes, etc.)

Environment Variables Required:
- SABER_SESSION_ID: Session identifier
- SABER_CLEANUP_TOKEN: Token file path to monitor for cleanup
- SABER_HOST_URL: URL of the SABER server for health checks
- SABER_POLL_INTERVAL: Polling interval in seconds (default: 30)
- SABER_COMPOSE_PROJECT: Docker Compose project name for the session
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
from typing import Any, List, Optional

import httpx

# Add src to path for local development
sys.path.insert(0, "/home/ms_test/repos/saber_vibin/src")  # noqa: E402

from saber.logging_config import (  # noqa: E402
    LogCategory,
    SaberLogger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)


class EpisodeContainerOrchestrator:
    """Orchestrates container lifecycle for SABER episodes."""

    def __init__(self) -> None:
        """Initialize the episode container orchestrator."""
        self.session_id = os.getenv("SABER_SESSION_ID")
        self.cleanup_token = os.getenv("SABER_CLEANUP_TOKEN")
        self.host_url = os.getenv("SABER_HOST_URL", "http://host.docker.internal:8000")
        self.poll_interval = int(os.getenv("SABER_POLL_INTERVAL", "30"))
        self.compose_project = os.getenv("SABER_COMPOSE_PROJECT")
        self.compose_file = os.getenv("SABER_COMPOSE_FILE")
        self.graceful_timeout = int(os.getenv("SABER_GRACEFUL_TIMEOUT", "30"))

        # Validate required environment variables
        if not self.session_id:
            raise ValueError("SABER_SESSION_ID environment variable is required")
        if not self.cleanup_token:
            raise ValueError("SABER_CLEANUP_TOKEN environment variable is required")
        if not self.compose_project:
            raise ValueError("SABER_COMPOSE_PROJECT environment variable is required")

        self.logger = self._setup_logging()
        self.logger.info("Episode Container Orchestrator initializing", self.session_id)
        self.running = True
        self.failure_count = 0
        # Increased tolerance for async execution - allow longer command execution times
        self.max_failures = int(os.getenv("SABER_MAX_ORCHESTRATOR_FAILURES", "10"))  # 30s × 10 = 5 minutes tolerance

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.logger.info("Episode Container Orchestrator initialized", self.session_id)
        self.logger.info(f"Managing Docker Compose project: {self.compose_project}")
        self.logger.info(f"Polling {self.host_url} every {self.poll_interval} seconds")

    def _setup_logging(self) -> SaberLogger:
        """Setup logging for the orchestrator with episode category."""
        # Create log directory if it doesn't exist
        log_dir = "/tmp/saber-orchestrator-logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = f"{log_dir}/orchestrator-{self.session_id}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode="a"),
                logging.FileHandler("/tmp/saber_orchestrator.log", mode="a"),  # Keep old location too
            ],
        )
        # Return a SaberLogger with the episode category
        return SaberLogger("saber.episode.orchestrator", LogCategory.EPISODE)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        self.logger.warning(f"Received signal {signum}, initiating episode cleanup", self.session_id)
        self.running = False

    async def check_episode_status(self) -> Optional[bool]:
        """
        Check if the episode is still active.

        Returns:
            True if active, False if inactive, None if check failed
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.host_url}/internal/episode-status/{self.session_id}", params={"token": self.cleanup_token}
                )

                if response.status_code == 200:
                    data = response.json()
                    is_active = bool(data.get("active", False))
                    self.logger.debug(f"Episode status check: active={is_active}")
                    return is_active
                else:
                    self.logger.warning(f"Episode status check failed: HTTP {response.status_code}")
                    return None

        except Exception as e:
            self.logger.error(f"Episode status check failed: {e}")
            return None

    def get_episode_containers(self) -> List[str]:
        """
        Get list of containers belonging to this episode.

        Returns:
            List of container IDs/names
        """
        try:
            # Use docker compose to list containers for this project
            if not self.compose_project:
                raise ValueError("Compose project not configured")

            cmd = ["docker", "compose", "-p", self.compose_project, "ps", "-q"]
            if self.compose_file:
                cmd.extend(["-f", self.compose_file])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                container_ids = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                self.logger.debug(f"Found {len(container_ids)} containers for project {self.compose_project}")
                return container_ids
            else:
                self.logger.warning(f"Failed to list containers: {result.stderr}")
                return []

        except Exception as e:
            self.logger.error(f"Error getting episode containers: {e}")
            return []

    def graceful_stop_containers(self) -> bool:
        """
        Gracefully stop all containers in the episode.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.compose_project:
                raise ValueError("Compose project not configured")

            log_operation_start(
                self.logger,
                "Graceful container stop",
                self.session_id,
                project=self.compose_project,
                timeout=self.graceful_timeout,
            )

            # Build docker compose stop command
            cmd = ["docker", "compose", "-p", self.compose_project, "stop", "-t", str(self.graceful_timeout)]
            if self.compose_file:
                cmd.extend(["-f", self.compose_file])

            self.logger.debug(f"Running command: {' '.join(cmd)}", self.session_id)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.graceful_timeout + 30)

            if result.returncode == 0:
                log_operation_success(
                    self.logger, "Graceful container stop", self.session_id, project=self.compose_project
                )
                return True
            else:
                log_operation_failure(self.logger, "Graceful container stop", result.stderr, self.session_id)
                return False

        except Exception as e:
            log_operation_failure(self.logger, "Graceful container stop", str(e), self.session_id)
            return False

    def force_cleanup_containers(self) -> bool:
        """
        Force cleanup all containers and resources.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.compose_project:
                raise ValueError("Compose project not configured")

            # Import cleanup logger for this specific operation
            from saber.logging_config import get_cleanup_logger

            cleanup_logger = get_cleanup_logger(__name__)

            log_operation_start(
                cleanup_logger, "Force container cleanup", self.session_id, project=self.compose_project
            )

            # Build docker compose down command with force cleanup
            cmd = [
                "docker",
                "compose",
                "-p",
                self.compose_project,
                "down",
                "--remove-orphans",
                "--volumes",
                "--timeout",
                "10",
            ]
            if self.compose_file:
                cmd.extend(["-f", self.compose_file])

            cleanup_logger.debug(f"Running command: {' '.join(cmd)}", self.session_id)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                log_operation_success(
                    cleanup_logger, "Force container cleanup", self.session_id, project=self.compose_project
                )
                return True
            else:
                cleanup_logger.warning(f"Force cleanup had issues: {result.stderr}", self.session_id)
                # Consider it successful even with warnings
                return True

        except Exception as e:
            log_operation_failure(cleanup_logger, "Force container cleanup", str(e), self.session_id)
            return False

    def nuclear_cleanup(self) -> None:
        """
        Nuclear option: find and kill all containers with episode labels.
        """
        try:
            # Import cleanup logger for this critical operation
            from saber.logging_config import get_cleanup_logger

            cleanup_logger = get_cleanup_logger(__name__)

            cleanup_logger.warning("Performing nuclear cleanup - finding containers by session label", self.session_id)

            # Find containers by SABER session label
            cmd = ["docker", "ps", "-aq", "--filter", f"label=saber.session_id={self.session_id}"]

            cleanup_logger.debug(f"Running container search: {' '.join(cmd)}", self.session_id)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout.strip():
                container_ids = result.stdout.strip().split("\n")
                cleanup_logger.warning(f"Found {len(container_ids)} containers with session label", self.session_id)

                # Force kill and remove
                for container_id in container_ids:
                    try:
                        cleanup_logger.info(f"Killing container {container_id}", self.session_id)
                        subprocess.run(["docker", "kill", container_id], timeout=10)
                        cleanup_logger.info(f"Removing container {container_id}", self.session_id)
                        subprocess.run(["docker", "rm", "-f", container_id], timeout=10)
                        cleanup_logger.info(f"Forcefully removed container {container_id}", self.session_id)
                    except Exception as e:
                        cleanup_logger.error(f"Failed to remove container {container_id}: {e}", self.session_id)
            else:
                cleanup_logger.info("No containers found with session label", self.session_id)

        except Exception as e:
            cleanup_logger.error(f"Nuclear cleanup failed: {e}", self.session_id)

    async def cleanup_episode(self, reason: str) -> None:
        """
        Perform complete episode cleanup with multiple fallback strategies.

        Args:
            reason: Reason for cleanup
        """
        self.logger.info(f"Starting episode cleanup: {reason}", self.session_id)

        containers_before = self.get_episode_containers()
        if containers_before:
            self.logger.info(f"Found {len(containers_before)} containers to clean up", self.session_id)
        else:
            self.logger.info("No containers found, cleanup may already be complete", self.session_id)

        # Strategy 1: Graceful stop
        self.logger.info("Attempting graceful container stop", self.session_id)
        if self.graceful_stop_containers():
            # Give containers time to stop gracefully
            await asyncio.sleep(5)

            # Check if containers are actually stopped
            containers_after = self.get_episode_containers()
            if not containers_after:
                self.logger.info("Graceful cleanup successful", self.session_id)
                return

        # Strategy 2: Force cleanup with docker compose down
        self.logger.info("Graceful stop failed, attempting force cleanup", self.session_id)
        if self.force_cleanup_containers():
            await asyncio.sleep(5)

            containers_after = self.get_episode_containers()
            if not containers_after:
                self.logger.info("Force cleanup successful", self.session_id)
                return

        # Strategy 3: Nuclear option
        self.logger.error("Force cleanup failed, performing nuclear cleanup", self.session_id)
        self.nuclear_cleanup()

        self.logger.info("Episode cleanup completed", self.session_id)

    async def monitor_episode(self) -> None:
        """Main monitoring loop for episode lifecycle."""
        self.logger.info("Starting episode monitoring", self.session_id)

        while self.running:
            try:
                status = await self.check_episode_status()

                if status is True:
                    # Episode is active, reset failure count
                    self.failure_count = 0
                    self.logger.debug("Episode is active, continuing monitoring...")

                elif status is False:
                    # Episode is no longer active, cleanup all containers
                    self.logger.info("Episode no longer active, triggering cleanup", self.session_id)
                    await self.cleanup_episode("Episode no longer active")
                    self.running = False
                    return

                else:
                    # Status check failed
                    self.failure_count += 1
                    self.logger.warning(
                        f"Status check failed ({self.failure_count}/{self.max_failures})", self.session_id
                    )

                    if self.failure_count >= self.max_failures:
                        self.logger.warning(
                            f"Max failures ({self.max_failures}) reached, triggering cleanup", self.session_id
                        )
                        await self.cleanup_episode(
                            f"Max failures ({self.max_failures}) reached, assuming episode ended"
                        )
                        self.running = False
                        return

                # Wait before next check
                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                self.logger.error(f"Unexpected error in monitoring loop: {e}", self.session_id)
                self.failure_count += 1

                if self.failure_count >= self.max_failures:
                    self.logger.warning("Max failures reached due to errors, triggering cleanup", self.session_id)
                    await self.cleanup_episode("Max failures reached due to errors")
                    self.running = False
                    return

                # Brief sleep before retrying
                await asyncio.sleep(5)

    async def run(self) -> None:
        """Run the episode container orchestrator."""
        self.logger.info("Episode orchestrator starting", self.session_id)
        try:
            await self.monitor_episode()
        except Exception as e:
            self.logger.error(f"Fatal error in episode orchestrator: {e}", self.session_id)
            await self.cleanup_episode("Fatal error occurred")
        finally:
            self.logger.info("Episode Container Orchestrator shutting down", self.session_id)


async def main() -> None:
    """Main entry point."""
    print("🎬 ORCHESTRATOR MAIN: Starting episode orchestrator main()", file=sys.stderr)
    try:
        orchestrator = EpisodeContainerOrchestrator()
        print(f"🎬 ORCHESTRATOR MAIN: Created orchestrator for session {orchestrator.session_id}", file=sys.stderr)
        await orchestrator.run()
    except ValueError as e:
        print(f"❌ ORCHESTRATOR MAIN ERROR: Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ORCHESTRATOR MAIN ERROR: Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print("🎬 ORCHESTRATOR MAIN: Main function completed", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
