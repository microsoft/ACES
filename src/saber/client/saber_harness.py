#!/usr/bin/env python3
"""
SABER Harness

Client-side orchestrator for running agent episodes using containerized execution
with a shared MCP sidecar.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .api import SABERRestClient
from .episode_executor import ContainerEpisodeExecutor
from .harness_models import HarnessRunResult, SABERHarnessConfig
from .llm import create_llm_client

logger = logging.getLogger(__name__)


class SABERHarness:
    """Client orchestrator for container-executed agents."""

    def __init__(self, config: Optional[SABERHarnessConfig] = None):
        """Initialize harness with configuration."""
        self.config = config or SABERHarnessConfig()

        # Core state
        self.session_id: Optional[str] = None
        self.rest_client: Optional[SABERRestClient] = None
        self.llm_client: Optional[Any] = None
        self.agent: Optional[Any] = None

        # Container execution infrastructure (optional)
        self.container_executor: Optional[ContainerEpisodeExecutor] = None

        # Configure logging
        logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper()))

    async def initialize(self, agent: Any, env_file: Optional[Path] = None) -> None:
        """Initialize the harness with an agent and optional environment file."""
        logger.info("🚀 Initializing SABER harness")
        self.agent = agent

        # Initialize REST client
        self.rest_client = SABERRestClient(
            base_url=self.config.server_url,
            client_id=self.config.client_id,
            request_timeout=self.config.request_timeout,
        )

        # Optional LLM client
        self.llm_client = create_llm_client(
            provider=self.config.llm_provider, env_file=env_file, **self.config.llm_config
        )
        logger.info("✅ LLM client initialized" if self.llm_client else "ℹ️ No LLM client configuration found")

        logger.info("✅ SABER harness initialized successfully")

    async def run(self) -> HarnessRunResult:
        """Run complete test with client-orchestrated episodes."""
        try:
            logger.info("🚀 Starting SABER benchmark")
            logger.info(f"🎯 Parallelism: {self.config.parallelism}")

            if self.config.task_ids:
                logger.info(f"🎯 Specific tasks: {self.config.task_ids}")
            else:
                logger.info("🎯 All available tasks")

            logger.info("=" * 60)

            if not self.rest_client:
                raise RuntimeError("REST client not initialized")

            # Step 1: Create session
            self.session_id = await self.rest_client.create_session()
            logger.info(f"📍 Created session: {self.session_id}")

            # Step 1.5: Initialize container infrastructure
            if self.container_executor is None:
                self.container_executor = ContainerEpisodeExecutor(
                    rest_client=self.rest_client,
                    session_id=self.session_id,
                    parallelism=self.config.parallelism,
                )
            # Always allow the executor to (re)initialize; custom injected executors can no-op
            await self.container_executor.initialize()
            logger.info("🐳 Container infrastructure initialized")

            # Determine tasks
            tasks = await self._resolve_tasks()
            logger.info(f"📋 Resolved {len(tasks)} tasks for execution")

            # Build episode queue (one attempt per task per requirement)
            episodes = [(task["task_id"], 1) for task in tasks]
            logger.info(f"🎬 Prepared {len(episodes)} episodes")

            # Execute episodes (container-only)
            all_results = await self.container_executor.execute_episodes(episodes, self.agent)

            await self._cleanup_session()

            # Return results
            successful_episodes = [r for r in all_results if r.success]
            return HarnessRunResult(
                session_id=self.session_id,
                total_episodes=len(all_results),
                successful_episodes=len(successful_episodes),
                episode_results=all_results,
                success=len(successful_episodes) > 0,
            )

        except Exception as e:
            logger.error(f"❌ SABER harness failed: {e}")
            await self._cleanup_session()
            raise

    async def _resolve_tasks(self) -> List[Dict[str, Any]]:
        """Resolve tasks according to R2/R2b."""
        assert self.rest_client is not None  # Ensured by run() method
        if self.config.task_ids:
            # R2: Use supplied tasks
            all_tasks = await self.rest_client.list_tasks()
            filtered_tasks = [task for task in all_tasks if task["task_id"] in self.config.task_ids]

            if not filtered_tasks:
                raise ValueError(f"No matching tasks found for: {self.config.task_ids}")

            missing_tasks = set(self.config.task_ids) - {t["task_id"] for t in filtered_tasks}
            if missing_tasks:
                logger.warning(f"⚠️ Tasks not found: {missing_tasks}")

            return filtered_tasks
        else:
            # R2b: Fetch and run all tasks (non-interactive default)
            tasks = await self.rest_client.list_tasks()
            if not tasks:
                logger.warning("⚠️ No tasks available")
            return tasks

    async def _cleanup_session(self) -> None:
        """Clean up SABER session and container infrastructure."""
        try:
            # Clean up container infrastructure first
            if self.container_executor:
                await self.container_executor.cleanup()
                logger.info("🐳 Container infrastructure cleaned up")

            # Clean up session
            if self.rest_client and self.session_id:
                await self.rest_client.terminate_session(self.session_id)
                logger.info(f"🧹 Cleaned up session: {self.session_id}")
            self.session_id = None
        except Exception as e:
            logger.error(f"❌ Session cleanup failed: {e}")
