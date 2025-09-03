#!/usr/bin/env python3
"""
SABER Harness

Client-side orchestrator for running agent episodes using containerized execution
with a shared MCP sidecar.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .api import SABERRestClient
from .episode_executor import ContainerEpisodeExecutor
from .harness_models import HarnessRunResult, SABERHarnessConfig
from .llm import create_llm_client
from .logging import (
    CleanupEvent,
    ClientLoggingConfig,
    EpisodeQueueEvent,
    ErrorEvent,
    ExecutionCompleteEvent,
    ExecutionStartEvent,
    HarnessExecutionLogger,
    HarnessInitializationEvent,
    InfrastructureEvent,
    SessionCreationEvent,
    TaskResolutionEvent,
    UIEvent,
)
from .ui import MessageType, TaskInfo, TaskStatus, UIConfig, UIManager

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

        # Agent file path for direct file-based loading
        self.agent_file_path: Optional[str] = None
        self.agent_class_name: Optional[str] = None

        # Container execution infrastructure (optional)
        self.container_executor: Optional[ContainerEpisodeExecutor] = None

        # Harness execution logger (initialized when session starts)
        self.harness_logger: Optional[HarnessExecutionLogger] = None

        # Session log directory (shared by harness and container logs)
        self.session_log_dir: Optional[Path] = None

        # UI Manager using new Inspect-AI flow
        self.ui_manager: Optional[UIManager] = None
        if self.config.ui_enabled:
            ui_config = UIConfig.for_inspect_ai(mode=self.config.ui_backend)
            self.ui_manager = UIManager(ui_config)
            logger.info(f"🎨 UI manager initialized: {self.config.ui_backend}")

        # SSE client for tool call progress updates (replaces old progress server)
        self.tool_call_sse_client: Optional[Any] = None

        # Configure logging
        logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper()))

    async def initialize(self, agent: Any, env_file: Optional[Path] = None) -> None:
        """Initialize the harness with an agent and optional environment file."""
        logger.info("🚀 Initializing SABER harness")
        self.agent = agent
        await self._initialize_common(env_file)

    async def initialize_with_file(
        self, agent_file_path: str, agent_class: Optional[str] = None, env_file: Optional[Path] = None
    ) -> None:
        """Initialize the harness with an agent file path directly."""
        logger.info("🚀 Initializing SABER harness with agent file")

        # Store agent file information for container packaging
        self.agent_file_path = agent_file_path
        self.agent_class_name = agent_class
        self.agent = None  # No need to load the agent in the client

        await self._initialize_common(env_file)

    async def _initialize_common(self, env_file: Optional[Path] = None) -> None:
        """Common initialization logic for both initialization methods."""

        # Initialize harness execution logging early to capture all events
        if self.config.enable_container_logging and self.config.client_log_dir:
            # Create session log directory with timestamp only
            session_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
            self.session_log_dir = self.config.client_log_dir / session_timestamp

            logging_config = ClientLoggingConfig(
                base_log_dir=self.config.client_log_dir,
                enable_container_logging=self.config.enable_container_logging,
                retention_days=self.config.log_retention_days,
                max_log_size_mb=self.config.max_log_size_mb,
                compress_old_logs=self.config.compress_old_logs,
            )

            self.harness_logger = HarnessExecutionLogger(
                config=logging_config, session_log_dir=self.session_log_dir, client_id=self.config.client_id
            )

            # Log harness initialization
            init_event = HarnessInitializationEvent(
                parallelism=self.config.parallelism,
                client_id=self.config.client_id,
                server_url=self.config.server_url,
                mcp_url=self.config.mcp_url,
                enable_container_logging=self.config.enable_container_logging,
                client_log_dir=self.config.client_log_dir,
                log_retention_days=self.config.log_retention_days,
                task_ids=self.config.task_ids,
                ui_enabled=self.config.ui_enabled,
                ui_backend=self.config.ui_backend,
            )
            self.harness_logger.log_harness_initialization_event(init_event)

            logger.info(f"📋 Harness execution logging enabled: {self.session_log_dir}")
        else:
            logger.info("📋 Harness execution logging disabled")

        # Initialize REST client with sidecar support
        # The sidecar runs in its own container named "saber-mcp-sidecar" on port 8002
        sidecar_url = "http://saber-mcp-sidecar:8002" if self.config.mcp_url else None
        self.rest_client = SABERRestClient(
            base_url=self.config.server_url,
            client_id=self.config.client_id,
            request_timeout=self.config.request_timeout,
            sidecar_url=sidecar_url,
        )

        # Optional LLM client
        self.llm_client = create_llm_client(
            provider=self.config.llm_provider, env_file=env_file, **self.config.llm_config
        )
        logger.info("✅ LLM client initialized" if self.llm_client else "ℹ️ No LLM client configuration found")

        # Initialize SSE client for real-time tool call updates from sidecar
        logger.info(
            f"🔧 SSE client initialization check: ui_enabled={self.config.ui_enabled}, "
            f"ui_manager={self.ui_manager is not None}"
        )
        if self.config.ui_enabled and self.ui_manager:
            try:
                # SSE client will connect to sidecar's /tool-call-events endpoint
                # This will be initialized when sidecar is available
                logger.info("� SSE client for tool call updates will be initialized when sidecar starts")
            except Exception as e:
                logger.error(f"Failed to prepare SSE client: {e}")
                self.tool_call_sse_client = None
        else:
            logger.info("ℹ️ Progress server not started: UI disabled or UI manager not available")

        logger.info("✅ SABER harness initialized successfully")

    async def run(self) -> HarnessRunResult:
        """Run complete test with client-orchestrated episodes."""
        start_time = datetime.now(timezone.utc)

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

            # Create session using rest api interfacing w/ saber server
            try:
                self.session_id = await self.rest_client.create_session()
                logger.info(f"📍 Created session: {self.session_id}")

                # Update logging context with real session ID
                if self.harness_logger:
                    self.harness_logger.set_session_id(self.session_id)

                # Log successful session creation
                if self.harness_logger:
                    session_event = SessionCreationEvent(
                        session_id=self.session_id, success=True, server_url=self.config.server_url
                    )
                    self.harness_logger.log_session_creation_event(session_event)

            except Exception as e:
                # Log failed session creation
                if self.harness_logger:
                    session_event = SessionCreationEvent(
                        session_id=None, success=False, error_message=str(e), server_url=self.config.server_url
                    )
                    self.harness_logger.log_session_creation_event(session_event)
                raise

            # Determine tasks first to decide UI flow
            tasks = await self._resolve_tasks()
            logger.info(f"📋 Resolved {len(tasks)} tasks for execution")

            # Check if textual mode is requested and available
            if (
                self.ui_manager
                and self.config.ui_backend == "textual"
                and hasattr(self.ui_manager, "run_textual_evaluation")
            ):

                logger.info("🎨 Starting SABER with textual UI mode")
                return await self._run_with_textual_ui(tasks, start_time)
            else:
                # Use standard UI flow
                return await self._run_with_standard_ui(tasks, start_time)

        except Exception as e:
            logger.error(f"❌ SABER harness failed: {e}")

            if self.harness_logger:
                error_event = ErrorEvent(
                    error_type="harness_execution",
                    error_message=f"SABER harness execution failed: {str(e)}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    session_id=self.session_id,
                    execution_phase="unknown",  # Could be enhanced to track current phase
                )
                self.harness_logger.log_error_event(error_event)

            await self._cleanup_session()
            raise

    async def _run_with_textual_ui(self, tasks: List[Dict[str, Any]], start_time: datetime) -> HarnessRunResult:
        """Run SABER with textual UI mode using the dedicated textual flow."""
        try:
            # Convert tasks to TaskInfo objects for textual UI
            task_infos = []
            for i, task in enumerate(tasks):
                task_info = TaskInfo(
                    task_id=task["task_id"],
                    name=task.get("name", f"Task {i+1}"),
                    status=TaskStatus.PENDING,
                    progress=0.0,
                    metadata={
                        "target": task.get("target", "saber_domain"),
                        "description": task.get("description", f"Execute task {task['task_id']}"),
                    },
                )
                task_infos.append(task_info)

            # Define the evaluation function that will run inside textual context
            async def evaluation_function(textual_session: Any) -> Any:
                logger.info("🎨 Starting evaluation within textual UI context")

                # Initialize container infrastructure within textual context
                await self._initialize_container_infrastructure()

                # Build episode queue
                episodes = [(task["task_id"], 1) for task in tasks]
                logger.info(f"🎬 Prepared {len(episodes)} episodes")

                # Execute episodes with UI progress tracking
                if not self.container_executor:
                    raise RuntimeError("Container executor not initialized")

                all_results = await self.container_executor.execute_episodes(episodes, self.agent)

                return all_results

            # Run the evaluation within textual UI
            logger.info("🎨 Launching textual UI...")
            if not self.ui_manager:
                raise RuntimeError("UI manager not initialized")
            if not self.session_id:
                raise RuntimeError("Session ID not set")

            all_results = self.ui_manager.run_textual_evaluation(
                session_id=self.session_id, tasks=task_infos, evaluation_func=evaluation_function
            )

            return self._create_harness_result(all_results, start_time)

        except Exception as e:
            logger.error(f"❌ Textual UI execution failed: {e}")
            await self._cleanup_session()
            raise

    async def _run_with_standard_ui(self, tasks: List[Dict[str, Any]], start_time: datetime) -> HarnessRunResult:
        """Run SABER with standard UI modes (plain, rich, etc.)."""
        try:
            # Initialize UI session using new UIManager
            if self.ui_manager:
                session_info = {
                    "session_id": self.session_id,
                    "total_tasks": len(tasks),
                    "parallel_episodes": self.config.parallelism,
                    "ui_tool_detail_level": self.config.ui_tool_detail_level,
                    "agent_name": getattr(self.agent, "__class__.__name__", "Unknown") if self.agent else "Agent",
                    "target": "saber_domain",
                    "start_time": start_time,
                    "tasks": [task["task_id"] for task in tasks],
                }
                await self.ui_manager.session_start(session_info)

                if self.harness_logger:
                    ui_event = UIEvent(
                        event_type="session_start",
                        ui_backend=self.config.ui_backend,
                        success=True,
                        details={"ui_tool_detail_level": self.config.ui_tool_detail_level},
                    )
                    self.harness_logger.log_ui_event_structured(ui_event)

            # Initialize container infrastructure
            await self._initialize_container_infrastructure()

            if self.harness_logger:
                task_ids = [task["task_id"] for task in tasks]
                task_resolution_event = TaskResolutionEvent(
                    total_tasks=len(tasks),
                    requested_task_ids=self.config.task_ids,
                    resolved_task_ids=task_ids,
                    resolution_mode="specific" if self.config.task_ids else "all_available",
                )
                self.harness_logger.log_task_resolution_event(task_resolution_event)

            # Update UI with actual task count and start tasks
            if self.ui_manager:
                # Create task info objects for the UI
                for i, task in enumerate(tasks):
                    task_info = {
                        "task_id": task["task_id"],
                        "name": task.get("name", f"Task {i+1}"),
                        "target": task.get("target", "saber_domain"),
                        "description": task.get("description", f"Execute task {task['task_id']}"),
                    }
                    await self.ui_manager.task_start(task_info)

                self.ui_manager.display_message(f"📋 Prepared {len(tasks)} tasks for execution", MessageType.INFO)

            # Build episode queue (one attempt per task per requirement)
            episodes = [(task["task_id"], 1) for task in tasks]
            logger.info(f"🎬 Prepared {len(episodes)} episodes")

            if self.harness_logger:
                episode_details = [{"task_id": task_id, "attempt": attempt} for task_id, attempt in episodes]
                episode_queue_event = EpisodeQueueEvent(episode_count=len(episodes), episodes=episode_details)
                self.harness_logger.log_episode_queue_event(episode_queue_event)

            # Execute episodes (container-only)
            if self.harness_logger:
                agent_info = {
                    "agent_type": type(self.agent).__name__ if self.agent else None,
                    "agent_config": getattr(self.agent, "config", None) if hasattr(self.agent, "config") else None,
                }
                execution_start_event = ExecutionStartEvent(
                    parallelism=self.config.parallelism, episode_count=len(episodes), agent_info=agent_info
                )
                self.harness_logger.log_execution_start_event(execution_start_event)

            if not self.container_executor:
                raise RuntimeError("Container executor not initialized")

            all_results = await self.container_executor.execute_episodes(episodes, self.agent)

            # Calculate execution duration
            end_time = datetime.now(timezone.utc)
            execution_duration = (end_time - start_time).total_seconds()

            if self.harness_logger:
                successful_episodes = len([r for r in all_results if r.success])
                execution_complete_event = ExecutionCompleteEvent(
                    total_episodes=len(all_results),
                    successful_episodes=successful_episodes,
                    success_rate=successful_episodes / len(all_results) if all_results else 0,
                    execution_duration_seconds=execution_duration,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                )
                self.harness_logger.log_execution_complete_event(execution_complete_event)

            await self._cleanup_session()

            # Return results
            result = self._create_harness_result(all_results, start_time)

            # Complete UI session
            if self.ui_manager:
                successful_episode_results = [r for r in all_results if r.success]
                session_summary = {
                    "session_id": self.session_id,
                    "total_episodes": len(all_results),
                    "successful_episodes": len(successful_episode_results),
                    "completion_time": datetime.now().isoformat(),
                    "final_success": result.success,
                    "success": result.success,
                    "tasks_completed": len(successful_episode_results),
                    "flags_captured": len(successful_episode_results),  # Approximate
                    "total_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                    "agent_performance": "excellent" if result.success else "needs_improvement",
                }
                await self.ui_manager.session_complete(session_summary)

                if result.success:
                    self.ui_manager.display_message("✅ SABER benchmark completed successfully!", MessageType.SUCCESS)
                else:
                    self.ui_manager.display_message("⚠️ SABER benchmark completed with issues", MessageType.WARNING)

            return result

        except Exception as e:
            logger.error(f"❌ Standard UI execution failed: {e}")
            await self._cleanup_session()
            raise

    async def _initialize_container_infrastructure(self) -> None:
        """Initialize container infrastructure."""
        if self.container_executor is None:
            if not self.rest_client:
                raise RuntimeError("REST client not initialized")
            if not self.session_id:
                raise RuntimeError("Session ID not set")

            self.container_executor = ContainerEpisodeExecutor(
                rest_client=self.rest_client,
                session_id=self.session_id,
                parallelism=self.config.parallelism,
                saber_server_url=self.config.server_url,
                saber_mcp_url=self.config.mcp_url,
                harness_config=self.config,
                ui_manager=self.ui_manager,  # Pass UI manager to executor
                session_log_dir=self.session_log_dir,  # Pass shared session log directory
                harness=self,  # Pass harness reference for agent file access
            )
        # Always allow the executor to (re)initialize; custom injected executors can no-op
        await self.container_executor.initialize()
        logger.info("🐳 Container infrastructure initialized")

        if self.harness_logger:
            infra_event = InfrastructureEvent(
                operation="initialized",
                component="container_infrastructure",
                success=True,
                details={"parallelism": self.config.parallelism},
            )
            self.harness_logger.log_infrastructure_event_structured(infra_event)

    def _create_harness_result(self, all_results: List, start_time: datetime) -> HarnessRunResult:
        """Create a HarnessRunResult from episode results."""
        if not self.session_id:
            raise RuntimeError("Session ID not set")

        successful_episode_results = [r for r in all_results if r.success]
        return HarnessRunResult(
            session_id=self.session_id,
            total_episodes=len(all_results),
            successful_episodes=len(successful_episode_results),
            episode_results=all_results,
            success=len(successful_episode_results) > 0,
        )

    async def _resolve_tasks(self) -> List[Dict[str, Any]]:
        """Resolve tasks according to R2/R2b."""
        assert self.rest_client is not None  # Ensured by run() method

        if self.config.task_ids:
            # R2: Use supplied tasks
            all_tasks = await self.rest_client.list_tasks()
            filtered_tasks = [task for task in all_tasks if task["task_id"] in self.config.task_ids]

            if not filtered_tasks:
                error_msg = f"No matching tasks found for: {self.config.task_ids}"
                if self.harness_logger:
                    error_event = ErrorEvent(
                        error_type="task_resolution", error_message=error_msg, session_id=self.session_id
                    )
                    self.harness_logger.log_error_event(error_event)
                raise ValueError(error_msg)

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
        cleanup_start_time = datetime.now(timezone.utc)
        cleanup_successful = True
        cleanup_errors = []

        try:
            # Clean up container infrastructure first
            if self.container_executor:
                try:
                    await self.container_executor.cleanup()
                    logger.info("🐳 Container infrastructure cleaned up")
                except Exception as e:
                    cleanup_successful = False
                    cleanup_errors.append(f"Container cleanup failed: {str(e)}")
                    logger.error(f"❌ Container cleanup failed: {e}")

            # Clean up session
            if self.rest_client and self.session_id:
                try:
                    await self.rest_client.terminate_session(self.session_id)
                    logger.info(f"🧹 Cleaned up session: {self.session_id}")
                except Exception as e:
                    cleanup_successful = False
                    cleanup_errors.append(f"Session termination failed: {str(e)}")
                    logger.error(f"❌ Session termination failed: {e}")

            # Clean up UI session - UIManager handles its own cleanup
            if self.ui_manager:
                try:
                    # UIManager doesn't need explicit cleanup in our new flow
                    logger.info("🎨 UI session cleaned up")
                except Exception as e:
                    cleanup_successful = False
                    cleanup_errors.append(f"UI cleanup failed: {str(e)}")
                    logger.error(f"❌ UI cleanup failed: {e}")

            # Clean up SSE client for tool call updates
            if self.tool_call_sse_client:
                try:
                    # Close SSE connection
                    if hasattr(self.tool_call_sse_client, "close"):
                        await self.tool_call_sse_client.close()
                    logger.info("� SSE client for tool call updates stopped")
                except Exception as e:
                    cleanup_successful = False
                    cleanup_errors.append(f"SSE client cleanup failed: {str(e)}")
                    logger.error(f"❌ SSE client cleanup failed: {e}")

            # Log cleanup completion
            if self.harness_logger:
                cleanup_end_time = datetime.now(timezone.utc)
                cleanup_duration = (cleanup_end_time - cleanup_start_time).total_seconds()

                cleanup_event = CleanupEvent(
                    cleanup_successful=cleanup_successful,
                    cleanup_duration_seconds=cleanup_duration,
                    cleanup_errors=cleanup_errors,
                    session_id=self.session_id,
                    cleanup_start_time=cleanup_start_time.isoformat(),
                    cleanup_end_time=cleanup_end_time.isoformat(),
                )
                self.harness_logger.log_cleanup_event(cleanup_event)

                # Generate session summary
                session_summary = self.harness_logger.get_session_summary()
                logger.info(f"📋 Session summary: {session_summary.get('event_count', 0)} events logged")

            self.session_id = None

        except Exception as e:
            logger.error(f"❌ Session cleanup failed: {e}")

            if self.harness_logger:
                error_event = ErrorEvent(
                    error_type="cleanup",
                    error_message=f"Session cleanup failed with exception: {str(e)}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    session_id=self.session_id,
                )
                self.harness_logger.log_error_event(error_event)
