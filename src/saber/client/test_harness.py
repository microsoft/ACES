"""
SABER Client - Test harness controller.

Main controller that orchestrates the test execution by coordinating
the server client, agent, and prompt builder components.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .context_logger import ContextLogger
from .prompt_builder import PromptBuilder
from .server_client import ServerClient


class StructuredFormatter(logging.Formatter):
    """JSON-based structured logging formatter for detailed file logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from log record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


@dataclass
class TestHarnessConfig:
    """Configuration for the test harness."""

    # Server connection
    server_url: str = "http://localhost:8000"
    request_timeout: float = 30.0

    # Test execution
    max_steps: int = 100
    step_delay: float = 0.0  # Delay between steps in seconds

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_structured: bool = False  # Enable structured JSON logging to file

    # Additional configuration
    agent_config: Dict[str, Any] = field(default_factory=dict)


class TestHarness:
    """
    Main test harness controller.

    Orchestrates the execution of security analysis tasks by:
    1. Managing server communication
    2. Building prompts for the agent
    3. Coordinating the test execution loop
    4. Handling errors and cleanup
    """

    def __init__(self, config: TestHarnessConfig):
        """
        Initialize the test harness.

        Args:
            config: Test harness configuration
        """
        self.config = config
        self.server_client: Optional[ServerClient] = None
        self.agent: Optional[Any] = None  # Any agent with process_prompt and reset methods
        self.logger = self._setup_logging()
        self.ctx_logger = ContextLogger(self.logger)  # Context-aware logger wrapper

        # Test state
        self.session_id: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.step_count = 0
        self.is_running = False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration with enhanced file logging support."""
        logger = logging.getLogger("saber.client")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler with standard formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        console_formatter = logging.Formatter(self.config.log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if self.config.log_file:
            try:
                # Ensure log directory exists
                self.config.log_file.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setLevel(getattr(logging, self.config.log_level.upper()))

                if self.config.log_structured:
                    # Use structured JSON formatter for file logging
                    file_formatter: Union[StructuredFormatter, logging.Formatter] = StructuredFormatter()
                else:
                    # Use standard formatter for file logging
                    file_formatter = logging.Formatter(self.config.log_format)

                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

                logger.info(f"File logging enabled: {self.config.log_file}")

            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")

        return logger

    async def initialize(self, agent: Any) -> None:
        """
        Initialize the test harness with an agent.

        The agent can be any object that has:
        - async process_prompt(prompt: str) -> str method
        - async reset() -> None method (optional)

        Args:
            agent: The agent to test (any object with required methods)
        """
        self.logger.info("Initializing SABER test harness")

        # Setup server client
        self.server_client = ServerClient(server_url=self.config.server_url, timeout=self.config.request_timeout)

        # Setup agent
        self.agent = agent
        await self.agent.reset()

        # Test server connection
        try:
            health = await self.server_client.health_check()
            self.logger.info(f"Server health check passed: {health}")
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            raise

        self.logger.info("Test harness initialized successfully")

    async def run_test(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the main test loop.

        Args:
            task_id: Optional task ID to execute. If None, uses server default.

        Returns:
            Dict[str, Any]: Test execution results and metadata
        """
        if not self.server_client or not self.agent:
            raise ValueError("Test harness not initialized. Call initialize() first.")

        self.logger.info("Starting test execution")
        self.is_running = True

        try:
            # Create session
            self.session_id = await self.server_client.create_session()
            self.ctx_logger.update_context(session_id=self.session_id)
            self.ctx_logger.log_session_created(self.session_id)

            # Start episode with optional task_id
            self.ctx_logger.log_episode_starting(task_id)
            if task_id:
                episode_info = await self.server_client.start_episode(task_id)
            else:
                episode_info = await self.server_client.start_episode()
            self.episode_id = episode_info.episode_id
            self.ctx_logger.log_episode_start(self.episode_id, task_id)

            # Get task and policy information
            task_info = await self.server_client.get_current_task()
            policy_info = await self.server_client.get_policy()

            self.ctx_logger.log_task_info(task_info.title, policy_info.available_commands)

            # Execute test loop
            initial_prompt = PromptBuilder.build_initial_prompt(task_info, policy_info, episode_info)
            current_prompt = initial_prompt
            step_response = None

            while self.is_running and self.step_count < self.config.max_steps:
                self.step_count += 1
                self.ctx_logger.log_step_start(self.step_count)

                # Get agent response
                try:
                    agent_response = await self.agent.process_prompt(current_prompt)
                    self.ctx_logger.log_agent_response(agent_response)

                    if not agent_response.strip():
                        self.ctx_logger.log_empty_response()
                        break

                except Exception as e:
                    self.ctx_logger.log_agent_error(str(e))
                    break

                # Execute step on server
                try:
                    step_response = await self.server_client.execute_step(agent_response)
                    self.ctx_logger.log_step_response(step_response.success, step_response.done, step_response.output)

                    if step_response.error:
                        self.ctx_logger.log_step_error(step_response.error)

                except Exception as e:
                    self.ctx_logger.log_server_error(str(e))
                    break

                # Check if done
                if step_response.done:
                    self.ctx_logger.log_episode_completed()
                    completion_prompt = PromptBuilder.build_completion_prompt(step_response.output)
                    self.ctx_logger.log_final_result(completion_prompt)
                    break

                # Build next prompt
                if step_response.success:
                    current_prompt = PromptBuilder.build_step_prompt(
                        step_response.output, agent_response, step_response
                    )
                else:
                    current_prompt = PromptBuilder.build_error_prompt(
                        step_response.error or "Unknown error", agent_response
                    )

                # Optional delay between steps
                if self.config.step_delay > 0:
                    await asyncio.sleep(self.config.step_delay)

            # Prepare results
            results = {
                "session_id": self.session_id,
                "episode_id": self.episode_id,
                "steps_executed": self.step_count,
                "completed": step_response.done if step_response else False,
                "final_step": step_response.model_dump() if step_response else None,
                "task_info": task_info.model_dump(),
            }

            self.ctx_logger.log_test_completed(results)
            return results

        except Exception as e:
            self.ctx_logger.log_test_failed(str(e))
            raise
        finally:
            self.is_running = False

    async def shutdown(self) -> None:
        """Close session and cleanup resources."""
        self.logger.info("Shutting down test harness")

        self.is_running = False

        if self.server_client:
            try:
                await self.server_client.close_session()
                self.logger.info("Session closed")
            except Exception as e:
                self.logger.warning(f"Error closing session: {e}")

        # Reset state
        self.session_id = None
        self.episode_id = None
        self.step_count = 0

        self.logger.info("Test harness shutdown complete")

    async def stop(self) -> None:
        """Stop the currently running test."""
        self.logger.info("Stopping test execution")
        self.is_running = False

    async def __aenter__(self) -> "TestHarness":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        await self.shutdown()
