"""
Get target transcript executor for AI red team testing.

Allows red team to retrieve blue team's transcript with state-aware waiting
and efficient delta retrieval. Prevents race conditions by waiting for
WAITING_FOR_USER state before reading.

This executor provides safe transcript retrieval in orchestrated scenarios
where red team needs to observe blue team's conversation.

Logging category: ``LogCategory.TASK_EXEC``.
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional

from .....logging_config import LogCategory, get_saber_logger
from ....base import CommandResult
from ...base import Parameter, ParameterType
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..docker_executor import DockerExecutor
from ..orchestration_utils import resolve_target_episode_id

logger = get_saber_logger(LogCategory.TASK_EXEC, __name__)

# Configuration constants
DAEMON_PORT = os.getenv("WEBSOCKET_DAEMON_PORT", "9999")
TIMEOUT_BUFFER_CURL = 5
TIMEOUT_BUFFER_ENV = 10
MAX_TAIL_COUNT = 1000
MAX_RETRIES = 3
RETRY_ERRORS = {"connection refused", "timed out", "network unreachable"}


class GetTargetTranscriptExecutor(DockerExecutor):
    """
    Retrieve target team's transcript with state-aware waiting and delta retrieval.

    This executor allows red team agents to safely retrieve blue team's transcript
    by waiting for completion of turns (WAITING_FOR_USER state) to avoid reading
    partial context. Features:

    1. **Automatic Target Resolution**: Target episode ID is automatically resolved
       from orchestration metadata (ORCHESTRATION_TARGET_EPISODES).

    2. **State-Aware Waiting**: Waits for target agent to enter WAITING_FOR_USER
       state via event-driven WebSocket notifications, ensuring complete turns.

    3. **Multiple Retrieval Modes**:
       - full: All messages in transcript
       - delta: Only new messages since version X
       - tail: Last N messages (1-1000)

    4. **Retry Logic**: Exponential backoff (1s, 2s, 4s) for transient failures.

    5. **Rewrite Detection**: Tracks last_operation metadata to detect rewrites.

    Communicates with WebSocket daemon via HTTP IPC to query transcript state.
    """

    _executor_metadata = {
        "name": "get_target_transcript",
        "description": "Get target team transcript with wait-for-user detection",
    }

    def __init__(
        self,
        sandbox_manager: SandboxEnvironmentManager,
        config: Optional[Dict[str, Any]] = None,
        session_manager: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize the get_target_transcript executor."""
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)
        self._session_manager = session_manager

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for get_target_transcript executor."""
        return {}

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: Optional[Dict[str, Any]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        session_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> "GetTargetTranscriptExecutor":
        """
        Create get_target_transcript executor with standardized configuration interface.

        Args:
            sandbox_manager: Sandbox manager for Docker execution
            config: Executor-specific configuration dictionary
            additional_params: Additional parameters for executor creation
            session_manager: Optional session manager for cross-episode operations
            **kwargs: Additional keyword arguments

        Returns:
            Configured GetTargetTranscriptExecutor instance
        """
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        return cls(
            sandbox_manager=sandbox_manager,
            config=config,
            session_manager=session_manager,
            **merged_kwargs,
        )

    def setup_parameters(self, config: Dict[str, Any]) -> None:
        """
        Set up executor-specific parameters.

        Args:
            config: The merged configuration dictionary
        """
        self.add_parameter(
            Parameter(
                name="wait_for_user",
                type=ParameterType.BOOLEAN,
                description="Wait until target is waiting for next user message (includes tool results)",
                required=False,
                default=True,
            )
        )

        self.add_parameter(
            Parameter(
                name="max_wait_seconds",
                type=ParameterType.NUMBER,
                description="Maximum seconds to wait for agent to be waiting on user",
                required=False,
                default=120.0,
            )
        )

        self.add_parameter(
            Parameter(
                name="retrieval_mode",
                type=ParameterType.STRING,
                description=(
                    "How to retrieve transcript:\n"
                    "  'full' - All messages\n"
                    "  'delta' - Only new since last read\n"
                    "  'tail' - Last N messages"
                ),
                required=False,
                default="full",
            )
        )

        self.add_parameter(
            Parameter(
                name="tail_count",
                type=ParameterType.INTEGER,
                description=f"Number of recent messages (1-{MAX_TAIL_COUNT}, tail mode only)",
                required=False,
                default=10,
            )
        )

        self.add_parameter(
            Parameter(
                name="since_version",
                type=ParameterType.INTEGER,
                description="Retrieve messages added since this version (delta mode)",
                required=False,
                default=0,
            )
        )

        self.add_parameter(
            Parameter(
                name="include_metadata",
                type=ParameterType.BOOLEAN,
                description="Include version and modification metadata",
                required=False,
                default=True,
            )
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Parameter]:
        """
        Define the parameter schema for the get_target_transcript executor.

        Returns:
            Dictionary mapping parameter names to Parameter definitions
        """
        return {
            "wait_for_user": Parameter(
                name="wait_for_user",
                type=ParameterType.BOOLEAN,
                description="Wait until target is waiting for next user message (includes tool results)",
                required=False,
                default=True,
            ),
            "max_wait_seconds": Parameter(
                name="max_wait_seconds",
                type=ParameterType.NUMBER,
                description="Maximum seconds to wait for agent to be waiting on user",
                required=False,
                default=120.0,
            ),
            "retrieval_mode": Parameter(
                name="retrieval_mode",
                type=ParameterType.STRING,
                description=(
                    "How to retrieve transcript:\n"
                    "  'full' - All messages\n"
                    "  'delta' - Only new since last read\n"
                    "  'tail' - Last N messages"
                ),
                required=False,
                default="full",
            ),
            "tail_count": Parameter(
                name="tail_count",
                type=ParameterType.INTEGER,
                description=f"Number of recent messages (1-{MAX_TAIL_COUNT}, tail mode only)",
                required=False,
                default=10,
            ),
            "since_version": Parameter(
                name="since_version",
                type=ParameterType.INTEGER,
                description="Retrieve messages added since this version (delta mode)",
                required=False,
                default=0,
            ),
            "include_metadata": Parameter(
                name="include_metadata",
                type=ParameterType.BOOLEAN,
                description="Include version and modification metadata",
                required=False,
                default=True,
            ),
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CommandResult:
        """
        Execute transcript retrieval via WebSocket daemon.

        Flow:
        1. Resolve target episode ID from orchestration metadata
        2. If wait_for_user=True, wait for WAITING_FOR_USER state (event-driven)
        3. Query transcript with specified mode (full/delta/tail)
        4. Return result to agent

        Args:
            parameters: Retrieval configuration with mode, tail_count, etc.
            context: Execution context with episode_id and session_id

        Returns:
            CommandResult with transcript messages and metadata
        """
        try:
            red_episode_id = context.get("episode_id")
            if not red_episode_id:
                return CommandResult.error_result("get_target_transcript: Missing episode_id")

            environment = self.get_episode_environment(red_episode_id)
            target_episode_id = await self._resolve_target_episode_id(parameters, context, red_episode_id)

            if not target_episode_id:
                return CommandResult.error_result(
                    "get_target_transcript: Could not resolve target_episode_id from orchestration metadata"
                )

            # Extract and validate parameters
            wait_for_user = parameters.get("wait_for_user", True)
            max_wait = parameters.get("max_wait_seconds", 120.0)
            retrieval_mode = parameters.get("retrieval_mode", "full")
            tail_count = parameters.get("tail_count", 10)
            since_version = parameters.get("since_version", 0)
            include_metadata = parameters.get("include_metadata", True)

            # Validate parameters
            if retrieval_mode not in ["full", "delta", "tail"]:
                return CommandResult.error_result(f"get_target_transcript: Invalid retrieval_mode: {retrieval_mode}")
            if tail_count < 1 or tail_count > MAX_TAIL_COUNT:
                return CommandResult.error_result(f"get_target_transcript: tail_count must be 1-{MAX_TAIL_COUNT}")

            # Event-driven wait for WAITING_FOR_USER state
            if wait_for_user:
                wait_result = await self._wait_for_user_with_retry(environment, target_episode_id, max_wait)
                if not wait_result.get("success"):
                    return CommandResult.error_result(
                        f"get_target_transcript: Failed to wait: {wait_result.get('error')}"
                    )

            # Query transcript via daemon IPC with retry
            result = await self._query_transcript_with_retry(
                environment, target_episode_id, retrieval_mode, tail_count, since_version
            )

            if not result.get("success"):
                return CommandResult.error_result(f"get_target_transcript: Query failed: {result.get('error')}")

            # Build result data
            messages = result.get("messages", [])
            result_data = {
                "messages": messages,
                "message_count": len(messages),
                "retrieval_mode": retrieval_mode,
            }

            if include_metadata:
                result_data["metadata"] = {
                    "current_version": result.get("current_version", 0),
                    "full_transcript_length": result.get("full_transcript_length", 0),
                    "last_operation": result.get("last_operation"),
                }

            logger.info(
                "Red team retrieved transcript",
                extra={
                    "event": "transcript_retrieved",
                    "red_episode_id": red_episode_id,
                    "target_episode_id": target_episode_id,
                    "retrieval_mode": retrieval_mode,
                    "message_count": len(messages),
                },
            )

            return CommandResult.success_result(
                result_data,
                metadata={
                    "target_episode_id": target_episode_id,
                    "current_version": result.get("current_version", 0),
                    "retrieved_count": len(messages),
                },
            )

        except Exception as e:
            logger.error(
                "Transcript retrieval failed",
                extra={
                    "event": "transcript_retrieval_failed",
                    "episode_id": context.get("episode_id"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return CommandResult.error_result(f"get_target_transcript: Retrieval failed: {str(e)}")

    async def _query_transcript_with_retry(
        self, environment: Any, target_episode_id: str, mode: str, tail_count: int, since_version: int
    ) -> Dict[str, Any]:
        """
        Query transcript with exponential backoff retry.

        Args:
            environment: Episode environment for command execution
            target_episode_id: Target episode to query
            mode: Retrieval mode (full/delta/tail)
            tail_count: Number of messages for tail mode
            since_version: Version for delta mode

        Returns:
            Dict with success status and transcript data
        """
        for attempt in range(MAX_RETRIES):
            result = await self._query_transcript_via_ipc(
                environment, target_episode_id, mode, tail_count, since_version
            )

            if result.get("success") or not self._is_retryable_error(result):
                return result

            if attempt < MAX_RETRIES - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"Retrying transcript query (attempt {attempt + 1}/{MAX_RETRIES})",
                    extra={
                        "target_episode_id": target_episode_id,
                        "attempt": attempt + 1,
                        "wait_time": wait_time,
                    },
                )
                await asyncio.sleep(wait_time)

        return result

    async def _query_transcript_via_ipc(
        self, environment: Any, target_episode_id: str, mode: str, tail_count: int, since_version: int
    ) -> Dict[str, Any]:
        """
        Query transcript from daemon via HTTP POST.

        Args:
            environment: Episode environment for command execution
            target_episode_id: Target episode to query
            mode: Retrieval mode (full/delta/tail)
            tail_count: Number of messages for tail mode
            since_version: Version for delta mode

        Returns:
            Dict with success status and transcript data
        """
        try:
            payload: Dict[str, Any] = {"target_episode_id": target_episode_id, "mode": mode}
            if mode == "tail":
                payload["tail_count"] = tail_count
            elif mode == "delta":
                payload["since_version"] = since_version

            payload_str = json.dumps(payload)
        except (TypeError, ValueError) as e:
            return {"success": False, "error": f"Invalid payload: {e}"}

        curl_cmd = [
            "curl",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            payload_str,
            "--max-time",
            "30",
            "-sS",
            f"http://localhost:{DAEMON_PORT}/get_transcript",
        ]

        exec_result = await environment.execute_command(command=curl_cmd, timeout=int(self.get_timeout()))

        if exec_result.exit_code != 0:
            return {"success": False, "error": exec_result.stderr}

        try:
            response: Dict[str, Any] = json.loads(exec_result.stdout)
            return response
        except json.JSONDecodeError:
            return {"success": False, "error": f"Invalid JSON: {exec_result.stdout}"}

    async def _wait_for_user_with_retry(
        self, environment: Any, target_episode_id: str, max_wait_seconds: float
    ) -> Dict[str, Any]:
        """
        Wait for WAITING_FOR_USER state with retry logic.

        Args:
            environment: Episode environment for command execution
            target_episode_id: Target episode to wait for
            max_wait_seconds: Maximum seconds to wait

        Returns:
            Dict with success status and state information
        """
        for attempt in range(MAX_RETRIES):
            result = await self._wait_for_user(environment, target_episode_id, max_wait_seconds)

            if result.get("success") or not self._is_retryable_error(result):
                return result

            if attempt < MAX_RETRIES - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"Retrying wait_for_user (attempt {attempt + 1}/{MAX_RETRIES})",
                    extra={
                        "target_episode_id": target_episode_id,
                        "attempt": attempt + 1,
                        "wait_time": wait_time,
                    },
                )
                await asyncio.sleep(wait_time)

        return result

    async def _wait_for_user(self, environment: Any, target_episode_id: str, max_wait_seconds: float) -> Dict[str, Any]:
        """
        Wait for WAITING_FOR_USER state via event-driven WebSocket notification.

        Sends request to daemon which subscribes to is_waiting_on_user events
        and blocks until the target agent enters WAITING_FOR_USER state.

        Args:
            environment: Episode environment for command execution
            target_episode_id: Target episode to wait for
            max_wait_seconds: Maximum seconds to wait

        Returns:
            Dict with success status and state information
        """
        try:
            payload = {
                "target_episode_id": target_episode_id,
                "max_wait_seconds": max_wait_seconds,
            }
            payload_str = json.dumps(payload)
        except (TypeError, ValueError) as e:
            return {"success": False, "error": f"Invalid payload: {e}"}

        # Add buffer to timeouts to account for network overhead
        curl_timeout_seconds = max_wait_seconds + TIMEOUT_BUFFER_CURL
        env_timeout_seconds = max_wait_seconds + TIMEOUT_BUFFER_ENV

        curl_cmd = [
            "curl",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            payload_str,
            "--max-time",
            str(int(curl_timeout_seconds)),
            "-sS",
            f"http://localhost:{DAEMON_PORT}/wait_for_user",
        ]

        exec_result = await environment.execute_command(command=curl_cmd, timeout=int(env_timeout_seconds))

        if exec_result.exit_code != 0:
            return {"success": False, "error": exec_result.stderr}

        try:
            response: Dict[str, Any] = json.loads(exec_result.stdout)
            return response
        except json.JSONDecodeError:
            return {"success": False, "error": f"Invalid JSON: {exec_result.stdout}"}

    def _is_retryable_error(self, result: Dict[str, Any]) -> bool:
        """
        Check if error is transient and should be retried.

        Args:
            result: Result dictionary from IPC call

        Returns:
            True if error should be retried, False otherwise
        """
        if result.get("success"):
            return False
        error = str(result.get("error", "")).lower()
        return any(retry_err in error for retry_err in RETRY_ERRORS)

    async def _resolve_target_episode_id(
        self, parameters: Dict[str, Any], context: Dict[str, Any], red_episode_id: str
    ) -> Optional[str]:
        """
        Resolve target episode ID from orchestration metadata.

        Args:
            parameters: Execution parameters (unused)
            context: Execution context (unused)
            red_episode_id: Red team's episode ID

        Returns:
            Target episode ID or None if not resolvable
        """
        if not self._session_manager:
            logger.error(
                "Session manager not available",
                extra={
                    "red_episode_id": red_episode_id,
                },
            )
            return None

        return await resolve_target_episode_id(
            session_manager=self._session_manager,
            red_episode_id=red_episode_id,
            parameters=parameters,
            context=context,
        )


# Register the executor
from ..executor_registry import register_executor

register_executor("get_target_transcript", GetTargetTranscriptExecutor, "standard")
