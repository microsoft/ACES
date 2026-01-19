"""
Prompt injection executor for AI red team testing.

Sends injection commands via IPC to the WebSocket daemon running in the red team
container. The daemon maintains a persistent WebSocket connection to the SABER
server and forwards injections using the push_message protocol.

This executor operates through the network boundary and does not require direct
server-side memory access.

The inject_prompt executor performs a complete "inject and observe" operation:
1. Waits for target agent to be ready (WAITING_FOR_USER state)
2. Injects the adversarial prompt into the target's transcript
3. Waits for the target agent to generate a response
4. Returns the target's response as part of the result

Simplified to two strategies:
- append: Add message to end of conversation (default)
- restart: Reset to initial transcript (system->user) and inject fresh message

Logging category: ``LogCategory.TASK_EXEC``.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from .....logging_config import LogCategory, get_saber_logger
from .....models.transcript import TranscriptPushOperation
from ....base import CommandResult
from ...base import ExecutionContext, ExecutorParameters, InjectPromptParameters, Parameter, ParameterType
from ...models import ExecutorConfig
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..base_executors import normalize_context, normalize_parameters
from ..docker_executor import DockerExecutor
from ..orchestration_utils import resolve_target_episode_id

logger = get_saber_logger(LogCategory.TASK_EXEC, __name__)

# Configuration constants
DAEMON_PORT = os.getenv("WEBSOCKET_DAEMON_PORT", "9999")
TIMEOUT_BUFFER_CURL = 10
TIMEOUT_BUFFER_ENV = 15
MAX_RETRIES = 3
RETRY_ERRORS = {"connection refused", "timed out", "network unreachable"}


class InjectPromptExecutor(DockerExecutor):
    """
    Execute prompt injection attacks via IPC to WebSocket daemon.

    This executor allows red team agents to inject hidden prompts into target
    agent's conversation context through a WebSocket connection. The executor
    performs a complete "inject and observe" cycle:

    1. **Wait for Ready**: Waits for target agent to be in WAITING_FOR_USER state
    2. **Inject Prompt**: Sends the adversarial prompt to target's transcript
    3. **Wait for Response**: Blocks until target agent generates a response
    4. **Return Response**: Returns the target's response as part of the result

    This approach ensures the red team can see exactly how the blue team
    responded to each injection, enabling iterative attack refinement.

    Features:
    - **Automatic Target Resolution**: Target episode ID is automatically resolved
      from orchestration metadata (ORCHESTRATION_TARGET_EPISODES).

    - **IPC Communication**: Sends injection via HTTP POST to localhost:9999/inject_and_wait
      where the WebSocket daemon handles the full inject-and-observe cycle.

    - **Two Simple Strategies**:
       - append: Add message at end (default) - continues the conversation
       - restart: Reset to initial transcript (system->user) then inject - fresh start

    The executor inherits from DockerExecutor to leverage container execution
    infrastructure for sending curl commands to the daemon.
    """

    _executor_metadata = {
        "name": "inject_prompt",
        "description": "Inject adversarial prompts into target agent context and observe the response.",
    }

    @classmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        """Get the parameter dataclass type for this executor."""
        return InjectPromptParameters

    def __init__(
        self,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the inject_prompt executor."""
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

    @classmethod
    def get_default_config(cls) -> ExecutorConfig:
        """Get default configuration for inject_prompt executor."""
        return ExecutorConfig(timeout=300.0)

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: Any = None,
        **kwargs: Any,
    ) -> InjectPromptExecutor:
        """
        Create inject_prompt executor with standardized configuration interface.

        Args:
            sandbox_manager: Sandbox manager for executing commands in container
            config: Injection-specific configuration dictionary
            additional_params: Additional parameters for executor creation
            session_manager: Session manager for cross-episode operations
            **kwargs: Additional keyword arguments

        Returns:
            Configured InjectPromptExecutor instance
        """
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        return cls(sandbox_manager=sandbox_manager, config=config, session_manager=session_manager, **merged_kwargs)

    def setup_parameters(self, config: ExecutorConfig) -> None:
        """
        Set up executor-specific parameters.

        Args:
            config: The executor configuration
        """
        # Define message parameter
        self.add_parameter(
            Parameter(
                name="message",
                type=ParameterType.STRING,
                description="The adversarial prompt content to inject into target agent context",
                required=True,
            )
        )

        # Define strategy parameter - simplified to append/restart
        self.add_parameter(
            Parameter(
                name="strategy",
                type=ParameterType.STRING,
                description=(
                    "Injection strategy: 'append' (add to conversation, default) or "
                    "'restart' (reset to initial transcript and inject fresh)"
                ),
                required=False,
                default=TranscriptPushOperation.APPEND.value,
            )
        )

    @classmethod
    def get_parameter_schema(cls) -> dict[str, Parameter]:
        """
        Define the parameter schema for the inject_prompt executor.

        Returns:
            Dictionary mapping parameter names to Parameter definitions
        """
        return {
            "message": Parameter(
                name="message",
                type=ParameterType.STRING,
                description="The adversarial prompt content to inject into target agent context",
                required=True,
            ),
            "strategy": Parameter(
                name="strategy",
                type=ParameterType.STRING,
                description=(
                    "Injection strategy: 'append' (add to conversation, default) or "
                    "'restart' (reset to initial transcript and inject fresh)"
                ),
                required=False,
                default=TranscriptPushOperation.APPEND.value,
            ),
            "wait_for_user": Parameter(
                name="wait_for_user",
                type=ParameterType.BOOLEAN,
                description="Wait for target agent to reach WAITING_FOR_USER state before injecting (default: True)",
                required=False,
                default=True,
            ),
            "max_wait_seconds": Parameter(
                name="max_wait_seconds",
                type=ParameterType.NUMBER,
                description="Maximum seconds to wait for WAITING_FOR_USER state (default: 120)",
                required=False,
                default=120.0,
            ),
        }

    async def execute(
        self,
        parameters: InjectPromptParameters | dict[str, Any],
        context: ExecutionContext | dict[str, Any],
    ) -> CommandResult:
        """
        Execute prompt injection via WebSocket daemon in red team container.

        This performs a complete "inject and observe" cycle:
        1. Wait for target agent to be ready (WAITING_FOR_USER state)
        2. Inject the adversarial prompt into target's transcript
        3. Wait for target agent to generate a response
        4. Return the target's response along with injection metadata

        The response includes the blue team's actual reply, allowing the red
        team to iteratively refine their attack strategy.

        Args:
            parameters: Injection parameters (InjectPromptParameters or dict for backward compat)
            context: Execution context (ExecutionContext or dict for backward compatibility)

        Returns:
            CommandResult with injection status and target's response
        """
        # Normalize inputs for backward compatibility
        try:
            ctx = normalize_context(context)
        except ValueError as e:
            return CommandResult.error_result(f"Invalid context: {e}")

        try:
            p = normalize_parameters(parameters, InjectPromptParameters)
        except ValueError as e:
            return CommandResult.error_result(f"Invalid parameters: {e}")
        try:
            # Validate context
            if not ctx.session_id:
                return CommandResult.error_result("Missing session_id in execution context")

            # Get Docker environment for episode
            environment = self.get_episode_environment(ctx.episode_id)

            # Resolve target episode ID from orchestration metadata
            target_episode_id = await self._resolve_target_episode_id(ctx)

            if not target_episode_id:
                return CommandResult.error_result("Could not resolve target_episode_id from orchestration metadata")

            # Validate strategy - only append and restart are valid
            valid_strategies = [s.value for s in TranscriptPushOperation]
            if p.strategy not in valid_strategies:
                return CommandResult.error_result(
                    f"Invalid strategy: {p.strategy}. Valid: {', '.join(valid_strategies)}"
                )

            # Wait for target to be in WAITING_FOR_USER state before injecting
            if p.wait_for_user:
                logger.info(
                    "Waiting for target agent WAITING_FOR_USER state before injection",
                    extra={
                        "event": "inject_wait_for_user_start",
                        "target_episode_id": target_episode_id,
                        "max_wait_seconds": p.max_wait_seconds,
                    },
                )
                wait_result = await self._wait_for_user_with_retry(environment, target_episode_id, p.max_wait_seconds)
                if not wait_result.get("success"):
                    return CommandResult.error_result(
                        f"inject_prompt: Target not ready - {wait_result.get('error', 'wait_for_user failed')}"
                    )
                logger.info(
                    "Target agent ready for injection",
                    extra={
                        "event": "inject_wait_for_user_success",
                        "target_episode_id": target_episode_id,
                        "state": wait_result.get("state"),
                    },
                )

            # Build IPC request payload for inject_and_wait
            ipc_payload = {
                "target_episode_id": target_episode_id,
                "message": p.message,
                "strategy": p.strategy,
                "max_wait_seconds": p.max_wait_seconds,
            }

            # Send to daemon via IPC using curl - use inject_and_wait endpoint
            ipc_url = f"http://localhost:{DAEMON_PORT}/inject_and_wait"

            # Calculate timeout: max_wait + buffer for injection and network overhead
            curl_timeout = int(p.max_wait_seconds + TIMEOUT_BUFFER_CURL + 30)  # +30 for injection overhead
            env_timeout = int(p.max_wait_seconds + TIMEOUT_BUFFER_ENV + 30)

            logger.info(
                "Sending inject_and_wait request",
                extra={
                    "event": "inject_and_wait_start",
                    "target_episode_id": target_episode_id,
                    "strategy": p.strategy,
                    "max_wait_seconds": p.max_wait_seconds,
                },
            )

            # Execute HTTP POST inside container using curl
            curl_cmd = [
                "curl",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(ipc_payload),
                "--max-time",
                str(curl_timeout),
                "-s",  # Silent mode
                ipc_url,
            ]

            exec_result = await environment.execute_command(command=curl_cmd, timeout=env_timeout)

            # Check if curl command failed
            if exec_result.exit_code != 0:
                return CommandResult.error_result(f"Failed to communicate with daemon: {exec_result.stderr}")

            # Parse response
            try:
                result_data = json.loads(exec_result.stdout)
            except json.JSONDecodeError:
                return CommandResult.error_result(f"Failed to parse daemon response: {exec_result.stdout}")

            # Check success
            if result_data.get("success"):
                injection_version = result_data.get("injection_version", 0)
                final_version = result_data.get("final_version", 0)
                response_messages = result_data.get("response_messages", [])
                response_count = result_data.get("response_count", 0)
                wait_time = result_data.get("wait_time_seconds", 0)
                warning = result_data.get("warning")

                # Format the response for the red team agent
                if response_count > 0:
                    # Extract the actual response content
                    response_text = self._format_response_messages(response_messages)
                    status_msg = (
                        f"Injection successful (v{injection_version}→v{final_version}). "
                        f"Target responded with {response_count} message(s) in {wait_time:.1f}s:\n\n"
                        f"{response_text}"
                    )
                else:
                    status_msg = (
                        f"Injection sent (v{injection_version}) but no response received within {wait_time:.1f}s. "
                        f"Target may still be processing or blocked."
                    )
                    if warning:
                        status_msg += f" Warning: {warning}"

                logger.info(
                    "Injection and observation complete",
                    extra={
                        "event": "inject_and_wait_complete",
                        "target_episode_id": target_episode_id,
                        "injection_version": injection_version,
                        "final_version": final_version,
                        "response_count": response_count,
                        "wait_time_seconds": wait_time,
                    },
                )

                return CommandResult.success_result(
                    {
                        "message": status_msg,
                        "target_episode_id": target_episode_id,
                        "strategy": p.strategy,
                        "injection_version": injection_version,
                        "final_version": final_version,
                        "response_messages": response_messages,
                        "response_count": response_count,
                        "wait_time_seconds": wait_time,
                    },
                    metadata={
                        "target_episode_id": target_episode_id,
                        "injection_version": injection_version,
                        "final_version": final_version,
                        "response_count": response_count,
                        "strategy": p.strategy,
                    },
                )
            else:
                return CommandResult.error_result(f"Injection failed: {result_data.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(
                "Injection execution failed",
                extra={
                    "event": "injection_failed",
                    "episode_id": ctx.episode_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return CommandResult.error_result(f"Injection execution failed: {str(e)}")

    def _format_response_messages(self, messages: list) -> str:
        """Format response messages for display to the red team agent.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Formatted string representation of the messages
        """
        if not messages:
            return "(no messages)"

        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Format based on role
            if role == "assistant":
                formatted_parts.append(f"[BLUE TEAM RESPONSE]:\n{content}")
            elif role == "user":
                formatted_parts.append(f"[USER MESSAGE]:\n{content}")
            elif role == "system":
                # System messages are typically the injected prompts - skip or note
                formatted_parts.append(
                    f"[SYSTEM]:\n{content[:200]}..." if len(content) > 200 else f"[SYSTEM]:\n{content}"
                )
            else:
                formatted_parts.append(f"[{role.upper()}]:\n{content}")

        return "\n\n".join(formatted_parts)

    async def _wait_for_user_with_retry(
        self, environment: Any, target_episode_id: str, max_wait_seconds: float
    ) -> dict[str, Any]:
        """
        Wait for WAITING_FOR_USER state with exponential backoff retry.

        Args:
            environment: Episode environment for command execution
            target_episode_id: Target episode to wait for
            max_wait_seconds: Maximum seconds to wait

        Returns:
            Dict with success status and state information
        """
        result: dict[str, Any] = {"success": False, "error": "Max retries exceeded"}

        for attempt in range(MAX_RETRIES):
            result = await self._wait_for_user(environment, target_episode_id, max_wait_seconds)

            if result.get("success"):
                return result

            if not self._is_retryable_error(result):
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

    async def _wait_for_user(self, environment: Any, target_episode_id: str, max_wait_seconds: float) -> dict[str, Any]:
        """
        Wait for WAITING_FOR_USER state via event-driven WebSocket notification.

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
            response: dict[str, Any] = json.loads(exec_result.stdout)
            return response
        except json.JSONDecodeError:
            return {"success": False, "error": f"Invalid JSON: {exec_result.stdout}"}

    def _is_retryable_error(self, result: dict[str, Any]) -> bool:
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

    async def _resolve_target_episode_id(self, context: ExecutionContext) -> str | None:
        """
        Resolve target episode ID from orchestration metadata.

        Args:
            context: Typed execution context

        Returns:
            Target episode ID or None if not resolvable
        """
        # Convert context to dict for resolve_target_episode_id utility
        context_dict = {
            "episode_id": context.episode_id,
            "session_id": context.session_id,
            "task_id": context.task_id,
            "target_episode_ids": context.target_episode_ids,
            "role": context.role,
        }
        if context.extra:
            context_dict.update(context.extra)

        return await resolve_target_episode_id(
            session_manager=self._session_manager,
            red_episode_id=context.episode_id,
            parameters={},
            context=context_dict,
        )


# Register the executor
from ..executor_registry import register_executor  # noqa: E402

register_executor("inject_prompt", InjectPromptExecutor, "standard")
