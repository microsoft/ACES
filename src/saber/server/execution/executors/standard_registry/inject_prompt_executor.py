"""
Prompt injection executor for AI red team testing.

Sends injection commands via IPC to the WebSocket daemon running in the red team
container. The daemon maintains a persistent WebSocket connection to the SABER
server and forwards injections using the push_message protocol.

This executor operates through the network boundary and does not require direct
server-side memory access.

Logging category: ``LogCategory.TASK_EXEC``.
"""

import json
from typing import Any, Dict, Optional

from .....logging_config import LogCategory, get_saber_logger
from .....models.transcript import TranscriptPushOperation
from ....base import CommandResult
from ...base import Parameter, ParameterType
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..docker_executor import DockerExecutor
from ..orchestration_utils import resolve_target_episode_id

logger = get_saber_logger(LogCategory.TASK_EXEC, __name__)


class InjectPromptExecutor(DockerExecutor):
    """
    Execute prompt injection attacks via IPC to WebSocket daemon.

    This executor allows red team agents to inject hidden prompts into target
    agent's conversation context through a WebSocket connection. Features:

    1. **Automatic Target Resolution**: Target episode ID is automatically resolved
       from orchestration metadata (ORCHESTRATION_TARGET_EPISODES).

    2. **IPC Communication**: Sends injection via HTTP POST to localhost:9999/inject
       where the WebSocket daemon forwards to SABER server.

    3. **Injection Strategies**: Multiple strategies for modifying transcript:
       - append: Add message at end (default)
       - rewind: Remove last N messages, then append
       - rewrite: Replace entire transcript
       - insert: Insert at specific position

    The executor inherits from DockerExecutor to leverage container execution
    infrastructure for sending curl commands to the daemon.
    """

    _executor_metadata = {
        "name": "inject_prompt",
        "description": "Inject adversarial prompts into target agent context via WebSocket daemon.",
    }

    def __init__(
        self,
        sandbox_manager: SandboxEnvironmentManager,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the inject_prompt executor."""
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for inject_prompt executor."""
        return {}

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: Optional[Dict[str, Any]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        session_manager: Any = None,
        **kwargs: Any,
    ) -> "InjectPromptExecutor":
        """
        Create inject_prompt executor with standardized configuration interface.

        Args:
            sandbox_manager: Sandbox manager for executing commands in container
            config: Injection-specific configuration dictionary
            additional_params: Additional parameters for executor creation
            **kwargs: Additional keyword arguments

        Returns:
            Configured InjectPromptExecutor instance
        """
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        return cls(sandbox_manager=sandbox_manager, config=config, **merged_kwargs)

    def setup_parameters(self, config: Dict[str, Any]) -> None:
        """
        Set up executor-specific parameters.

        Args:
            config: The merged configuration dictionary
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

        # Define strategy parameter
        self.add_parameter(
            Parameter(
                name="strategy",
                type=ParameterType.STRING,
                description=(
                    f"Injection strategy: {', '.join([s.value for s in TranscriptPushOperation])} " "(default: append)"
                ),
                required=False,
                default=TranscriptPushOperation.APPEND.value,
            )
        )

        # Define rewind_count parameter (for rewind strategy)
        self.add_parameter(
            Parameter(
                name="rewind_count",
                type=ParameterType.INTEGER,
                description="Number of messages to remove from end before appending (rewind strategy only)",
                required=False,
                default=1,
            )
        )

        # Define insert_position parameter (for insert strategy)
        self.add_parameter(
            Parameter(
                name="insert_position",
                type=ParameterType.INTEGER,
                description="Zero-based position to insert message at (insert strategy only)",
                required=False,
                default=0,
            )
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Parameter]:
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
                    f"Injection strategy: {', '.join([s.value for s in TranscriptPushOperation])} " "(default: append)"
                ),
                required=False,
                default=TranscriptPushOperation.APPEND.value,
            ),
            "rewind_count": Parameter(
                name="rewind_count",
                type=ParameterType.INTEGER,
                description="Number of messages to remove from end before appending (rewind strategy only)",
                required=False,
                default=1,
            ),
            "insert_position": Parameter(
                name="insert_position",
                type=ParameterType.INTEGER,
                description="Zero-based position to insert message at (insert strategy only)",
                required=False,
                default=0,
            ),
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CommandResult:
        """
        Execute prompt injection via WebSocket daemon in red team container.

        Flow:
        1. Resolve target episode ID from orchestration metadata
        2. Build IPC payload with injection parameters
        3. Send HTTP POST to daemon IPC (localhost:9999) via curl
        4. Daemon forwards to WebSocket → server processes → sends push_ack
        5. Return result to agent

        No privileged server access - operates through network boundary.

        Args:
            parameters: Injection config (message, strategy, etc.)
            context: Execution context (episode_id, session_id)

        Returns:
            CommandResult with injection status
        """
        try:
            # Extract context
            red_episode_id = context.get("episode_id")
            session_id = context.get("session_id")

            if not red_episode_id or not session_id:
                return CommandResult.error_result("Missing episode_id or session_id in execution context")

            # Get Docker environment for episode
            environment = self.get_episode_environment(red_episode_id)

            # Resolve target episode ID from orchestration metadata
            target_episode_id = await self._resolve_target_episode_id(parameters, context, red_episode_id)

            if not target_episode_id:
                return CommandResult.error_result("Could not resolve target_episode_id from orchestration metadata")

            # Get injection parameters
            message = parameters.get("message", "")
            strategy = parameters.get("strategy", TranscriptPushOperation.APPEND.value)
            rewind_count = parameters.get("rewind_count", 1)
            insert_position = parameters.get("insert_position", 0)

            if not message:
                return CommandResult.error_result("Missing required field: message")

            # Validate strategy
            valid_strategies = [s.value for s in TranscriptPushOperation]
            if strategy not in valid_strategies:
                return CommandResult.error_result(f"Invalid strategy: {strategy}. Valid: {', '.join(valid_strategies)}")

            # Build IPC request payload
            ipc_payload = {
                "target_episode_id": target_episode_id,
                "message": message,
                "strategy": strategy,
            }

            if strategy == TranscriptPushOperation.REWIND.value:
                ipc_payload["rewind_count"] = rewind_count
            elif strategy == TranscriptPushOperation.INSERT.value:
                ipc_payload["insert_position"] = insert_position

            # Send to daemon via IPC using curl
            ipc_url = "http://localhost:9999/inject"
            timeout = int(self.get_timeout())

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
                "30",
                "-s",  # Silent mode
                ipc_url,
            ]

            exec_result = await environment.execute_command(command=curl_cmd, timeout=timeout)

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
                version = result_data.get("version", 0)
                return CommandResult.success_result(
                    {
                        "message": f"Injected message (version #{version}) using '{strategy}' strategy",
                        "target_episode_id": target_episode_id,
                        "strategy": strategy,
                        "version": version,
                    },
                    metadata={
                        "target_episode_id": target_episode_id,
                        "version": version,
                        "strategy": strategy,
                    },
                )
            else:
                return CommandResult.error_result(f"Injection failed: {result_data.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(
                "Injection execution failed",
                extra={
                    "event": "injection_failed",
                    "episode_id": context.get("episode_id"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return CommandResult.error_result(f"Injection execution failed: {str(e)}")

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
        return await resolve_target_episode_id(
            session_manager=self._session_manager,
            red_episode_id=red_episode_id,
            parameters=parameters,
            context=context,
        )


# Register the executor
from ..executor_registry import register_executor  # noqa: E402

register_executor("inject_prompt", InjectPromptExecutor, "standard")
