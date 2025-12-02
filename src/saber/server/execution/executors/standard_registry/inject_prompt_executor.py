"""
Prompt injection executor for AI red team testing.

Allows red team to inject adversarial prompts into blue team's conversation
context by directly modifying the blue team's CLIENT_TRANSCRIPT.

This executor provides the key mechanism for testing AI safety guardrails
through hidden prompt injection attacks in the blocking transcript solver architecture.

Logging category: ``LogCategory.TASK_EXEC``.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .....logging_config import LogCategory, get_saber_logger
from .....models.constants import MetadataKeys
from ....base import CommandResult
from ...base import Parameter, ParameterType
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..base_executors import CommandExecutor

logger = get_saber_logger(LogCategory.TASK_EXEC, __name__)


class InjectionStrategy(str, Enum):
    """Strategies for injecting messages into target agent's transcript."""

    APPEND = "append"  # Add message at end (default)
    REWIND = "rewind"  # Remove last N messages, then append
    REWRITE = "rewrite"  # Replace entire transcript
    INSERT = "insert"  # Insert at specific position


class InjectPromptExecutor(CommandExecutor):
    """
    Execute prompt injection attacks by directly modifying target agent's transcript.

    This executor allows red team agents to inject hidden prompts into target
    agent's conversation context. Features:

    1. **Automatic Target Resolution**: Target episode ID is automatically resolved
       from orchestration metadata (ORCHESTRATION_TARGET_EPISODES).

    2. **Injection Strategies**: Multiple strategies for modifying transcript:
       - append: Add message at end (default)
       - rewind: Remove last N messages, then append
       - rewrite: Replace entire transcript
       - insert: Insert at specific position

    3. **Timestamp Signaling**: Sets TRANSCRIPT_LAST_MODIFIED_AT to signal
       target agent that transcript has been modified.

    4. **Audit Trail**: Increments TRANSCRIPT_MODIFICATION_COUNT for tracking.

    The blocking target agent solver polls for timestamp changes to detect modifications.
    """

    _executor_metadata = {
        "name": "inject_prompt",
        "description": "Inject adversarial prompts into target agent context for safety testing.",
    }

    def __init__(
        self,
        sandbox_manager: Optional[SandboxEnvironmentManager] = None,
        config: Optional[Dict[str, Any]] = None,
        session_manager: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize the inject_prompt executor."""
        super().__init__(config=config, session_manager=session_manager, **kwargs)

        self.sandbox_manager = sandbox_manager

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
        session_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> "InjectPromptExecutor":
        """
        Create inject_prompt executor with standardized configuration interface.

        Args:
            sandbox_manager: Sandbox manager (not used but required by interface)
            config: Injection-specific configuration dictionary
            additional_params: Additional parameters for executor creation
            session_manager: Optional session manager for cross-episode operations
            **kwargs: Additional keyword arguments

        Returns:
            Configured InjectPromptExecutor instance
        """
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        return cls(sandbox_manager=sandbox_manager, config=config, session_manager=session_manager, **merged_kwargs)

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
                description=f"Injection strategy: {', '.join([s.value for s in InjectionStrategy])} (default: append)",
                required=False,
                default=InjectionStrategy.APPEND.value,
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
                description=f"Injection strategy: {', '.join([s.value for s in InjectionStrategy])} (default: append)",
                required=False,
                default=InjectionStrategy.APPEND.value,
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
        Inject adversarial prompt into target agent's CLIENT_TRANSCRIPT.

        This executor:
        1. Auto-resolves target_episode_id from orchestration metadata
        2. Retrieves target episode via session_manager.episode_manager
        3. Applies injection strategy (append/rewind/rewrite/insert)
        4. Sets TRANSCRIPT_LAST_MODIFIED_AT timestamp atomically
        5. Increments TRANSCRIPT_MODIFICATION_COUNT

        Args:
            parameters: Injection configuration with 'message', 'strategy', etc.
            context: Execution context with episode_id and session_id

        Returns:
            CommandResult indicating success/failure of injection
        """
        try:
            # Extract context
            red_episode_id = context.get("episode_id")
            session_id = context.get("session_id")

            if not red_episode_id or not session_id:
                return CommandResult.error_result("Missing episode_id or session_id in execution context")

            # Phase 2c: Automatic target resolution from orchestration metadata
            target_episode_id = await self._resolve_target_episode_id(parameters, context, red_episode_id)

            if not target_episode_id:
                return CommandResult.error_result(
                    "Could not resolve target_episode_id from orchestration metadata. "
                    "Episode must have ORCHESTRATION_TARGET_EPISODES in context."
                )

            # Get injection parameters
            message = parameters.get("message", "")
            strategy = parameters.get("strategy", InjectionStrategy.APPEND.value)
            rewind_count = parameters.get("rewind_count", 1)
            insert_position = parameters.get("insert_position", 0)

            if not message:
                return CommandResult.error_result("Missing required field: message")

            # Validate strategy
            valid_strategies = [s.value for s in InjectionStrategy]
            if strategy not in valid_strategies:
                return CommandResult.error_result(
                    f"Invalid strategy: {strategy}. " f"Valid strategies: {', '.join(valid_strategies)}"
                )

            # Get target episode via session manager
            if not self._session_manager:
                return CommandResult.error_result("Session manager not available (executor not properly initialized)")

            target_episode = self._session_manager.episode_manager.get_episode_by_id(target_episode_id)
            if not target_episode:
                return CommandResult.error_result(f"Target episode {target_episode_id} not found")

            # Create injection message
            injected_message = {"role": "system", "content": message, "source": "red_team_injection"}

            # Get current target transcript
            target_transcript = target_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])

            # Apply injection strategy
            modified_transcript = self._apply_injection_strategy(
                target_transcript, injected_message, strategy, rewind_count, insert_position
            )

            # Increment modification counter
            modification_count = target_episode.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0)
            modification_count += 1

            # Update target episode atomically
            # This sets: transcript, timestamp, and counter in one atomic operation
            context_updates = {
                MetadataKeys.CLIENT_TRANSCRIPT: modified_transcript,
                MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT: datetime.utcnow().isoformat(),
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: modification_count,
            }
            await target_episode.update_context_atomic(context_updates)

            logger.info(
                "Red team modified transcript and set timestamp",
                extra={
                    "event": "injection_created",
                    "red_episode_id": red_episode_id,
                    "target_episode_id": target_episode_id,
                    "modification_count": modification_count,
                    "message_length": len(message),
                    "strategy": strategy,
                },
            )

            return CommandResult.success_result(
                {
                    "message": f"Injected message (modification #{modification_count}) using '{strategy}' strategy",
                    "target_episode_id": target_episode_id,
                    "strategy": strategy,
                    "modification_count": modification_count,
                },
                metadata={
                    "target_episode_id": target_episode_id,
                    "modification_count": modification_count,
                    "strategy": strategy,
                },
            )

        except Exception as e:
            logger.error(
                "Injection failed",
                extra={
                    "event": "injection_failed",
                    "episode_id": context.get("episode_id"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return CommandResult.error_result(f"Injection failed: {str(e)}")

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
        # Auto-resolve from orchestration metadata
        if self._session_manager:
            red_episode = self._session_manager.episode_manager.get_episode_by_id(red_episode_id)
            if red_episode:
                target_episodes = red_episode.context.get(MetadataKeys.ORCHESTRATION_TARGET_EPISODES)
                if target_episodes and isinstance(target_episodes, list) and len(target_episodes) > 0:
                    target_id: str = str(target_episodes[0])  # Use first target
                    logger.debug(
                        "Auto-resolved target from orchestration metadata",
                        extra={
                            "red_episode_id": red_episode_id,
                            "target_episode_id": target_id,
                            "total_targets": len(target_episodes),
                        },
                    )
                    return target_id

        return None

    def _apply_injection_strategy(
        self,
        target_transcript: List[Dict[str, Any]],
        injected_message: Dict[str, Any],
        strategy: str,
        rewind_count: int,
        insert_position: int,
    ) -> List[Dict[str, Any]]:
        """
        Apply injection strategy to modify target transcript.

        Strategies:
        - append: Add message at end (default)
        - rewind: Remove last N messages, then append
        - rewrite: Replace entire transcript with injected message
        - insert: Insert message at specific position

        Args:
            target_transcript: Current target transcript
            injected_message: Message to inject
            strategy: Injection strategy (InjectionStrategy value)
            rewind_count: Number of messages to remove (rewind only)
            insert_position: Position to insert at (insert only)

        Returns:
            Modified transcript
        """
        if strategy == InjectionStrategy.APPEND.value:
            # Simple append at end
            return target_transcript + [injected_message]

        elif strategy == InjectionStrategy.REWIND.value:
            # Remove last N messages, then append
            rewind_count = max(0, rewind_count)  # Ensure non-negative
            rewind_count = min(rewind_count, len(target_transcript))  # Don't rewind more than exists
            rewound_transcript = target_transcript[:-rewind_count] if rewind_count > 0 else target_transcript
            return rewound_transcript + [injected_message]

        elif strategy == InjectionStrategy.REWRITE.value:
            # Replace entire transcript
            return [injected_message]

        elif strategy == InjectionStrategy.INSERT.value:
            # Insert at specific position
            insert_position = max(0, insert_position)  # Ensure non-negative
            insert_position = min(insert_position, len(target_transcript))  # Clamp to valid range
            return target_transcript[:insert_position] + [injected_message] + target_transcript[insert_position:]

        else:
            # Fallback to append (should never reach here due to validation)
            return target_transcript + [injected_message]


# Register the executor
from ..executor_registry import register_executor  # noqa: E402

register_executor("inject_prompt", InjectPromptExecutor, "standard")
