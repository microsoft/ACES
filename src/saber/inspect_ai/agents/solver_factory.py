"""SABER Solver Factory for Inspect AI.

This module provides solver creation functionality for SABER agents,
including role-based model assignment, dynamic prompt injection,
debug logging integration, model context management, and per-iteration
transcript synchronization.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from inspect_ai.model import Model, get_model
from inspect_ai.model._model import active_model, active_model_context_var
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ...models.rest.websocket_config import WebSocketConfig
from ..constants import InspectStoreKeys
from ..integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper
from ..server.domain_manager import get_active_domain

logger = get_saber_logger(LogCategory.AGENT, __name__)


@dataclass
class SABERExecutionContext:
    """Encapsulates all SABER execution context extracted from TaskState.

    This class handles the complexity of merging metadata from two sources:
    - state.metadata: Original sample metadata (pre-sample_init)
    - state.store: Runtime context added by SABERSandboxEnvironment.sample_init()
    """

    # Required prompts
    instruction_prompt: str
    assistant_prompt: str
    submit_prompt: str

    # Episode context
    session_id: Optional[str]
    episode_id: Optional[str]
    domain_slug: Optional[str]
    rest_url: Optional[str]

    # Task metadata
    task_id: Optional[str]
    sample_id: Optional[str]

    # Role & transcript coordination
    role: Optional[str]
    transcript_config: Optional[dict[str, Any]]

    @classmethod
    def from_task_state(cls, state: TaskState) -> "SABERExecutionContext":
        """Extract SABER context from TaskState, handling metadata/store split.

        Note: Inspect AI creates state.metadata as a deepcopy of sample.metadata BEFORE
        sample_init runs, so SABER context added during sample_init is NOT in state.metadata.
        We must retrieve it from state.store where sample_init stores it.

        Args:
            state: TaskState containing metadata and store

        Returns:
            SABERExecutionContext with all required fields

        Raises:
            ValueError: If required prompts are missing
        """
        # Start with state.metadata (original sample metadata)
        metadata = state.metadata or {}

        # Merge in runtime context from state.store (added by sample_init)
        domain_slug = state.store.get(InspectStoreKeys.DOMAIN_SLUG)
        if domain_slug:
            metadata[MetadataKeys.SABER_DOMAIN_SLUG] = domain_slug

        session_id = state.store.get(InspectStoreKeys.SESSION_ID)
        if session_id:
            metadata[MetadataKeys.SESSION_ID] = session_id

        # Resolve episode_id from mapping
        sample_id = metadata.get(MetadataKeys.SAMPLE_ID)
        episode_id = None
        if sample_id:
            episode_mapping = state.store.get(InspectStoreKeys.EPISODE_MAPPING, {})
            episode_info = episode_mapping.get(sample_id)
            if episode_info:
                episode_id = episode_info.episode_id
                metadata[MetadataKeys.EPISODE_ID] = episode_id
            else:
                logger.warning(
                    f"No episode_id found for sample_id={sample_id}",
                    extra={"sample_id": sample_id, "available_keys": list(episode_mapping.keys())},
                )

        # Update state.metadata with merged values for downstream consumers
        state.metadata = metadata

        # Extract required prompts
        instruction_prompt = metadata.get(MetadataKeys.INSTRUCTION_PROMPT)
        assistant_prompt = metadata.get(MetadataKeys.ASSISTANT_PROMPT)
        submit_prompt = metadata.get(MetadataKeys.SUBMIT_PROMPT)

        # Validate required prompts
        if not instruction_prompt:
            raise ValueError(
                "Missing 'instruction_prompt' in sample metadata. "
                "Ensure the SABER server is providing all three prompts."
            )
        if not assistant_prompt:
            raise ValueError(
                "Missing 'assistant_prompt' in sample metadata. "
                "Ensure the SABER server is providing all three prompts."
            )
        if not submit_prompt:
            raise ValueError(
                "Missing 'submit_prompt' in sample metadata. " "Ensure the SABER server is providing all three prompts."
            )

        # Resolve REST URL from domain
        rest_url = None
        if domain_slug:
            domain_context = get_active_domain(domain_slug)
            if domain_context:
                rest_url = domain_context.get("rest_url")
            else:
                logger.warning(
                    f"Domain '{domain_slug}' not found in active domains registry",
                    extra={"domain_slug": domain_slug},
                )

        return cls(
            instruction_prompt=instruction_prompt,
            assistant_prompt=assistant_prompt,
            submit_prompt=submit_prompt,
            session_id=session_id,
            episode_id=episode_id,
            domain_slug=domain_slug,
            rest_url=rest_url,
            task_id=metadata.get(MetadataKeys.TASK_ID),
            sample_id=sample_id,
            role=metadata.get(MetadataKeys.SUB_TASK_ROLE),
            transcript_config=metadata.get("transcript_config"),
        )

    def has_episode_context(self) -> bool:
        """Check if complete episode context is available for transcript sync."""
        return all([self.session_id, self.episode_id, self.rest_url])

    def log_summary(self, agent_name: str) -> None:
        """Log context summary for debugging."""
        logger.info(
            f"SABER agent '{agent_name}' starting with prompts from metadata",
            extra={
                "agent": agent_name,
                "task_id": self.task_id,
                "instruction_length": len(self.instruction_prompt),
                "assistant_length": len(self.assistant_prompt),
                "submit_length": len(self.submit_prompt),
                "has_episode_context": self.has_episode_context(),
            },
        )


def _select_model(context: SABERExecutionContext, role_config: Optional[Any]) -> tuple[Model, str]:
    """Select model based on role configuration or use default.

    Args:
        context: Execution context containing role information
        role_config: Optional role-based configuration

    Returns:
        Tuple of (selected_model, model_name_for_logging)
    """
    if context.role and role_config:
        try:
            role_agent_config = role_config.get_config_for_role(context.role)
            role_model_name = role_agent_config.model

            if role_model_name:
                logger.info(
                    f"Using role-specific model for '{context.role}' role: {role_model_name}",
                    extra={
                        "role": context.role,
                        "model": role_model_name,
                        "task_id": context.task_id,
                    },
                )
                return get_model(role_model_name), role_model_name

        except Exception as e:
            logger.warning(
                "Failed to get role-based model, using default",
                extra={
                    "role": context.role,
                    "error": str(e),
                    "task_id": context.task_id,
                },
            )

    # Use default model
    return active_model(), "default"


def _wrap_model_for_transcript_sync(
    model: Model,
    model_name: str,
    context: SABERExecutionContext,
    state: TaskState,
) -> Model:
    """Wrap model with transcript synchronization if episode context available.

    Args:
        model: Base model to wrap
        model_name: Model name for logging
        context: Execution context containing episode info and blocking config
        state: TaskState for storing wrapper reference

    Returns:
        Wrapped model (or original if no episode context)
    """
    if not context.has_episode_context():
        logger.warning(
            "SABER context incomplete, transcript sync disabled for this sample",
            extra={
                "has_session_id": bool(context.session_id),
                "has_episode_id": bool(context.episode_id),
                "has_rest_url": bool(context.rest_url),
            },
        )
        return model

    # ALWAYS use WebSocket wrapper for unified coordination infrastructure
    # Configuration controls behavior (pull.blocking controls whether to wait)
    websocket_config = context.transcript_config.get("websocket", {}) if context.transcript_config else {}

    # Build WebSocketConfig from YAML
    # Handle nested push/pull configuration or flat legacy configuration
    from ...models.rest.websocket_config import PullConfig, PushConfig

    ws_config_kwargs = {}

    # Extract top-level connection settings
    for key in [
        "connection_timeout",
        "ping_interval",
        "pong_timeout",
        "reconnect_enabled",
        "max_reconnect_attempts",
        "initial_reconnect_delay",
        "max_reconnect_delay",
        "reconnect_backoff_multiplier",
    ]:
        if key in websocket_config and websocket_config[key] is not None:
            ws_config_kwargs[key] = websocket_config[key]

    # Handle push configuration (nested or flat)
    if "push" in websocket_config:
        push_dict = websocket_config["push"]
        ws_config_kwargs["push"] = PushConfig(**{k: v for k, v in push_dict.items() if v is not None})

    # Handle pull configuration (nested or flat)
    if "pull" in websocket_config:
        pull_dict = websocket_config["pull"]
        ws_config_kwargs["pull"] = PullConfig(**{k: v for k, v in pull_dict.items() if v is not None})
    elif "event_timeout" in websocket_config:
        # Legacy flat configuration - map to pull config
        ws_config_kwargs["pull"] = PullConfig(
            event_timeout=websocket_config.get("event_timeout", 300.0),
            sync_timeout=websocket_config.get("sync_timeout", 5.0),
        )

    ws_config = WebSocketConfig(**ws_config_kwargs)

    # Determine skip_first_iteration from pull.blocking (skip if not blocking)
    skip_first_iteration = not ws_config.pull.blocking if ws_config.pull else True

    wrapped_model = WebSocketTranscriptSyncingModelWrapper(
        base_model=model,
        session_id=context.session_id,  # type: ignore[arg-type]
        episode_id=context.episode_id,  # type: ignore[arg-type]
        rest_url=context.rest_url,  # type: ignore[arg-type]
        skip_first_iteration=skip_first_iteration,
        ws_config=ws_config,
    )

    logger.info(
        "Created WebSocket transcript syncing model wrapper",
        extra={
            "event": "solver_factory_websocket_wrapper_created",
            "session_id": context.session_id,
            "episode_id": context.episode_id,
            "model": model_name,
            "skip_first_iteration": skip_first_iteration,
            "pull_blocking": ws_config.pull.blocking,
            "pull_event_timeout": ws_config.pull.event_timeout,
            "push_confirmation_timeout": ws_config.push.confirmation_timeout,
        },
    )

    # Store wrapper in state for explicit cleanup
    state.store.set(InspectStoreKeys.MODEL_WRAPPER, wrapped_model)

    return wrapped_model


def create_saber_solver(agent_name: str, agent_factory: Callable, role_config: Optional[Any] = None) -> Solver:
    """Create solver with SABER agent that extracts prompts from metadata.

    This solver:
    1. Extracts the three prompts from sample metadata
    2. Uses saber_tools() to get MCP client from sandbox
    3. Creates agent with prompts and tools (using provided factory)
    4. Executes the agent

    Args:
        agent_name: Name of the agent implementation
        agent_factory: Callable that creates the agent with prompts
        role_config: Optional role-based configuration for model selection

    Returns:
        Solver that runs SABER agent with dynamic prompts
    """

    @solver  # type: ignore[misc]
    def saber_agent_solver() -> Solver:
        """SABER agent solver with dynamic prompt injection."""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            """Execute SABER agent with prompts from metadata."""

            # Extract all execution context in one clean operation
            context = SABERExecutionContext.from_task_state(state)
            context.log_summary(agent_name)

            # Select model (role-specific or default)
            model_to_use, model_name = _select_model(context, role_config)

            # Wrap model for transcript synchronization
            model_to_use = _wrap_model_for_transcript_sync(model_to_use, model_name, context, state)

            # Save current model to restore later
            previous_model = active_model()

            try:
                # Set the (possibly wrapped) model as active BEFORE creating agent
                active_model_context_var.set(model_to_use)

                # Create agent using the factory
                # agent_factory() returns a function that accepts prompts
                # Call it with prompts to get the actual agent
                create_with_prompts = agent_factory()

                agent = create_with_prompts(
                    instruction_prompt=context.instruction_prompt,
                    assistant_prompt=context.assistant_prompt,
                    submit_prompt=context.submit_prompt,
                )

                # Execute agent - it will use the wrapped model internally
                # The wrapper will push transcripts after each model.generate() call
                result = await agent(state)

            finally:
                # Restore previous model context
                active_model_context_var.set(previous_model)

            logger.info(
                f"SABER agent '{agent_name}' execution complete",
                extra={
                    "agent": agent_name,
                    "model": model_name,
                    "task_id": context.task_id,
                    "completion_length": len(result.output.completion) if result.output.completion else 0,
                },
            )

            return result

        return solve

    return saber_agent_solver()
