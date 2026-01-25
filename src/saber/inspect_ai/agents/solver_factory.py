"""SABER Solver Factory for Inspect AI.

This module provides solver creation functionality for SABER agents,
including role-based model assignment, dynamic prompt injection,
debug logging integration, model context management, and per-iteration
transcript synchronization.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from inspect_ai.model import Model, get_model
from inspect_ai.model._model import active_model, active_model_context_var
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ...models.rest.websocket_config import WebSocketConfig
from ..constants import InspectStoreKeys
from ..integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper
from ..server.domain_manager import get_active_domain
from .registry.models import ModelPrefix

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
    continue_prompt: str

    # Episode context
    session_id: str | None
    episode_id: str | None
    domain_slug: str | None
    rest_url: str | None

    # Task metadata
    task_id: str | None
    sample_id: str | None

    # Role & transcript coordination
    role: str | None
    transcript_config: dict[str, Any] | None

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
        continue_prompt = metadata.get(MetadataKeys.CONTINUE_PROMPT)

        # Log prompt extraction for debugging
        logger.info(
            "Extracting prompts from sample metadata",
            extra={
                "event": "solver_factory_extract_prompts",
                "sample_id": sample_id,
                "has_instruction_prompt": bool(instruction_prompt),
                "has_assistant_prompt": bool(assistant_prompt),
                "has_submit_prompt": bool(submit_prompt),
                "has_continue_prompt": bool(continue_prompt),
                "metadata_keys": list(metadata.keys()),
                "continue_prompt_value": continue_prompt[:100] if continue_prompt else "<MISSING>",
            },
        )

        # Validate required prompts
        if not instruction_prompt:
            raise ValueError(
                "Missing 'instruction_prompt' in sample metadata. "
                "Ensure the SABER server is providing all four prompts."
            )
        if not assistant_prompt:
            raise ValueError(
                "Missing 'assistant_prompt' in sample metadata. Ensure the SABER server is providing all four prompts."
            )
        if not submit_prompt:
            raise ValueError(
                "Missing 'submit_prompt' in sample metadata. Ensure the SABER server is providing all four prompts."
            )
        if not continue_prompt:
            raise ValueError(
                "Missing 'continue_prompt' in sample metadata. Ensure the SABER server is providing all four prompts."
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
            continue_prompt=continue_prompt,
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


def _select_model(context: SABERExecutionContext, role_config: Any | None, agent_name: str) -> tuple[Model, str]:
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

    # Get the actual model from active_model() - this will be the model from --model flag
    # or whatever Inspect AI has configured
    model = active_model()
    if model is not None:
        actual_model_name = model.api.model_name
        # Check if this is a real model (not a fake agent model)
        if any(actual_model_name.startswith(prefix.value) for prefix in ModelPrefix):
            logger.info(
                "Using model from --model flag for provider config",
                extra={"agent": agent_name, "model_name": actual_model_name},
            )
            return model, actual_model_name
        # Also check if it's NOT an agent/ model (fallback)
        elif not actual_model_name.startswith("agent/"):
            logger.info(
                "Using non-agent model from active_model()",
                extra={"agent": agent_name, "model_name": actual_model_name},
            )
            return model, actual_model_name

    # Fall back to agent name as model identifier when no real model specified
    model_identifier = f"agent/{agent_name}"
    logger.info(
        "Using agent name as model identifier (no real model specified via --model)",
        extra={"agent": agent_name, "model_identifier": model_identifier},
    )
    return active_model(), model_identifier


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
    websocket_config = context.transcript_config.get("websocket", {}) if context.transcript_config else {}

    # Build WebSocketConfig from YAML - only enabled flags are configurable
    from ...models.rest.websocket_config import PullConfig, PushConfig

    ws_config_kwargs: dict[str, Any] = {}

    # Handle push configuration - only enabled is configurable
    if "push" in websocket_config:
        push_dict = websocket_config["push"]
        if "enabled" in push_dict:
            ws_config_kwargs["push"] = PushConfig(enabled=push_dict["enabled"])

    # Handle pull configuration - only enabled is configurable
    if "pull" in websocket_config:
        pull_dict = websocket_config["pull"]
        if "enabled" in pull_dict:
            ws_config_kwargs["pull"] = PullConfig(enabled=pull_dict["enabled"])

    ws_config = WebSocketConfig(**ws_config_kwargs)

    wrapped_model = WebSocketTranscriptSyncingModelWrapper(
        base_model=model,
        session_id=context.session_id,  # type: ignore[arg-type]
        episode_id=context.episode_id,  # type: ignore[arg-type]
        rest_url=context.rest_url,  # type: ignore[arg-type]
        ws_config=ws_config,
    )

    logger.info(
        "Created WebSocket transcript syncing model wrapper",
        extra={
            "event": "solver_factory_websocket_wrapper_created",
            "session_id": context.session_id,
            "episode_id": context.episode_id,
            "model": model_name,
            "pull_enabled": ws_config.pull.enabled,
            "push_enabled": ws_config.push.enabled,
        },
    )

    # Store wrapper in state for explicit cleanup
    state.store.set(InspectStoreKeys.MODEL_WRAPPER, wrapped_model)

    return wrapped_model


def create_saber_solver(
    agent_name: str,
    agent_factory: Callable,
    role_config: Any | None = None,
    skills_dir: str | None = None,
) -> Solver:
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
        skills_dir: Optional path to directory containing Copilot skill files

    Returns:
        Solver that runs SABER agent with dynamic prompts
    """

    @solver
    def saber_agent_solver() -> Solver:
        """SABER agent solver with dynamic prompt injection."""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            """Execute SABER agent with prompts from metadata."""

            # Extract all execution context in one clean operation
            context = SABERExecutionContext.from_task_state(state)
            context.log_summary(agent_name)

            # Select model (role-specific or default)
            model_to_use, model_name = _select_model(context, role_config, agent_name)

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

                # Get submit configuration from role config if available
                submit_enabled: bool | None = None
                if context.role and role_config:
                    try:
                        role_agent_config = role_config.get_config_for_role(context.role)
                        submit_enabled = role_agent_config.submit
                        if submit_enabled is False:
                            logger.info(
                                f"Submit tool DISABLED for role '{context.role}' via role config",
                                extra={
                                    "role": context.role,
                                    "task_id": context.task_id,
                                    "submit_enabled": False,
                                },
                            )
                    except Exception as e:
                        logger.debug(
                            f"Could not get submit config for role '{context.role}': {e}",
                            extra={"role": context.role},
                        )

                agent = create_with_prompts(
                    instruction_prompt=context.instruction_prompt,
                    assistant_prompt=context.assistant_prompt,
                    submit_prompt=context.submit_prompt,
                    continue_prompt=context.continue_prompt,
                    transcript_config=context.transcript_config,
                    submit=submit_enabled,
                    skill_directories=[p.strip() for p in skills_dir.split(",")] if skills_dir else None,
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
