"""SABER Solver Factory for Inspect AI.

This module provides solver creation functionality for SABER agents,
including role-based model assignment, dynamic prompt injection,
debug logging integration, model context management, and per-iteration
transcript synchronization.
"""

from typing import Any, Callable, Optional

from inspect_ai.model import get_model
from inspect_ai.model._model import active_model, active_model_context_var
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ..constants import InspectStoreKeys
from ..integration.model_wrapper import TranscriptSyncingModelWrapper
from ..integration.transcript_sync import pull_injected_messages
from ..server.domain_manager import get_active_domain

logger = get_saber_logger(LogCategory.AGENT, __name__)


async def _pull_injections_if_enabled(
    state: TaskState,
    metadata: dict[str, Any],
    get_active_domain_func: Callable,
) -> None:
    """Helper to pull injected messages if episode context is available.

    Extracts episode context from metadata and active domain, then pulls any
    pending messages injected by the red team via the REST API.

    Args:
        state: TaskState to inject messages into
        metadata: Sample metadata containing episode/session IDs and domain slug
        get_active_domain_func: Function to get active domain context

    Returns:
        None (modifies state.messages in-place, gracefully handles missing context)
    """
    try:
        # Extract episode context from metadata
        session_id = metadata.get(MetadataKeys.SESSION_ID)
        episode_id = metadata.get(MetadataKeys.EPISODE_ID)
        domain_slug = metadata.get(MetadataKeys.SABER_DOMAIN_SLUG)

        # Validate required context
        if not all([session_id, episode_id, domain_slug]):
            logger.debug(
                "Skipping message injection pull - incomplete episode context",
                extra={
                    "has_session_id": bool(session_id),
                    "has_episode_id": bool(episode_id),
                    "has_domain_slug": bool(domain_slug),
                },
            )
            return

        # Get active domain configuration
        domain_ctx = get_active_domain_func(domain_slug)
        if not domain_ctx:
            logger.debug(
                "Skipping message injection pull - no active domain",
                extra={"domain_slug": domain_slug},
            )
            return

        rest_url = domain_ctx.get("rest_url")
        if not rest_url:
            logger.debug(
                "Skipping message injection pull - no REST URL",
                extra={"domain_slug": domain_slug},
            )
            return

        # Pull injected messages
        # Type assertion: we validated these are not None above
        await pull_injected_messages(
            state=state,
            session_id=str(session_id),
            episode_id=str(episode_id),
            rest_url=rest_url,
        )

    except Exception as e:
        logger.warning(
            "Failed to pull injected messages",
            extra={
                "error": str(e),
                "session_id": metadata.get(MetadataKeys.SESSION_ID),
                "episode_id": metadata.get(MetadataKeys.EPISODE_ID),
            },
        )
        # Don't raise - graceful degradation


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

            # Extract prompts from sample metadata
            # Note: inspect_ai creates state.metadata as a deepcopy of sample.metadata BEFORE
            # sample_init runs, so SABER context added during sample_init is NOT in state.metadata.
            # We must retrieve it from state.store where sample_init stores it.
            metadata = state.metadata or {}

            # Retrieve SABER context from state.store (set by SABERSandboxEnvironment.sample_init)
            domain_slug = state.store.get(InspectStoreKeys.DOMAIN_SLUG)
            if domain_slug:
                metadata[MetadataKeys.SABER_DOMAIN_SLUG] = domain_slug

            session_id = state.store.get(InspectStoreKeys.SESSION_ID)
            if session_id:
                metadata[MetadataKeys.SESSION_ID] = session_id

            # Episode mapping is keyed by sample_id (includes attempt suffix like "task_1__attempt_1")
            sample_id = metadata.get(MetadataKeys.SAMPLE_ID)
            if sample_id:
                episode_mapping = state.store.get(InspectStoreKeys.EPISODE_MAPPING, {})
                episode_info = episode_mapping.get(sample_id)
                if episode_info:
                    metadata[MetadataKeys.EPISODE_ID] = episode_info.episode_id
                else:
                    logger.warning(
                        f"No episode_id found for sample_id={sample_id}",
                        extra={"sample_id": sample_id, "available_keys": list(episode_mapping.keys())},
                    )

            # Update state.metadata with merged values
            state.metadata = metadata

            instruction_prompt = metadata.get(MetadataKeys.INSTRUCTION_PROMPT)
            assistant_prompt = metadata.get(MetadataKeys.ASSISTANT_PROMPT)
            submit_prompt = metadata.get(MetadataKeys.SUBMIT_PROMPT)

            # Validate prompts are present
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
                    "Missing 'submit_prompt' in sample metadata. "
                    "Ensure the SABER server is providing all three prompts."
                )

            logger.info(
                f"SABER agent '{agent_name}' starting with prompts from metadata",
                extra={
                    "agent": agent_name,
                    "task_id": metadata.get(MetadataKeys.TASK_ID),
                    "instruction_length": len(instruction_prompt),
                    "assistant_length": len(assistant_prompt),
                    "submit_length": len(submit_prompt),
                },
            )

            # Extract episode context for transcript sync and message injection
            session_id = metadata.get(MetadataKeys.SESSION_ID)
            episode_id = metadata.get(MetadataKeys.EPISODE_ID)
            domain_slug = metadata.get(MetadataKeys.SABER_DOMAIN_SLUG)

            # Get REST URL from active domain
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

            # Check for role-based model selection
            role = metadata.get(MetadataKeys.SUB_TASK_ROLE)

            # Determine which model to use (role-specific or default)
            model_to_use = None
            model_name = None

            if role and role_config:
                # Get role-specific model from configuration
                try:
                    role_agent_config = role_config.get_config_for_role(role)
                    role_model_name = role_agent_config.model

                    if role_model_name:
                        logger.info(
                            f"Using role-specific model for '{role}' role: {role_model_name}",
                            extra={
                                "role": role,
                                "model": role_model_name,
                                "task_id": metadata.get(MetadataKeys.TASK_ID),
                            },
                        )
                        model_to_use = get_model(role_model_name)
                        model_name = role_model_name

                except Exception as e:
                    logger.warning(
                        "Failed to get role-based model, using default",
                        extra={
                            "role": role,
                            "error": str(e),
                            "task_id": metadata.get(MetadataKeys.TASK_ID),
                        },
                    )

            # Use default model if no role-specific model
            if model_to_use is None:
                model_to_use = active_model()
                model_name = "default"

            # Wrap model for transcript sync BEFORE creating agent
            # This wrapper intercepts model.generate() calls and pushes transcripts
            # BEFORE tools execute, solving the race condition
            if session_id and episode_id and rest_url:
                model_to_use = TranscriptSyncingModelWrapper(
                    base_model=model_to_use,
                    session_id=session_id,
                    episode_id=episode_id,
                    rest_url=rest_url,
                )
                logger.debug(
                    "Wrapped model for transcript synchronization",
                    extra={
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "model": model_name,
                    },
                )
            else:
                logger.warning(
                    "SABER context incomplete, transcript sync disabled for this sample",
                    extra={
                        "has_session_id": bool(session_id),
                        "has_episode_id": bool(episode_id),
                        "has_rest_url": bool(rest_url),
                    },
                )

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
                    instruction_prompt=instruction_prompt,
                    assistant_prompt=assistant_prompt,
                    submit_prompt=submit_prompt,
                )

                # Pull injected messages from server before agent execution
                await _pull_injections_if_enabled(state, metadata, get_active_domain)

                # Execute agent - it will use the wrapped model internally
                # The wrapper will push transcripts after each model.generate() call
                result = await agent(state)

            finally:
                # Restore previous model
                if previous_model:
                    active_model_context_var.set(previous_model)
                else:
                    # If no previous model, clear the context
                    active_model_context_var.set(None)

            logger.info(
                f"SABER agent '{agent_name}' execution complete",
                extra={
                    "agent": agent_name,
                    "model": model_name,
                    "task_id": metadata.get(MetadataKeys.TASK_ID),
                    "completion_length": len(result.output.completion) if result.output.completion else 0,
                },
            )

            return result

        return solve

    return saber_agent_solver()
