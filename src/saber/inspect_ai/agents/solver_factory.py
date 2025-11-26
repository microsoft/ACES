"""SABER Solver Factory for Inspect AI.

This module provides solver creation functionality for SABER agents,
including role-based model assignment, dynamic prompt injection,
debug logging integration, and model context management.
"""

from typing import Any, Callable, Optional

from inspect_ai.model import get_model
from inspect_ai.model._model import active_model, active_model_context_var
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ...debug_logging import EpisodeDebugLogger
from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ..integration.transcript_sync import pull_injected_messages, push_transcript_if_enabled
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
        domain_ctx = get_active_domain_func()
        if not domain_ctx or not hasattr(domain_ctx, "rest_url"):
            logger.debug(
                "Skipping message injection pull - no active domain or REST URL",
                extra={"domain_slug": domain_slug},
            )
            return

        rest_url = domain_ctx.rest_url

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

            # DEBUG: Log solver invocation
            debug_logger = EpisodeDebugLogger("solver")
            debug_logger.info(
                "🎯 SOLVER_INVOKED: SABER solver called by Inspect AI",
                agent_name=agent_name,
                state_metadata_keys=list(state.metadata.keys()) if state.metadata else [],
                state_messages_count=len(state.messages) if state.messages else 0,
            )

            # Extract prompts from sample metadata
            metadata = state.metadata
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

            # DEBUG: Log before calling agent factory
            debug_logger.info(
                "🏭 SOLVER: Calling agent_factory to create agent",
                agent_name=agent_name,
                agent_factory_type=type(agent_factory).__name__,
            )

            # Create agent using the factory
            # agent_factory() returns a function that accepts prompts
            # Call it with prompts to get the actual agent
            create_with_prompts = agent_factory()

            # DEBUG: Log factory result
            debug_logger.info(
                "🏗️ SOLVER: agent_factory returned, calling with prompts",
                agent_name=agent_name,
                create_with_prompts_type=type(create_with_prompts).__name__,
            )

            agent = create_with_prompts(
                instruction_prompt=instruction_prompt,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            )

            # DEBUG: Log agent creation complete
            debug_logger.info(
                "✅ SOLVER: Agent created, about to execute",
                agent_name=agent_name,
                agent_type=type(agent).__name__,
            )

            # Check for role-based model selection
            role = metadata.get(MetadataKeys.SUB_TASK_ROLE)

            # DEBUG: Log role detection
            debug_logger.info(
                "🔍 ROLE CHECK",
                role=role,
                has_role_config=role_config is not None,
                metadata_keys=list(metadata.keys()) if metadata else [],
            )

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

                        # Get the model instance
                        role_model = get_model(role_model_name)

                        # Save current model to restore later
                        previous_model = active_model()

                        try:
                            # Set role-specific model as active using context var
                            active_model_context_var.set(role_model)

                            # Pull injected messages from server before agent execution
                            await _pull_injections_if_enabled(state, metadata, get_active_domain)

                            # Execute agent with role-specific model context
                            result = await agent(state)

                            # Push transcript to server after agent execution
                            await push_transcript_if_enabled(result, metadata, get_active_domain)
                        finally:
                            # Restore previous model
                            if previous_model:
                                active_model_context_var.set(previous_model)
                            else:
                                # If no previous model, clear the context
                                active_model_context_var.set(None)

                        logger.info(
                            f"SABER agent '{agent_name}' execution complete with role model",
                            extra={
                                "agent": agent_name,
                                "role": role,
                                "model": role_model_name,
                                "task_id": metadata.get(MetadataKeys.TASK_ID),
                            },
                        )
                        return result

                except Exception as e:
                    logger.warning(
                        "Failed to apply role-based model, using default",
                        extra={
                            "role": role,
                            "error": str(e),
                            "task_id": metadata.get(MetadataKeys.TASK_ID),
                        },
                    )

            # Execute the agent with default model
            logger.debug(
                f"Executing SABER agent '{agent_name}'",
                extra={"agent": agent_name, "task_id": metadata.get(MetadataKeys.TASK_ID)},
            )

            # Pull injected messages from server before agent execution
            await _pull_injections_if_enabled(state, metadata, get_active_domain)

            result = await agent(state)

            # Push transcript to server after agent execution
            await push_transcript_if_enabled(result, metadata, get_active_domain)

            logger.info(
                f"SABER agent '{agent_name}' execution complete",
                extra={
                    "agent": agent_name,
                    "task_id": metadata.get(MetadataKeys.TASK_ID),
                    "completion_length": len(result.output.completion) if result.output.completion else 0,
                },
            )

            return result

        return solve

    return saber_agent_solver()
