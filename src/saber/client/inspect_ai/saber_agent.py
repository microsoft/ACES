"""
SABER-Wrapped Inspect AI Agents

Following SABER best practices:
- Composition over inheritance
- Clean factory pattern
- Fail-fast design with strict validation
- Type-safe implementation
- Clear separation of SABER infrastructure from agent logic

This module creates SABER-aware agents that wrap inspect_ai implementations
with SABER infrastructure (MCP tools, session management, episode handling).
"""

import logging
from typing import Any, List

from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import ModelOutput
from inspect_ai.tool import Tool, mcp_server_http

from ...models import EvalSubmission, HTTPHeaders, OrchestrationEnvironment
from ..agent.registry import register_saber_agent
from ..client_session import ClientSessionManager
from ..models import SABERConfig
from .agent_implementations import InspectAIImplementationNotFoundError, InspectAIImplementationRegistry

logger = logging.getLogger(__name__)


@register_saber_agent(
    name="inspect_react",
    description="React agent with native SABER MCP integration",
    implementation_type="inspect_ai",
    capabilities=["reasoning", "tool_use", "step_by_step"],
    tags=["saber", "react", "security"],
)
async def create_react_agent(
    config: SABERConfig,
    session_manager: ClientSessionManager,
    agent_id: str = "inspect_react",
    **kwargs: Any,
) -> Agent:
    """Create a React agent with SABER MCP integration.

    Args:
        config: SABER configuration
        session_manager: Session manager for SABER infrastructure
        agent_id: Agent identifier
        **kwargs: Additional agent parameters

    Returns:
        SABER-integrated inspect_ai Agent
    """
    return await create_saber_inspect_agent(
        config=config,
        session_manager=session_manager,
        agent_id=agent_id,
        implementation_name="react",
        **kwargs,
    )


async def create_saber_inspect_agent(
    config: SABERConfig,
    session_manager: ClientSessionManager,
    agent_id: str = "saber_agent",
    implementation_name: str = "react",
    **kwargs: Any,
) -> Agent:
    """
    Create a SABER-integrated inspect_ai agent.

    This is the core composition function that:
    1. Gets raw inspect_ai implementation from low-level registry
    2. Wraps it with SABER infrastructure (MCP, sessions, episodes)
    3. Returns a SABER-aware agent

    Args:
        config: SABER configuration
        session_manager: Session manager for SABER infrastructure
        agent_id: Agent identifier
        implementation_name: Name of inspect_ai implementation to use
        **kwargs: Additional agent parameters

    Returns:
        SABER-integrated Agent

    Raises:
        InspectAIImplementationNotFoundError: If implementation not found
        ValueError: If required parameters are missing
    """
    logger.info(f"Creating SABER-integrated inspect_ai agent: {agent_id} (implementation: {implementation_name})")

    # Get the raw inspect_ai implementation from low-level registry
    try:
        agent_implementation = InspectAIImplementationRegistry.get_implementation(implementation_name)
    except InspectAIImplementationNotFoundError as e:
        logger.error(f"Failed to get inspect_ai implementation '{implementation_name}': {e}")
        raise

    # Validate required parameters
    if session_manager is None:
        raise ValueError("Session manager is required but not provided")

    # Create a SABER-aware agent using inspect_ai's @agent decorator
    @agent  # type: ignore[misc]
    def saber_inspect_agent() -> Agent:
        """SABER-aware inspect_ai agent with deferred context creation."""

        async def execute(state: AgentState, tools: List[Tool]) -> AgentState:
            """Agent execution function - called for each sample when sample metadata is available."""

            # Import what we need for SABER context initialization
            from inspect_ai.solver._task_state import sample_state
            from inspect_ai.util import store

            # Initialize SABER context
            task_store = store()
            task_store.set("saber_session_manager", session_manager)

            # Get session ID
            session_id = session_manager.get_current_session_id()
            if session_id is None:
                raise ValueError("Session ID not available from session manager")
            task_store.set("saber_session_id", session_id)

            # Get task_id from current sample metadata
            current_state = sample_state()
            if current_state is None:
                raise ValueError("Current task state is not available")

            task_id = current_state.metadata.get("task_id")
            if task_id is None:
                raise ValueError("Task ID not found in sample metadata")
            task_store.set("saber_task_id", task_id)

            # Get initial prompt from sample metadata
            initial_prompt = current_state.metadata.get("initial_prompt")
            if initial_prompt is None:
                raise ValueError("Initial prompt not found in sample metadata")
            logger.info(f"Using initial prompt from metadata: {initial_prompt[:100]}...")

            # Create episode
            episode_response = await session_manager.create_episode(session_id, task_id)
            task_store.set("saber_current_episode", episode_response)

            # Log episode dependency information for agent context
            if episode_response.attached_to_episode_id:
                logger.info(
                    f"🤖 Agent episode context: {episode_response.episode_id} attached to parent episode "
                    f"{episode_response.attached_to_episode_id}"
                )
                logger.info(
                    f"🔗 This agent will operate in a dependent episode context with shared state from episode "
                    f"{episode_response.attached_to_episode_id}"
                )
                # Store attachment info for potential agent use
                task_store.set("saber_attached_to_episode_id", episode_response.attached_to_episode_id)
            else:
                logger.info(f"🤖 Agent episode context: {episode_response.episode_id} created as independent episode")
                task_store.set("saber_attached_to_episode_id", None)

            # Create MCP server connection with episode headers
            mcp_headers = {
                HTTPHeaders.SESSION_ID: session_id,
                HTTPHeaders.EPISODE_ID: episode_response.episode_id,
                HTTPHeaders.TASK_ID: task_id,
                HTTPHeaders.ORCHESTRATION_ENV: OrchestrationEnvironment.INSPECT,
            }

            saber_server = mcp_server_http(
                name="SABER Security Tools",
                url=f"{config.saber_mcp_url}/mcp",
                headers=mcp_headers,
            )

            # Combine all tools (passed tools + SABER MCP tools)
            all_tools = list(tools) + [saber_server]

            # Create the actual agent with SABER tools using the specified implementation
            actual_agent = agent_implementation(
                name=f"SABER {agent_id.title()} Agent",
                prompt=initial_prompt,
                tools=all_tools,
                **kwargs,
            )

            try:
                # Run the agent
                result: AgentState = await actual_agent(state)
                output: ModelOutput = result.output

                # Create EvalSubmission object from ModelOutput
                eval_submission = EvalSubmission(
                    episode_id=episode_response.episode_id,
                    task_id=task_id,
                    model=getattr(output, "model", "unknown"),
                    choices=[
                        choice.dict() if hasattr(choice, "dict") else choice
                        for choice in getattr(output, "choices", [])
                    ],
                    submission=output.completion if output.completion else "No submission provided",
                    tokens=output.usage.model_dump() if output.usage else {},
                    time=getattr(output, "time", 0.0),
                )

                # End episode with the EvalSubmission object
                # If this episode is attached to another episode, cascade-end the parent episode
                cascade_end = episode_response.attached_to_episode_id is not None
                if cascade_end:
                    logger.info(
                        f"🔗 Episode {episode_response.episode_id} will cascade-end parent episode "
                        f"{episode_response.attached_to_episode_id}"
                    )

                await session_manager.end_episode(
                    episode_response.session_id,
                    episode_response.episode_id,
                    reason="completed",
                    result=eval_submission,
                    cascade_end_attached_episodes=cascade_end,
                )

                logger.info(f"Successfully completed episode {episode_response.episode_id} for agent {agent_id}")
                return result

            except Exception as e:
                # Create EvalSubmission object for error case
                error_submission = EvalSubmission(
                    episode_id=episode_response.episode_id,
                    task_id=task_id,
                    model="unknown",
                    choices=[],
                    submission=f"Episode failed: {str(e)}",
                    tokens={},
                    time=0.0,
                )

                # End episode with error - still pass EvalSubmission object
                # If this episode is attached to another episode, cascade-end the parent episode
                cascade_end = episode_response.attached_to_episode_id is not None
                if cascade_end:
                    logger.info(
                        f"🔗 Episode {episode_response.episode_id} failed - will cascade-end parent episode "
                        f"{episode_response.attached_to_episode_id}"
                    )

                await session_manager.end_episode(
                    episode_response.session_id,
                    episode_response.episode_id,
                    reason="error",
                    result=error_submission,
                    cascade_end_attached_episodes=cascade_end,
                )
                logger.error(f"Episode {episode_response.episode_id} failed for agent {agent_id}: {e}")
                raise

        return execute

    # Return the SABER-aware agent
    return saber_inspect_agent()


# Register additional inspect_ai implementations as they become available
# Example:
# @register_saber_agent(
#     name="chain",
#     description="Chain agent with SABER integration",
#     implementation_type="inspect_ai",
#     capabilities=["chaining", "sequential_reasoning"],
#     tags=["saber", "chain"],
# )
# async def create_chain_agent(config, session_manager, agent_id="chain", **kwargs):
#     return await create_saber_inspect_agent(
#         config=config,
#         session_manager=session_manager,
#         agent_id=agent_id,
#         implementation_name="chain",
#         **kwargs,
#     )
