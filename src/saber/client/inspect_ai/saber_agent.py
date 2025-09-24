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

from typing import Any, Dict, List

from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.agent._types import AgentPrompt
from inspect_ai.model import ModelOutput
from inspect_ai.tool import Tool, mcp_server_http

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_context,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...models import EvalSubmission, HTTPHeaders, OrchestrationEnvironment
from ..agent.registry import register_saber_agent
from ..client_session import ClientSessionManager
from ..models import SABERConfig
from .agent_implementations import InspectAIImplementationNotFoundError, InspectAIImplementationRegistry

logger = get_saber_logger(LogCategory.AGENT, __name__)


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
    session_manager: ClientSessionManager | None,
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
    log_operation_start(
        logger,
        "create_saber_inspect_agent",
        agent_id=agent_id,
        implementation=implementation_name,
    )

    try:
        # Get the raw inspect_ai implementation from low-level registry
        agent_implementation = InspectAIImplementationRegistry.get_implementation(implementation_name)
    except InspectAIImplementationNotFoundError as exc:
        log_operation_failure(
            logger,
            "create_saber_inspect_agent",
            exc,
            agent_id=agent_id,
            implementation=implementation_name,
        )
        raise

    # Validate required parameters
    if session_manager is None:
        error = ValueError("Session manager is required but not provided")
        log_operation_failure(
            logger,
            "create_saber_inspect_agent",
            error,
            agent_id=agent_id,
            implementation=implementation_name,
        )
        raise error

    # Create a SABER-aware agent using inspect_ai's @agent decorator
    @agent  # type: ignore[misc]
    def saber_inspect_agent() -> Agent:
        """SABER-aware inspect_ai agent with deferred context creation."""

        async def execute(state: AgentState, tools: List[Tool]) -> AgentState:
            """Agent execution function - called for each sample when sample metadata is available."""

            from inspect_ai.solver._task_state import sample_state
            from inspect_ai.util import store

            task_store = store()
            task_store.set("saber_session_manager", session_manager)

            session_id = session_manager.get_current_session_id()
            if session_id is None:
                raise ValueError("Session ID not available from session manager")
            task_store.set("saber_session_id", session_id)

            current_state = sample_state()
            if current_state is None:
                raise ValueError("Current task state is not available")

            task_id = current_state.metadata.get("task_id")
            if task_id is None:
                raise ValueError("Task ID not found in sample metadata")
            task_store.set("saber_task_id", task_id)

            # Get all three prompts from sample metadata
            instruction_prompt = current_state.metadata.get("instruction_prompt")
            assistant_prompt = current_state.metadata.get("assistant_prompt")
            submit_prompt = current_state.metadata.get("submit_prompt")

            # Fail fast if any prompts are missing
            if instruction_prompt is None:
                raise ValueError("Instruction prompt not found in sample metadata")
            if assistant_prompt is None:
                raise ValueError("Assistant prompt not found in sample metadata")
            if submit_prompt is None:
                raise ValueError("Submit prompt not found in sample metadata")

            with log_context(session_id=session_id, agent_id=agent_id, task_id=task_id):
                logger.debug(
                    "Multi-prompt prompts retrieved from metadata",
                    extra={
                        "event": "agent_prompts_loaded",
                        "instruction_preview": instruction_prompt[:100],
                        "assistant_preview": assistant_prompt[:100],
                        "submit_preview": submit_prompt[:100],
                    },
                )

                episode_response = await session_manager.create_episode(session_id, task_id)
                task_store.set("saber_current_episode", episode_response)
                task_store.set("saber_attached_to_episode_id", episode_response.attached_to_episode_id)

                with log_context(episode_id=episode_response.episode_id):
                    logger.info(
                        "Episode created for agent execution",
                        extra={
                            "event": "agent_episode_created",
                            "episode_id": episode_response.episode_id,
                            "attached_episode_id": episode_response.attached_to_episode_id,
                        },
                    )

                    if not config.session_config:
                        raise ValueError("session_config is required for SABER agents")

                    mcp_headers = {
                        HTTPHeaders.SESSION_ID: session_id,
                        HTTPHeaders.EPISODE_ID: episode_response.episode_id,
                        HTTPHeaders.TASK_ID: task_id,
                        HTTPHeaders.ORCHESTRATION_ENV: OrchestrationEnvironment.INSPECT,
                    }

                    saber_server = mcp_server_http(
                        name="SABER Security Tools",
                        url=f"{config.session_config.mcp_server_url}/mcp",
                        headers=mcp_headers,
                    )

                    all_tools = list(tools) + [saber_server]

                    # Create the actual agent with SABER tools using multi-prompt structure
                    agent_kwargs = {
                        "name": f"SABER {agent_id.title()} Agent",
                        "prompt": AgentPrompt(
                            instructions=instruction_prompt,
                            handoff_prompt=None,
                            assistant_prompt=assistant_prompt,
                            submit_prompt=submit_prompt,
                        ),
                        "tools": all_tools,
                        **kwargs,
                    }

                    log_operation_start(
                        logger,
                        "agent_execution",
                        agent_id=agent_id,
                        implementation=implementation_name,
                        session_id=session_id,
                        task_id=task_id,
                        episode_id=episode_response.episode_id,
                    )

                    actual_agent = agent_implementation(**agent_kwargs)
                    try:
                        result: AgentState = await actual_agent(state)
                        output: ModelOutput = result.output

                        processed_model = getattr(output, "model", "unknown") or "unknown"
                        processed_choices = getattr(output, "choices", []) or []
                        processed_submission = output.completion if output.completion else "No submission provided"
                        processed_tokens = output.usage.model_dump() if output.usage else {}
                        processed_time = getattr(output, "time", 0.0) or 0.0

                        def clean_dict(payload: Any) -> Any:
                            """Remove None/False values to prevent server-side type coercion issues."""

                            if isinstance(payload, dict):
                                cleaned: Dict[str, Any] = {}
                                for key, value in payload.items():
                                    if value is None or value is False:
                                        continue
                                    cleaned[key] = clean_dict(value)
                                return cleaned
                            if isinstance(payload, list):
                                return [clean_dict(item) for item in payload if item is not None and item is not False]
                            return payload

                        cleaned_tokens: Dict[str, Any] = clean_dict(processed_tokens)

                        processed_choices_for_eval: List[Dict[str, Any]] = []
                        for choice in processed_choices:
                            if hasattr(choice, "dict"):
                                choice_dict = choice.dict()
                                cleaned_choice_dict = clean_dict(choice_dict)
                                processed_choices_for_eval.append(cleaned_choice_dict)
                            else:
                                processed_choices_for_eval.append(choice)

                        eval_submission = EvalSubmission(
                            episode_id=episode_response.episode_id,
                            task_id=task_id,
                            model=processed_model,
                            choices=processed_choices_for_eval,
                            submission=processed_submission,
                            tokens=cleaned_tokens,
                            time=processed_time,
                        )

                        try:
                            json_result = eval_submission.model_dump_json()
                            logger.debug(
                                "EvalSubmission serialized to JSON",
                                extra={
                                    "event": "eval_submission_serialized",
                                    "payload_bytes": len(json_result),
                                },
                            )
                        except Exception as json_error:
                            logger.error(
                                "EvalSubmission serialization failed",
                                extra={
                                    "event": "eval_submission_serialization_failed",
                                    "error": str(json_error),
                                    "episode_id": episode_response.episode_id,
                                },
                            )
                            logger.debug(
                                "EvalSubmission payload snapshot",
                                extra={
                                    "event": "eval_submission_serialization_payload",
                                    "payload": eval_submission.model_dump(),
                                },
                            )
                            raise

                        cascade_end = episode_response.attached_to_episode_id is not None
                        if cascade_end:
                            logger.info(
                                "Episode completion will cascade-end parent episode",
                                extra={
                                    "event": "episode_cascade_completion",
                                    "episode_id": episode_response.episode_id,
                                    "parent_episode_id": episode_response.attached_to_episode_id,
                                },
                            )

                        logger.info(
                            "Ending episode with evaluation submission",
                            extra={
                                "event": "end_episode_invocation",
                                "episode_id": episode_response.episode_id,
                                "cascade_end": cascade_end,
                            },
                        )
                        await session_manager.end_episode(
                            episode_response.session_id,
                            episode_response.episode_id,
                            reason="completed",
                            result=eval_submission,
                            cascade_end_attached_episodes=cascade_end,
                        )
                        logger.info(
                            "Episode ended successfully",
                            extra={
                                "event": "episode_completed",
                                "episode_id": episode_response.episode_id,
                                "cascade_end": cascade_end,
                            },
                        )

                        log_operation_success(
                            logger,
                            "agent_execution",
                            agent_id=agent_id,
                            implementation=implementation_name,
                            episode_id=episode_response.episode_id,
                            task_id=task_id,
                        )
                        return result

                    except Exception as exc:
                        log_operation_failure(
                            logger,
                            "agent_execution",
                            exc,
                            agent_id=agent_id,
                            implementation=implementation_name,
                            episode_id=episode_response.episode_id,
                            task_id=task_id,
                        )

                        logger.error(
                            "Agent execution failed",
                            extra={
                                "event": "agent_execution_failed",
                                "error": str(exc),
                                "exception_type": type(exc).__name__,
                                "episode_id": episode_response.episode_id,
                            },
                        )

                        error_submission = EvalSubmission(
                            episode_id=episode_response.episode_id,
                            task_id=task_id,
                            model="unknown",
                            choices=[],
                            submission=f"Episode failed: {str(exc)}",
                            tokens={},
                            time=0.0,
                        )

                        cascade_end = episode_response.attached_to_episode_id is not None
                        if cascade_end:
                            logger.info(
                                "Episode failure will cascade-end parent episode",
                                extra={
                                    "event": "episode_cascade_failure",
                                    "episode_id": episode_response.episode_id,
                                    "parent_episode_id": episode_response.attached_to_episode_id,
                                },
                            )

                        logger.info(
                            "Ending episode after failure",
                            extra={
                                "event": "end_episode_after_failure",
                                "episode_id": episode_response.episode_id,
                                "cascade_end": cascade_end,
                            },
                        )
                        await session_manager.end_episode(
                            episode_response.session_id,
                            episode_response.episode_id,
                            reason="error",
                            result=error_submission,
                            cascade_end_attached_episodes=cascade_end,
                        )
                        logger.error(
                            "Episode ended with failure",
                            extra={
                                "event": "episode_failed",
                                "episode_id": episode_response.episode_id,
                                "cascade_end": cascade_end,
                            },
                        )
                        raise

        return execute

    try:
        created_agent = saber_inspect_agent()
    except Exception as exc:
        log_operation_failure(
            logger,
            "create_saber_inspect_agent",
            exc,
            agent_id=agent_id,
            implementation=implementation_name,
        )
        raise

    log_operation_success(
        logger,
        "create_saber_inspect_agent",
        agent_id=agent_id,
        implementation=implementation_name,
    )

    return created_agent


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
