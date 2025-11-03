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

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

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

# Import context injection - patches applied at import time for backward compatibility
# TODO: Migrate to scoped patching using saber_context_injection_patch() context manager
from .context_injection import saber_execute_tools, saber_tool_params

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Monkey-patch execute_tools AND tool_params for context injection
# Note: This is applied at import time for backward compatibility.
# For better test isolation and explicit scope control, use the
# saber_context_injection_patch() context manager instead.
import inspect_ai.agent._react
import inspect_ai.model._call_tools

inspect_ai.agent._react.execute_tools = saber_execute_tools
inspect_ai.model._call_tools.execute_tools = saber_execute_tools
inspect_ai.model._call_tools.tool_params = saber_tool_params

logger.info("🔧 Monkey-patched execute_tools + tool_params with SABER context injection")


async def create_mcp_client_with_retry(
    url: str,
    headers: Dict[str, str],
    name: str,
    timeout: float = 3600.0,  # 1 hour default for long-running commands
    sse_read_timeout: float = 3600.0,  # 1 hour default for long-running commands
    max_retries: int = 3,
) -> Tool:
    """
    Create MCP client with retry logic for transient failures.

    Handles transient failures like event loop starvation by retrying with
    exponential backoff before eventually failing.

    Args:
        url: MCP server URL
        headers: HTTP headers for session/episode context
        name: Display name for the MCP server
        timeout: Timeout for HTTP operations (default: 3600s = 1 hour for long commands)
        sse_read_timeout: Timeout for SSE read operations (default: 3600s = 1 hour)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Tool object from mcp_server_http()

    Raises:
        Exception: After max_retries attempts have failed
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Creating MCP client (attempt {attempt + 1}/{max_retries}) with {timeout}s timeout",
                extra={
                    "event": "mcp_client_create_attempt",
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "url": url,
                    "timeout": timeout,
                    "sse_read_timeout": sse_read_timeout,
                },
            )

            mcp_tool = mcp_server_http(
                name=name,
                url=url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=sse_read_timeout,
            )

            logger.info(
                "MCP client created successfully",
                extra={
                    "event": "mcp_client_created",
                    "attempt": attempt + 1,
                    "url": url,
                    "timeout": timeout,
                },
            )
            return mcp_tool

        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s, etc.
                wait_time = 2**attempt
                logger.warning(
                    f"MCP client creation failed, retrying in {wait_time}s",
                    extra={
                        "event": "mcp_client_create_retry",
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "wait_time": wait_time,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "MCP client creation failed after all retries",
                    extra={
                        "event": "mcp_client_create_failed",
                        "attempts": max_retries,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

    # All retries exhausted
    raise Exception(
        f"Failed to create MCP client after {max_retries} attempts. "
        f"Last error: {type(last_exception).__name__}: {str(last_exception)}"
    ) from last_exception


@asynccontextmanager
async def mcp_client_lifecycle(
    url: str,
    headers: Dict[str, str],
    name: str = "MCP Tools",
    timeout: float = 3600.0,  # 1 hour default for long-running commands
    sse_read_timeout: float = 3600.0,  # 1 hour default for long-running commands
    max_retries: int = 3,
) -> AsyncIterator[Tool]:
    """
    Async context manager for MCP client lifecycle management with retry logic.

    This wrapper ensures the MCP client from mcp_server_http() is properly cleaned up
    in the same async task context where it was created, preventing anyio cancel scope errors.

    The issue: mcp_server_http() creates an async HTTP client internally that uses anyio.create_task_group().
    When cleanup happens in a different asyncio task (e.g., during eval_async shutdown), anyio raises:
    "RuntimeError: Attempted to exit cancel scope in a different task than it was entered in"

    This wrapper forces cleanup to happen in the correct task context by:
    1. Creating the MCP client within the async context with retry logic
    2. Yielding it for use
    3. Ensuring cleanup happens in the same task via __aexit__

    Args:
        url: MCP server URL
        headers: HTTP headers for session/episode context
        name: Display name for the MCP server
        timeout: Timeout for HTTP operations (default: 3600s = 1 hour for long commands)
        sse_read_timeout: Timeout for SSE read operations (default: 3600s = 1 hour)
        max_retries: Maximum number of retry attempts (default: 3)

    Yields:
        Tool object from mcp_server_http()

    Raises:
        Exception: After max_retries connection attempts have failed
    """
    mcp_tool = None
    try:
        # Create MCP client with retry logic
        mcp_tool = await create_mcp_client_with_retry(
            url=url,
            headers=headers,
            name=name,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            max_retries=max_retries,
        )
        logger.debug(
            "MCP client created",
            extra={
                "event": "mcp_client_created",
                "url": url,
                "name": name,
            },
        )
        yield mcp_tool

    finally:
        # Explicit cleanup in same async task context
        if mcp_tool is not None:
            try:
                # Try various cleanup strategies
                if hasattr(mcp_tool, "__aexit__"):
                    # If it's an async context manager
                    await mcp_tool.__aexit__(None, None, None)
                elif hasattr(mcp_tool, "cleanup"):
                    cleanup_method = getattr(mcp_tool, "cleanup")
                    if asyncio.iscoroutinefunction(cleanup_method):
                        await cleanup_method()
                    else:
                        cleanup_method()
                elif hasattr(mcp_tool, "close"):
                    close_method = getattr(mcp_tool, "close")
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()

                # Force garbage collection to trigger async generator cleanup NOW
                # while we're still in the correct async task context
                import gc

                del mcp_tool
                gc.collect()

                logger.debug(
                    "MCP client cleanup completed",
                    extra={"event": "mcp_client_cleanup_completed"},
                )

            except Exception as cleanup_exc:
                # Log but don't raise - cleanup errors shouldn't mask actual errors
                logger.debug(
                    "MCP client cleanup error (non-fatal)",
                    extra={
                        "event": "mcp_client_cleanup_error",
                        "error": str(cleanup_exc),
                        "error_type": type(cleanup_exc).__name__,
                    },
                )


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

            # Wrap initialization in try/except to provide clear error messages
            try:
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

            except Exception as init_error:
                logger.error(
                    "Agent initialization failed before episode creation",
                    extra={
                        "event": "agent_initialization_failed",
                        "error": str(init_error),
                        "exception_type": type(init_error).__name__,
                        "phase": "pre_episode_setup",
                    },
                )
                raise ValueError(
                    f"Agent failed during initialization (before episode creation): {init_error}. "
                    "This means the agent could not set up properly to begin work."
                ) from init_error

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

                # Create episode - wrap in try/except to provide better error context
                try:
                    # Use async episode creation with wait for ready state
                    episode_response = await session_manager.create_episode_and_wait(session_id, task_id)
                    task_store.set("saber_current_episode", episode_response)
                    task_store.set("saber_attached_to_episode_id", episode_response.attached_to_episode_id)
                except TimeoutError as timeout_error:
                    logger.error(
                        "Episode creation timed out waiting for ready state",
                        extra={
                            "event": "agent_episode_creation_timeout",
                            "error": str(timeout_error),
                            "session_id": session_id,
                            "task_id": task_id,
                        },
                    )
                    raise ValueError(
                        f"Episode creation timed out for task {task_id}: {timeout_error}. "
                        "The episode may have failed during environment setup."
                    ) from timeout_error
                except Exception as episode_error:
                    logger.error(
                        "Failed to create episode for agent execution",
                        extra={
                            "event": "agent_episode_creation_failed",
                            "error": str(episode_error),
                            "exception_type": type(episode_error).__name__,
                            "session_id": session_id,
                            "task_id": task_id,
                        },
                    )
                    raise ValueError(
                        f"Episode creation failed for task {task_id}: {episode_error}. "
                        "This will prevent any tool calls from working."
                    ) from episode_error

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

                    # Use async context manager for MCP client lifecycle management
                    # This ensures cleanup happens in the same async task context,
                    # preventing anyio cancel scope errors during shutdown
                    async with mcp_client_lifecycle(
                        url=f"{config.session_config.mcp_server_url}/mcp",
                        headers=mcp_headers,
                        name="SABER Security Tools",
                        timeout=config.session_config.mcp_timeout,
                        sse_read_timeout=config.session_config.mcp_sse_read_timeout,
                        max_retries=config.session_config.mcp_max_retries,
                    ) as saber_server:
                        # Context injection happens automatically via monkey-patched execute_tools
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

                        # Execute agent - MCP client cleanup will happen automatically
                        # when we exit the async with block above
                        try:
                            result: AgentState = await actual_agent(state)
                            output: ModelOutput = result.output

                            processed_model = getattr(output, "model", "unknown") or "unknown"
                            processed_choices = getattr(output, "choices", []) or []
                            processed_submission = output.completion if output.completion else "No submission provided"
                            processed_tokens = output.usage.model_dump() if output.usage else {}
                            processed_time = getattr(output, "time", 0.0) or 0.0

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

                            # Success path - end episode and return result
                            cascade_end = episode_response.attached_to_episode_id is not None

                            logger.info(
                                "Agent execution completed successfully - ending episode",
                                extra={
                                    "event": "agent_execution_success",
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

                            log_operation_success(
                                logger,
                                "agent_execution",
                                agent_id=agent_id,
                                implementation=implementation_name,
                                episode_id=episode_response.episode_id,
                                task_id=task_id,
                            )

                            # Return the result from successful agent execution
                            return result

                        except Exception as exc:
                            # Unified exception handler for agent execution failures
                            error_type = type(exc).__name__
                            error_message = str(exc)

                            # Check if this is a tool call limit error - treat it as completion, not failure
                            is_tool_call_limit = (
                                "tool call limit" in error_message.lower()
                                or "exhausted available tool calls" in error_message.lower()
                                or "LimitExceededError" in error_type
                            )

                            # Check if it's a timeout error
                            is_timeout = "timeout" in error_type.lower() or "timeout" in error_message.lower()

                            if is_tool_call_limit:
                                # Tool call limit is a normal completion condition, not an error
                                logger.info(
                                    "Agent reached tool call limit - ending episode normally",
                                    extra={
                                        "event": "agent_tool_call_limit_reached",
                                        "error": error_message,
                                        "exception_type": error_type,
                                        "episode_id": episode_response.episode_id,
                                    },
                                )

                                # Create submission indicating limit reached
                                eval_submission = EvalSubmission(
                                    episode_id=episode_response.episode_id,
                                    task_id=task_id,
                                    model="unknown",
                                    choices=[],
                                    submission="Agent reached tool call limit",
                                    tokens={},
                                    time=0.0,
                                )

                                reason = "completed"
                                completion_reason = "tool_call_limit"

                            elif is_timeout:
                                # Timeout errors
                                logger.error(
                                    "Agent execution timed out - command may have completed on "
                                    "server but client timed out waiting for response",
                                    extra={
                                        "event": "agent_execution_timeout",
                                        "error": error_message,
                                        "error_type": error_type,
                                        "session_id": session_id,
                                        "task_id": task_id,
                                        "episode_id": episode_response.episode_id,
                                        "timeout_hint": "Consider breaking long commands into "
                                        "smaller chunks or increasing mcp_timeout",
                                    },
                                )

                                eval_submission = EvalSubmission(
                                    episode_id=episode_response.episode_id,
                                    task_id=task_id,
                                    model="unknown",
                                    choices=[],
                                    submission=f"Agent execution timed out: {error_message}",
                                    tokens={},
                                    time=0.0,
                                )

                                reason = "error"
                                completion_reason = "timeout"

                            else:
                                # General agent execution errors
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
                                        "error": error_message,
                                        "error_type": error_type,
                                        "session_id": session_id,
                                        "task_id": task_id,
                                        "episode_id": episode_response.episode_id,
                                    },
                                )

                                eval_submission = EvalSubmission(
                                    episode_id=episode_response.episode_id,
                                    task_id=task_id,
                                    model="unknown",
                                    choices=[],
                                    submission=f"Agent execution failed: {error_type}: {error_message}",
                                    tokens={},
                                    time=0.0,
                                )

                                reason = "error"
                                completion_reason = "execution_error"

                            # Common handling for all exception types
                            logger.info(
                                "Created failure/completion submission",
                                extra={
                                    "event": "graceful_failure_submission",
                                    "episode_id": episode_response.episode_id,
                                    "error_type": error_type,
                                    "reason": reason,
                                },
                            )

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
                                    "reason": reason,
                                },
                            )

                            # End episode with submission - wrap in try/except for graceful error handling
                            try:
                                await session_manager.end_episode(
                                    episode_response.session_id,
                                    episode_response.episode_id,
                                    reason=reason,
                                    result=eval_submission,
                                    cascade_end_attached_episodes=cascade_end,
                                )
                                logger.info(
                                    "Episode ended after agent execution error",
                                    extra={
                                        "event": "episode_completed",
                                        "episode_id": episode_response.episode_id,
                                        "cascade_end": cascade_end,
                                        "reason": reason,
                                    },
                                )

                                log_operation_success(
                                    logger,
                                    "agent_execution",
                                    agent_id=agent_id,
                                    implementation=implementation_name,
                                    episode_id=episode_response.episode_id,
                                    task_id=task_id,
                                    completion_reason=completion_reason,
                                )

                            except Exception as cleanup_error:
                                # Gracefully handle episode cleanup errors
                                logger.error(
                                    "Episode cleanup failed after agent execution",
                                    extra={
                                        "event": "episode_cleanup_failed",
                                        "error": str(cleanup_error),
                                        "error_type": type(cleanup_error).__name__,
                                        "session_id": session_id,
                                        "episode_id": episode_response.episode_id,
                                    },
                                )
                                # Don't re-raise - we want to return the state even if cleanup fails
                                logger.warning(
                                    "Continuing despite cleanup error - state was captured",
                                    extra={
                                        "event": "episode_cleanup_error_ignored",
                                        "episode_id": episode_response.episode_id,
                                    },
                                )

                            # Return state since agent execution failed before result was created
                            # For tool call limit, this allows the scorer to still run
                            return state

                    # MCP client cleanup happens automatically when exiting async with block

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
