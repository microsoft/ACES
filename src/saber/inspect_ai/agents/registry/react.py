"""React agent implementation for SABER.

This is the default agent implementation based on inspect_ai's react agent.
It extracts prompts from sample metadata and uses SABER's MCP tools.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from inspect_ai.agent import react
from inspect_ai.agent._agent import AgentState
from inspect_ai.agent._types import AgentPrompt
from inspect_ai.model._model import active_model

from ....logging_config import LogCategory, get_saber_logger
from ...integration.tools import saber_tools

logger = get_saber_logger(LogCategory.AGENT, __name__)


async def _server_controlled_on_continue(state: AgentState) -> AgentState | bool:
    """AgentContinue callback that waits for server to inject messages.

    This callback is used when transcript_config.websocket.pull.enabled=True.
    Instead of client-side continue prompt injection, it:
    1. Waits specifically for red team to inject a user message (is_waiting_on_assistant event)
    2. Syncs transcript from server
    3. Returns AgentState with server-provided messages (if injection happened)
    4. Returns False to stop the loop (if no injection within timeout)

    This allows the server's AutoContinueManager or red team to inject
    messages into the transcript without client-side interference.

    Args:
        state: Current agent state

    Returns:
        AgentState with server-synced messages (continue), or False (stop)
    """
    from ...integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper

    logger.debug(
        "Server-controlled on_continue callback invoked",
        extra={"current_message_count": len(state.messages)},
    )

    # Get the active model (should be our wrapper)
    model = active_model()

    logger.debug(
        "Active model retrieved",
        extra={"model_type": type(model).__name__ if model else "None"},
    )

    if not isinstance(model, WebSocketTranscriptSyncingModelWrapper):
        logger.warning(
            "Server-controlled continue called but model is not WebSocket wrapper, "
            "falling back to default behavior (stop if no tool_calls)",
            extra={"model_type": type(model).__name__ if model else "None"},
        )
        # Default behavior: stop if last assistant message has no tool_calls
        if state.messages and hasattr(state.messages[-1], "tool_calls"):
            if not state.messages[-1].tool_calls:
                return False
        return state

    # Wait for red team to inject a user message (wait indefinitely)
    # The evaluation's task timeout or red team submission will terminate if needed
    try:
        logger.debug(
            "Calling model.wait_for_injection_and_sync() - waiting indefinitely",
        )
        synced_messages = await model.wait_for_injection_and_sync()

        logger.info(
            "Injection received, continuing with synced transcript",
            extra={
                "message_count": len(synced_messages),
                "last_message_role": synced_messages[-1].role if synced_messages else "none",
            },
        )

        # Return new AgentState with server-provided messages
        new_state = AgentState(messages=synced_messages)
        # Preserve output from current state
        if state._output is not None:
            new_state.output = state.output

        return new_state

    except asyncio.CancelledError:
        logger.info(
            "Cancelled (shutdown signal), stopping loop gracefully",
        )
        return False

    except Exception as e:
        logger.error(
            "Error in server-controlled continue, stopping loop",
            extra={"error": str(e)},
        )
        return False


def create_agent(**kwargs: Any) -> Callable[..., Any]:
    """Create a React agent with SABER integration.

    This agent:
    - Extracts four prompts from sample metadata (instruction, assistant, submit, continue)
    - Uses saber_tools() to get MCP client from sandbox
    - Runs standard React loop with tool calling
    - Supports server-controlled continue when transcript_config.pull.enabled=True
    - Can disable the submit tool for continuous monitoring agents

    Args:
        **kwargs: Additional parameters passed to react()

    Returns:
        React agent configured for SABER

    Usage:
        This function is called by the task factory with metadata-extracted prompts.
        The actual prompts are injected at runtime from sample metadata.
    """

    def create_with_prompts(
        instruction_prompt: str,
        assistant_prompt: str,
        submit_prompt: str,
        continue_prompt: str,
        transcript_config: dict | None = None,
        submit: bool | None = None,
    ) -> Any:
        """Inner factory that receives prompts from task execution.

        Args:
            instruction_prompt: The main instructions for the agent
            assistant_prompt: Assistant behavior prompt
            submit_prompt: Instructions about submission
            continue_prompt: Message shown after each step to guide the agent
            transcript_config: Optional transcript configuration dict
            submit: Whether to enable the submit tool. None=True (default), False=disabled.
                   Disable for continuous monitoring agents that should never submit.
        """
        # Determine if server-controlled continue should be used
        # This is enabled when websocket.pull.enabled=True
        use_server_continue = False
        if transcript_config:
            ws_config = transcript_config.get("websocket", {})
            pull_config = ws_config.get("pull", {})
            use_server_continue = pull_config.get("enabled", False)

        # Determine submit tool behavior
        # submit=None means use default (True), submit=False disables the tool
        submit_enabled = submit if submit is not None else True

        logger.debug(
            "Creating React agent with SABER prompts",
            extra={
                "instruction_length": len(instruction_prompt),
                "assistant_length": len(assistant_prompt),
                "submit_length": len(submit_prompt),
                "continue_length": len(continue_prompt),
                "use_server_continue": use_server_continue,
                "submit_enabled": submit_enabled,
            },
        )

        # Choose on_continue based on configuration
        # - Server-controlled: Use callback that syncs with server
        # - Client-controlled with submit: Use string prompt (default Inspect AI behavior)
        # - Client-controlled without submit: Use callback that always continues with prompt
        #   (inspect_ai requires callback when submit=False, otherwise agent would terminate)
        on_continue: Any
        if use_server_continue:
            on_continue = _server_controlled_on_continue
            logger.info(
                "Using server-controlled continue (pull.enabled=True)",
                extra={"continue_prompt_stored_server_side": True},
            )
        elif not submit_enabled:
            # When submit is disabled, we must use a callback function.
            # The callback should:
            # - Return True when tools were called (let model reason about output naturally)
            # - Return continue_prompt string only when model stops calling tools
            #   (this nudges it to keep monitoring instead of stopping)
            async def _no_submit_continue(state: AgentState) -> bool | str:
                """Continue callback for no-submit agents (continuous monitoring).

                Returns True if tools were called (natural continuation).
                Returns continue_prompt if no tools called (nudge to keep working).
                """
                # Check if the last assistant message had tool calls
                has_tool_calls = state.output.message.tool_calls if state.output and state.output.message else False

                if has_tool_calls:
                    # Tools were called - let the model reason naturally about results
                    logger.debug(
                        "No-submit continue: tools called, continuing naturally",
                        extra={"message_count": len(state.messages), "has_tool_calls": True},
                    )
                    return True
                else:
                    # No tools called - nudge the model to keep monitoring
                    logger.debug(
                        "No-submit continue: no tools called, sending continue prompt",
                        extra={"message_count": len(state.messages), "has_tool_calls": False},
                    )
                    return continue_prompt

            on_continue = _no_submit_continue
            logger.info(
                "Using no-submit continue callback (submit=False requires callback)",
                extra={"continue_prompt_length": len(continue_prompt)},
            )
        else:
            on_continue = continue_prompt
            logger.debug(
                "Using client-controlled continue (pull.enabled=False or not configured)",
            )

        # Log when submit is disabled (important for debugging)
        if not submit_enabled:
            logger.info(
                "Submit tool DISABLED for this agent (submit=False)",
                extra={"submit_enabled": False},
            )

        return react(
            prompt=AgentPrompt(
                instructions=instruction_prompt,
                handoff_prompt=None,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt if submit_enabled else None,
            ),
            tools=[saber_tools()],
            on_continue=on_continue,
            submit=submit_enabled,
            **kwargs,
        )

    return create_with_prompts


__all__ = ["create_agent"]
