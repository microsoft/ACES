"""React agent implementation for SABER.

This is the default agent implementation based on inspect_ai's react agent.
It extracts prompts from sample metadata and uses SABER's MCP tools.
"""

from typing import Any, Callable, Optional

from inspect_ai.agent import react
from inspect_ai.agent._agent import AgentState
from inspect_ai.agent._types import AgentPrompt
from inspect_ai.model._model import active_model

from ....logging_config import LogCategory, get_saber_logger
from ...integration.tools import saber_tools

logger = get_saber_logger(LogCategory.AGENT, __name__)


async def _server_controlled_on_continue(state: AgentState) -> AgentState:
    """AgentContinue callback that waits for server to inject messages.

    This callback is used when transcript_config.websocket.pull.enabled=True.
    Instead of client-side continue prompt injection, it:
    1. Waits for server transcript modification (via WebSocket)
    2. Syncs transcript from server
    3. Returns AgentState with server-provided messages

    This allows the server's AutoContinueManager or red team to inject
    messages into the transcript without client-side interference.

    Args:
        state: Current agent state

    Returns:
        AgentState with server-synced messages
    """
    from ...integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper

    # Get the active model (should be our wrapper)
    model = active_model()

    if not isinstance(model, WebSocketTranscriptSyncingModelWrapper):
        logger.warning(
            "Server-controlled continue called but model is not WebSocket wrapper, " "falling back to no-op continue",
            extra={"model_type": type(model).__name__ if model else "None"},
        )
        # Return True-equivalent: continue loop but let generate() handle sync
        # We return the state unchanged - the next generate() will sync
        return state

    # Wait for server to inject message and sync transcript
    try:
        synced_messages = await model.wait_and_sync_transcript()

        logger.debug(
            "Server-controlled continue: synced transcript from server",
            extra={"message_count": len(synced_messages)},
        )

        # Return new AgentState with server-provided messages
        new_state = AgentState(messages=synced_messages)
        # Preserve output from current state
        if state._output is not None:
            new_state.output = state.output
        return new_state

    except Exception as e:
        logger.error(
            "Error in server-controlled continue, returning current state",
            extra={"error": str(e)},
        )
        return state


def create_agent(**kwargs: Any) -> Callable[..., Any]:
    """Create a React agent with SABER integration.

    This agent:
    - Extracts four prompts from sample metadata (instruction, assistant, submit, continue)
    - Uses saber_tools() to get MCP client from sandbox
    - Runs standard React loop with tool calling
    - Supports server-controlled continue when transcript_config.pull.enabled=True

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
        transcript_config: Optional[dict] = None,
    ) -> Any:
        """Inner factory that receives prompts from task execution.

        Args:
            instruction_prompt: The main instructions for the agent
            assistant_prompt: Assistant behavior prompt
            submit_prompt: Instructions about submission
            continue_prompt: Message shown after each step to guide the agent
            transcript_config: Optional transcript configuration dict
        """
        # Determine if server-controlled continue should be used
        # This is enabled when websocket.pull.enabled=True
        use_server_continue = False
        if transcript_config:
            ws_config = transcript_config.get("websocket", {})
            pull_config = ws_config.get("pull", {})
            use_server_continue = pull_config.get("enabled", False)

        logger.debug(
            "Creating React agent with SABER prompts",
            extra={
                "instruction_length": len(instruction_prompt),
                "assistant_length": len(assistant_prompt),
                "submit_length": len(submit_prompt),
                "continue_length": len(continue_prompt),
                "use_server_continue": use_server_continue,
            },
        )

        # Choose on_continue based on configuration
        # - Server-controlled: Use callback that syncs with server
        # - Client-controlled: Use string prompt (default Inspect AI behavior)
        on_continue: Any
        if use_server_continue:
            on_continue = _server_controlled_on_continue
            logger.info(
                "Using server-controlled continue (pull.enabled=True)",
                extra={"continue_prompt_stored_server_side": True},
            )
        else:
            on_continue = continue_prompt
            logger.debug(
                "Using client-controlled continue (pull.enabled=False or not configured)",
            )

        return react(
            prompt=AgentPrompt(
                instructions=instruction_prompt,
                handoff_prompt=None,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            ),
            tools=[saber_tools()],
            on_continue=on_continue,
            **kwargs,
        )

    return create_with_prompts


__all__ = ["create_agent"]
