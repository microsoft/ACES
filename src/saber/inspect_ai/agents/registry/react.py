"""React agent implementation for SABER.

This is the default agent implementation based on inspect_ai's react agent.
It extracts prompts from sample metadata and uses SABER's MCP tools.
"""

from typing import Any, Callable

from inspect_ai.agent import react
from inspect_ai.agent._types import AgentPrompt

from ....logging_config import LogCategory, get_saber_logger
from ...integration.tools import saber_tools

logger = get_saber_logger(LogCategory.AGENT, __name__)

MODIFIED_CONTINUE_PROMPT = """
Please proceed to the next step using your best judgement. If you believe you
have completed the task, please call the `{submit}()` tool with your final answer
or call it with no arguments to indicate that you have given up.
"""


def create_agent(**kwargs: Any) -> Callable[..., Any]:
    """Create a React agent with SABER integration.

    This agent:
    - Extracts three prompts from sample metadata (instruction, assistant, submit)
    - Uses saber_tools() to get MCP client from sandbox
    - Runs standard React loop with tool calling

    Args:
        **kwargs: Additional parameters passed to react()

    Returns:
        React agent configured for SABER

    Usage:
        This function is called by the task factory with metadata-extracted prompts.
        The actual prompts are injected at runtime from sample metadata.
    """

    def create_with_prompts(instruction_prompt: str, assistant_prompt: str, submit_prompt: str) -> Any:
        """Inner factory that receives prompts from task execution."""
        logger.debug(
            "Creating React agent with SABER prompts",
            extra={
                "instruction_length": len(instruction_prompt),
                "assistant_length": len(assistant_prompt),
                "submit_length": len(submit_prompt),
            },
        )

        return react(
            prompt=AgentPrompt(
                instructions=instruction_prompt,
                handoff_prompt=None,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
            ),
            tools=[saber_tools()],
            on_continue=MODIFIED_CONTINUE_PROMPT,
            **kwargs,
        )

    return create_with_prompts


__all__ = ["create_agent"]
