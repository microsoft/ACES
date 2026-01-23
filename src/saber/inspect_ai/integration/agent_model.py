"""Agent Model API for SABER.

This module provides a mock ModelAPI that allows using "agent/<agent_name>"
as a model identifier for tracking purposes. The actual model generation
is delegated to the active model in the context.

This is useful for agentic workflows where the agent (e.g., claude_code)
manages its own model calls and we just need a valid model identifier
for Inspect AI's tracking and logging.
"""

from inspect_ai.model import ModelAPI
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolChoice, ToolInfo

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


@modelapi(name="agent")  # type: ignore[misc]
def agent() -> type[ModelAPI]:
    """Register the agent model API."""
    return AgentModelAPI


class AgentModelAPI(ModelAPI):
    """A pass-through ModelAPI for agent-based workflows.

    This API is used when the agent (like claude_code) manages its own
    model interactions. The generate method is a no-op that returns
    a placeholder output - actual generation happens through the agent.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig | None = None,
    ) -> None:
        """Initialize the agent model API.

        Args:
            model_name: The agent name (e.g., "claude_code")
            base_url: Ignored (agents manage their own connections)
            api_key: Ignored (agents manage their own auth)
            config: Generation config (passed through but typically unused)
        """
        super().__init__(model_name, base_url, api_key, [], config or GenerateConfig())
        logger.info(
            "Initialized agent model API",
            extra={"agent_name": model_name},
        )

    async def generate(
        self,
        input: list,  # ChatMessage list
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Agent models do not support direct generation.

        The actual generation is handled by the agent implementation.
        This method should never be called directly.

        Raises:
            NotImplementedError: Always raised - agent models delegate
                generation to the agent implementation.
        """
        raise NotImplementedError(
            f"Agent model '{self.model_name}' does not support direct generation. "
            "Generation is delegated to the agent implementation."
        )
