"""Step backfill service for correlating transcript messages with episode steps.

This module provides functionality to backfill episode steps with assistant context
(message and reasoning) from a synced transcript. The correlation is done using
tool_call_id as the primary key.

Algorithm:
1. Build index: tool_call_id → AssistantContext from assistant messages
2. Extract tool responses from transcript (in execution order)
3. Match step[i] to tool_response[i] by position
4. Extract tool_call_id from tool_response, look up context in index
5. Update step.action with assistant_message and reasoning

Logging category: EPISODE
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ..base import Episode

logger = get_saber_logger(LogCategory.EPISODE, __name__)


class AssistantContext(BaseModel):
    """Context extracted from an assistant message for step backfill.

    This model captures the assistant's message content and optional reasoning
    that precedes a tool call. It is used to correlate transcript messages
    with execution steps.
    """

    model_config = ConfigDict(frozen=True)

    assistant_message: str | None = Field(None, description="The assistant's full message content")
    reasoning: str | None = Field(None, description="The assistant's reasoning/thinking")


class BackfillResult(BaseModel):
    """Result of a step backfill operation.

    Provides detailed statistics about the backfill operation including
    how many steps were updated, skipped, or already filled.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["success", "no_transcript", "no_steps", "error"] = Field(
        ..., description="Overall status of the backfill operation"
    )
    steps_updated: int = Field(default=0, description="Number of steps that were updated with context")
    steps_skipped: int = Field(default=0, description="Number of steps skipped due to missing correlation data")
    steps_already_filled: int = Field(default=0, description="Number of steps already having context (idempotency)")
    total_steps: int = Field(default=0, description="Total number of steps in the episode")
    error_message: str | None = Field(default=None, description="Error message if status is 'error'")


# Type alias for transcript messages
TranscriptMessage = dict[str, Any]


def build_tool_call_index(
    transcript: list[TranscriptMessage],
) -> dict[str, AssistantContext]:
    """Build an index mapping tool_call_id to the assistant context that initiated it.

    Handles:
    - Single tool calls: 1 assistant message → 1 tool_call
    - Parallel tool calls: 1 assistant message → N tool_calls (same context for all)
    - Multi-turn: Multiple assistant messages with their own tool_calls

    Args:
        transcript: List of transcript messages from the episode context.

    Returns:
        Dictionary mapping tool_call_id to AssistantContext.
    """
    index: dict[str, AssistantContext] = {}

    for msg in transcript:
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue

        # Create context from this assistant message
        context = AssistantContext(
            assistant_message=msg.get("content"),
            reasoning=msg.get("reasoning"),
        )

        # Map each tool_call_id to this context
        for tc in tool_calls:
            tc_id = tc.get("id")
            if tc_id:
                index[tc_id] = context
                logger.debug(
                    "Indexed tool_call_id",
                    extra={"tool_call_id": tc_id, "has_message": context.assistant_message is not None},
                )

    logger.info(
        "Built tool call index",
        extra={"index_size": len(index)},
    )
    return index


def extract_tool_responses(transcript: list[TranscriptMessage]) -> list[TranscriptMessage]:
    """Extract tool role messages from transcript in execution order.

    Args:
        transcript: List of transcript messages from the episode context.

    Returns:
        List of tool response messages in the order they appear.
    """
    responses = [msg for msg in transcript if msg.get("role") == "tool"]
    logger.debug(
        "Extracted tool responses",
        extra={"count": len(responses)},
    )
    return responses


class StepBackfillService:
    """Service for backfilling episode steps with assistant context from transcript.

    This service correlates transcript messages with existing episode steps,
    updating each step's Action.assistant_message and Action.reasoning
    from the synced transcript.

    Usage:
        service = StepBackfillService()
        result = service.backfill_steps(episode)
    """

    def backfill_steps(self, episode: Episode) -> BackfillResult:
        """Correlate transcript messages with steps and update step context.

        Algorithm:
        1. Extract transcript from episode.context
        2. Build index: tool_call_id → AssistantContext (from assistant messages)
        3. Extract tool responses from transcript (in execution order)
        4. For each step (by index):
           a. Get corresponding tool_response by index (step N → tool_response N)
           b. Extract tool_call_id from tool_response
           c. Look up AssistantContext in index by tool_call_id
           d. Update step.action with assistant_message and reasoning (if not set)
        5. Return statistics

        Args:
            episode: The episode containing steps to backfill.

        Returns:
            BackfillResult with statistics about the operation.
        """
        # Get transcript from episode context
        transcript_key = MetadataKeys.CLIENT_TRANSCRIPT.value
        transcript: list[TranscriptMessage] = episode.context.get(transcript_key, [])

        if not transcript:
            logger.info(
                "No transcript available for backfill",
                extra={"episode_id": episode.episode_id},
            )
            return BackfillResult(status="no_transcript")

        if not episode.steps:
            logger.info(
                "No steps to backfill",
                extra={"episode_id": episode.episode_id},
            )
            return BackfillResult(status="no_steps", total_steps=0)

        # Build tool_call_id → AssistantContext index
        index = build_tool_call_index(transcript)

        # Extract tool responses in order
        tool_responses = extract_tool_responses(transcript)

        # Correlate steps with tool responses by position
        steps_updated = 0
        steps_skipped = 0
        steps_already_filled = 0

        for i, step in enumerate(episode.steps):
            # Check if step is already filled
            if step.action.assistant_message is not None or step.action.reasoning is not None:
                steps_already_filled += 1
                logger.debug(
                    "Step already has context",
                    extra={"step_number": step.step_number, "episode_id": episode.episode_id},
                )
                continue

            # Match step to tool_response by index (step N → tool_response N)
            if i >= len(tool_responses):
                steps_skipped += 1
                logger.debug(
                    "No tool response for step",
                    extra={"step_index": i, "tool_response_count": len(tool_responses)},
                )
                continue

            tool_response = tool_responses[i]
            tool_call_id = tool_response.get("tool_call_id")

            if not tool_call_id:
                steps_skipped += 1
                logger.debug(
                    "Tool response missing tool_call_id",
                    extra={"step_index": i},
                )
                continue

            if tool_call_id not in index:
                steps_skipped += 1
                logger.debug(
                    "Tool call ID not found in index",
                    extra={"tool_call_id": tool_call_id, "step_index": i},
                )
                continue

            context = index[tool_call_id]

            # Update step action with context
            step.action.assistant_message = context.assistant_message
            step.action.reasoning = context.reasoning
            steps_updated += 1

            logger.debug(
                "Updated step with context",
                extra={
                    "step_number": step.step_number,
                    "tool_call_id": tool_call_id,
                    "has_message": context.assistant_message is not None,
                    "has_reasoning": context.reasoning is not None,
                },
            )

        logger.info(
            "Backfill completed",
            extra={
                "episode_id": episode.episode_id,
                "steps_updated": steps_updated,
                "steps_skipped": steps_skipped,
                "steps_already_filled": steps_already_filled,
                "total_steps": len(episode.steps),
            },
        )

        return BackfillResult(
            status="success",
            steps_updated=steps_updated,
            steps_skipped=steps_skipped,
            steps_already_filled=steps_already_filled,
            total_steps=len(episode.steps),
        )


__all__ = [
    "AssistantContext",
    "BackfillResult",
    "StepBackfillService",
    "build_tool_call_index",
    "extract_tool_responses",
]
