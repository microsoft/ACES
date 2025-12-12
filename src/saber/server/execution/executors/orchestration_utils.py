"""
Shared utilities for orchestration-based executors.

This module provides reusable functions for resolving target episodes
in orchestrated multi-agent scenarios.

Logging category: ``LogCategory.TASK_EXEC``.
"""

from typing import Any, Dict, Optional

from ....logging_config import LogCategory, get_saber_logger
from ....models.constants import MetadataKeys

logger = get_saber_logger(LogCategory.TASK_EXEC, __name__)


async def resolve_target_episode_id(
    session_manager: Any,
    red_episode_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Resolve target episode ID from orchestration metadata.

    This function implements the standard target resolution pattern used by
    orchestration-aware executors (inject_prompt, get_target_transcript, etc.).

    Resolution strategy:
    1. Get red team episode from session manager
    2. Read ORCHESTRATION_TARGET_EPISODES from red team's context
    3. Return first target episode ID (assumes single target for now)

    Args:
        session_manager: Session manager for accessing episodes
        red_episode_id: Red team's episode ID
        parameters: Execution parameters (unused, for future extensibility)
        context: Execution context (unused, for future extensibility)

    Returns:
        Target episode ID or None if not resolvable

    Example:
        >>> target_id = await resolve_target_episode_id(
        ...     session_manager=self._session_manager,
        ...     red_episode_id=context["episode_id"]
        ... )
        >>> if target_id:
        ...     target_episode = session_manager.episode_manager.get_episode_by_id(target_id)
    """
    if not session_manager:
        logger.warning(
            "Cannot resolve target: session_manager is None",
            extra={
                "red_episode_id": red_episode_id,
            },
        )
        return None

    red_episode = session_manager.episode_manager.get_episode_by_id(red_episode_id)
    if not red_episode:
        logger.warning(
            "Cannot resolve target: red team episode not found",
            extra={
                "red_episode_id": red_episode_id,
            },
        )
        return None

    # DEBUG: Log all context keys for troubleshooting
    logger.info(
        "Checking orchestration metadata in episode context",
        extra={
            "red_episode_id": red_episode_id,
            "context_keys": list(red_episode.context.keys()) if red_episode.context else [],
            "has_orchestration_key": MetadataKeys.ORCHESTRATION_TARGET_EPISODES in (red_episode.context or {}),
        },
    )

    target_episodes = red_episode.context.get(MetadataKeys.ORCHESTRATION_TARGET_EPISODES)
    if not target_episodes or not isinstance(target_episodes, list) or len(target_episodes) == 0:
        logger.warning(
            "No target episodes in orchestration metadata",
            extra={
                "red_episode_id": red_episode_id,
                "has_target_episodes": target_episodes is not None,
                "is_list": isinstance(target_episodes, list) if target_episodes else False,
                "target_episodes_value": str(target_episodes),
            },
        )
        return None

    # Use first target (single blue team assumption)
    target_id: str = str(target_episodes[0])

    logger.debug(
        "Auto-resolved target from orchestration metadata",
        extra={
            "red_episode_id": red_episode_id,
            "target_episode_id": target_id,
            "total_targets": len(target_episodes),
        },
    )

    return target_id
