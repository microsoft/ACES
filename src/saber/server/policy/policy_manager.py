"""
Stub PolicyManager implementation for SABER domain server.

Logging Category: POLICY

This is a minimal implementation to support SessionManager development.
"""

from typing import Any

from pydantic import BaseModel, Field

from saber.logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.POLICY, __name__)


class PolicyDocument(BaseModel):
    """Represents a domain policy document containing the initial agent prompt."""

    prompt: str = Field(..., description="Initial prompt for agents")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
        }


class PolicyManager:
    """PolicyManager stores episode-specific policy documents.

    All prompt generation must be performed upstream (BenchmarkManager + PromptGenerator).
    No legacy fallback or prompt construction exists here by design (fail-fast principle).
    """

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self._episode_policies: dict[str, PolicyDocument] = {}
        logger.info(
            "Policy manager initialized",
            extra={
                "event": "policy_manager_initialized",
                "domain": domain_name,
            },
        )

    def set_episode_policy(
        self,
        episode_id: str,
        session_id: str,
        pre_generated_prompt: str,
    ) -> None:
        """Register a pre-generated prompt for an episode.

        Args:
            episode_id: Episode identifier
            session_id: Session identifier
            pre_generated_prompt: Rendered prompt text (must be non-empty)
        Raises:
            ValueError: If prompt is empty or only whitespace
        """
        if not pre_generated_prompt or not pre_generated_prompt.strip():
            raise ValueError(
                f"Non-empty pre_generated_prompt required for episode {episode_id} (received empty/whitespace)."
            )
        self._episode_policies[episode_id] = PolicyDocument(prompt=pre_generated_prompt)
        logger.info(
            "Episode policy stored",
            extra={
                "event": "episode_policy_stored",
                "episode_id": episode_id,
                "session_id": session_id,
                "prompt_length": len(pre_generated_prompt),
            },
        )

    def get_policy(self, episode_id: str | None = None) -> PolicyDocument:
        """
        Get domain policy document for a specific episode.

        Args:
            episode_id: Episode identifier to get policy for

        Returns:
            PolicyDocument containing the prompt for agents with task-specific timeout
        """
        if episode_id is None:
            raise ValueError("episode_id is required for policy retrieval")

        if episode_id not in self._episode_policies:
            raise ValueError(f"No policy configured for episode {episode_id}")

        return self._episode_policies[episode_id]

    def cleanup_episode_policy(self, episode_id: str) -> None:
        """
        Remove policy for a completed episode.

        Args:
            episode_id: Episode identifier to clean up
        """
        if episode_id in self._episode_policies:
            del self._episode_policies[episode_id]
            logger.debug(
                "Episode policy removed",
                extra={
                    "event": "episode_policy_removed",
                    "episode_id": episode_id,
                },
            )

    async def validate_action(self, action: str) -> bool:
        """
        Validate if an action complies with domain policies.

        Args:
            action: Action to validate

        Returns:
            True if action is valid (stub always returns True)
        """
        return True
