"""Constants for SABER Inspect AI integration.

This module contains constants specific to the Inspect AI integration,
including store keys and other integration-specific identifiers.
"""


# Inspect AI store key constants
class InspectStoreKeys:
    """Keys used for storing SABER context in Inspect AI's task store."""

    SESSION_MANAGER = "saber_session_manager"
    SESSION_ID = "saber_session_id"
    TASK_ID = "saber_task_id"
    DOMAIN_SLUG = "saber_domain_slug"
    EPISODE_MAPPING = "saber_episode_mapping"
    EPISODE_SUBMISSION = "saber_episode_submission"


__all__ = [
    "InspectStoreKeys",
]
