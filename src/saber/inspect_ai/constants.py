"""Constants for SABER Inspect AI integration.

This module contains constants specific to the Inspect AI integration,
including store keys, timeout values, and other integration-specific identifiers.
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
    MODEL_WRAPPER = "saber_model_wrapper"  # WebSocketTranscriptSyncingModelWrapper for cleanup


# =============================================================================
# Timeout and Duration Constants
# =============================================================================
# Centralizes all timeout values used throughout the SABER sandbox lifecycle
# to improve maintainability and make performance tuning easier.
#
# Timeout Categories:
# - Network operations (HTTP requests, MCP connections)
# - Episode lifecycle (creation, readiness polling)
# - Orchestration coordination (dependency waiting, score synchronization)
# - Cleanup operations (graceful shutdown, termination)
#
# Design Notes:
# - Episode creation timeout (400s) MUST exceed server health check timeout (200s)
# - Dependency coordination timeouts should be generous for multi-step orchestrations
# - Network timeouts are shorter to fail fast on connection issues
# =============================================================================


class SandboxTimeouts:
    """Timeout constants for SABER sandbox lifecycle operations."""

    # Episode Lifecycle Timeouts
    # ---------------------------
    # Episode creation timeout (must exceed server health check timeout of 200s)
    # Timeline: POST /episodes returns immediately, but Docker compose + health checks
    # can take 200s+. Client polls for is_ready=True. Add buffer for other finalization.
    EPISODE_CREATE_SECONDS = 400  # 6.67 minutes
    EPISODE_READY_SECONDS = 400  # Match creation timeout
    MCP_CONNECTION_SECONDS = 300  # 5 minutes default MCP client timeout

    # Orchestration Coordination Timeouts
    # ------------------------------------
    # Time for dependent sample to wait for root sample registration
    # Note: Currently unused parameter in register_dependent_sample()
    ORCHESTRATION_REGISTRATION_SECONDS = 60  # 1 minute

    # Time for dependent sample to wait for dependency to become ready
    # Used in multi-sample orchestrations (e.g., blue team waiting for red team episode)
    ORCHESTRATION_DEPENDENCY_WAIT_SECONDS = 300  # 5 minutes

    # Time to wait for all orchestrated samples to complete scoring
    # Used in score coordination barrier to ensure all siblings finish before returning
    ORCHESTRATION_SCORE_SYNC_SECONDS = 300  # 5 minutes

    # Network Request Timeouts
    # ------------------------
    REST_API_SECONDS = 180  # 3 minutes - REST API request timeout for episode creation
    SESSION_CREATE_SECONDS = 30  # 30 seconds - Session creation HTTP request
    SESSION_TERMINATE_SECONDS = 2.0  # 2 seconds - Session termination (fire-and-forget)
    SESSION_TERMINATE_THREAD_JOIN_SECONDS = 1.5  # 1.5 seconds - Thread join for cleanup

    # MCP Client Configuration
    # ------------------------
    MCP_CLIENT_MAX_RETRIES = 3  # Maximum retry attempts for transient failures

    # Validation
    # ----------
    _SERVER_HEALTH_CHECK_TIMEOUT_SECONDS = 200  # Reference value from server config

    @classmethod
    def validate(cls) -> None:
        """Validate timeout consistency constraints."""
        assert cls.EPISODE_CREATE_SECONDS > cls._SERVER_HEALTH_CHECK_TIMEOUT_SECONDS, (
            f"EPISODE_CREATE_SECONDS ({cls.EPISODE_CREATE_SECONDS}s) must exceed "
            f"server health check timeout ({cls._SERVER_HEALTH_CHECK_TIMEOUT_SECONDS}s)"
        )


# Validate on module load
SandboxTimeouts.validate()


__all__ = [
    "InspectStoreKeys",
    "SandboxTimeouts",
]
