"""REST API endpoint path constants.

This module defines all REST API endpoint paths used for client-server communication.
"""


class APIEndpoints:
    """REST API endpoint path constants."""

    # Session endpoints
    SESSION = "/api/v1/session"
    SESSION_BY_ID = "/api/v1/session/{session_id}"

    # Episode endpoints
    EPISODES = "/api/v1/session/{session_id}/episodes"
    EPISODE_BY_ID = "/api/v1/session/{session_id}/episodes/{episode_id}"
    EPISODE_STATUS = "/api/v1/session/{session_id}/episodes/{episode_id}/status"
    EPISODE_TASK = "/api/v1/session/{session_id}/episodes/{episode_id}/task"
    EPISODE_POLICY = "/api/v1/session/{session_id}/episodes/{episode_id}/policy"
    EPISODE_STEPS = "/api/v1/session/{session_id}/episodes/{episode_id}/steps"
    EPISODE_TRANSCRIPT = "/api/v1/session/{session_id}/episodes/{episode_id}/transcript"
    EPISODE_SUBMISSION = "/api/v1/session/{session_id}/episodes/{episode_id}/submission"
    EPISODE_MESSAGES_INJECT = "/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject"
    EPISODE_SUBMISSION_EVALUATION_CRITERIA = (
        "/api/v1/session/{session_id}/episodes/{episode_id}/submission-evaluation-criteria"
    )
    EPISODE_SUBTASK_EVALUATION_CRITERIA = (
        "/api/v1/session/{session_id}/episodes/{episode_id}/subtask-evaluation-criteria"
    )
    EPISODE_EVALUATION = "/api/v1/session/{session_id}/episodes/{episode_id}/evaluation"
    EPISODE_FILES = "/api/v1/session/{session_id}/episodes/{episode_id}/files"
    EPISODE_SANDBOX_FILE_READ = "/api/v1/session/{session_id}/episodes/{episode_id}/sandbox/files"

    # Evaluation endpoints
    EVALUATION_BY_EPISODE = "/api/v1/session/{session_id}/evaluations/{episode_id}"
    EVALUATIONS_LIST = "/api/v1/session/{session_id}/evaluations"
    EVALUATIONS_SUMMARY = "/api/v1/session/{session_id}/evaluations/summary"
    EVALUATIONS_UPLOAD = "/api/v1/session/{session_id}/evaluations/upload"

    # Task endpoints
    TASKS = "/api/v1/tasks"

    # Template endpoints
    TEMPLATE_CONTENT = "/api/v1/templates/{template_path:path}"

    # Health endpoint
    HEALTH = "/api/v1/health"


__all__ = ["APIEndpoints"]
