"""SABER episode lifecycle management for Inspect AI sandbox.

This module handles episode creation and cleanup with robust retry logic.
Episodes are created per-sample and cleaned up after sample execution.

Key features:
- Episode creation with health check waiting
- Robust cleanup with retry and verification
- Fire-and-forget HTTP for interrupted cleanup (shutdown safety)
- Async cleanup for happy path with submission support
"""

import asyncio
import threading
import time

import requests
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.util import store

from saber.client.client_session import ClientSessionManager
from saber.debug_logging import log_episode_end_complete, log_episode_end_request
from saber.logging_config import LogCategory, get_saber_logger
from saber.models import APIEndpoints
from saber.server.episodes.constants import EpisodeTerminationReason

from ..constants import InspectStoreKeys, SandboxTimeouts

logger = get_saber_logger(LogCategory.AGENT, __name__)


class SandboxError(Exception):
    """Exception raised for SABER sandbox lifecycle errors."""

    pass


class EpisodeLifecycleManager:
    """Manages SABER episode creation and cleanup.

    Handles:
    - Episode creation with timeout and health check waiting
    - Happy path cleanup (async with submission support)
    - Interrupted cleanup (fire-and-forget with retry)
    - Episode verification after deletion
    """

    def __init__(self, session_manager: ClientSessionManager):
        """Initialize episode lifecycle manager.

        Args:
            session_manager: Client session manager for API calls
        """
        self.session_manager = session_manager

    async def create_episode(self, session_id: str, task_id: str) -> str:
        """Create SABER episode via REST API and wait for it to be ready.

        Episode creation is asynchronous - the POST returns immediately with an episode ID,
        but Docker container setup can take several minutes. We use the ClientSessionManager's
        create_episode_and_wait method which has proper timeout and polling logic.

        Timeline:
        - Episode POST returns immediately with episode_id
        - Docker compose starts (serialized by global lock)
        - Background finalization: health checks (200s timeout), prompt generation, etc.
        - Client polls for is_ready=True

        Timeout must exceed server-side health check timeout (200s) plus buffer for other
        finalization steps. See SandboxTimeouts.EPISODE_CREATE_SECONDS for configured value.

        Args:
            session_id: SABER session ID
            task_id: Task identifier for the episode

        Returns:
            Episode ID

        Raises:
            PrerequisiteError: If episode creation fails or times out
        """
        if self.session_manager is None:
            raise SandboxError("Session manager not initialized")

        try:
            response = await self.session_manager.create_episode_and_wait(
                session_id=session_id,
                task_id=task_id,
                timeout_seconds=SandboxTimeouts.EPISODE_CREATE_SECONDS,
            )
            episode_id: str = response.episode_id
            return episode_id
        except TimeoutError as e:
            # Provide clear timeout diagnostic
            error_msg = (
                f"Episode creation timed out after {SandboxTimeouts.EPISODE_CREATE_SECONDS}s for task '{task_id}'. "
                f"This usually indicates Docker health checks are taking too long or containers failed to start. "
                f"Check server logs for details. Error: {str(e)}"
            )
            logger.error(
                error_msg,
                extra={
                    "event": "episode_creation_timeout",
                    "task_id": task_id,
                    "session_id": session_id,
                    "timeout_seconds": SandboxTimeouts.EPISODE_CREATE_SECONDS,
                },
            )
            raise PrerequisiteError(error_msg) from e
        except Exception as e:
            # Provide context for other errors
            error_msg = f"Failed to create episode for task '{task_id}': {type(e).__name__}: {str(e)}"
            logger.error(
                error_msg,
                extra={
                    "event": "episode_creation_failed",
                    "task_id": task_id,
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                },
            )
            raise PrerequisiteError(error_msg) from e

    async def cleanup_episode(
        self,
        session_id: str,
        episode_id: str,
        interrupted: bool = False,
        submission: dict | None = None,
    ) -> None:
        """Best-effort cleanup of episode with robust retry logic.

        Unified episode ending for all paths - this is the single place where episodes end.
        Note: Session is NOT terminated here - it's shared across all samples and cleaned up in task_cleanup.

        Args:
            session_id: SABER session ID
            episode_id: Episode ID to cleanup
            interrupted: Whether this is an interrupted (non-happy path) cleanup
            submission: Optional submission data (for happy path after scorer)
        """
        # End episode if it exists (both happy and interrupted paths)
        if episode_id and self.session_manager:
            try:
                # Determine reason based on interrupted flag
                reason = EpisodeTerminationReason.INTERRUPTED if interrupted else EpisodeTerminationReason.COMPLETED

                logger.info(
                    f"Ending episode (reason={reason})",
                    extra={
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "reason": reason,
                        "event": "sandbox_cleanup_end_episode",
                    },
                )

                # Log episode end request
                log_episode_end_request(
                    reason=reason,
                    interrupted=interrupted,
                )

                # For interrupted cleanup, use fire-and-forget HTTP to avoid event loop shutdown issues
                if interrupted:
                    # Use synchronous requests library for reliability during shutdown
                    # NOW WITH RETRY for better reliability
                    base_url = self.session_manager.base_url
                    url = f"{base_url}{APIEndpoints.EPISODE_BY_ID.format(session_id=session_id, episode_id=episode_id)}"
                    verify_url = (
                        f"{base_url}{APIEndpoints.EPISODE_STATUS.format(session_id=session_id, episode_id=episode_id)}"
                    )
                    params = {
                        "reason": reason,
                        "cascade_end_attached_episodes": "false",
                    }

                    def send_delete_request_with_retry() -> bool:
                        """Send delete request with retry logic in background thread."""
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                # Longer timeout than before - 5s instead of 2s
                                response = requests.delete(url, params=params, timeout=5.0)

                                if response.status_code == 200:
                                    # Verify episode actually ended
                                    time.sleep(0.5)  # Brief wait for server processing
                                    verify_response = requests.get(verify_url, timeout=2.0)
                                    if verify_response.status_code == 404:
                                        logger.info(
                                            f"Episode end verified successfully (attempt {attempt + 1}/{max_retries})",
                                            extra={
                                                "session_id": session_id,
                                                "episode_id": episode_id,
                                                "reason": reason,
                                            },
                                        )
                                        return True
                                    else:
                                        logger.debug(
                                            f"Episode ended but verification inconclusive "
                                            f"(status {verify_response.status_code})",
                                            extra={
                                                "session_id": session_id,
                                                "episode_id": episode_id,
                                                "verify_status": verify_response.status_code,
                                            },
                                        )
                                        return True  # Accept success even if verification unclear

                                elif response.status_code == 400:
                                    # Might be already ended
                                    logger.info(
                                        f"Episode already ended (attempt {attempt + 1}/{max_retries})",
                                        extra={
                                            "session_id": session_id,
                                            "episode_id": episode_id,
                                        },
                                    )
                                    return True

                                else:
                                    logger.debug(
                                        f"Episode end request returned {response.status_code} "
                                        f"(attempt {attempt + 1}/{max_retries})",
                                        extra={
                                            "session_id": session_id,
                                            "episode_id": episode_id,
                                            "status_code": response.status_code,
                                            "attempt": attempt + 1,
                                        },
                                    )

                            except Exception as e:
                                logger.debug(
                                    f"Episode end request attempt {attempt + 1}/{max_retries} failed: {e}",
                                    extra={
                                        "session_id": session_id,
                                        "episode_id": episode_id,
                                        "attempt": attempt + 1,
                                        "error": str(e),
                                    },
                                )

                            # Retry with backoff
                            if attempt < max_retries - 1:
                                backoff = 1.0 * (attempt + 1)
                                time.sleep(backoff)

                        # All retries exhausted
                        logger.warning(
                            f"Episode end failed after {max_retries} attempts (interrupted cleanup)",
                            extra={
                                "session_id": session_id,
                                "episode_id": episode_id,
                            },
                        )
                        return False

                    # Start in non-daemon thread and wait for completion
                    # Non-daemon prevents process exit until thread completes
                    # Increased timeout to allow for retries: 3 attempts * ~6s each = 18s max
                    thread = threading.Thread(target=send_delete_request_with_retry, daemon=False)
                    thread.start()
                    thread.join(timeout=20.0)  # Wait up to 20s for retries to complete

                    logger.info(
                        f"Episode end request initiated with retry in background (reason={reason})",
                        extra={
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "reason": reason,
                            "note": "Used retry logic with up to 3 attempts",
                        },
                    )
                else:
                    # Happy path - use async session manager WITH RETRY
                    # Get submission from store if not provided
                    if submission is None:
                        try:
                            task_store = store()
                            # Scorer may have stored the submission for us
                            submission = task_store.get(InspectStoreKeys.EPISODE_SUBMISSION)
                        except Exception:
                            # Store might not be available or submission not set - that's ok
                            pass

                    # End episode with submission - USING ROBUST RETRY
                    success = await self.session_manager.end_episode_with_retry(
                        session_id=session_id,
                        episode_id=episode_id,
                        reason=reason,
                        result=submission,
                        cascade_end_attached_episodes=False,
                        max_retries=5,  # More retries for happy path
                        initial_backoff=1.0,
                        max_backoff=30.0,
                    )

                    if success:
                        logger.info(
                            f"Episode ended successfully with verification (reason={reason})",
                            extra={
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "reason": reason,
                            },
                        )

                        # Log successful episode end
                        log_episode_end_complete(success=True)
                    else:
                        logger.error(
                            f"Episode ending failed after retries - may be orphaned (reason={reason})",
                            extra={
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "reason": reason,
                                "event": "episode_end_failed_with_retry",
                            },
                        )

                        # Log failed episode end
                        log_episode_end_complete(success=False)

            except (asyncio.CancelledError, asyncio.TimeoutError):
                # Cancellation/timeout during shutdown - log but don't raise
                # The server will clean up the episode eventually
                reason_str = reason if "reason" in locals() else "unknown"
                logger.info(
                    f"Episode ending cancelled/timed out during shutdown (reason={reason_str})",
                    extra={
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "event": "episode_end_cancelled",
                        "note": "Server will clean up episode eventually",
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Failed to end episode with retry (server will eventually clean up): {e}",
                    extra={
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "reason": reason if "reason" in locals() else "unknown",
                    },
                    exc_info=True,
                )
