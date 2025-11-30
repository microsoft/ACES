"""SABER session lifecycle management for Inspect AI sandbox.

This module handles session creation and termination for SABER tasks.
Sessions are created once per task (shared across all samples) and terminated
during task cleanup.

Key features:
- Async session creation via REST API
- Fire-and-forget synchronous termination (safe during shutdown)
- Thread-based termination to avoid event loop cancellation issues
"""

import threading

import aiohttp
import requests  # type: ignore[import-untyped]
from inspect_ai._util.error import PrerequisiteError

from saber.logging_config import LogCategory, get_saber_logger
from saber.models import APIEndpoints, ClientIdentifiers

from ..constants import SandboxTimeouts

logger = get_saber_logger(LogCategory.AGENT, __name__)


class SessionLifecycleManager:
    """Manages SABER session creation and termination.

    Sessions are created once per task and shared across all samples.
    Termination uses fire-and-forget HTTP to handle event loop cancellation
    during shutdown gracefully.
    """

    def __init__(self, rest_base_url: str):
        """Initialize session lifecycle manager.

        Args:
            rest_base_url: Base URL for SABER REST API
        """
        self.rest_base_url = rest_base_url

    async def create_session(self, task_name: str) -> str:
        """Create SABER session for a task (shared across all samples).

        Args:
            task_name: Name of the task

        Returns:
            Session ID

        Raises:
            PrerequisiteError: If session creation fails
        """
        url = f"{self.rest_base_url}{APIEndpoints.SESSION}"
        params = {"client_id": f"{ClientIdentifiers.INSPECT_AI_PREFIX}{task_name}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, params=params, timeout=aiohttp.ClientTimeout(total=SandboxTimeouts.SESSION_CREATE_SECONDS)
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise PrerequisiteError(f"Failed to create SABER session: {response.status} - {text}")
                data = await response.json()
                return str(data["session_id"])

    def terminate_session_sync(self, session_id: str) -> None:
        """Terminate SABER session synchronously with fire-and-forget.

        Uses synchronous requests library with background thread to avoid event loop
        cancellation issues during shutdown. Doesn't wait for response since this is
        called during cleanup.

        Args:
            session_id: Session ID to terminate
        """
        url = f"{self.rest_base_url}{APIEndpoints.SESSION_BY_ID.format(session_id=session_id)}"

        def send_delete_request() -> None:
            """Send DELETE request in background thread."""
            try:
                response = requests.delete(url, timeout=2.0)
                if response.status_code == 200:
                    logger.info(
                        "Session terminated successfully",
                        extra={
                            "session_id": session_id,
                            "event": "session_terminated_sync",
                        },
                    )
                else:
                    logger.debug(
                        f"Session termination returned {response.status_code}",
                        extra={
                            "session_id": session_id,
                            "status_code": response.status_code,
                        },
                    )
            except Exception as e:
                logger.debug(
                    f"Session termination exception (server may still process): {e}",
                    extra={"session_id": session_id},
                )

        # Start in non-daemon thread and wait briefly for completion
        thread = threading.Thread(target=send_delete_request, daemon=False)
        thread.start()
        thread.join(timeout=SandboxTimeouts.SESSION_TERMINATE_THREAD_JOIN_SECONDS)

        logger.info(
            "Session termination request sent",
            extra={
                "session_id": session_id,
                "event": "session_termination_initiated",
            },
        )
