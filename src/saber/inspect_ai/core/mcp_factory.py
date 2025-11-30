"""MCP client factory for SABER tool execution.

This module creates and manages MCP clients for SABER tool execution, handling:
- Client creation with headers and retry logic
- Unique naming for cache key management
- Cache cleanup to prevent memory leaks
"""

import asyncio

import anyio
from inspect_ai.tool import Tool, mcp_server_http
from inspect_ai.tool._mcp._local import MCPServerLocal

from saber.logging_config import LogCategory, get_saber_logger
from saber.models.mcp import OrchestrationEnvironment

logger = get_saber_logger(LogCategory.AGENT, __name__)

# SABER MCP Header Constants
HEADER_SESSION_ID = "X-SABER-Session-ID"
HEADER_EPISODE_ID = "X-SABER-Episode-ID"
HEADER_TASK_ID = "X-SABER-Task-ID"
HEADER_ORCHESTRATION_ENV = "X-SABER-Orchestration-Env"


class MCPClientFactory:
    """Creates and manages MCP clients for SABER tool execution.

    Handles:
    - MCP client creation with headers and retry
    - Unique naming for cache key management
    - Cache cleanup to prevent memory leaks
    """

    def __init__(self, mcp_url_base: str, domain_slug: str, timeout: float):
        """Initialize MCP client factory.

        Args:
            mcp_url_base: Base URL for MCP API
            domain_slug: Domain slug for naming
            timeout: Timeout for MCP operations
        """
        self.mcp_url_base = mcp_url_base
        self.domain_slug = domain_slug
        self.timeout = timeout

    async def create_mcp_client(
        self,
        session_id: str,
        episode_id: str,
        task_id: str,
        sample_id: str,
        max_retries: int = 3,
    ) -> Tool:
        """Create MCP client with retry and unique naming.

        Creates an MCP HTTP client with proper headers for SABER tool execution.
        Uses unique naming per sample to prevent cache collisions.

        Args:
            session_id: SABER session ID
            episode_id: SABER episode ID
            task_id: Task identifier
            sample_id: Unique sample identifier
            max_retries: Maximum retry attempts (default: 3)

        Returns:
            MCP Tool instance

        Raises:
            Exception: If client creation fails after all retries
        """
        # Construct MCP headers
        mcp_headers = {
            HEADER_SESSION_ID: session_id,
            HEADER_EPISODE_ID: episode_id,
            HEADER_TASK_ID: task_id,
            HEADER_ORCHESTRATION_ENV: OrchestrationEnvironment.INSPECT.value,
        }

        mcp_url = f"{self.mcp_url_base}/mcp"
        logger.debug(
            f"Creating MCP client for URL: {mcp_url}",
            extra={
                "mcp_url": mcp_url,
                "headers": mcp_headers,
            },
        )

        # Create MCP client with retry logic for transient failures
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Creating MCP client (attempt {attempt + 1}/{max_retries})",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "session_id": session_id,
                        "episode_id": episode_id,
                    },
                )

                # CRITICAL: Name must be unique per sample to prevent MCP client caching!
                mcp_client = mcp_server_http(
                    name=f"SABER {self.domain_slug} Tools - {sample_id}",
                    url=mcp_url,
                    headers=mcp_headers,
                    timeout=self.timeout,
                )

                logger.debug(
                    "MCP client created successfully",
                    extra={
                        "attempt": attempt + 1,
                        "session_id": session_id,
                        "episode_id": episode_id,
                    },
                )

                return mcp_client

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**attempt
                    logger.warning(
                        f"MCP client creation failed, retrying in {wait_time}s",
                        extra={
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "wait_time": wait_time,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "MCP client creation failed after all retries",
                        extra={
                            "attempts": max_retries,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    raise Exception(
                        f"Failed to create MCP client after {max_retries} attempts. "
                        f"Last error: {type(e).__name__}: {str(e)}"
                    ) from e

        # Should never reach here due to raise in last except block
        raise Exception("Unexpected: MCP client creation loop completed without success or error")

    async def cleanup_mcp_cache(self, sample_id: str) -> None:
        """Clean up MCP session cache to prevent memory leaks.

        MEMORY LEAK FIX: With unique server names, each sample creates a cache entry
        that never gets cleaned. For 10,000+ samples, this would cause significant
        memory leaks. This method properly cleans up the cache entry.

        Args:
            sample_id: Unique sample identifier
        """
        try:
            # Build the cache key that would have been used for this sample
            task_id = anyio.get_current_task().id
            server_name = f"SABER {self.domain_slug} Tools - {sample_id}"
            cache_key = f"{task_id}_{server_name}"

            # Remove from cache if it exists
            if cache_key in MCPServerLocal._task_sessions:
                cached_session = MCPServerLocal._task_sessions.pop(cache_key)

                # Properly close the cached session if it's active
                if hasattr(cached_session, "_session") and cached_session._session is not None:
                    await cached_session.__aexit__(None, None, None)

                logger.debug(
                    f"Cleaned up MCP session cache for sample {sample_id}",
                    extra={
                        "cache_key": cache_key,
                        "remaining_cache_entries": len(MCPServerLocal._task_sessions),
                        "event": "mcp_cache_cleanup",
                    },
                )
        except Exception as cache_cleanup_err:
            logger.warning(
                f"Failed to clean up MCP session cache for sample {sample_id}: {cache_cleanup_err}",
                extra={"sample_id": sample_id},
            )
