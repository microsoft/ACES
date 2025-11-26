"""SABER Server Health Check with Retry Logic.

This module provides health check functionality for SABER server instances,
with exponential backoff and retry logic to handle server startup delays.
"""

import asyncio

import aiohttp
from inspect_ai._util.error import PrerequisiteError

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


async def wait_for_server_health(
    rest_url: str,
    max_retries: int = 30,
    backoff: float = 2.0,
) -> None:
    """Wait for SABER server to become healthy with retry/backoff.

    Polls the health endpoint until successful or max retries exceeded.

    Args:
        rest_url: Base URL for REST API
        max_retries: Maximum number of retry attempts (default: 30)
        backoff: Backoff multiplier between retries (default: 2.0s)

    Raises:
        PrerequisiteError: If server doesn't become healthy within retries
    """
    health_url = f"{rest_url}/api/v1/health"

    for attempt in range(1, max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(
                            f"SABER server health check passed (attempt {attempt})",
                            extra={
                                "rest_url": rest_url,
                                "attempt": attempt,
                                "domain": data.get("domain", "unknown"),
                            },
                        )
                        return
                    else:
                        logger.warning(
                            f"Health check returned {response.status} (attempt {attempt}/{max_retries})",
                            extra={"rest_url": rest_url, "attempt": attempt},
                        )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(
                f"Health check failed (attempt {attempt}/{max_retries}): {e}",
                extra={"rest_url": rest_url, "attempt": attempt},
            )

        if attempt < max_retries:
            await asyncio.sleep(backoff)

    # Max retries exceeded
    raise PrerequisiteError(
        f"SABER server health check failed after {max_retries} attempts.\n\n"
        f"REST URL: {rest_url}\n\n"
        "The server may have failed to start or is not responding.\n"
        "Check server logs for details:\n"
        "  docker logs <container_name>\n"
        "  docker compose -p saber-<domain> logs server"
    )
