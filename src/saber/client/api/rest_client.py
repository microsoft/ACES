"""
SABER REST Client - Dataset Support

Minimal REST client focused on fetching task data for dataset conversion.
This replaces the full REST client for eval_async integration.
"""

import logging
from typing import Any, Dict, cast

import aiohttp

from ...models import BenchmarkInfo

logger = logging.getLogger(__name__)


class SABERRestClient:
    """
    REST client for SABER client operations.
    """

    def __init__(self, saber_server_url: str, request_timeout: float = 30.0):
        """
        Initialize REST client.

        Args:
            saber_server_url: SABER server URL
            request_timeout: Request timeout in seconds
        """
        self.saber_server_url = saber_server_url.rstrip("/")
        self.request_timeout = request_timeout

    async def get_benchmark_info(self) -> BenchmarkInfo:
        """
        Get complete benchmark information from SABER server.

        Returns:
            BenchmarkInfo object with all benchmark data

        Raises:
            Exception: If request fails or server returns error
        """
        url = f"{self.saber_server_url}/benchmark"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    benchmark_info = BenchmarkInfo(**data)
                    logger.debug(
                        f"Retrieved benchmark info: {benchmark_info.domain} with {benchmark_info.total_tasks} tasks"
                    )
                    return benchmark_info
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get benchmark info: {response.status} - {error_text}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check against the server.

        Returns:
            Health status data

        Raises:
            Exception: If health check fails
        """
        url = f"{self.saber_server_url}/health"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    return cast(Dict[str, Any], result)
                else:
                    raise Exception(f"Health check failed: {response.status}")
