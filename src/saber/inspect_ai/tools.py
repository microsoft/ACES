"""SABER ToolSource implementation for lazy tool provision.

This module provides the ToolSource implementation that exposes SABER's MCP client
as tools to Inspect AI solvers. The ToolSource pattern enables lazy evaluation,
meaning tools are retrieved during solver execution after the sandbox lifecycle
has completed (task_init and sample_init).

Phase 4 of the SABER sandbox integration.
"""

from typing import TYPE_CHECKING

from inspect_ai.tool import Tool, ToolSource
from inspect_ai.util import sandbox

from ..logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)

if TYPE_CHECKING:
    from .saber import SABERSandboxEnvironment  # noqa: F401


class SABERToolSource(ToolSource):
    """ToolSource that provides MCP tools from SABER sandbox environment.

    This class implements the ToolSource protocol to expose SABER's MCP client
    as tools during solver execution. It performs lifecycle-aware validation to
    ensure the sandbox has been properly initialized before tools are accessed.

    Key design points:
    - Lazy evaluation: tools() called during solver execution (after sample_init)
    - Validates sandbox type and MCP client availability
    - Provides clear error messages for lifecycle ordering issues
    - Returns list containing the MCP client Tool for use by solvers

    Usage:
        from inspect_ai.solver import use_tools
        from saber.inspect_ai import saber_tools

        solver = use_tools(saber_tools())

    The ToolSource is evaluated lazily, so saber_tools() can be called during task
    setup, but the actual MCP client is only retrieved when the solver executes
    (after sample_init has created the session/episode).
    """

    def __init__(self, sandbox_name: str = "saber") -> None:
        """Initialize SABER ToolSource.

        Args:
            sandbox_name: Name of the sandbox to retrieve (default: "saber")
        """
        self._sandbox_name = sandbox_name

    async def tools(self) -> list[Tool]:
        """Dynamically retrieve SABER tools from the active sandbox's MCP client.

        This method is called by Inspect AI when tools are needed for agent execution.
        It retrieves the active sandbox and returns its MCP client, which provides
        access to SABER domain tools.

        Returns:
            List containing the MCP client Tool for SABER tool execution

        Raises:
            TypeError: If the sandbox is not a SABERSandboxEnvironment instance
            RuntimeError: If the MCP client is not available (sample_init not completed)
            ProcessLookupError: If no sandbox is available (propagated from sandbox())
        """
        logger.debug("SABERToolSource.tools() called - retrieving MCP client from sandbox")

        # Import here to avoid circular dependency (runtime only)
        if not TYPE_CHECKING:
            from .saber import SABERSandboxEnvironment  # noqa: F811

        # Retrieve active sandbox
        try:
            sb = sandbox(self._sandbox_name)
            logger.debug(f"Retrieved sandbox: {type(sb).__name__}")
        except Exception as e:
            logger.error(f"Failed to retrieve sandbox '{self._sandbox_name}': {e}", exc_info=True)
            raise

        # Unwrap proxy to get actual sandbox instance
        # Inspect AI wraps sandboxes in SandboxEnvironmentProxy
        actual_sandbox = sb
        if hasattr(sb, "_sandbox"):
            actual_sandbox = sb._sandbox
            logger.debug(f"Unwrapped proxy, actual sandbox: {type(actual_sandbox).__name__}")

        # Validate sandbox type
        if not isinstance(actual_sandbox, SABERSandboxEnvironment):
            raise TypeError(
                f"Expected SABERSandboxEnvironment, got {type(actual_sandbox).__name__}. "
                f"The sandbox '{self._sandbox_name}' is not a SABER sandbox. "
                "Ensure you configured the task with a SABER sandbox using "
                "sandbox='saber' or a custom SABER sandbox configuration."
            )

        logger.debug("Validated sandbox is SABERSandboxEnvironment")

        # Validate MCP client availability
        if actual_sandbox._mcp_client is None:
            raise RuntimeError(
                "SABER MCP client not available. The sandbox lifecycle has not completed. "
                "Ensure sample_init() has been called and completed successfully before "
                "accessing tools. This typically means:\n"
                "1. The task must provide 'task_id' in sample metadata\n"
                "2. sample_init() must complete session/episode creation\n"
                "3. Tools are accessed during solver execution (not during setup)\n\n"
                "If you're seeing this error, check that your dataset provides task_id "
                "and that the SABER REST API is accessible."
            )

        logger.debug(
            f"MCP client available: {type(actual_sandbox._mcp_client).__name__}",
            extra={
                "session_id": actual_sandbox._session_id,
                "episode_id": actual_sandbox._episode_id,
                "task_id": actual_sandbox._task_id,
            },
        )

        # Call .tools() on the MCP server to get actual tools
        # The MCP server will enter its context automatically when needed
        logger.debug("Calling .tools() on MCP server to get domain tools")
        try:
            mcp_tools = await actual_sandbox._mcp_client.tools()
            logger.info(
                f"MCP server returned {len(mcp_tools)} tools",
                extra={
                    "tool_count": len(mcp_tools),
                    "tool_types": [type(t).__name__ for t in mcp_tools],
                    "tool_names": [getattr(t, "name", getattr(t, "__name__", "unnamed")) for t in mcp_tools],
                },
            )
            return list(mcp_tools)
        except Exception as e:
            logger.error(
                f"❌ Error calling .tools() on MCP server: {e}", extra={"error_type": type(e).__name__}, exc_info=True
            )
            raise


def saber_tools(sandbox_name: str = "saber") -> SABERToolSource:
    """Create a ToolSource for SABER tools.

    This is a convenience function that returns a SABERToolSource instance for use
    in solver tool stacks. The ToolSource pattern enables lazy evaluation, meaning
    the MCP client is only retrieved during solver execution (after sample_init).

    Args:
        sandbox_name: Name of the SABER sandbox to retrieve tools from (default: "saber")

    Returns:
        SABERToolSource instance that provides SABER MCP tools

    Example:
        from inspect_ai import Task
        from inspect_ai.solver import generate, use_tools
        from saber.inspect_ai import saber_tools

        task = Task(
            dataset=dataset,
            sandbox="saber",
            solver=[
                use_tools(saber_tools()),
                generate()
            ]
        )

    Note:
        The ToolSource is evaluated lazily during solver execution, so this function
        can be called during task setup without triggering sandbox initialization.
    """
    return SABERToolSource(sandbox_name=sandbox_name)
