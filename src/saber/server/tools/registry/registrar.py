"""
ToolRegistrar handles tool registration, storage, and analytics.

This component manages the core registry of security tools and provides
analytics and reporting capabilities.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from ..base import DuplicateToolError, SecurityTool, ToolNotFoundError, ValidationResult

logger = logging.getLogger(__name__)


class ToolRegistrar:
    """
    Manages tool registration, storage, and analytics.

    Provides thread-safe registration and retrieval of security tools
    along with analytics and reporting capabilities.
    """

    def __init__(self, domain_filter: Optional[str] = None):
        """
        Initialize the ToolRegistrar.

        Args:
            domain_filter: Optional domain filter for tools
        """
        self._tools: Dict[str, SecurityTool] = {}
        self._domain_filter = domain_filter
        self._lock = threading.RLock()

        logger.info(f"ToolRegistrar initialized for domain: {domain_filter or 'all'}")

    def register_tool(self, tool: SecurityTool) -> None:
        """
        Register a single security tool.

        Args:
            tool: SecurityTool instance to register

        Raises:
            DuplicateToolError: If tool name already exists
            ValueError: If tool validation fails
        """
        with self._lock:
            # Validate tool
            try:
                validation_result = tool.validate()
                if not validation_result.valid:
                    raise ValueError(f"Tool validation failed: {', '.join(validation_result.errors)}")
            except Exception as e:
                logger.error(f"Tool validation error for '{tool.name}': {e}")
                raise ValueError(f"Tool validation failed: {str(e)}")

            # Check for domain filter
            if self._domain_filter and tool.domain != self._domain_filter:
                logger.debug(f"Tool '{tool.name}' domain '{tool.domain}' does not match filter '{self._domain_filter}'")
                return

            # Check for duplicates
            if tool.name in self._tools:
                raise DuplicateToolError(f"Tool '{tool.name}' is already registered")

            # Register the tool
            self._tools[tool.name] = tool

            logger.info(f"Registered tool '{tool.name}' in domain '{tool.domain}'")

    def register_tools(self, tools: List[SecurityTool]) -> None:
        """
        Register multiple security tools.

        Args:
            tools: List of SecurityTool instances to register
        """
        for tool in tools:
            try:
                self.register_tool(tool)
            except Exception as e:
                logger.error(f"Failed to register tool '{tool.name}': {e}")

    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a security tool.

        Args:
            tool_name: Name of the tool to unregister

        Raises:
            ToolNotFoundError: If tool is not found
        """
        with self._lock:
            if tool_name not in self._tools:
                raise ToolNotFoundError(f"Tool '{tool_name}' is not registered")

            del self._tools[tool_name]

            logger.info(f"Unregistered tool '{tool_name}'")

    def get_tool(self, name: str) -> SecurityTool:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            SecurityTool instance

        Raises:
            ToolNotFoundError: If tool is not found
        """
        with self._lock:
            if name not in self._tools:
                raise ToolNotFoundError(f"Tool '{name}' is not registered")
            return self._tools[name]

    def get_tools_by_domain(self, domain: str) -> List[SecurityTool]:
        """
        Get all tools for a specific domain.

        Args:
            domain: Domain name

        Returns:
            List of SecurityTool instances
        """
        with self._lock:
            return [tool for tool in self._tools.values() if tool.domain == domain]

    def list_tools(self) -> List[SecurityTool]:
        """
        List all registered tools.

        Returns:
            List of SecurityTool instances
        """
        with self._lock:
            return list(self._tools.values())

    def get_available_domains(self) -> List[str]:
        """
        Get list of domains that have registered tools.

        Returns:
            List of domain names
        """
        with self._lock:
            return list(set(tool.domain for tool in self._tools.values()))

    # Analytics and Reporting Methods

    def validate_tool(self, tool: SecurityTool) -> ValidationResult:
        """
        Validate a tool configuration.

        Args:
            tool: SecurityTool to validate

        Returns:
            ValidationResult
        """
        try:
            return tool.validate()
        except Exception as e:
            logger.error(f"Tool validation error: {e}")
            # Return a failed validation result
            return ValidationResult(valid=False, errors=[f"Validation failed: {str(e)}"])

    def get_tool_count(self) -> Dict[str, int]:
        """
        Get count of tools by status.

        Returns:
            Dictionary with counts by status
        """
        with self._lock:
            return {
                "total": len(self._tools),
            }

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            stats: Dict[str, Any] = {
                "total_tools": len(self._tools),
                "domains": {},
                "authors": {},
            }

            # Count by domain
            for tool in self._tools.values():
                domain = tool.domain
                stats["domains"][domain] = stats["domains"].get(domain, 0) + 1

                author = tool.author
                stats["authors"][author] = stats["authors"].get(author, 0) + 1

            return stats

    def to_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all enabled tools to MCP format.

        Returns:
            List of tools in MCP format
        """
        with self._lock:
            mcp_tools = []
            for tool in self._tools.values():
                try:
                    mcp_tools.append(tool.to_mcp_tool())
                except Exception as e:
                    logger.error(f"Failed to convert tool '{tool.name}' to MCP format: {e}")
            return mcp_tools

    # Magic Methods

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools

    def __repr__(self) -> str:
        """String representation of the registrar."""
        return f"ToolRegistrar(tools={len(self._tools)}, domain_filter={self._domain_filter})"
