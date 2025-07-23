"""
ToolRegistry implementation using modular components.

This is the main orchestration layer that coordinates between the specialized
components: ToolRegistrar, ToolDiscoverer, ToolExecutionManager, and RegistryConfiguration.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import SecurityTool, ToolResult, ValidationResult
from .registry import RegistryConfiguration, ToolDiscoverer, ToolExecutionManager, ToolRegistrar

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Orchestration layer for domain-organized security tools.

    Coordinates between specialized components to provide a unified interface
    for tool registration, discovery, validation, and execution.
    """

    def __init__(
        self, domain: Optional[str] = None, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None
    ):
        """
        Initialize the ToolRegistry with modular components.

        Args:
            domain: Optional domain filter for tools
            config: Optional configuration dictionary
            config_file: Optional path to YAML configuration file
        """
        self._configuration = RegistryConfiguration(config, config_file)

        self._registrar = ToolRegistrar(domain)
        self._discoverer = ToolDiscoverer(self._registrar)

        execution_config = self._configuration.get_execution_config()
        default_timeout = execution_config.get("timeout", 300.0)
        max_concurrent = execution_config.get("max_concurrent", 10)
        self._execution_manager = ToolExecutionManager(self._registrar, default_timeout, max_concurrent)

        self._auto_load_tools_from_config()

        logger.info(f"ToolRegistry initialized for domain: {domain or 'all'}")

    def _auto_load_tools_from_config(self) -> None:
        """Auto-load tools based on configuration settings."""
        try:
            # Load entire domains if specified
            domains = self._configuration.get_domains_to_load()
            for domain in domains:
                try:
                    self._discoverer.auto_discover_domain_tools(domain)
                except Exception as e:
                    logger.error(f"Failed to auto-discover domain '{domain}': {e}")

            # Load specific tools if specified
            specific_tools = self._configuration.get_specific_tools_to_load()
            if specific_tools:
                try:
                    self._discoverer.auto_discover_specific_tools(specific_tools)
                except Exception as e:
                    logger.error(f"Failed to auto-discover specific tools: {e}")

        except Exception as e:
            logger.error(f"Failed to auto-load tools from configuration: {e}")

    def register_tool(self, tool: SecurityTool) -> None:
        """
        Register a single security tool.

        Args:
            tool: SecurityTool instance to register

        Raises:
            DuplicateToolError: If tool name already exists
            ValueError: If tool validation fails
        """
        self._registrar.register_tool(tool)

    def register_tools(self, tools: List[SecurityTool]) -> None:
        """
        Register multiple security tools.

        Args:
            tools: List of SecurityTool instances to register
        """
        self._registrar.register_tools(tools)

    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a security tool.

        Args:
            tool_name: Name of the tool to unregister

        Raises:
            ToolNotFoundError: If tool is not found
        """
        self._registrar.unregister_tool(tool_name)

    def auto_discover_domain_tools(self, domain: Optional[str] = None) -> None:
        """
        Auto-discover and register tools from domain packages.

        Args:
            domain: Specific domain to discover, or None for all domains
        """
        self._discoverer.auto_discover_domain_tools(domain)

    def auto_discover_specific_tools(self, tool_specs: List[str]) -> None:
        """
        Auto-discover and register specific tools by their paths.

        Args:
            tool_specs: List of tool specifications like:
                       - "malware.static_analysis.File"
                       - "malware.static_analysis.Strings"
        """
        self._discoverer.auto_discover_specific_tools(tool_specs)

    def discover_tool_by_path(self, tool_path: str) -> None:
        """
        Discover and register a single tool by its module path.

        Args:
            tool_path: Tool path like "malware.static_analysis.File"

        Raises:
            ValueError: If tool path is invalid
            ImportError: If module cannot be imported
            AttributeError: If class not found in module
        """
        self._discoverer.discover_tool_by_path(tool_path)

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
        return self._registrar.get_tool(name)

    def get_tools_by_domain(self, domain: str) -> List[SecurityTool]:
        """
        Get all tools for a specific domain.

        Args:
            domain: Domain name

        Returns:
            List of SecurityTool instances
        """
        return self._registrar.get_tools_by_domain(domain)

    def list_tools(self) -> List[SecurityTool]:
        """
        List all registered tools.

        Returns:
            List of SecurityTool instances
        """
        return self._registrar.list_tools()

    def get_available_domains(self) -> List[str]:
        """
        Get list of domains that have registered tools.

        Returns:
            List of domain names
        """
        return self._registrar.get_available_domains()

    async def execute_tool(
        self, name: str, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute a tool with the given parameters.

        Args:
            name: Tool name
            parameters: Tool parameters
            context: Optional execution context

        Returns:
            ToolResult with execution results

        Raises:
            ToolNotFoundError: If tool is not found
            ToolExecutionError: If execution fails
        """
        return await self._execution_manager.execute_tool(name, parameters, context)

    def validate_tool(self, tool: SecurityTool) -> ValidationResult:
        """
        Validate a tool configuration.

        Args:
            tool: SecurityTool to validate

        Returns:
            ValidationResult
        """
        return self._registrar.validate_tool(tool)

    def get_tool_count(self) -> Dict[str, int]:
        """
        Get count of tools by status.

        Returns:
            Dictionary with counts by status
        """
        return self._registrar.get_tool_count()

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        return self._registrar.get_registry_stats()

    def to_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all enabled tools to MCP format.

        Returns:
            List of tools in MCP format
        """
        return self._registrar.to_mcp_tools()

    def load_configuration(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        self._configuration.load_configuration(config_path)

        # Update execution manager with new settings
        execution_config = self._configuration.get_execution_config()
        default_timeout_val = execution_config.get("timeout")
        max_concurrent_val = execution_config.get("max_concurrent")

        # Convert to proper types if present
        default_timeout: Optional[float] = float(default_timeout_val) if default_timeout_val is not None else None
        max_concurrent: Optional[int] = int(max_concurrent_val) if max_concurrent_val is not None else None

        self._execution_manager.update_execution_settings(default_timeout, max_concurrent)

        # Re-run auto-discovery if new tools specified
        self._auto_load_tools_from_config()

    def get_configuration(self) -> RegistryConfiguration:
        """
        Get the configuration manager.

        Returns:
            RegistryConfiguration instance
        """
        return self._configuration

    def get_registrar(self) -> ToolRegistrar:
        """
        Get the tool registrar component.

        Returns:
            ToolRegistrar instance
        """
        return self._registrar

    def get_discoverer(self) -> ToolDiscoverer:
        """
        Get the tool discoverer component.

        Returns:
            ToolDiscoverer instance
        """
        return self._discoverer

    def get_execution_manager(self) -> ToolExecutionManager:
        """
        Get the execution manager component.

        Returns:
            ToolExecutionManager instance
        """
        return self._execution_manager

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._registrar)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._registrar

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ToolRegistry(tools={len(self._registrar)}, components=[registrar, discoverer, execution_manager, \
            configuration])"
