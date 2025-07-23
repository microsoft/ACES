"""
ToolRegistry implementation for domain-organized security tools.

This module provides the main ToolRegistry class that manages registration,
discovery, validation, and execution of security tools across different domains.
"""

import asyncio
import importlib
import inspect
import logging
import pkgutil
import threading
from typing import Any, Dict, List, Optional, Set

import yaml

from ..tools.base import (
    DuplicateToolError,
    SecurityTool,
    ToolExecutionError,
    ToolExecutor,
    ToolNotFoundError,
    ToolResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing security tools across different domains.

    Provides thread-safe registration, auto-discovery, validation, and execution
    of security tools organized by domain (malware, threat_investigation, forensics).
    """

    def __init__(
        self, domain: Optional[str] = None, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None
    ):
        """
        Initialize the ToolRegistry.

        Args:
            domain: Optional domain filter for tools
            config: Optional configuration dictionary
            config_file: Optional path to YAML configuration file
        """
        self._tools: Dict[str, SecurityTool] = {}
        self._domain_filter = domain

        # Load configuration from file if provided, otherwise use provided config or empty dict
        if config_file:
            self._config = self._load_config_from_file(config_file)
        else:
            self._config = config or {}

        self._lock = threading.RLock()
        self._enabled_tools: Set[str] = set()
        self._disabled_tools: Set[str] = set()

        # Execution settings from configuration
        tools_config = self._config.get("tools", {})
        execution_config = tools_config.get("execution", {})
        self._default_timeout = execution_config.get("timeout", 300.0)
        self._max_concurrent = execution_config.get("max_concurrent", 10)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        logger.info(f"ToolRegistry initialized for domain: {domain or 'all'}")

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
            validation_result = tool.validate()
            if not validation_result.valid:
                raise ValueError(f"Tool validation failed: {', '.join(validation_result.errors)}")

            # Check for domain filter
            if self._domain_filter and tool.domain != self._domain_filter:
                logger.debug(f"Tool '{tool.name}' domain '{tool.domain}' does not match filter '{self._domain_filter}'")
                return

            # Check for duplicates
            if tool.name in self._tools:
                raise DuplicateToolError(f"Tool '{tool.name}' is already registered")

            # Register the tool
            self._tools[tool.name] = tool
            if tool.enabled:
                self._enabled_tools.add(tool.name)

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
            self._enabled_tools.discard(tool_name)
            self._disabled_tools.discard(tool_name)

            logger.info(f"Unregistered tool '{tool_name}'")

    def auto_discover_domain_tools(self, domain: Optional[str] = None) -> None:
        """
        Auto-discover and register tools from domain packages.

        Args:
            domain: Specific domain to discover, or None for all domains
        """
        domains_to_scan = [domain] if domain else ["malware", "threat_investigation", "forensics"]

        for domain_name in domains_to_scan:
            try:
                self._discover_tools_in_domain(domain_name)
            except Exception as e:
                logger.error(f"Failed to discover tools in domain '{domain_name}': {e}")

    def _discover_tools_in_domain(self, domain: str) -> None:
        """
        Discover tools in a specific domain package.

        Args:
            domain: Domain name to scan
        """
        try:
            # Import the domain package
            domain_package = f"saber.server.tools.domains.{domain}"
            module = importlib.import_module(domain_package)

            # Walk through all modules in the domain package
            if hasattr(module, "__path__"):
                for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
                    try:
                        full_modname = f"{domain_package}.{modname}"
                        submodule = importlib.import_module(full_modname)
                        self._extract_tools_from_module(submodule, domain)
                    except Exception as e:
                        logger.error(f"Error importing module '{full_modname}': {e}")

        except ImportError as e:
            logger.warning(f"Domain package '{domain}' not found: {e}")

    def _extract_tools_from_module(self, module: Any, domain: str) -> None:
        """
        Extract security tools from a module.

        Args:
            module: Python module to scan
            domain: Domain name for the tools
        """
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, ToolExecutor)
                and obj is not ToolExecutor
                and hasattr(obj, "_security_tool_metadata")
            ):

                try:
                    # Create tool instance
                    metadata = obj._security_tool_metadata
                    executor = obj()

                    # Extract parameters from executor if it has a get_parameters method
                    parameters = {}
                    if hasattr(executor, "get_parameters"):
                        parameters = executor.get_parameters()

                    # Create SecurityTool
                    tool = SecurityTool(
                        name=metadata["name"],
                        domain=metadata["domain"],
                        description=metadata["description"],
                        author=metadata["author"],
                        parameters=parameters,
                        executor=executor,
                    )

                    self.register_tool(tool)

                except Exception as e:
                    logger.error(f"Failed to create tool from class '{name}': {e}")

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

    def list_tools(self, enabled_only: bool = True) -> List[SecurityTool]:
        """
        List all registered tools.

        Args:
            enabled_only: If True, only return enabled tools

        Returns:
            List of SecurityTool instances
        """
        with self._lock:
            if enabled_only:
                return [tool for tool in self._tools.values() if tool.enabled]
            return list(self._tools.values())

    def get_available_domains(self) -> List[str]:
        """
        Get list of domains that have registered tools.

        Returns:
            List of domain names
        """
        with self._lock:
            return list(set(tool.domain for tool in self._tools.values()))

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
        tool = self.get_tool(name)

        if not tool.enabled:
            raise ToolExecutionError(f"Tool '{name}' is disabled")

        # Use semaphore to limit concurrent executions
        async with self._semaphore:
            try:
                # Set up execution context
                exec_context = context or {}
                exec_context.setdefault("tool_name", name)
                exec_context.setdefault("domain", tool.domain)

                # Execute with timeout
                timeout = tool.executor.get_timeout() or self._default_timeout
                result = await asyncio.wait_for(tool.execute(parameters, exec_context), timeout=timeout)

                return result

            except asyncio.TimeoutError:
                raise ToolExecutionError(f"Tool '{name}' execution timed out after {timeout}s")
            except Exception as e:
                logger.error(f"Tool execution error for '{name}': {e}")
                raise ToolExecutionError(f"Tool '{name}' execution failed: {str(e)}")

    def validate_tool(self, tool: SecurityTool) -> ValidationResult:
        """
        Validate a tool configuration.

        Args:
            tool: SecurityTool to validate

        Returns:
            ValidationResult
        """
        return tool.validate()

    def enable_tool(self, tool_name: str) -> None:
        """
        Enable a tool for execution.

        Args:
            tool_name: Name of the tool to enable

        Raises:
            ToolNotFoundError: If tool is not found
        """
        with self._lock:
            if tool_name not in self._tools:
                raise ToolNotFoundError(f"Tool '{tool_name}' is not registered")

            self._tools[tool_name].enabled = True
            self._enabled_tools.add(tool_name)
            self._disabled_tools.discard(tool_name)

            logger.info(f"Enabled tool '{tool_name}'")

    def disable_tool(self, tool_name: str) -> None:
        """
        Disable a tool from execution.

        Args:
            tool_name: Name of the tool to disable

        Raises:
            ToolNotFoundError: If tool is not found
        """
        with self._lock:
            if tool_name not in self._tools:
                raise ToolNotFoundError(f"Tool '{tool_name}' is not registered")

            self._tools[tool_name].enabled = False
            self._enabled_tools.discard(tool_name)
            self._disabled_tools.add(tool_name)

            logger.info(f"Disabled tool '{tool_name}'")

    def get_tool_count(self) -> Dict[str, int]:
        """
        Get count of tools by status.

        Returns:
            Dictionary with counts by status
        """
        with self._lock:
            return {
                "total": len(self._tools),
                "enabled": len(self._enabled_tools),
                "disabled": len(self._disabled_tools),
            }

    def to_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all enabled tools to MCP format.

        Returns:
            List of tools in MCP format
        """
        with self._lock:
            return [tool.to_mcp_tool() for tool in self._tools.values() if tool.enabled]

    def load_configuration(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            self._config.update(config)

            # Apply tool-specific configurations
            if "tools" in config:
                tool_configs = config["tools"]

                # Enable/disable tools based on configuration
                if "enabled" in tool_configs:
                    for tool_name in tool_configs["enabled"]:
                        try:
                            self.enable_tool(tool_name)
                        except ToolNotFoundError:
                            logger.warning(f"Tool '{tool_name}' not found for enabling")

                if "disabled" in tool_configs:
                    for tool_name in tool_configs["disabled"]:
                        try:
                            self.disable_tool(tool_name)
                        except ToolNotFoundError:
                            logger.warning(f"Tool '{tool_name}' not found for disabling")

            logger.info(f"Loaded configuration from '{config_path}'")

        except Exception as e:
            logger.error(f"Failed to load configuration from '{config_path}': {e}")

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            stats: Dict[str, Any] = {
                "total_tools": len(self._tools),
                "enabled_tools": len(self._enabled_tools),
                "disabled_tools": len(self._disabled_tools),
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

    def _load_config_from_file(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        from pathlib import Path

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {config_file}: {e}")
            raise

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ToolRegistry(tools={len(self._tools)}, domain_filter={self._domain_filter})"
