"""
ToolDiscoverer handles auto-discovery of security tools.

This component manages the discovery and instantiation of tools from
domain packages and specific tool paths.
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Any, List, Optional

from ..base import SecurityTool, ToolExecutor
from .registrar import ToolRegistrar

logger = logging.getLogger(__name__)


class ToolDiscoverer:
    """
    Handles auto-discovery and registration of security tools.

    Provides methods to discover tools from domain packages or specific
    tool paths and register them with a ToolRegistrar.
    """

    def __init__(self, registrar: ToolRegistrar):
        """
        Initialize the ToolDiscoverer.

        Args:
            registrar: ToolRegistrar instance to register discovered tools
        """
        self._registrar = registrar
        logger.info("ToolDiscoverer initialized")

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

    def auto_discover_specific_tools(self, tool_specs: List[str]) -> None:
        """
        Auto-discover and register specific tools by their paths.

        Args:
            tool_specs: List of tool specifications like:
                       - "malware.static_analysis.File"
                       - "malware.static_analysis.Strings"
        """
        for tool_spec in tool_specs:
            try:
                self.discover_tool_by_path(tool_spec)
            except Exception as e:
                logger.error(f"Failed to discover tool '{tool_spec}': {e}")

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
        try:
            domain, module_name, class_name = self._resolve_tool_path(tool_path)

            # Import specific module
            full_module_path = f"saber.server.tools.domains.{domain}.{module_name}"
            module = importlib.import_module(full_module_path)

            # Get specific class
            if not hasattr(module, class_name):
                logger.error(f"Class '{class_name}' not found in module {full_module_path}")
                raise AttributeError(f"Class '{class_name}' not found in {full_module_path}")

            tool_class = getattr(module, class_name)

            # Validate it's a tool
            if (
                not inspect.isclass(tool_class)
                or not issubclass(tool_class, ToolExecutor)
                or not hasattr(tool_class, "_security_tool_metadata")
            ):
                logger.error(f"'{class_name}' is not a valid security tool class")
                raise ValueError(f"'{class_name}' is not a valid security tool")

            # Create and register the tool
            self._create_tool_from_class(tool_class, domain)
            logger.info(f"Discovered and registered tool '{tool_path}'")

        except Exception as e:
            logger.error(f"Failed to discover tool '{tool_path}': {e}")
            raise

    def _resolve_tool_path(self, tool_path: str) -> tuple[str, str, str]:
        """
        Resolve tool path to domain, module, and class.

        Args:
            tool_path: Tool path like "malware.static_analysis.File"

        Returns:
            Tuple of (domain, module, class_name)

        Raises:
            ValueError: If tool path format is invalid
        """
        parts = tool_path.split(".")
        if len(parts) != 3:
            logger.error(f"Invalid tool path format: {tool_path}")
            raise ValueError(f"Tool path must be 'domain.module.class', got: {tool_path}")

        domain, module, class_name = parts

        # For now, only support malware domain
        if domain != "malware":
            logger.error(f"Unsupported domain in tool path: {domain}")
            raise ValueError(f"Currently only 'malware' domain is supported, got: {domain}")

        return domain, module, class_name

    def _create_tool_from_class(self, tool_class: type, domain: str) -> None:
        """
        Create and register a tool from a class.

        Args:
            tool_class: Tool executor class
            domain: Domain name

        Raises:
            Exception: If tool creation or registration fails
        """
        try:
            metadata = getattr(tool_class, "_security_tool_metadata")
            executor = tool_class()

            # Extract parameters from executor if it has a get_parameters method
            parameters = {}
            if hasattr(executor, "get_parameters"):
                try:
                    parameters = executor.get_parameters()
                except Exception as e:
                    logger.warning(f"Failed to get parameters from executor: {e}")

            # Create SecurityTool
            tool = SecurityTool(
                name=metadata["name"],
                domain=metadata["domain"],
                description=metadata["description"],
                author=metadata["author"],
                parameters=parameters,
                executor=executor,
            )

            self._registrar.register_tool(tool)
        except Exception as e:
            logger.error(f"Failed to create tool from class {tool_class.__name__}: {e}")
            raise

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
                        try:
                            parameters = executor.get_parameters()
                        except Exception as e:
                            logger.warning(f"Failed to get parameters from executor '{name}': {e}")

                    # Create SecurityTool
                    tool = SecurityTool(
                        name=metadata["name"],
                        domain=metadata["domain"],
                        description=metadata["description"],
                        author=metadata["author"],
                        parameters=parameters,
                        executor=executor,
                    )

                    self._registrar.register_tool(tool)

                except Exception as e:
                    logger.error(f"Failed to create tool from class '{name}': {e}")
